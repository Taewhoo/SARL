"""
Tool-calling search environment for SARL training.

Unlike SearchEnv (search-R1 style), the model is given two explicit tools and
returns structured tool calls instead of XML <search>/<answer> tags:

  - search(query: str)     — search the knowledge base
  - browse(url: str)       — retrieve full document text

Tool names and descriptions match snapshot-po Solar Open 100B (search_bench.py).
Legacy names (local_knowledge_base_retrieval, get_document) are also accepted
in model output for backward compatibility.

The episode ends when the model emits a response **without** a tool-call block
(i.e. a final answer). All reward and self-critique feedback logic from SearchEnv
is preserved exactly.

Training configuration notes:
  - Use generator.use_conversation_multi_turn=true
  - Set stop token per format:
      Qwen:  generator.sampling_params.stop='["</tool_call>"]'
      Solar: generator.sampling_params.stop='["<|tool_call:end|>"]'
  - Set environment.skyrl_gym.search_tool_calling.tool_call_format="solar" for Solar Open 100B.
  - The environment injects the tool system prompt and reformats the user message
    during init(), so the upstream dataset format is unchanged.
"""

import json
import random
import re
import logging
import requests
import string

from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.search_tool_calling.utils import compute_score_tool_calling, extract_solution_tool_calling
from skyrl_gym.tools import SearchToolGroup
from skyrl_gym.tools.search import call_search_api

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — Qwen format
# ---------------------------------------------------------------------------

# Qwen-native tool-calling system prompt.
# Uses search/browse tool names matching snapshot-po Solar Open 100B.
_TOOL_SYSTEM_PROMPT = """\
# Tools

You may call one or more functions to assist with the user request.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Search the local knowledge base for documents relevant to the query. Returns the top results with their URLs and text snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Query to search the local knowledge base for relevant information"}}, "required": ["query"], "additionalProperties": false}}}
{"type": "function", "function": {"name": "browse", "description": "Retrieve the full text of a document by its URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Document URL to retrieve"}}, "required": ["url"], "additionalProperties": false}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>\
"""

# ---------------------------------------------------------------------------
# Prompt templates — Solar Open format
# ---------------------------------------------------------------------------
# Tool definitions in OpenAI Chat Completions format ({"type": "function", "function": {...}}).
# These are passed to tokenizer.apply_chat_template(tools=SOLAR_TOOL_DEFS) so the Solar
# tokenizer's Jinja2 template renders the canonical
#   <|begin|>system<|content|>## Provider System Prompt...\n\n## Tools\n...<|end|>
# system section — identical to what vLLM produces when the eval script passes tools=[...]
# via the Responses API.  Do NOT embed these in a system message string; that would produce
# a different token sequence (extra "## System Prompt" header).

SOLAR_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the local knowledge base for documents relevant to the query. "
                "Returns the top results with their URLs and text snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the local knowledge base for relevant information",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Retrieve the full text of a document by its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Document URL to retrieve",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

# User-message template — matches snapshot-po Solar Open 100B QUERY_TEMPLATE.
_USER_PROMPT_TEMPLATE = (
    "You are a deep research agent. You need to answer the given question by interacting "
    "with a search engine, using the search and browse tools provided. Please perform "
    "reasoning and use the tools step by step, in an interleaved manner. You may use the "
    "search and browse tools multiple times.\n\n"
    "Question: {question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer.}}\n"
    "Exact Answer: {{your succinct, final answer}}"
)

# Alias for Solar Open tokenizer path (tool_call_format="solar").
_SOLAR_QUERY_TEMPLATE = _USER_PROMPT_TEMPLATE

# Injected into chat history (as a user message) after an INCOMPLETE feedback.
# {feedback} is the only format field; the rest are literal instructions.
# Matches snapshot-po Solar Open 100B: FEEDBACK_FORMAT_TEMPLATE in search_bench.py.
FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR = (
    "\n\nFeedback:\n{feedback}\n\n"
    "Now, revise your research process based on the feedback above. "
    "You must first perform reasoning, discussing the validity of the given feedback. "
    "After reasoning, use the search and browse tools to "
    "gather more information. You may search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide "
    "your final answer in the format:\n"
    "Explanation: {your explanation}\n"
    "Exact Answer: {your succinct final answer}"
)


class SearchToolCallingEnv(BaseTextEnv):
    """
    Tool-calling search environment with self-critique feedback loop.

    Reward formula (training):
        R_total = (base_reward * feedback_penalty^(T-1)) + feedback_coef * R_fb

    where T = number of feedback iterations, R_fb tracks trajectory quality.
    During evaluation, R_total = base_reward (no penalties or feedback reward).
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras.get("max_turns", 2)

        # ── feedback state ──────────────────────────────────────────────
        self.awaiting_feedback = False
        self.last_answer_turn: Optional[int] = None
        self.feedback_count = 0

        # First-answer tracking for R_fb computation
        self.first_answer_correctness: Optional[bool] = None

        # ── env config ──────────────────────────────────────────────────
        self.feedback_model = env_config.get("feedback_model", "self")
        self.feedback_url = env_config.get("feedback_url", None)

        # mask_zero_format: if True, invalid-format trajectories get reward=0
        # AND the loss is masked (no gradient). If False, compute reward normally.
        self.mask_zero_format = env_config.get("mask_zero_format", True)

        self.feedback_penalty = env_config.get("feedback_penalty", 0.8)
        self.enable_feedback_reward = env_config.get("enable_feedback_reward", False)
        self.feedback_coef = env_config.get("feedback_coef", 0.5)

        self.is_training = extras.get("is_training", True)

        # Search parameters (mirrors SearchEnv)
        self.search_url = env_config.search_url
        self.topk = env_config.topk
        self.timeout = env_config.timeout
        self.log_requests = env_config.log_requests

        # tool_call_format: "qwen" (default, <tool_call> XML) or "solar" (<|tool_call:begin|> tags)
        self.tool_call_format: str = env_config.get("tool_call_format", "qwen")

        # Optional document-access endpoint for get_document.
        # Defaults to search_url with /retrieve → /access substitution.
        doc_access_url = env_config.get("doc_access_url", None)
        if not doc_access_url and "/retrieve" in self.search_url:
            doc_access_url = self.search_url.replace("/retrieve", "/access")
        self.doc_access_url: Optional[str] = doc_access_url

        # Initialise the shared search tool (for connection pooling)
        self.tool_group = SearchToolGroup(
            search_url=self.search_url,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []

    # ------------------------------------------------------------------ #
    #  Prompt construction helpers                                         #
    # ------------------------------------------------------------------ #

    def _extract_question(self, init_prompt: ConversationType) -> str:
        """
        Extract the bare question text from the upstream dataset prompt.

        The dataset uses search-R1 format, e.g.:
          "...For example, <answer> Beijing </answer>. Question: {question}"
        We strip the preamble and return just {question}.
        Falls back to the full user message content if the pattern is not found.
        """
        for msg in init_prompt:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            # search-R1 dataset preamble ends with "Question: {question}"
            m = re.search(r"Question:\s*(.*?)(?:\s*$|\Z)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            # Fallback: strip any trailing format instructions
            m = re.match(r"^(.*?)(?:\n\nYour final response|\Z)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            return content.strip()
        return ""

    # ------------------------------------------------------------------ #
    #  Environment lifecycle                                               #
    # ------------------------------------------------------------------ #

    def init(self, init_prompt: ConversationType):
        """Rebuild prompt in tool-calling format and reset episode state."""
        # Reset all episode state
        self.awaiting_feedback = False
        self.last_answer_turn = None
        self.feedback_count = 0
        self.first_answer_correctness = None

        question = self._extract_question(init_prompt)

        if self.tool_call_format == "solar":
            # Solar Open: no explicit system message — the tokenizer's chat template renders
            # the tools section automatically when tools=SOLAR_TOOL_DEFS is passed to
            # apply_chat_template.  We signal this via _chat_template_tools in the metadata
            # returned by init(), which the generator uses ONLY for the initial tokenization.
            new_prompt: ConversationType = [
                {"role": "user", "content": _SOLAR_QUERY_TEMPLATE.format(question=question)},
            ]
            self.chat_history = new_prompt.copy()
            prompt, metadata = super().init(new_prompt)
            metadata["_chat_template_tools"] = SOLAR_TOOL_DEFS
            return prompt, metadata
        else:
            new_prompt = [
                {"role": "system", "content": _TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(question=question)},
            ]
            self.chat_history = new_prompt.copy()
            return super().init(new_prompt)

    # ------------------------------------------------------------------ #
    #  Tool execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute_search(self, query: str) -> str:
        """
        Execute search tool and return JSON result list.

        Returns a JSON array: [{"docid": str, "score": float|null, "snippet": str}, ...]
        """
        if not query or not query.strip():
            return json.dumps([])

        api_response, error_msg = call_search_api(
            retrieval_service_url=self.search_url,
            query=query.strip(),
            topk=self.topk,
            return_scores=True,
            timeout=self.timeout,
            log_requests=self.log_requests,
            session=self.tool_group.session,
        )

        if error_msg or not api_response:
            return json.dumps({"error": error_msg or "No response from search API"})

        items = []
        raw_results = api_response.get("result", [])
        if raw_results and isinstance(raw_results[0], list):
            for i, doc_item in enumerate(raw_results[0]):
                # When return_scores=True each item is {"document": {...}, "score": float}.
                # When return_scores=False each item is the doc dict directly.
                if "document" in doc_item:
                    doc = doc_item.get("document", {})
                    score = doc_item.get("score", None)
                else:
                    doc = doc_item
                    score = None
                content = doc.get("contents", "").strip()
                # URL is the stable docid the model passes back to get_document.
                doc_id = doc.get("url") or doc.get("id") or doc.get("doc_id") or f"doc-{i}"
                items.append({"docid": str(doc_id), "score": score, "snippet": content})

        return json.dumps(items, indent=2) if items else json.dumps([])

    def _execute_get_document(self, docid: str) -> str:
        """
        Execute get_document by calling the server's /access endpoint.

        If doc_access_url is not configured, returns an error message.
        """
        if not self.doc_access_url:
            return json.dumps(
                {"error": "get_document is not available (doc_access_url not configured)"}
            )
        try:
            resp = requests.post(
                self.doc_access_url,
                # Server (local_retrieval_server.py) expects {"urls": [...]}
                json={"urls": [docid]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            docs = result.get("result", [])
            if docs:
                # Server returns the raw page object {"url":..., "title":..., "contents":...}
                # (same structure as corpus entries). Use "contents" for the full text.
                page = (docs[0] or {}).get("contents", "") or ""
                return json.dumps({"docid": docid, "text": page}, indent=2)
            return json.dumps({"error": f"Document '{docid}' not found"})
        except Exception as e:
            return json.dumps({"error": f"get_document failed: {str(e)}"})

    # ------------------------------------------------------------------ #
    #  Action parsing                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _random_call_id() -> str:
        """Generate a 10-char random alphanumeric id as required by Solar's format."""
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

    def _parse_tool_call(self, action: str) -> Optional[Dict[str, Any]]:
        """
        Parse the first tool-call block from the model output.

        Qwen format:  <tool_call>{"name": ..., "arguments": ...}</tool_call>
        Solar format: <|tool_call:begin|>{id}<|tool_call:name|>{name}<|tool_call:args|>{args}<|tool_call:end|>

        Returns a dict with keys "name", "arguments", and (Solar only) "_call_id", or None.
        """
        if self.tool_call_format == "solar":
            m = re.search(
                r"<\|tool_call:begin\|>(.*?)<\|tool_call:name\|>(.*?)<\|tool_call:args\|>(.*?)<\|tool_call:end\|>",
                action,
                re.DOTALL,
            )
            if not m:
                return None
            call_id = m.group(1).strip() or self._random_call_id()
            name = m.group(2).strip()
            args_str = m.group(3).strip()
            try:
                arguments = json.loads(args_str)
                if not isinstance(arguments, dict):
                    arguments = {}
            except (json.JSONDecodeError, ValueError):
                arguments = {}
            return {"name": name, "arguments": arguments, "_call_id": call_id}
        else:
            # Qwen format
            m = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", action, re.DOTALL)
            if not m:
                return None
            try:
                parsed = json.loads(m.group(1))
                if not isinstance(parsed, dict):
                    return None
                return parsed
            except (json.JSONDecodeError, ValueError):
                return None

    # ------------------------------------------------------------------ #
    #  Feedback helpers (mirrors SearchEnv)                               #
    # ------------------------------------------------------------------ #

    def _get_feedback_prompt(self) -> str:
        """Build the self-critique evaluation prompt (adapted for tool calls)."""
        # Extract question from the user message in the rebuilt prompt.
        # The user prompt is: "...Question: {question}\n\nYour response should be..."
        question = ""
        for msg in self.chat_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Try "Question: {question}" pattern first (Solar-style prompt)
                m = re.search(r"Question:\s*(.*?)(?:\n\nYour response|\Z)", content, re.DOTALL)
                if m:
                    question = m.group(1).strip()
                    break
                # Fallback: everything before "\n\nYour final response"
                m = re.match(r"^(.*?)(?:\n\nYour final response|\Z)", content, re.DOTALL)
                if m:
                    question = m.group(1).strip()
                    break
                question = content.strip()
                break

        # Collect assistant turns (tool calls + final answer) as the reasoning trace
        reasoning_parts = [
            msg.get("content", "")
            for msg in self.chat_history
            if msg.get("role") == "assistant"
        ]
        reasoning = "\n".join(reasoning_parts)
        search_process = f"Question: {question}\nReasoning: {reasoning}"

        eval_prompt = (
            "The following is a search process that uses tool calls to reason and collect "
            "evidence.\n"
            "Your task is to judge the integrity of the search process itself, "
            "NOT whether the final answer is correct.\n\n"
            "A search process has integrity only if it demonstrates that all information "
            "required to answer the question was systematically and transparently sought "
            "before committing to a conclusion.\n\n"
            "Evaluate the process using the criteria below.\n\n"
            "--------------------------------\n"
            "Evaluation Criteria\n"
            "--------------------------------\n\n"
            "- An essential constraint is a minimal factual dependency that must be resolved "
            "in order to compute the answer.\n"
            "- Direct questions have a single essential constraint; multi-step questions "
            "require multiple essential constraints.\n"
            "- Each essential constraint must be constructed directly from the question, "
            "not from the reasoning or search behavior.\n"
            "- The reasoning behavior must explicitly attempt to resolve each essential "
            "constraint.\n"
            "- The reasoning behavior must not commit to a candidate answer before all "
            "essential constraints have been addressed.\n"
            "- Explicit acknowledgment of missing information followed by an answer counts "
            "as failure.\n"
            "- Any missing, skipped, implicitly assumed, or prematurely resolved constraint "
            "results in failure.\n\n"
            "--------------------------------\n"
            "Output Requirements\n"
            "--------------------------------\n\n"
            "Your response MUST follow this exact order:\n\n"
            "1. Essential Constraint(s):\n"
            "   - List each essential constraint implied by the question as short, minimal, "
            "concrete items.\n"
            "   - Do NOT include any additional constraint that is not directly derived from "
            "the question. The provided search process must not affect the construction of "
            "the essential constraint(s).\n\n"
            "2. Analysis:\n"
            "   - Analyze the integrity of the search process based on the criteria defined "
            "above.\n"
            "   - Analyze whether the search process attempted to resolve each essential "
            "constraint before committing to an answer.\n\n"
            "3. Judgment:\n"
            "   - Output exactly one word: COMPLETE or INCOMPLETE\n\n"
            "--------------------------------\n"
            "Important Restrictions\n"
            "--------------------------------\n\n"
            "- Focus solely on the integrity of the search process.\n"
            "- DO NOT perform any searches yourself.\n"
            "- DO NOT answer the question.\n"
            "- DO NOT assume facts that are not explicitly supported by the search process.\n"
            "- If there is reasonable doubt that all essential constraints were properly "
            "addressed, you MUST choose INCOMPLETE.\n\n"
            "--------------------------------\n"
            "Search Process:\n"
            "--------------------------------\n\n"
            "{search_process}"
        )
        return eval_prompt.format(search_process=search_process)

    def _parse_feedback_judgment(self, feedback_text: str) -> bool:
        """Return True if the feedback says INCOMPLETE (needs more research)."""
        if re.search(r"Judgment:\s*[-\s]*INCOMPLETE", feedback_text, re.IGNORECASE):
            return True
        if re.search(r"\bINCOMPLETE\b", feedback_text, re.IGNORECASE):
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Reward helpers (identical logic to SearchEnv)                      #
    # ------------------------------------------------------------------ #

    def _compute_feedback_reward(self, first_correct: bool, final_correct: bool) -> float:
        """
        R_fb:
          +1.0  correct → correct  (maintained)
          +1.0  wrong   → correct  (improved)
          -0.1  wrong   → wrong    (still wrong)
          -2.0  correct → wrong    (degraded)
        """
        if first_correct and final_correct:
            return 1.0
        elif not first_correct and final_correct:
            return 1.0
        elif not first_correct and not final_correct:
            return -0.1
        else:
            return -2.0

    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0.0

        chat_history_str = " ".join(item.get("content", "") for item in self.chat_history)
        base_reward = compute_score_tool_calling(chat_history_str, self.ground_truth)

        # Evaluation: pure correctness, no penalties
        if not self.is_training:
            logger.info(
                f"Eval reward — Base: {base_reward:.4f}, Feedbacks: {self.feedback_count}"
            )
            return base_reward

        # Training: apply feedback penalty
        if self.feedback_count > 0:
            penalty_factor = self.feedback_penalty ** (self.feedback_count - 1)
            penalized_reward = base_reward * penalty_factor
        else:
            penalized_reward = base_reward

        # Optional R_fb component
        feedback_reward_component = 0.0
        if (
            self.enable_feedback_reward
            and self.feedback_count > 0
            and self.first_answer_correctness is not None
        ):
            final_correct = base_reward > 0
            R_fb = self._compute_feedback_reward(self.first_answer_correctness, final_correct)
            feedback_reward_component = self.feedback_coef * R_fb

        final_reward = penalized_reward + feedback_reward_component

        if self.feedback_count > 0:
            logger.info(
                f"Reward — Base: {base_reward:.4f}, Feedbacks: {self.feedback_count}, "
                f"Penalty: {penalty_factor:.4f}, Penalized: {penalized_reward:.4f}, "
                f"Feedback component: {feedback_reward_component:.4f}, "
                f"Final: {final_reward:.4f}"
            )
        else:
            logger.info(
                f"Reward — Base: {base_reward:.4f}, Final: {base_reward:.4f} (no feedback)"
            )

        return final_reward

    # ------------------------------------------------------------------ #
    #  Core step logic                                                     #
    # ------------------------------------------------------------------ #

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # ── Feedback response path ──────────────────────────────────────
        if self.awaiting_feedback:
            self.feedback_count += 1
            is_incomplete = self._parse_feedback_judgment(action)

            if is_incomplete:
                self.awaiting_feedback = False

                # Max-turns guard: if we're at the limit, end the episode
                if self.turns >= self.max_turns:
                    reward = self._get_reward(action, done=True)
                    return BaseTextEnvStepOutput(
                        observations=[],
                        reward=reward,
                        done=True,
                        metadata={
                            "feedback_incomplete": True,
                            "max_turns_reached": True,
                            "total_feedbacks": self.feedback_count,
                        },
                    )

                # Continue the episode — generator will inject the feedback
                # text as a user message using FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=False,
                    metadata={
                        "feedback_incomplete": True,
                        "continuing_search": True,
                    },
                )
            else:
                # COMPLETE (or unparsed) → end episode
                reward = self._get_reward(action, done=True)
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=reward,
                    done=True,
                    metadata={
                        "feedback_incomplete": False,
                        "total_feedbacks": self.feedback_count,
                    },
                )

        # ── Normal generation path ──────────────────────────────────────
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        tool_call = self._parse_tool_call(action)
        has_tool_call = tool_call is not None

        # Max-turns always terminates
        if self.turns >= self.max_turns:
            reward = self._get_reward(action, done=True)
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"total_feedbacks": self.feedback_count},
            )

        # ── Final answer (no tool call) ─────────────────────────────────
        if not has_tool_call:
            # Track first-answer correctness for R_fb
            if self.enable_feedback_reward and self.first_answer_correctness is None:
                chat_str = " ".join(m.get("content", "") for m in self.chat_history)
                first_reward = compute_score_tool_calling(chat_str, self.ground_truth)
                self.first_answer_correctness = first_reward > 0
                logger.info(
                    f"First answer — Correctness: {self.first_answer_correctness}, "
                    f"Score: {first_reward:.4f}"
                )

            # Check for invalid format (no extractable answer) before requesting feedback
            chat_str = " ".join(m.get("content", "") for m in self.chat_history)
            has_answer = extract_solution_tool_calling(chat_str) is not None

            if not has_answer:
                # No recognisable answer format
                if self.mask_zero_format or not self.is_training:
                    return BaseTextEnvStepOutput(
                        observations=[],
                        reward=0.0,
                        done=True,
                        metadata={
                            "error": "no_valid_answer",
                            "reason": "No 'Exact Answer:' or <answer> tag detected",
                            "total_feedbacks": self.feedback_count,
                            "mask_loss": True,
                        },
                    )
                else:
                    # Train on invalid trajectory with R_fb penalty if enabled
                    if self.enable_feedback_reward:
                        reward = self.feedback_coef * (-3.0)
                        logger.info(
                            f"Reward — INVALID FORMAT, R_fb=-3.0, "
                            f"Final: {reward:.4f}"
                        )
                    else:
                        reward = 0.0
                    return BaseTextEnvStepOutput(
                        observations=[],
                        reward=reward,
                        done=True,
                        metadata={
                            "error": "no_valid_answer",
                            "total_feedbacks": self.feedback_count,
                            "mask_loss": False,
                        },
                    )

            # Request self-critique feedback
            self.awaiting_feedback = True
            self.last_answer_turn = self.turns

            feedback_prompt = self._get_feedback_prompt()
            # Embed evaluator role in user message — Qwen3.5 (and similar models)
            # require system messages only at the start of the conversation.
            user_msg = {
                "role": "user",
                "content": (
                    "You are a careful and strict evaluator of research processes.\n\n"
                    + feedback_prompt
                ),
            }

            metadata: Dict[str, Any] = {
                "awaiting_feedback": True,
                "answer_turn": self.last_answer_turn,
                "use_only_observations": True,
                "feedback_model": self.feedback_model,
                "feedback_url": self.feedback_url,
                # Tell the generator to use this template when injecting the feedback
                # as a user message after an INCOMPLETE judgment.
                "feedback_format_template": FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR,
            }
            if self.feedback_model == "self":
                metadata["response_prefix"] = "1. Essential Constraint(s):\n"

            return BaseTextEnvStepOutput(
                observations=[user_msg],
                reward=0.0,
                done=False,
                metadata=metadata,
            )

        # ── Tool call path ──────────────────────────────────────────────
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Coerce argument values to str — the model may emit nested objects
        # or other non-string types instead of plain strings.
        def _arg_str(key: str) -> str:
            val = arguments.get(key, "")
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                # e.g. {"user_query": {"query": "..."}} — try common sub-keys
                for sub in ("query", "user_query", "q", "text", "docid", "id"):
                    if sub in val:
                        return str(val[sub])
                # fallback: first string value
                for v in val.values():
                    if isinstance(v, str):
                        return v
            return str(val) if val else ""

        if tool_name in ("search", "local_knowledge_base_retrieval"):
            result = self._execute_search(_arg_str("query") or _arg_str("user_query"))
        elif tool_name in ("browse", "get_document"):
            result = self._execute_get_document(_arg_str("url") or _arg_str("docid"))
        else:
            result = json.dumps({"error": f"Unknown tool: {tool_name!r}"})

        if self.tool_call_format == "solar":
            call_id = tool_call.get("_call_id") or self._random_call_id()
            # Use the native Solar chat-template format:
            #   role="tool", tool_call_id=id, content=plain_result
            # The tokenizer's Jinja2 template renders this as:
            #   <|begin|>tool<|tool_response|><|tool_response:begin|>{id}
            #   <|tool_response:name|>{name}<|tool_response:result|>{result}
            #   <|tool_response:end|><|end|>
            # Do NOT embed <|tool_response:...|> tokens manually — that would
            # double-encode them when apply_chat_template processes the message.
            obs: Dict[str, str] = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            }
        else:
            # Tool results are returned as user-role messages so they:
            # (a) work with any Qwen tokenizer (no "tool" role dependency)
            # (b) are automatically excluded from the loss mask (user turns)
            obs: Dict[str, str] = {
                "role": "user",
                "content": f"<tool_response>\n{result}\n</tool_response>",
            }
        self.chat_history.append(obs)

        info: Dict[str, Any] = {
            "tool_name": tool_name,
            "tool_input": arguments,
        }

        return BaseTextEnvStepOutput(
            observations=[obs],
            reward=0.0,
            done=False,
            metadata=info,
        )
