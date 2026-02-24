from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from skyrl_gym.envs.search.utils import compute_score
from skyrl_gym.tools import SearchToolGroup
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class SearchEnv(BaseTextEnv):
    """
    Environment for Search execution tasks with self-critique feedback loop.

    Based on Verl + Search-R1 integration
    
    When an answer is provided, the agent receives a feedback evaluation prompt.
    The agent then evaluates its own search process. If the evaluation is INCOMPLETE,
    the agent continues searching. If COMPLETE, the episode ends.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2
        
        # Feedback evaluation state tracking
        self.awaiting_feedback = False
        self.last_answer_turn = None  # Track which turn the answer was given
        self.feedback_count = 0  # Track number of feedbacks in trajectory
        
        # Track first answer for feedback reward computation
        self.first_answer_correctness = None  # True/False/None (not yet answered)
        self.first_answer_text = None  # Store first answer text
        
        # Feedback model configuration
        self.feedback_model = env_config.get("feedback_model", "self")
        self.feedback_url = env_config.get("feedback_url", None)
        
        # Masking behavior when no valid format is detected
        # If True: reward=0 AND mask loss (no gradient computation)
        # If False: compute reward from trajectory AND train normally
        self.mask_zero_format = env_config.get("mask_zero_format", True)
        
        # Feedback penalty: final_reward = r * a^(T-1) where T is number of feedbacks
        self.feedback_penalty = env_config.get("feedback_penalty", 0.8)
        
        # Feedback reward: R_total = (r * penalty) + β * R_fb
        self.enable_feedback_reward = env_config.get("enable_feedback_reward", False)
        self.feedback_coef = env_config.get("feedback_coef", 0.5)
        
        # Check if we're in evaluation mode (no feedback loop during eval)
        self.is_training = extras.get("is_training", True)

        # Initialize the tools
        # name is hardcoded to "SearchToolGroup", with tool name "search"
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _get_feedback_prompt(self) -> str:
        """Generate feedback evaluation prompt"""
        # Build search_process in format: "Question: {question}\nReasoning: {reasoning}"
        # chat_history contains structured messages, extract question and reasoning
        
        # Find the user message containing the question
        question = ""
        for msg in self.chat_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Parse question from the formatted prompt using regex
                question_pattern = re.compile(
                    r"For example, <answer> Beijing </answer>. Question:\s*(.*?)(?:\s*$|<\|im_end\|>)",
                    re.DOTALL
                )
                match = question_pattern.search(content)
                if match:
                    question = match.group(1).strip()
                    break
        
        # Collect all assistant messages as reasoning
        reasoning_parts = []
        for msg in self.chat_history:
            if msg.get("role") == "assistant":
                reasoning_parts.append(msg.get("content", ""))
        
        reasoning = "\n".join(reasoning_parts)
        search_process = f"Question: {question}\nReasoning: {reasoning}"

        ## self-critique evaluation prompt
        eval_prompt = """
        The following is a search process that uses <think> to reason and <search> to collect evidence.
Your task is to judge the integrity of the search process itself, NOT whether the final answer is correct.

A search process has integrity only if it demonstrates that all information required to answer the question was systematically and transparently sought before committing to a conclusion.

Evaluate the process using the criteria below.

--------------------------------
Evaluation Criteria
--------------------------------

- An essential constraint is a minimal factual dependency that must be resolved in order to compute the answer.
- Direct questions have a single essential constraint; multi-step questions require multiple essential constraints.
- Each essential constraint must be constructed directly from the question, not from the reasoning or search behavior.
- The reasoning behavior must explicitly attempt to resolve each essential constraint.
- The reasoning behavior must not commit to a candidate answer before all essential constraints have been addressed.
- Explicit acknowledgment of missing information followed by an answer counts as failure.
- Any missing, skipped, implicitly assumed, or prematurely resolved constraint results in failure.

--------------------------------
Output Requirements
--------------------------------

Your response MUST follow this exact order:

1. Essential Constraint(s):
   - List each essential constraint implied by the question as short, minimal, concrete items.
   - Do NOT include any additional constraint that is not directly derived from the question. The provided search process must not affect the construction of the essential constraint(s).

2. Analysis:
   - Analyze the integrity of the search process based on the criteria defined above.
   - Analyze whether the search process attempted to resolve each essential constraint before committing to an answer.

3. Judgment:
   - Output exactly one word: COMPLETE or INCOMPLETE

--------------------------------
Important Restrictions
--------------------------------

- Focus solely on the integrity of the search process.
- DO NOT perform any searches yourself.
- DO NOT answer the question.
- DO NOT assume facts that are not explicitly supported by the search process.
- If there is reasonable doubt that all essential constraints were properly addressed, you MUST choose INCOMPLETE.

--------------------------------
Search Process:
--------------------------------

{search_process}
"""
#         # gpt-oss-120b evaluation prompt
#         eval_prompt = """The following is a search process that uses <think> to reason and <search> to collect evidence.
# Your task is to judge the integrity of the search process itself, NOT whether the final answer is correct.

# A search process has integrity if it demonstrates that the information needed to answer the question was obtained before reaching a conclusion.

# Evaluate the process using the criteria below.

# --------------------------------
# Evaluation Criteria
# --------------------------------

# - An essential constraint is a minimal factual dependency that must be resolved in order to compute the answer.
# - Direct questions have a single essential constraint; multi-step questions require multiple essential constraints.
# - Each essential constraint must be constructed directly from the question, not from the reasoning or search behavior.
# - The reasoning process should address each essential constraint, either through targeted searches or by obtaining relevant information through related queries.
# - For a constraint to be satisfied, relevant information must be both retrieved in search results and identified by the reasoning process to inform the conclusion.
# - When evaluating processes with revisions, focus on the final state: if a constraint was initially overlooked but later addressed through corrections, consider it resolved.
# - Only if a constraint remains unaddressed in the final reasoning should it be considered missing.

# --------------------------------
# Output Requirements
# --------------------------------

# Your response MUST follow this exact order:

# 1. Essential Constraint(s):
#    - List each essential constraint implied by the question as short, minimal, concrete items.
#    - Do NOT include any additional constraint that is not directly derived from the question. The provided search process must not affect the construction of the essential constraint(s).

# 2. Analysis:
#    - Analyze the integrity of the search process based on the criteria defined above.
#    - For each essential constraint, verify that the necessary information was not only retrieved but also recognized and utilized in the reasoning.
#    - Consider whether each constraint was addressed by the end of the reasoning process, accounting for any revisions or corrections made along the way.

# 3. Judgment:
#    - Output exactly one word: COMPLETE or INCOMPLETE

# --------------------------------
# Important Restrictions
# --------------------------------

# - Focus on the overall integrity of the search process as a whole.
# - DO NOT perform any searches yourself.
# - DO NOT answer the question.
# - DO NOT assume facts that are not explicitly supported by the search process.

# --------------------------------
# Search Process:
# --------------------------------

# {search_process}
# """
        
        return eval_prompt.format(search_process=search_process)

    def _parse_feedback_judgment(self, feedback_text: str) -> bool:
        """
        Check if feedback indicates INCOMPLETE (needs more searching).
        
        Returns:
            True if INCOMPLETE, False otherwise (including COMPLETE or unparsed)
        """
        # Look for INCOMPLETE in the judgment line
        judgment_match = re.search(r'Judgment:\s*[-\s]*INCOMPLETE', feedback_text, re.IGNORECASE)
        if judgment_match:
            return True
        
        # Fallback: look for standalone INCOMPLETE word
        incomplete_match = re.search(r'\bINCOMPLETE\b', feedback_text, re.IGNORECASE)
        if incomplete_match:
            return True
        
        # Everything else (COMPLETE, unparsed, malformed) is treated as complete
        return False

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def init(self, init_prompt):
        """Initialize episode - reset state"""
        # Reset feedback state for new episode
        self.awaiting_feedback = False
        self.last_answer_turn = None
        self.feedback_count = 0
        
        # Reset first answer tracking for feedback reward
        self.first_answer_correctness = None
        self.first_answer_text = None
        
        # Store the initial prompt in chat_history so we can access the question later
        self.chat_history = init_prompt.copy() if init_prompt else []
        
        # Call parent init
        return super().init(init_prompt)
    
    def _compute_feedback_reward(self, first_correct: bool, final_correct: bool) -> float:
        """
        Compute feedback reward based on first and final answer correctness.
        
        R_fb categories:
        - 1.0: correct at first answer, correct after feedback (maintained correctness)
        - 1.0: wrong at first answer, correct after feedback (improved)
        - -0.1: wrong at first answer, wrong after feedback (still wrong)
        - -2.0: correct at first answer, wrong after feedback (degraded)
        
        Note: Invalid format trajectories (no valid tags) get R_fb = -3.0, handled
        separately in step() method. This function only handles valid format cases.
        
        Args:
            first_correct: Whether first answer was correct
            final_correct: Whether final answer is correct
            
        Returns:
            Feedback reward value
        """
        if first_correct and final_correct:
            return 1.0  # Maintained correctness
        elif not first_correct and final_correct:
            return 1.0  # Improved
        elif not first_correct and not final_correct:
            return -0.1  # Still wrong
        else:  # first_correct and not final_correct
            return -2.0  # Degraded
    
    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            base_reward = compute_score(chat_history_str, self.ground_truth)
            
            # During evaluation: use pure base reward (no penalties or feedback components)
            # During training: apply feedback penalties and reward components
            if not self.is_training:
                final_reward = base_reward
                logger.info(f"Eval reward - Base: {base_reward:.4f}, Feedbacks: {self.feedback_count}")
                return final_reward
            
            # Training reward calculation
            # Apply feedback penalty: penalized_reward = r * a^(T-1)
            # where r = base_reward, a = feedback_penalty, T = feedback_count
            if self.feedback_count > 0:
                penalty_factor = self.feedback_penalty ** (self.feedback_count - 1)
                penalized_reward = base_reward * penalty_factor
            else:
                penalized_reward = base_reward
            
            # Compute feedback reward if enabled and we had feedback iterations
            feedback_reward_component = 0.0
            if self.enable_feedback_reward and self.feedback_count > 0 and self.first_answer_correctness is not None:
                # Determine final correctness
                final_correct = base_reward > 0
                R_fb = self._compute_feedback_reward(self.first_answer_correctness, final_correct)
                feedback_reward_component = self.feedback_coef * R_fb
            
            # Total reward: R_total = (r * a^(T-1)) + β * R_fb
            final_reward = penalized_reward + feedback_reward_component
            
            # Log reward breakdown
            if self.feedback_count > 0:
                if self.enable_feedback_reward and self.first_answer_correctness is not None:
                    final_correct = base_reward > 0
                    logger.info(
                        f"Reward calculation - Base: {base_reward:.4f}, "
                        f"Feedbacks: {self.feedback_count}, "
                        f"Penalty factor: {penalty_factor:.4f} ({self.feedback_penalty}^{self.feedback_count-1}), "
                        f"Penalized: {penalized_reward:.4f}, "
                        f"First correct: {self.first_answer_correctness}, Final correct: {final_correct}, "
                        f"R_fb: {R_fb:.2f}, Feedback component: {feedback_reward_component:.4f} ({self.feedback_coef}*{R_fb:.2f}), "
                        f"Final: {final_reward:.4f}"
                    )
                else:
                    logger.info(
                        f"Reward calculation - Base: {base_reward:.4f}, "
                        f"Feedbacks: {self.feedback_count}, "
                        f"Penalty factor: {penalty_factor:.4f} ({self.feedback_penalty}^{self.feedback_count-1}), "
                        f"Final: {final_reward:.4f}"
                    )
            else:
                logger.info(
                    f"Reward calculation - Base: {base_reward:.4f}, "
                    f"Feedbacks: {self.feedback_count}, "
                    f"Final: {base_reward:.4f} (no penalty, no feedback reward)"
                )
            
            return final_reward
        else:
            # No reward for intermediate steps for Search tasks
            return 0

    def _is_done(self, action: str) -> bool:
        # Max turns always takes precedence
        if self.turns >= self.max_turns:
            return True
        
        # If answer is detected, don't end yet - we need feedback first
        # Note: Model often doesn't generate </answer>, so we only check for <answer>
        if "<answer>" in action:
            return False
        
        return False

    def _validate_action(self, action: str):
        stop_tags = ["</search>", "</answer>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)

        return "\n<information>" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Check if this is a feedback response (generated externally via use_only_observations)
        if self.awaiting_feedback:
            # Increment feedback count - we've now received and are processing a feedback
            self.feedback_count += 1
            
            # Skip validation for feedback - it's analyzing the search process and may reference tags
            # Parse the judgment without validating stop tags
            is_incomplete = self._parse_feedback_judgment(action)
            
            if is_incomplete:
                # Feedback indicates INCOMPLETE, continue searching
                self.awaiting_feedback = False
                
                # Check if we've exceeded max_turns
                if self.turns >= self.max_turns:
                    # Episode must end due to max_turns limit
                    reward = self._get_reward(action, done=True)
                    return BaseTextEnvStepOutput(
                        observations=[],
                        reward=reward,
                        done=True,
                        metadata={
                            "feedback_incomplete": True,
                            "max_turns_reached": True,
                            "total_feedbacks": self.feedback_count
                        }
                    )
                
                metadata = {
                    "feedback_incomplete": True,
                    "continuing_search": True,
                    # Note: response_prefix_for_revision is now handled by generator through saved state
                    # The generator saves it from the original use_only_observations step and applies it here
                }
                
                # Return empty observations - generator will append feedback tokens with loss_mask=0
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0,
                    done=False,
                    metadata=metadata
                )
            else:
                # Everything else (COMPLETE or unparsed) ends the episode
                done = True
                reward = self._get_reward(action, done)
                
                metadata = {
                    "feedback_incomplete": False,
                    "total_feedbacks": self.feedback_count
                }
                
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=reward,
                    done=True,
                    metadata=metadata
                )
        
        # Validate action for normal (non-feedback) responses
        self._validate_action(action)
        
        # Increment turn counter - only for actual model generations, not feedback
        self.turns += 1
        
        # Normal case: add action as assistant message
        self.chat_history.append({"role": "assistant", "content": action})
        
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        # Check if answer was just provided and we need to request feedback
        # Note: Model often doesn't generate </answer>, so we only check for <answer>
        # But don't request feedback if we've already hit max_turns
        if "<answer>" in action and not self.awaiting_feedback and not done:
            # Track first answer for feedback reward computation
            if self.first_answer_correctness is None and self.enable_feedback_reward:
                # This is the first answer - compute its correctness
                chat_history_str = "".join([item["content"] for item in self.chat_history])
                first_answer_reward = compute_score(chat_history_str, self.ground_truth)
                self.first_answer_correctness = first_answer_reward > 0
                self.first_answer_text = chat_history_str
                logger.info(
                    f"First answer captured - Correctness: {self.first_answer_correctness}, "
                    f"Reward: {first_answer_reward:.4f}"
                )
            
            # Set flag to indicate we're now waiting for feedback
            self.awaiting_feedback = True
            self.last_answer_turn = self.turns
            
            # Generate feedback prompt - return it as observations so model generates feedback
            # But DON'T add it to chat_history (it's a separate generation process)
            # The model will only see this prompt, not the full chat history
            feedback_prompt = self._get_feedback_prompt()
            
            system_msg = {"role": "system", "content": "You are a careful and strict evaluator of search processes."}
            user_msg = {"role": "user", "content": feedback_prompt}
            
            # Return observations so generator uses ONLY the feedback prompt for next generation
            # Don't add to chat_history - we'll add the formatted feedback response later
            # Metadata flag signals that generator should use ONLY these observations (not full chat_history)
            metadata = {
                "awaiting_feedback": True, 
                "answer_turn": self.last_answer_turn,
                "use_only_observations": True,  # Signal to generator to use only observations, not full chat_history
                # No response_prefix_for_revision - let the model generate freely after feedback
                "feedback_model": self.feedback_model,  # "self" or "external"
                "feedback_url": self.feedback_url,  # URL for external feedback model
            }
            
            # Only use response prefix for self-feedback, not external feedback
            if self.feedback_model == "self":
                metadata["response_prefix"] = "1. Essential Constraint(s):\n"  # Force feedback generation to start with this format
            
            return BaseTextEnvStepOutput(
                observations=[system_msg, user_msg],
                reward=0,  # No reward yet, episode not done
                done=False,
                metadata=metadata
            )

        if done:
            return BaseTextEnvStepOutput(
                observations=[], 
                reward=reward, 
                done=done, 
                metadata={"total_feedbacks": self.feedback_count}
            )

        error = None
        observation = None
        query = self._parse_action(action)
        
        # Check if no valid tags were generated (neither <answer> nor <search>)
        has_answer_tag = "<answer>" in action
        has_search_tag = query and query[0] is not None
        
        if not has_answer_tag and not has_search_tag:
            # No valid tags detected - end episode
            # Behavior depends on mask_zero_format and enable_feedback_reward:
            # - mask_zero_format=True: reward=0, mask_loss=True (no gradient)
            # - mask_zero_format=False + enable_feedback_reward=False: reward=0, train on it
            # - mask_zero_format=False + enable_feedback_reward=True: R_fb=-3.0 penalty (worst outcome)
            # During evaluation, always mask (no training anyway)
            
            if self.mask_zero_format or not self.is_training:
                # Don't train on invalid format
                reward = 0
                mask_loss = True
            else:
                # Train on invalid format trajectory
                mask_loss = False
                # If feedback reward is enabled, apply worst penalty (even without feedback)
                if self.enable_feedback_reward:
                    R_fb = -3.0
                    reward = self.feedback_coef * R_fb
                    logger.info(
                        f"Reward calculation - INVALID FORMAT (feedbacks: {self.feedback_count}) - "
                        f"R_fb: {R_fb:.2f}, Final reward: {reward:.4f} ({self.feedback_coef}*{R_fb:.2f})"
                    )
                else:
                    # No feedback reward enabled: just give 0
                    reward = 0
            
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={
                    "error": "no_valid_tags", 
                    "reason": "Neither <answer> nor <search> tag detected",
                    "total_feedbacks": self.feedback_count,
                    "mask_loss": mask_loss,
                }
            )
        
        # Only execute search if a valid query was found
        if query and query[0] is not None:
            try:
                observation = self._execute_tool("SearchToolGroup", "search", query)
            except Exception as e:
                error = str(e)

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
        }
        
        # Add feedback count to metadata when episode ends
        if done:
            info["total_feedbacks"] = self.feedback_count

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
