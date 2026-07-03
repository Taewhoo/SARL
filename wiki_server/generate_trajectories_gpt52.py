"""
Generate search-agent trajectories with GPT-5.2 (or any OpenAI-compatible model)
on the same QA samples used for the SSA evaluation.

The output is written in the same JSONL schema as the existing trajectories under
human_annotation/, so it can be fed directly into llm_ssa_eval.py.

Usage:
    python generate_trajectories_gpt52.py \
        --input_file  human_annotation/self_critique_multihop_rand50_1.jsonl \
        --output_file gpt52_multihop_rand50_1.jsonl \
        --model       gpt-5.2 \
        --api_key     $OPENAI_API_KEY \
        --search_url  http://<host>:8010/retrieve \
        [--max_turns 32] \
        [--topk 5] \
        [--temperature 0.0] \
        [--concurrency 8]
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI


# Identical to the user-message used by the trained-Qwen trajectories.
INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by <search> "
    "query </search> and it will return the top searched results between "
    "<information> and </information>. You can search as many times as your "
    "want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: {question}"
)

# Same Qwen-template prompt string the existing trajectories carry. Kept verbatim
# so the file is drop-in compatible with llm_ssa_eval.py's extract_question().
QWEN_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant"
)


SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_question(prompt: str) -> str:
    m = re.search(r"Question:\s*(.*?)(?:\n|$)", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip().split("\n")[-1].strip()


def parse_action(text: str) -> tuple[str, str]:
    """Return ('search', query) | ('answer', content) | ('none', '')."""
    a = ANSWER_RE.search(text)
    s = SEARCH_RE.search(text)
    if a and (not s or a.start() < s.start()):
        return ("answer", a.group(1).strip())
    if s:
        return ("search", s.group(1).strip())
    return ("none", "")


async def search(client: httpx.AsyncClient, search_url: str, query: str, topk: int) -> list[str]:
    """Hit the local FAISS retriever and return up to `topk` document strings."""
    payload = {"queries": [query], "topk": topk, "return_scores": False}
    resp = await client.post(search_url, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    # The local_retrieval_server returns: {"result": [[{"document": {...}}, ...], ...]}
    docs = []
    result = data.get("result") or data.get("documents") or []
    if result and isinstance(result, list) and isinstance(result[0], list):
        for hit in result[0]:
            if isinstance(hit, dict):
                doc = hit.get("document") or hit
                if isinstance(doc, dict):
                    text = doc.get("contents") or doc.get("text") or json.dumps(doc)
                else:
                    text = str(doc)
            else:
                text = str(hit)
            docs.append(text)
    return docs[:topk]


async def run_one(
    sem: asyncio.Semaphore,
    oai: AsyncOpenAI,
    http: httpx.AsyncClient,
    model: str,
    search_url: str,
    question: str,
    answer: str,
    max_turns: int,
    topk: int,
    temperature: float,
) -> dict[str, Any]:
    """Run a single multi-turn search-agent trajectory."""
    instruction = INSTRUCTION.format(question=question)
    prompt_str = QWEN_PROMPT_TEMPLATE.format(instruction=instruction)
    history: list[dict[str, Any]] = []

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]

    async with sem:
        for turn in range(max_turns):
            try:
                resp = await oai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stop=["</search>", "</answer>"],
                )
            except Exception as e:
                history.append({"type": "error", "error": f"openai: {e!r}"})
                break

            text = resp.choices[0].message.content or ""
            stop_reason = resp.choices[0].finish_reason

            # The stop tokens are dropped by the API — re-append so the
            # action parsers see complete tags.
            if stop_reason == "stop":
                if SEARCH_RE.search(text + "</search>") and not SEARCH_RE.search(text):
                    text = text + "</search>"
                elif ANSWER_RE.search(text + "</answer>") and not ANSWER_RE.search(text):
                    text = text + "</answer>"

            kind, payload = parse_action(text)

            tool_calls = []
            if kind == "search":
                tool_calls = [{"type": "search", "query": payload}]
            elif kind == "answer":
                tool_calls = [{"type": "answer", "content": payload}]

            history.append({"type": "llm_response", "text": text, "tool_calls": tool_calls})

            if kind == "answer":
                break

            if kind == "search":
                try:
                    docs = await search(http, search_url, payload, topk)
                except Exception as e:
                    docs = [f"[retriever error: {e!r}]"]
                history.append(
                    {"type": "search_result", "query": payload, "documents": docs}
                )
                # Append model output + retrieved info to the running chat.
                info_block = "<information>\n" + "\n\n".join(docs) + "\n</information>"
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": info_block})
                continue

            # `none` — model emitted neither tag. Stop early to avoid loops.
            break

    return {"prompt": prompt_str, "answer": answer, "history": history}


async def main_async(args: argparse.Namespace) -> None:
    # Load (question, answer) pairs from the input JSONL.
    samples = []
    with open(args.input_file) as f:
        for line in f:
            d = json.loads(line)
            q = extract_question(d["prompt"])
            samples.append((q, d["answer"]))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Generating {len(samples)} trajectories with {args.model}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    oai = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    async with httpx.AsyncClient() as http:
        coros = [
            run_one(
                sem, oai, http, args.model, args.search_url,
                q, a, args.max_turns, args.topk, args.temperature,
            )
            for q, a in samples
        ]

        # Stream results to disk so a partial run still produces output.
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as out:
            for i, coro in enumerate(asyncio.as_completed(coros)):
                try:
                    rec = await coro
                except Exception as e:
                    rec = {"prompt": "", "answer": "", "history": [],
                           "error": f"{e!r}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 5 == 0 or i + 1 == len(samples):
                    print(f"  done: {i + 1}/{len(samples)}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True,
                   help="JSONL of existing samples; only `prompt` and `answer` are used.")
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", required=True, help="e.g. gpt-5.2")
    p.add_argument("--api_key", required=True)
    p.add_argument("--base_url", default=None,
                   help="OpenAI-compatible base URL; default is openai.com.")
    p.add_argument("--search_url", required=True,
                   help="Retriever endpoint, e.g. http://<host>:8010/retrieve")
    p.add_argument("--max_turns", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0,
                   help="Cap on how many samples to run; 0 = all.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
