"""
Convert SARL training data from Qwen search-R1 format to Solar Open 100B tool-calling format.

Changes per row:
  1. env_class: 'search' → 'search_tool_calling'
  2. extra_info['tools_kwargs']: {'search': {...}} → {'search_tool_calling': {'create_kwargs': {...}}}
  3. prompt[0]['content']: Qwen search-R1 instruction → QUERY_TEMPLATE (Solar tool-calling style)

Usage:
    python convert_data_solar.py \
        --input  train_data/combined_train_45k_train_converted.parquet \
        --output train_data/solar_train_45k_train.parquet

    python convert_data_solar.py \
        --input  train_data/combined_train_45k_test_converted.parquet \
        --output train_data/solar_train_45k_test.parquet
"""

import argparse
import copy
import json

import pandas as pd

# ---------------------------------------------------------------------------
# Prompt template matching the Solar Open 100B eval script (oss_eval_async_clean.py)
# ---------------------------------------------------------------------------
# Exact copy of QUERY_TEMPLATE from oss_eval_async_clean.py (strip() already applied).
QUERY_TEMPLATE = (
    "You are a deep research agent. You need to answer the given question by interacting "
    "with a search engine, using the search and get_document tools provided. Please perform "
    "reasoning and use the tools step by step, in an interleaved manner. You may use the "
    "search and get_document tools multiple times.\n\n"
    "Question: {Question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer.}}\n"
    "Exact Answer: {{your succinct, final answer}}"
)


def _extract_question(prompt) -> str:
    """Extract the bare question text from the Qwen search-R1 user message."""
    if isinstance(prompt, (list, tuple)) or hasattr(prompt, '__iter__') and not isinstance(prompt, (str, bytes)):
        for msg in prompt:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Qwen format: "Answer the given question. You must conduct reasoning
                #               inside <think>...</think> ... Question: {question}"
                for prefix in ("Question:", "question:"):
                    idx = content.find(prefix)
                    if idx != -1:
                        return content[idx + len(prefix):].strip()
                return content.strip()
    return str(prompt)


def convert_row(row: dict) -> dict:
    row = copy.deepcopy(row)

    # 1. env_class
    row["env_class"] = "search_tool_calling"

    # 2. Re-key tools_kwargs
    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}

    if isinstance(extra, dict):
        tw = extra.get("tools_kwargs", {})
        if isinstance(tw, str):
            try:
                tw = json.loads(tw)
            except Exception:
                tw = {}

        # Rename the 'search' key to 'search_tool_calling'
        old_key = "search"
        new_key = "search_tool_calling"
        if old_key in tw and new_key not in tw:
            tw[new_key] = tw.pop(old_key)
        elif old_key in tw:
            tw[new_key] = tw[old_key]
            del tw[old_key]

        extra["tools_kwargs"] = tw
        row["extra_info"] = extra

    # 3. Rewrite user message to Solar tool-calling format
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except Exception:
            prompt = [{"role": "user", "content": prompt}]

    question = _extract_question(prompt)
    solar_user_content = QUERY_TEMPLATE.format(Question=question)

    new_prompt = [{"role": "user", "content": solar_user_content}]
    row["prompt"] = new_prompt

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Columns: {list(df.columns)}")

    records = df.to_dict("records")
    converted = [convert_row(r) for r in records]
    out_df = pd.DataFrame(converted)

    out_df.to_parquet(args.output, index=False)
    print(f"Saved {len(out_df)} rows → {args.output}")

    # Sanity-check first row
    sample = converted[0]
    print("\n=== Sample row (first) ===")
    print("env_class:", sample.get("env_class"))
    ei = sample.get("extra_info", {})
    if isinstance(ei, str):
        ei = json.loads(ei)
    print("tools_kwargs keys:", list(ei.get("tools_kwargs", {}).keys()))
    prompt = sample.get("prompt", [])
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    print("prompt[0] content (first 200 chars):", prompt[0]["content"][:200])


if __name__ == "__main__":
    main()
