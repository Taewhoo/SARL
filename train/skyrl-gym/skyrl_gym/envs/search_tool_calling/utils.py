"""
Reward utilities for the tool-calling search environment.

Mirrors skyrl_gym/envs/search/utils.py but handles tool-calling answer format:
  - "Exact Answer: {answer}"   (primary, matches QUERY_TEMPLATE format)
  - "<answer>...</answer>"     (fallback, for compatibility)
"""

import re
import string
from typing import Optional


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def extract_solution_tool_calling(solution_str: str) -> Optional[str]:
    """
    Extract the final answer from a tool-calling trajectory string.

    Tries in order:
    1. "Exact Answer: {answer}"  (last occurrence)
    2. "<answer>...</answer>"    (last occurrence, for fallback compatibility)
    """
    # 1. "Exact Answer: {answer}" — capture to end-of-line
    matches = list(re.finditer(r"Exact Answer:\s*(.+?)(?:\n|$)", solution_str, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()

    # 2. <answer>...</answer> fallback
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    return None


def compute_score_tool_calling(
    solution_str: str,
    ground_truth,
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    Exact-match scoring for tool-calling trajectories.

    Args:
        solution_str:  Full concatenated chat history as a single string.
        ground_truth:  Dict with key "target" (str or list[str]).
        format_score:  Score when answer is present but wrong.
        score:         Score when answer is correct.
    """
    answer = extract_solution_tool_calling(solution_str)
    if answer is None:
        return 0.0
    if em_check(answer, ground_truth["target"]):
        return score
    return format_score
