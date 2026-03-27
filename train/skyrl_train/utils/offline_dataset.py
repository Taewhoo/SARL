"""
Utility for converting SARL online rollouts to a torchtitan-rl offline HuggingFace dataset.

The torchtitan-rl dataset format (multi-response form):
    prompt              : list of chat messages (the original prompt)
    generated_responses : list of response strings, one per rollout
    advantages          : list of float scalars (GRPO-normalized), one per rollout
    conversations       : JSON-encoded list of full chat histories (for multi-turn tool-call)
    is_truncated        : list of booleans, True when stop_reason == "length"

Usage — call `save_batch_as_hf_dataset()` from within the SARL trainer loop after
`compute_advantages_and_returns()` returns.

Example in trainer.py::train():

    from skyrl_train.utils.offline_dataset import save_batch_as_hf_dataset

    # after compute_advantages_and_returns, before training_input.pop("rewards")
    save_batch_as_hf_dataset(
        generator_input=generator_input,
        generator_output=generator_output,
        training_input=training_input,  # must have ["advantages"] and ["response_mask"]
        uids=uids,
        tokenizer=self.tokenizer,
        output_dir=cfg.trainer.offline_dataset_dir,
        global_step=self.global_step,
    )
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch


def _extract_response_scalar_advantage(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    idx: int,
) -> float:
    """
    GRPO broadcasts one scalar advantage to every response token.
    Recover the scalar by averaging the (identical) non-zero values under the response mask.
    Falls back to 0.0 for all-masked rows (padding rows).
    """
    mask = response_mask[idx].bool()
    vals = advantages[idx][mask]
    if vals.numel() == 0:
        return 0.0
    return vals.mean().item()


def convert_to_hf_records(
    generator_input: GeneratorInput,
    generator_output: GeneratorOutput,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    uids: List[str],
    tokenizer,
) -> List[Dict[str, Any]]:
    """
    Convert a single SARL training batch into a list of records for a HuggingFace dataset.

    Each record corresponds to one unique prompt (uid) and contains all N rollouts for
    that prompt, ready to be consumed by torchtitan-rl's GRPOHuggingFaceDataset.

    Args:
        generator_input:  The GeneratorInput passed to `generate()`.
        generator_output: The GeneratorOutput returned by `generate()`, after reward
                          postprocessing but before any further mutation.
        advantages:       Float tensor of shape (batch_size, seq_len) — per-token
                          advantages from `compute_advantages_and_returns()`.
        response_mask:    Binary tensor of shape (batch_size, seq_len) — 1 at response
                          token positions.
        uids:             List[str] of length batch_size; maps each row to a prompt group.
        tokenizer:        HuggingFace tokenizer (for decode).

    Returns:
        List of dicts, one per unique prompt:
            prompt              : List[dict]  (original chat messages)
            generated_responses : List[str]
            advantages          : List[float]
            conversations       : str  (JSON-encoded List[List[dict]])
            is_truncated        : List[bool]
    """
    n = len(uids)
    assert advantages.shape[0] == n, (
        f"advantages batch dim {advantages.shape[0]} != len(uids) {n}"
    )
    assert response_mask.shape[0] == n, (
        f"response_mask batch dim {response_mask.shape[0]} != len(uids) {n}"
    )

    response_ids: List[List[int]] = generator_output["response_ids"]
    stop_reasons: List[str] = generator_output.get("stop_reasons") or [""] * n
    main_trajectories: Optional[List[List[Dict]]] = generator_output.get("main_trajectories")
    prompts: List[Any] = generator_input["prompts"]
    trajectory_ids = generator_output.get("trajectory_ids")

    # Map uid → index of the corresponding original prompt in generator_input["prompts"]
    # trajectory_ids[i].instance_id is the uid; the instance_id is also the index into prompts
    # when dynamic_sampling is not used. Fall back to uid == str index.
    def _uid_to_prompt_idx(uid: str, i: int) -> int:
        if trajectory_ids is not None:
            try:
                # trajectory_ids[i].instance_id is the uid; the prompt index equals i //
                # n_samples_per_prompt for the plain (non-step-wise) case.
                # Use the uid directly as an int index when possible.
                return int(uid)
            except (ValueError, TypeError):
                pass
        return i  # fallback: same order as prompts

    # Group rows by uid
    uid_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_rows[uid].append(i)

    records = []
    for uid, row_indices in uid_to_rows.items():
        # Prompt messages — use the first row's index to look up the original prompt.
        first_i = row_indices[0]
        prompt_idx = _uid_to_prompt_idx(uid, first_i)
        try:
            prompt_messages = prompts[prompt_idx]
        except (IndexError, TypeError):
            prompt_messages = prompts[first_i] if first_i < len(prompts) else []

        generated_responses: List[str] = []
        adv_scalars: List[float] = []
        conversations_list: List[Optional[List[Dict]]] = []
        is_truncated_list: List[bool] = []

        for i in row_indices:
            # Decode response tokens back to text
            resp_text = tokenizer.decode(response_ids[i], skip_special_tokens=False)
            generated_responses.append(resp_text)

            # Extract scalar advantage (all response tokens share the same value)
            adv_scalars.append(_extract_response_scalar_advantage(advantages, response_mask, i))

            # Full conversation history for multi-turn tool-call data
            traj = main_trajectories[i] if main_trajectories is not None and i < len(main_trajectories) else None
            conversations_list.append(traj)

            # Truncation flag: "length" stop means the response was cut short
            is_truncated_list.append(stop_reasons[i] == "length" if i < len(stop_reasons) else False)

        # Encode conversations as JSON string if any trajectory is non-None
        has_conversations = any(c is not None for c in conversations_list)
        conversations_json = (
            json.dumps(conversations_list, ensure_ascii=False) if has_conversations else None
        )

        record: Dict[str, Any] = {
            "prompt": prompt_messages,
            "generated_responses": generated_responses,
            "advantages": adv_scalars,
            "is_truncated": is_truncated_list,
        }
        if conversations_json is not None:
            record["conversations"] = conversations_json

        records.append(record)

    return records


def save_batch_as_hf_dataset(
    generator_input: GeneratorInput,
    generator_output: GeneratorOutput,
    training_input: TrainingInputBatch,
    uids: List[str],
    tokenizer,
    output_dir: str,
    global_step: int,
    shard_id: int = 0,
) -> None:
    """
    Convert one SARL batch to HuggingFace dataset records and append them to an Arrow
    shard on disk.  Call this from the trainer loop after `compute_advantages_and_returns`.

    Creates ``{output_dir}/data-{global_step:06d}-{shard_id:04d}.arrow`` for each call.
    After all steps, run ``consolidate_shards_to_hf_dataset(output_dir)`` to produce the
    final pooled dataset that torchtitan-rl can load with ``load_from_disk``.

    Args:
        generator_input:  GeneratorInput passed to generate().
        generator_output: GeneratorOutput after reward postprocessing.
        training_input:   TrainingInputBatch after compute_advantages_and_returns().
        uids:             List[str] of length batch_size.
        tokenizer:        HuggingFace tokenizer.
        output_dir:       Root directory for shards.
        global_step:      Current training step (used for shard filename).
        shard_id:         Intra-step shard counter (default 0 for single-node).
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets>=2.0 is required. Install with: pip install datasets")

    advantages: torch.Tensor = training_input["advantages"].detach().cpu()
    response_mask: torch.Tensor = training_input["response_mask"].detach().cpu()

    records = convert_to_hf_records(
        generator_input=generator_input,
        generator_output=generator_output,
        advantages=advantages,
        response_mask=response_mask,
        uids=uids,
        tokenizer=tokenizer,
    )

    if not records:
        logger.warning(f"[offline_dataset] step={global_step}: no records to save, skipping.")
        return

    ds = Dataset.from_list(records)

    shard_path = os.path.join(output_dir, "shards", f"step-{global_step:06d}-shard-{shard_id:04d}")
    os.makedirs(shard_path, exist_ok=True)
    ds.save_to_disk(shard_path)
    logger.info(
        f"[offline_dataset] Saved {len(records)} prompt groups "
        f"({sum(len(r['generated_responses']) for r in records)} responses) "
        f"→ {shard_path}"
    )


def consolidate_shards_to_hf_dataset(
    output_dir: str,
    pool_ready_subdir: str = "pool_ready",
) -> None:
    """
    Merge all per-step Arrow shards written by ``save_batch_as_hf_dataset`` into one
    HuggingFace dataset saved to ``{output_dir}/{pool_ready_subdir}``.

    This is the directory you point torchtitan-rl's ``dataset_path`` at.

    Run once after all rollout generation steps are complete.
    """
    try:
        from datasets import concatenate_datasets, load_from_disk
    except ImportError:
        raise ImportError("datasets>=2.0 is required.")

    shards_dir = Path(output_dir) / "shards"
    shard_dirs = sorted(shards_dir.glob("step-*-shard-*"))

    if not shard_dirs:
        raise FileNotFoundError(f"No shards found under {shards_dir}")

    logger.info(f"[offline_dataset] Merging {len(shard_dirs)} shards …")
    datasets = [load_from_disk(str(d)) for d in shard_dirs]
    merged = concatenate_datasets(datasets)

    out_path = Path(output_dir) / pool_ready_subdir
    merged.save_to_disk(str(out_path))
    logger.info(
        f"[offline_dataset] Consolidated {len(merged)} prompt groups "
        f"→ {out_path}"
    )
