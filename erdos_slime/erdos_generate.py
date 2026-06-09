"""Custom Slime generate function for the Erdos TTT discovery loop."""
from __future__ import annotations

import os
import uuid

_ARCHIVE = None
_TOKENIZER = None


def _archive_path() -> str:
    return os.environ.get("ERDOS_ARCHIVE_PATH", "/root/workspace/erdos/data/archive.json")


def _get_archive():
    global _ARCHIVE
    if _ARCHIVE is None:
        from erdos_slime.puct_archive import create_initial_archive

        _ARCHIVE = create_initial_archive(
            _archive_path(),
            num_states=int(os.environ.get("ERDOS_NUM_INIT_STATES", "4")),
            rollout_n=int(os.environ.get("ERDOS_ROLLOUT_N", "8")),
            puct_c=float(os.environ.get("ERDOS_PUCT_C", "1.0")),
            topk_children=int(os.environ.get("ERDOS_TOPK_CHILDREN", "2")),
        )
    return _ARCHIVE


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        model_path = os.environ.get("ERDOS_MODEL_PATH", "")
        if not model_path:
            for candidate in (
                "/root/workspace/models/Qwen3-8B",
                "/root/workspace/models/gpt-oss-20b-bf16-clean",
                "/root/workspace/models/gpt-oss-20b",
            ):
                if os.path.isdir(candidate):
                    model_path = candidate
                    break
        if not model_path:
            raise RuntimeError("Cannot find tokenizer. Set ERDOS_MODEL_PATH to the model directory.")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return _TOKENIZER


async def generate(args, sample, sampling_params):
    from slime.utils.http_utils import post

    archive = _get_archive()
    tok = _get_tokenizer()

    budget_s = int(os.environ.get("ERDOS_BUDGET_S", "60"))
    cpus = int(os.environ.get("ERDOS_CPUS", "1"))
    target_c5 = float(os.environ.get("ERDOS_TARGET_C5", "0.3808"))
    rollout_n = int(os.environ.get("ERDOS_ROLLOUT_N", "8"))

    global_step = getattr(args, "global_step", 0)
    group_index = getattr(sample, "group_index", None)
    if group_index is None:
        sample_index = getattr(sample, "index", None)
        if sample_index is not None:
            group_index = int(sample_index) // max(int(getattr(args, "n_samples_per_prompt", rollout_n)), 1)
    slot_uid = group_index if group_index is not None else (
        getattr(sample, "uid", None) or getattr(sample, "data_id", None) or uuid.uuid4().hex
    )
    run_id = os.environ.get("ERDOS_RUN_ID", "").strip()
    base_group_uid = f"{global_step}:{slot_uid}"
    group_uid = f"{run_id}:{base_group_uid}" if run_id else base_group_uid

    state = archive.acquire_group(group_uid)

    from erdos_slime.erdos_env import build_erdos_prompt

    prompt_text = build_erdos_prompt(state, budget_s=budget_s, cpus=cpus, target_c5=target_c5)
    messages = [{"role": "user", "content": prompt_text}]
    enable_thinking = os.environ.get("ERDOS_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}
    try:
        prompt_str = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt_str = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=getattr(args, "reasoning_effort", "high"),
        )
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(
        url,
        {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
    )

    meta_info = output.get("meta_info", {})
    if "output_token_logprobs" in meta_info:
        new_response_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in meta_info["output_token_logprobs"]]
    else:
        new_response_tokens = []
        new_response_log_probs = []

    response_text = output.get("text", "")
    train_cap = int(os.environ.get("ERDOS_TRAIN_MAX_RESPONSE_TOKENS", "0") or "0")
    capped_for_train = False
    if train_cap > 0 and len(new_response_tokens) > train_cap:
        new_response_tokens = new_response_tokens[:train_cap]
        new_response_log_probs = new_response_log_probs[:train_cap]
        response_text = tok.decode(new_response_tokens, skip_special_tokens=False)
        capped_for_train = True

    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata.update(
        {
            "erdos_archive_path": _archive_path(),
            "erdos_group_uid": group_uid,
            "erdos_state": state.to_dict(),
            "erdos_budget_s": budget_s,
            "erdos_global_step": int(global_step) if str(global_step).isdigit() else global_step,
        }
    )

    sample.prompt = prompt_str
    sample.tokens = prompt_ids + new_response_tokens
    sample.response_length = len(new_response_tokens)
    sample.response = response_text
    sample.rollout_log_probs = new_response_log_probs

    finish_reason = meta_info.get("finish_reason", {})
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type", "stop")
    if finish_reason == "length" or capped_for_train:
        sample.status = sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = sample.Status.ABORTED
    else:
        sample.status = sample.Status.COMPLETED

    return sample


__all__ = ["generate"]
