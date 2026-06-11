from __future__ import annotations

import asyncio
import math
import os
from collections import defaultdict
from typing import Any


DEFAULT_ARCHIVE_PATH = "/root/workspace/erdos/data/archive.json"
DEFAULT_MODEL_PATH = "/root/workspace/models/Qwen3-8B"

_TOKENIZER = None
_TOKENIZER_PATH = None
_ARCHIVES: dict[tuple[str, int, float, int, int, int], Any] = {}


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _get_tokenizer(args: Any):
    global _TOKENIZER, _TOKENIZER_PATH
    from transformers import AutoTokenizer

    tokenizer_path = (
        _get_arg(args, "tokenizer_model", None)
        or _get_arg(args, "hf_checkpoint", None)
        or os.environ.get("ERDOS_MODEL_PATH")
        or os.environ.get("GPT_OSS_TOKENIZER")
        or DEFAULT_MODEL_PATH
    )
    if _TOKENIZER is None or _TOKENIZER_PATH != tokenizer_path:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        _TOKENIZER_PATH = tokenizer_path
    return _TOKENIZER


def _initial_states(args: Any):
    from erdos_slime.ttt_discover.erdos_env import create_random_initial_state

    seed = int(_get_arg(args, "seed", 1234))
    count = int(os.environ.get("ERDOS_NUM_INIT_STATES", _get_arg(args, "ttt_num_initial_states", 4)))
    return [create_random_initial_state(seed=seed + i) for i in range(count)]


def _get_archive(args: Any):
    from erdos_slime.ttt_discover.archive import PUCTArchive

    archive_path = str(_get_arg(args, "ttt_archive_path", os.environ.get("ERDOS_ARCHIVE_PATH", DEFAULT_ARCHIVE_PATH)))
    rollout_n = int(_get_arg(args, "n_samples_per_prompt", os.environ.get("ERDOS_ROLLOUT_N", 1)))
    puct_c = float(_get_arg(args, "ttt_puct_c", os.environ.get("ERDOS_PUCT_C", 1.0)))
    topk_children = int(_get_arg(args, "ttt_topk_children", os.environ.get("ERDOS_TOPK_CHILDREN", 2)))
    max_buffer_size = int(_get_arg(args, "ttt_max_buffer_size", 1000))
    num_initial_states = int(os.environ.get("ERDOS_NUM_INIT_STATES", _get_arg(args, "ttt_num_initial_states", 4)))
    key = (archive_path, rollout_n, puct_c, topk_children, max_buffer_size, num_initial_states)
    archive = _ARCHIVES.get(key)
    if archive is None:
        archive = PUCTArchive(
            archive_path,
            initial_states=_initial_states(args),
            rollout_n=rollout_n,
            puct_c=puct_c,
            topk_children=topk_children,
            max_buffer_size=max_buffer_size,
        )
        _ARCHIVES[key] = archive
    return archive


def _sample_slot(sample: Any, args: Any) -> int | str:
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return int(group_index)
    sample_index = getattr(sample, "index", None)
    if sample_index is not None:
        group_size = max(int(_get_arg(args, "n_samples_per_prompt", 1) or 1), 1)
        return int(sample_index) // group_size
    return getattr(sample, "session_id", None) or "unknown"


def _group_uid(args: Any, sample: Any) -> str:
    metadata = getattr(sample, "metadata", None) if isinstance(getattr(sample, "metadata", None), dict) else {}
    if metadata and metadata.get("ttt_group_uid"):
        return str(metadata["ttt_group_uid"])

    run_id = os.environ.get("ERDOS_RUN_ID", "").strip()
    rollout_step = (
        metadata.get("start_rollout_id")
        or metadata.get("rollout_id")
        or _get_arg(args, "global_step", None)
        or _get_arg(args, "rollout_id", None)
        or "step"
    )
    return f"{run_id}:{rollout_step}:group:{_sample_slot(sample, args)}" if run_id else f"{rollout_step}:group:{_sample_slot(sample, args)}"


def _build_prompt(args: Any, state: Any) -> str:
    from erdos_slime.ttt_discover.erdos_env import build_erdos_prompt

    return build_erdos_prompt(
        state,
        budget_s=int(_get_arg(args, "ttt_sandbox_timeout_s", os.environ.get("ERDOS_BUDGET_S", 1000))),
        cpus=int(_get_arg(args, "ttt_sandbox_cpus", os.environ.get("ERDOS_CPUS", 2))),
        target_c5=float(_get_arg(args, "ttt_target_c5", os.environ.get("ERDOS_TARGET_C5", 0.38080))),
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], args: Any) -> str:
    reasoning_effort = _get_arg(args, "reasoning_effort", "high")
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _decode_response(tokenizer: Any, response_tokens: list[int], fallback: str) -> str:
    if response_tokens:
        return tokenizer.decode(response_tokens, skip_special_tokens=False)
    return fallback


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any]):
    from slime.utils.http_utils import post

    archive = _get_archive(args)
    group_uid = _group_uid(args, sample)
    parent_state = archive.acquire_group(group_uid)
    tokenizer = _get_tokenizer(args)

    prompt_text = _build_prompt(args, parent_state)
    prompt_str = _apply_chat_template(tokenizer, [{"role": "user", "content": prompt_text}], args)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

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
    output_logprobs = meta_info.get("output_token_logprobs") or []
    response_tokens = [item[1] for item in output_logprobs]
    response_log_probs = [item[0] for item in output_logprobs]

    train_cap = int(os.environ.get("ERDOS_TRAIN_MAX_RESPONSE_TOKENS", "0") or "0")
    capped_for_train = train_cap > 0 and len(response_tokens) > train_cap
    if capped_for_train:
        response_tokens = response_tokens[:train_cap]
        response_log_probs = response_log_probs[:train_cap]

    response_text = _decode_response(tokenizer, response_tokens, output.get("text", ""))
    sample.prompt = prompt_str
    sample.tokens = prompt_ids + response_tokens
    sample.response_length = len(response_tokens)
    sample.response = response_text
    sample.rollout_log_probs = response_log_probs
    sample.rollout_id = int(_sample_slot(sample, args)) if isinstance(_sample_slot(sample, args), int) else sample.index
    sample.train_metadata = dict(sample.train_metadata or {})
    sample.train_metadata.update(
        {
            "ttt_archive_path": str(_get_arg(args, "ttt_archive_path", os.environ.get("ERDOS_ARCHIVE_PATH", DEFAULT_ARCHIVE_PATH))),
            "ttt_group_uid": group_uid,
            "ttt_state_id": parent_state.id,
            "ttt_parent_state": parent_state.to_dict(),
            "ttt_parent_value": float(parent_state.value),
            "ttt_parent_raw_score": parent_state.raw_score,
            "reasoning_effort": _get_arg(args, "reasoning_effort", "high"),
        }
    )
    sample.metadata.update(sample.train_metadata)

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


def _extract_code(response: str) -> str | None:
    from erdos_slime.ttt_discover.sandbox import extract_python_code

    return extract_python_code(response or "")


def _score_sample(args: Any, sample: Any) -> float:
    from erdos_slime.ttt_discover.erdos_env import score_erdos_result
    from erdos_slime.ttt_discover.sandbox import evaluate_python_code

    archive = _get_archive(args)
    metadata = getattr(sample, "train_metadata", None) or getattr(sample, "metadata", None) or {}
    group_uid = metadata.get("ttt_group_uid") or _group_uid(args, sample)
    parent_state = archive.acquire_group(group_uid)
    code = _extract_code(getattr(sample, "response", ""))

    extra = {
        "ttt_group_uid": group_uid,
        "ttt_state_id": parent_state.id,
        "ttt_valid": False,
        "ttt_raw_score": None,
        "ttt_reward": 0.0,
        "ttt_error": "",
        "raw_reward": 1e9,
    }

    if not code:
        extra["ttt_error"] = "No python code block found"
        archive.submit_child(group_uid, None)
        sample.metadata.update(extra)
        return 0.0

    result = evaluate_python_code(
        code,
        state=parent_state,
        timeout_s=int(_get_arg(args, "ttt_sandbox_timeout_s", os.environ.get("ERDOS_BUDGET_S", 1000))),
        work_dir=_get_arg(args, "ttt_sandbox_work_dir", None),
    )
    if result.error is not None:
        extra["ttt_error"] = result.error[:1000]
        archive.submit_child(group_uid, None)
        sample.metadata.update(extra)
        return 0.0

    try:
        scored = score_erdos_result(
            result.output,
            code=code,
            timestep=int(_get_arg(args, "global_step", getattr(sample, "index", 0) or 0) or 0),
            stdout=result.stdout,
        )
    except Exception as exc:
        extra["ttt_error"] = str(exc)
        archive.submit_child(group_uid, None)
        sample.metadata.update(extra)
        return 0.0

    archive.submit_child(group_uid, scored.state)
    extra.update(
        {
            "ttt_valid": True,
            "ttt_raw_score": scored.raw_score,
            "ttt_reward": scored.reward,
            "ttt_message": scored.message,
            "raw_reward": scored.raw_score,
        }
    )
    sample.metadata.update(extra)
    if sample.train_metadata is not None:
        sample.train_metadata.update(extra)
    return float(scored.reward)


async def reward(args: Any, sample: Any) -> float:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score_sample, args, sample)


def _solve_adaptive_beta(values: list[float], target_kl: float) -> float:
    if len(values) < 2 or max(values) - min(values) < 1e-12 or target_kl <= 0:
        return 0.0

    log_k = math.log(len(values))
    target = min(float(target_kl), log_k - 1e-8)

    def kl_for(beta: float) -> float:
        shifted = [beta * (value - max(values)) for value in values]
        log_z = math.log(sum(math.exp(x) for x in shifted))
        return sum(math.exp(x - log_z) * ((x - log_z) + log_k) for x in shifted)

    lo = 0.0
    hi = 1.0
    while hi < 1e8 and kl_for(hi) < target:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if kl_for(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _entropic_loo_advantages(values: list[float], target_kl: float, clip: float) -> list[float]:
    if len(values) < 2 or max(values) - min(values) < 1e-12:
        return [0.0] * len(values)
    beta = _solve_adaptive_beta(values, target_kl)
    if beta == 0.0:
        return [0.0] * len(values)
    max_value = max(values)
    exp_values = [math.exp(max(-80.0, min(80.0, beta * (value - max_value)))) for value in values]
    total = sum(exp_values)
    advantages = []
    for exp_value in exp_values:
        loo = (total - exp_value) / max(1, len(values) - 1)
        advantage = exp_value / (loo + 1e-12) - 1.0
        if clip > 0:
            advantage = max(-clip, min(clip, advantage))
        advantages.append(float(advantage))
    return advantages


def ttt_reward_post_process(args: Any, samples: list[Any]):
    reward_values = [float(sample.get_reward_value(args) or 0.0) for sample in samples]
    raw_scores = []
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for pos, sample in enumerate(samples):
        metadata = getattr(sample, "train_metadata", None) or getattr(sample, "metadata", None) or {}
        raw_scores.append(float(metadata.get("ttt_raw_score") or metadata.get("raw_reward") or reward_values[pos]))
        group_uid = metadata.get("ttt_group_uid") or _group_uid(args, sample)
        grouped[str(group_uid)].append((pos, reward_values[pos]))

    advantages = [0.0] * len(samples)
    target_kl = float(_get_arg(args, "ttt_entropic_target_kl", math.log(2.0)))
    advantage_clip = float(_get_arg(args, "ttt_advantage_clip", 20.0))
    for members in grouped.values():
        values = [reward for _, reward in members]
        group_advantages = _entropic_loo_advantages(values, target_kl, advantage_clip)
        for (sample_pos, _), advantage in zip(members, group_advantages, strict=False):
            advantages[sample_pos] = advantage
            samples[sample_pos].metadata["ttt_advantage"] = advantage
            if samples[sample_pos].train_metadata is not None:
                samples[sample_pos].train_metadata["ttt_advantage"] = advantage

    return raw_scores, advantages


def ttt_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    import torch

    rewards = rollout_data.get("rewards") or []
    kl = rollout_data.get("kl") or []
    loss_masks = rollout_data.get("loss_masks") or []
    response_lengths = rollout_data.get("response_lengths") or []
    advantages = []
    for idx, reward_value in enumerate(rewards):
        if idx < len(kl):
            template = kl[idx]
            adv = torch.ones_like(template, dtype=torch.float32) * float(reward_value)
        elif idx < len(loss_masks):
            adv = torch.ones_like(loss_masks[idx], dtype=torch.float32) * float(reward_value)
        else:
            length = int(response_lengths[idx]) if idx < len(response_lengths) else 1
            adv = torch.full((length,), float(reward_value), dtype=torch.float32)
        advantages.append(adv)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [adv.clone() for adv in advantages]


def ttt_reinforce_loss(args: Any, batch: dict[str, Any], logits: torch.Tensor, sum_of_sample_mean: Any):
    import torch

    from slime.backends.megatron_utils.loss import compute_approx_kl, get_log_probs_and_entropy

    advantages = torch.cat(batch["advantages"], dim=0).detach()
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=batch.get("max_seq_lens", None),
    )[1]
    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    old_log_probs = None
    if getattr(args, "use_rollout_logprobs", False) and batch.get("rollout_log_probs"):
        old_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
    elif batch.get("log_probs"):
        old_log_probs = torch.cat(batch["log_probs"], dim=0)

    if old_log_probs is not None and getattr(args, "use_rollout_logprobs", False):
        is_weights = torch.exp(log_probs.detach() - old_log_probs).detach()
        clip = float(_get_arg(args, "ttt_is_clip", 0.0) or 0.0)
        if clip > 0:
            is_weights = is_weights.clamp(max=clip)
    else:
        is_weights = torch.ones_like(log_probs)

    pg_loss = sum_of_sample_mean(-is_weights * log_probs * advantages)
    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
    loss = pg_loss - float(getattr(args, "entropy_coef", 0.0)) * entropy_loss

    ppo_kl = sum_of_sample_mean(old_log_probs - log_probs) if old_log_probs is not None else torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    log = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ttt_is_mean": sum_of_sample_mean(is_weights).clone().detach(),
        "ttt_advantage_mean": sum_of_sample_mean(advantages).clone().detach(),
    }

    if getattr(args, "use_kl_loss", False):
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + float(args.kl_loss_coef) * kl_loss
        log["loss"] = loss.clone().detach()
        log["kl_loss"] = kl_loss.clone().detach()

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()
    return loss, log
