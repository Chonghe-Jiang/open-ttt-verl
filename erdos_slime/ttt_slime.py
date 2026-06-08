from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any

import torch


DEFAULT_ARCHIVE_PATH = "/root/workspace/erdos/data/archive.json"
DEFAULT_MODEL_PATH = "/root/workspace/models/gpt-oss-20b"

_TOKENIZER = None
_TOKENIZER_PATH = None
_ARCHIVES: dict[tuple[str, int, float, int, int], Any] = {}


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _get_tokenizer(args: Any):
    global _TOKENIZER, _TOKENIZER_PATH
    from transformers import AutoTokenizer

    tokenizer_path = (
        _get_arg(args, "tokenizer_model", None)
        or _get_arg(args, "hf_checkpoint", None)
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
    return [create_random_initial_state(seed=seed + i) for i in range(4)]


def _get_archive(args: Any):
    from erdos_slime.ttt_discover.archive import PUCTArchive

    archive_path = str(_get_arg(args, "ttt_archive_path", DEFAULT_ARCHIVE_PATH))
    rollout_n = int(_get_arg(args, "n_samples_per_prompt", 1))
    puct_c = float(_get_arg(args, "ttt_puct_c", 1.0))
    topk_children = int(_get_arg(args, "ttt_topk_children", 2))
    max_buffer_size = int(_get_arg(args, "ttt_max_buffer_size", 1000))
    key = (archive_path, rollout_n, puct_c, topk_children, max_buffer_size)
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


def _group_uid(sample: Any) -> str:
    group_index = sample.group_index if getattr(sample, "group_index", None) is not None else sample.index
    return f"group:{group_index}"


def _build_prompt(args: Any, state: Any) -> str:
    from erdos_slime.ttt_discover.erdos_env import build_erdos_prompt

    reasoning_effort = _get_arg(args, "reasoning_effort", "high")
    prompt = build_erdos_prompt(
        state,
        budget_s=int(_get_arg(args, "ttt_sandbox_timeout_s", 60)),
        cpus=int(_get_arg(args, "ttt_sandbox_cpus", 1)),
        target_c5=float(_get_arg(args, "ttt_target_c5", 0.3808)),
    )
    return f"Reasoning: {reasoning_effort}\n\n{prompt}"


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


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any]):
    from slime.utils.http_utils import post

    archive = _get_archive(args)
    group_uid = _group_uid(sample)
    parent_state = archive.acquire_group(group_uid)
    tokenizer = _get_tokenizer(args)

    prompt_text = _build_prompt(args, parent_state)
    prompt_str = _apply_chat_template(tokenizer, [{"role": "user", "content": prompt_text}], args)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    output = await post(url, payload)
    meta_info = output.get("meta_info", {})

    if "output_token_logprobs" in meta_info:
        new_response_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in meta_info["output_token_logprobs"]]
    else:
        new_response_tokens = []
        new_response_log_probs = []

    sample.prompt = prompt_str
    sample.tokens = prompt_ids + new_response_tokens
    sample.response_length = len(new_response_tokens)
    sample.response = output.get("text", "")
    sample.rollout_log_probs = new_response_log_probs
    sample.rollout_id = sample.group_index if getattr(sample, "group_index", None) is not None else sample.index
    sample.train_metadata = dict(sample.train_metadata or {})
    sample.train_metadata.update(
        {
            "ttt_group_uid": group_uid,
            "ttt_state_id": parent_state.id,
            "ttt_parent_value": float(parent_state.value),
            "ttt_parent_raw_score": parent_state.raw_score,
            "reasoning_effort": _get_arg(args, "reasoning_effort", "high"),
        }
    )
    sample.metadata.update(sample.train_metadata)

    finish_reason = meta_info.get("finish_reason", {}).get("type", "stop")
    if finish_reason == "length":
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
    group_uid = (getattr(sample, "train_metadata", None) or sample.metadata or {}).get("ttt_group_uid") or _group_uid(sample)
    parent_state = archive.acquire_group(group_uid)
    code = _extract_code(getattr(sample, "response", ""))

    extra = {
        "ttt_group_uid": group_uid,
        "ttt_state_id": parent_state.id,
        "ttt_valid": False,
        "ttt_raw_score": None,
        "ttt_error": "",
    }

    if not code:
        extra["ttt_error"] = "No python code block found"
        archive.submit_child(group_uid, None)
        sample.metadata.update(extra)
        return 0.0

    result = evaluate_python_code(
        code,
        state=parent_state,
        timeout_s=int(_get_arg(args, "ttt_sandbox_timeout_s", 60)),
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
            timestep=int(getattr(sample, "index", 0) or 0),
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
        }
    )
    sample.metadata.update(extra)
    return float(scored.reward)


async def reward(args: Any, sample: Any) -> float:
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score_sample, args, sample)


def _solve_adaptive_beta(values: list[float], target_kl: float) -> float:
    if len(values) < 2 or all(math.isclose(values[0], value) for value in values):
        return 0.0

    log_k = math.log(len(values))

    def kl_for(beta: float) -> float:
        shifted = [beta * (value - max(values)) for value in values]
        log_z = math.log(sum(math.exp(x) for x in shifted))
        probs = [math.exp(x - log_z) for x in shifted]
        return sum(p * ((x - log_z) + log_k) for p, x in zip(probs, shifted))

    lo = 0.0
    hi = 1.0
    while hi < 1e6 and kl_for(hi) < target_kl:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_for(mid) < target_kl:
            lo = mid
        else:
            hi = mid
    return hi


def ttt_reward_post_process(args: Any, samples: list[Any]):
    raw_rewards = [float(sample.get_reward_value(args) or 0.0) for sample in samples]
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for pos, sample in enumerate(samples):
        metadata = getattr(sample, "train_metadata", None) or getattr(sample, "metadata", None) or {}
        group_uid = metadata.get("ttt_group_uid") or f"group:{getattr(sample, 'group_index', pos)}"
        grouped[group_uid].append((pos, raw_rewards[pos]))

    advantages = [0.0] * len(samples)
    target_kl = float(_get_arg(args, "ttt_entropic_target_kl", math.log(2.0)))
    advantage_clip = float(_get_arg(args, "ttt_advantage_clip", 20.0))
    for members in grouped.values():
        values = [reward for _, reward in members]
        if len(values) < 2 or all(math.isclose(values[0], value) for value in values):
            continue
        beta = _solve_adaptive_beta(values, min(target_kl, math.log(len(values)) - 1e-6))
        max_value = max(values)
        exp_values = [math.exp(beta * (value - max_value)) for value in values]
        total = sum(exp_values)
        for local_pos, (sample_pos, _) in enumerate(members):
            loo = (total - exp_values[local_pos]) / max(1, len(values) - 1)
            advantages[sample_pos] = exp_values[local_pos] / (loo + 1e-12) - 1.0
            if advantage_clip > 0:
                advantages[sample_pos] = max(-advantage_clip, min(advantage_clip, advantages[sample_pos]))
            samples[sample_pos].metadata["ttt_advantage"] = advantages[sample_pos]

    return raw_rewards, advantages


def ttt_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    rewards = rollout_data.get("rewards") or []
    kl = rollout_data.get("kl") or []
    advantages = []
    for reward, local_kl in zip(rewards, kl, strict=False):
        scalar = torch.as_tensor(float(reward), dtype=torch.float32, device=local_kl.device)
        adv = torch.ones_like(local_kl, dtype=torch.float32) * scalar
        if float(_get_arg(args, "kl_coef", 0.0)) != 0.0:
            adv = adv - float(args.kl_coef) * local_kl.float()
        advantages.append(adv)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [adv.clone() for adv in advantages]


def ttt_reinforce_loss(args: Any, batch: dict[str, Any], logits: torch.Tensor, sum_of_sample_mean: Any):
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
        clip = float(_get_arg(args, "ttt_is_clip", 10.0))
        is_weights = is_weights.clamp(max=clip)
    else:
        is_weights = torch.ones_like(log_probs)

    pg_token_loss = -is_weights * log_probs * advantages
    pg_loss = sum_of_sample_mean(pg_token_loss)

    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
    loss = pg_loss - float(getattr(args, "entropy_coef", 0.0)) * entropy_loss

    if old_log_probs is not None:
        ppo_kl = sum_of_sample_mean(old_log_probs - log_probs)
    else:
        ppo_kl = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)

    log = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ttt_is_mean": sum_of_sample_mean(is_weights).clone().detach(),
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
