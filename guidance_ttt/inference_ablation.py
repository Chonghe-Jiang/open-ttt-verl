from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.llm_client import BaseLLMClient, make_llm_client
from guidance_ttt.prompts import build_polyomino_inference_ablation_prompt, extract_tag_or_none
from guidance_ttt.state import LibraryEntry, LLMRequest, LLMResponse, VerificationResult
from guidance_ttt.tasks import TaskSpec, get_task_spec


@dataclass
class GeneratedCandidate:
    group_index: int
    index: int
    response: LLMResponse | None
    error: str | None
    queue_seconds: float
    generation_seconds: float


@dataclass
class EvaluatedCandidate:
    generation: GeneratedCandidate
    verification: VerificationResult
    queue_seconds: float
    evaluation_seconds: float


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _frontier_config(task_config: dict[str, Any]) -> dict[str, Any]:
    return dict(task_config.get("frontiercs") or {})


def _fallback_summary(thinking: str) -> str:
    if thinking.strip():
        return "[Fallback summary: model omitted <summary>.]\n" + thinking.strip()
    return "[Fallback summary: model omitted both <summary> and usable <think> content.]"


def _entry_candidate_index(entry: dict[str, Any], *, group_uid: str) -> int | None:
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict) or metadata.get("group_uid") != group_uid:
        return None
    try:
        return int(metadata["candidate_index"])
    except (KeyError, TypeError, ValueError):
        return None


class PolyominoInferenceAblationRunner:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_base_dir: str | Path = ".",
        client: BaseLLMClient | None = None,
        verifier: Callable[..., VerificationResult] | None = None,
        prompt_token_counter: Callable[[str, str], int] | None = None,
    ) -> None:
        self.config = config
        self.base_dir = Path(config_base_dir).resolve()
        self.run_config = dict(config.get("run") or {})
        self.search_config = dict(config.get("search") or {})
        self.task_config = dict(config.get("task") or {})
        self.execution_config = dict((config.get("llm") or {}).get("execution") or {})
        self.task_spec: TaskSpec = get_task_spec(str(self.task_config.get("id", "polyomino_packing")))
        if self.task_spec.task_id != "polyomino_packing":
            raise ValueError("Inference ablation only supports task.id=polyomino_packing")

        self.output_dir = _resolve_path(self.run_config["output_dir"], base_dir=self.base_dir)
        self.library_path = self.output_dir / "library.json"
        self.seed_library_path = _resolve_path(self.run_config["seed_library_path"], base_dir=self.base_dir)
        self.num_steps = int(self.run_config.get("num_steps", 50))
        self.groups_per_batch = int(self.search_config.get("groups_per_batch", 8))
        self.group_size = int(self.search_config.get("group_size", 16))
        self.generation_concurrency = int(self.search_config.get("generation_concurrency", 8))
        self.evaluation_concurrency = int(self.search_config.get("evaluation_concurrency", 16))
        self.max_model_len = int(self.execution_config.get("max_model_len", 32768))
        self.max_tokens = self.execution_config.get("max_tokens")
        if self.max_tokens is None:
            raise ValueError("llm.execution.max_tokens must be explicit for the inference ablation")
        self.max_tokens = int(self.max_tokens)
        if (
            self.groups_per_batch <= 0
            or self.group_size <= 0
            or self.generation_concurrency <= 0
            or self.evaluation_concurrency <= 0
        ):
            raise ValueError("group counts, group size, and concurrency values must be positive")
        expected_sampling = {
            "discover_compat": True,
            "puct_c": 1.0,
            "puct_q_mode": "best_child",
            "max_buffer_size": 1000,
            "topk_children": 2,
        }
        mismatches = [
            f"{key}={self.search_config[key]!r} (expected {expected!r})"
            for key, expected in expected_sampling.items()
            if key in self.search_config and self.search_config[key] != expected
        ]
        if mismatches:
            raise ValueError(
                "Polyomino inference sampling must use the guidance-ttt Discover-compatible profile: "
                + "; ".join(mismatches)
            )

        self.client = client or make_llm_client(self.execution_config)
        self.verifier = verifier or self.task_spec.verify_execution_text
        self.prompt_token_counter = prompt_token_counter or self._count_prompt_tokens
        self._tokenizer: Any | None = None

    def prepare(self) -> GuidanceLibrary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        library_config = {
            "rollout_n": self.group_size,
            "puct_c": 1.0,
            "puct_q_mode": "best_child",
            "max_buffer_size": 1000,
            "topk_children": 2,
            "discover_compat": True,
            "groups_per_batch": self.groups_per_batch,
            "score_direction": self.task_spec.score_direction,
        }
        if not self.library_path.exists():
            if not self.seed_library_path.exists():
                raise FileNotFoundError(f"Seed library not found: {self.seed_library_path}")
            temporary = self.library_path.with_name(f".{self.library_path.name}.{uuid4().hex}.tmp")
            try:
                shutil.copyfile(self.seed_library_path, temporary)
                temporary.replace(self.library_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            library = GuidanceLibrary(self.library_path)
            library.configure_pristine_archive(**library_config)
            library.ensure_pristine_root_count(self.groups_per_batch)
        else:
            library = GuidanceLibrary(self.library_path, **library_config)
        _atomic_write_text(self.output_dir / "inference_config.yaml", yaml.safe_dump(self.config, sort_keys=False))
        library.assert_runtime_config(**library_config)
        return library

    async def run(self) -> dict[str, Any]:
        library = self.prepare()
        for timestep in range(1, self.num_steps + 1):
            await self._run_step(library, timestep)
            self.write_summary(library)
        return self.write_summary(library)

    async def _run_step(self, library: GuidanceLibrary, timestep: int) -> None:
        step_started = time.perf_counter()
        group_states: dict[int, dict[str, Any]] = {}
        for group_index in range(self.groups_per_batch):
            group_uid = f"inference:{timestep}:group{group_index}"
            selected_node = library.acquire_group(
                group_uid,
                visible_timestep_exclusive=timestep,
                require_solution=True,
            )
            context = library.context_for_node(selected_node, visible_timestep_exclusive=timestep)
            selected_entry = context["selected_entry"]
            if selected_entry is None or not selected_entry.solution.strip():
                raise RuntimeError(f"Selected node {selected_node.id} has no attached parent solution")
            prompt = build_polyomino_inference_ablation_prompt(
                problem_prompt=self.task_spec.problem_prompt,
                selected_node=selected_node,
                selected_entry=selected_entry,
            )
            prompt_tokens = self.prompt_token_counter(prompt.system, prompt.user)
            if prompt_tokens + self.max_tokens > self.max_model_len:
                raise RuntimeError(
                    "Fixed prompt exceeds the model context budget: "
                    f"group={group_index}, prompt_tokens={prompt_tokens}, "
                    f"max_tokens={self.max_tokens}, max_model_len={self.max_model_len}"
                )
            group_states[group_index] = {
                "group_uid": group_uid,
                "selected_node": selected_node,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }

        snapshot = library.snapshot()
        generation_jobs: list[tuple[int, str, str, int, int]] = []
        resumed_candidate_count = 0
        for group_index, state in group_states.items():
            group_uid = state["group_uid"]
            group = (snapshot.get("groups") or {}).get(group_uid) or {}
            existing_indices = {
                index
                for entry in (snapshot.get("entries") or {}).values()
                if isinstance(entry, dict)
                and (index := _entry_candidate_index(entry, group_uid=group_uid)) is not None
            }
            resumed_candidate_count += len(existing_indices)
            if group.get("finalized"):
                if len(existing_indices) != self.group_size:
                    raise RuntimeError(f"Finalized group {group_uid} has {len(existing_indices)} indexed entries")
                state["existing_indices"] = existing_indices
                continue
            missing_indices = [index for index in range(self.group_size) if index not in existing_indices]
            expected_missing = self.group_size - int(group.get("submitted", 0))
            if len(missing_indices) != expected_missing:
                raise RuntimeError(
                    f"Cannot safely resume {group_uid}: submitted={group.get('submitted', 0)}, "
                    f"indexed_entries={len(existing_indices)}"
                )
            state["existing_indices"] = existing_indices
            prompt = state["prompt"]
            generation_jobs.extend(
                (group_index, prompt.system, prompt.user, timestep, index)
                for index in missing_indices
            )

        generation_started = time.perf_counter()
        generated = await self._generate_candidates(generation_jobs)
        generation_wall_seconds = time.perf_counter() - generation_started
        evaluation_started = time.perf_counter()
        evaluated = await self._evaluate_candidates(generated)
        evaluation_wall_seconds = time.perf_counter() - evaluation_started

        for candidate in sorted(
            evaluated,
            key=lambda item: (item.generation.group_index, item.generation.index),
        ):
            state = group_states[candidate.generation.group_index]
            prompt = state["prompt"]
            entry = self._build_entry(
                candidate,
                timestep=timestep,
                group_uid=state["group_uid"],
                selected_node_id=state["selected_node"].id,
                prompt_system=prompt.system,
                prompt_user=prompt.user,
                prompt_tokens=state["prompt_tokens"],
            )
            library.submit_child(state["group_uid"], entry)

        total_seconds = time.perf_counter() - step_started
        self._record_step_attempt(
            {
                "timestep": timestep,
                "group_uids": [state["group_uid"] for state in group_states.values()],
                "selected_node_ids": [state["selected_node"].id for state in group_states.values()],
                "resumed_candidate_count": resumed_candidate_count,
                "generated_candidate_count": len(generation_jobs),
                "prompt_tokens": [state["prompt_tokens"] for state in group_states.values()],
                "generation_wall_seconds": generation_wall_seconds,
                "evaluation_wall_seconds": evaluation_wall_seconds,
                "step_wall_seconds": total_seconds,
            }
        )

    async def _generate_candidates(
        self,
        jobs: list[tuple[int, str, str, int, int]],
    ) -> list[GeneratedCandidate]:
        semaphore = asyncio.Semaphore(self.generation_concurrency)

        async def generate(
            group_index: int,
            system: str,
            user: str,
            timestep: int,
            index: int,
        ) -> GeneratedCandidate:
            queued_at = time.perf_counter()
            async with semaphore:
                started_at = time.perf_counter()
                try:
                    response = await self.client.complete(
                        LLMRequest(
                            system=system,
                            user=user,
                            model=str(self.execution_config.get("model", "openai/gpt-oss-120b")),
                            temperature=float(self.execution_config.get("temperature", 0.0)),
                            max_tokens=self.max_tokens,
                            metadata={
                                "purpose": "polyomino_inference_ablation",
                                "group_index": group_index,
                                "candidate_index": index,
                                "timestep": timestep,
                            },
                        )
                    )
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)
                return GeneratedCandidate(
                    group_index=group_index,
                    index=index,
                    response=response,
                    error=error,
                    queue_seconds=started_at - queued_at,
                    generation_seconds=time.perf_counter() - started_at,
                )

        return list(await asyncio.gather(*(generate(*job) for job in jobs)))

    async def _evaluate_candidates(self, generated: list[GeneratedCandidate]) -> list[EvaluatedCandidate]:
        semaphore = asyncio.Semaphore(self.evaluation_concurrency)
        executor = ThreadPoolExecutor(max_workers=self.evaluation_concurrency, thread_name_prefix="polyomino-eval")
        loop = asyncio.get_running_loop()

        async def evaluate(candidate: GeneratedCandidate) -> EvaluatedCandidate:
            if candidate.response is None:
                return EvaluatedCandidate(
                    generation=candidate,
                    verification=VerificationResult.execution_error(candidate.error or "LLM generation failed"),
                    queue_seconds=0.0,
                    evaluation_seconds=0.0,
                )
            queued_at = time.perf_counter()
            async with semaphore:
                started_at = time.perf_counter()
                try:
                    verification = await loop.run_in_executor(
                        executor,
                        partial(
                            self.verifier,
                            candidate.response.text,
                            timeout_s=int(self.search_config.get("eval_timeout", 340)),
                            config=_frontier_config(self.task_config),
                        ),
                    )
                except Exception as exc:
                    verification = VerificationResult.execution_error(str(exc))
                return EvaluatedCandidate(
                    generation=candidate,
                    verification=verification,
                    queue_seconds=started_at - queued_at,
                    evaluation_seconds=time.perf_counter() - started_at,
                )

        try:
            return list(await asyncio.gather(*(evaluate(candidate) for candidate in generated)))
        finally:
            executor.shutdown(wait=True)

    def _build_entry(
        self,
        candidate: EvaluatedCandidate,
        *,
        timestep: int,
        group_uid: str,
        selected_node_id: str,
        prompt_system: str,
        prompt_user: str,
        prompt_tokens: int,
    ) -> LibraryEntry:
        response = candidate.generation.response
        text = response.text if response is not None else ""
        thinking = extract_tag_or_none(text, "think") or ""
        model_summary = extract_tag_or_none(text, "summary")
        summary = model_summary.strip() if model_summary and model_summary.strip() else _fallback_summary(thinking)
        solution = self.task_spec.solution_extractor(text) or ""
        format_ok = bool(
            model_summary is not None
            and extract_tag_or_none(text, "solution") is not None
            and solution
        )
        verification = candidate.verification
        return LibraryEntry(
            id=str(uuid4()),
            parent_id=selected_node_id,
            problem_id="polyomino_packing",
            timestep=timestep,
            guidance="",
            execution_thinking=thinking,
            solution=solution,
            verifier_reward=verification.reward,
            verifier_raw_score=verification.raw_score,
            verifier_status=verification.status,
            verifier_message=verification.message,
            summary=summary,
            reusable_idea=summary,
            failure_mode=None if verification.valid else verification.status,
            metadata={
                "inference_ablation": True,
                "group_uid": group_uid,
                "group_index": candidate.generation.group_index,
                "candidate_index": candidate.generation.index,
                "selected_node_id": selected_node_id,
                "fixed_prompt": {"system": prompt_system, "user": prompt_user},
                "execution_prompt": {"system": prompt_system, "user": prompt_user},
                "prompt_tokens": prompt_tokens,
                "execution_text": text,
                "raw_model_summary": model_summary or "",
                "response_format_ok": format_ok,
                "execution_provider": self.execution_config.get("provider"),
                "execution_model": self.execution_config.get("model"),
                "execution_finish_reason": response.finish_reason if response else "error",
                "execution_response_usage": response.usage if response else {},
                "execution_response_metadata": response.metadata if response else {},
                "generation_error": candidate.generation.error,
                "generation_queue_seconds": candidate.generation.queue_seconds,
                "generation_seconds": candidate.generation.generation_seconds,
                "evaluation_queue_seconds": candidate.queue_seconds,
                "evaluation_seconds": candidate.evaluation_seconds,
                "verification_artifacts": verification.artifacts,
                "task": self.task_config,
            },
        )

    def _count_prompt_tokens(self, system: str, user: str) -> int:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.execution_config.get("tokenizer") or self.execution_config["model"]),
                trust_remote_code=bool(self.execution_config.get("trust_remote_code", True)),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        token_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(token_ids)

    def _record_step_attempt(self, record: dict[str, Any]) -> None:
        path = self.output_dir / "step_timings.json"
        payload = json.loads(path.read_text()) if path.exists() else {"attempts": []}
        payload.setdefault("attempts", []).append(record)
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))

    def write_summary(self, library: GuidanceLibrary) -> dict[str, Any]:
        snapshot = library.snapshot()
        timings_path = self.output_dir / "step_timings.json"
        timings = json.loads(timings_path.read_text()) if timings_path.exists() else {"attempts": []}
        latest_timing = {int(item["timestep"]): item for item in timings.get("attempts", [])}
        entries_by_step: dict[int, list[dict[str, Any]]] = {}
        seed_entries: list[dict[str, Any]] = []
        for entry in (snapshot.get("entries") or {}).values():
            if not isinstance(entry, dict):
                continue
            if (entry.get("metadata") or {}).get("inference_ablation"):
                entries_by_step.setdefault(int(entry["timestep"]), []).append(entry)
            elif int(entry.get("timestep", 0)) == 0:
                seed_entries.append(entry)

        seed_scores = [
            float(entry["verifier_raw_score"])
            for entry in seed_entries
            if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
        ]
        valid_scores = list(seed_scores)
        steps: list[dict[str, Any]] = []
        for timestep in sorted(entries_by_step):
            entries = sorted(
                entries_by_step[timestep],
                key=lambda item: (
                    int((item.get("metadata") or {}).get("group_index", 0)),
                    int((item.get("metadata") or {}).get("candidate_index", -1)),
                ),
            )
            step_scores = [
                float(entry["verifier_raw_score"])
                for entry in entries
                if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
            ]
            valid_scores.extend(step_scores)
            steps.append(
                {
                    "timestep": timestep,
                    "selected_node_ids": list(
                        dict.fromkeys(
                            (entry.get("metadata") or {}).get("selected_node_id")
                            for entry in entries
                            if (entry.get("metadata") or {}).get("selected_node_id")
                        )
                    ),
                    "candidate_count": len(entries),
                    "valid_count": len(step_scores),
                    "best_frontiercs_score": max(step_scores) if step_scores else None,
                    "cumulative_best_frontiercs_score": max(valid_scores) if valid_scores else None,
                    "timing": latest_timing.get(timestep),
                    "candidates": [self._summary_candidate(entry) for entry in entries],
                }
            )
        best_node = (snapshot.get("nodes") or {}).get(snapshot.get("best_node_id")) or {}
        summary = {
            "run": {
                "output_dir": str(self.output_dir),
                "model": self.execution_config.get("model"),
                "temperature": self.execution_config.get("temperature"),
                "num_steps": self.num_steps,
                "groups_per_batch": self.groups_per_batch,
                "group_size": self.group_size,
                "generation_concurrency": self.generation_concurrency,
                "evaluation_concurrency": self.evaluation_concurrency,
                "training_enabled": False,
            },
            "seed": {
                "entry_count": len(seed_entries),
                "best_frontiercs_score": max(seed_scores) if seed_scores else None,
            },
            "steps": steps,
            "best": {
                "node_id": snapshot.get("best_node_id"),
                "frontiercs_score": best_node.get("raw_score"),
                "value": best_node.get("value"),
            },
        }
        _atomic_write_text(self.output_dir / "run_summary.json", json.dumps(summary, indent=2, sort_keys=True))
        _atomic_write_text(self.output_dir / "run_summary.md", self._summary_markdown(summary))
        return summary

    @staticmethod
    def _summary_candidate(entry: dict[str, Any]) -> dict[str, Any]:
        metadata = entry.get("metadata") or {}
        return {
            "group_index": metadata.get("group_index"),
            "candidate_index": metadata.get("candidate_index"),
            "entry_id": entry.get("id"),
            "status": entry.get("verifier_status"),
            "frontiercs_score": entry.get("verifier_raw_score"),
            "response_format_ok": metadata.get("response_format_ok"),
            "finish_reason": metadata.get("execution_finish_reason"),
            "generation_seconds": metadata.get("generation_seconds"),
            "evaluation_seconds": metadata.get("evaluation_seconds"),
            "generation_error": metadata.get("generation_error"),
        }

    @staticmethod
    def _summary_markdown(summary: dict[str, Any]) -> str:
        run = summary["run"]
        lines = [
            "# Polyomino Inference-Only PUCT Run",
            "",
            f"- Model: `{run['model']}`",
            f"- Temperature: `{run['temperature']}`",
            f"- Geometry: `{run['groups_per_batch']} x {run['group_size']}` candidates per step",
            (
                f"- Concurrency: generation `{run['generation_concurrency']}`, "
                f"evaluation `{run['evaluation_concurrency']}`"
            ),
            f"- Training enabled: `{run['training_enabled']}`",
            "",
            "## Per-Step FrontierCS Scores",
            "",
            "| Step | Parent | Valid | Best score | Cumulative best | Generation s | Evaluation s | Total s |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for step in summary["steps"]:
            timing = step.get("timing") or {}
            generation_seconds = _format_seconds(timing.get("generation_wall_seconds"))
            evaluation_seconds = _format_seconds(timing.get("evaluation_wall_seconds"))
            step_seconds = _format_seconds(timing.get("step_wall_seconds"))
            best_score = _format_score(step["best_frontiercs_score"])
            cumulative_best = _format_score(step["cumulative_best_frontiercs_score"])
            lines.append(
                f"| {step['timestep']} | `{','.join(step['selected_node_ids'])}` | "
                f"{step['valid_count']}/{step['candidate_count']} | {best_score} | {cumulative_best} | "
                f"{generation_seconds} | {evaluation_seconds} | {step_seconds} |"
            )
        lines.extend(
            [
                "",
                "Full prompts, responses, verifier artifacts, and per-candidate timings are stored in `library.json`.",
                "",
            ]
        )
        return "\n".join(lines)


def _format_seconds(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _format_score(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference-only PUCT search for Polyomino Packing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.num_steps is not None:
        config.setdefault("run", {})["num_steps"] = args.num_steps
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir
    runner = PolyominoInferenceAblationRunner(config, config_base_dir=Path.cwd())
    summary = asyncio.run(runner.run())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
