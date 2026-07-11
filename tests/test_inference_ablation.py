import asyncio
import json
import re
import threading
import time
from pathlib import Path

import yaml

from guidance_ttt.inference_ablation import PolyominoInferenceAblationRunner
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.prompts import build_polyomino_inference_ablation_prompt
from guidance_ttt.state import LibraryEntry, LLMResponse, VerificationResult, make_root_node
from guidance_ttt.tasks import get_task_spec


def _write_seed(path: Path) -> tuple[object, LibraryEntry]:
    root = make_root_node(problem_id="polyomino_packing", raw_score=74.5, reward=74.5)
    root.id = "seed-root"
    entry = LibraryEntry(
        id="seed-entry",
        parent_id=root.id,
        problem_id="polyomino_packing",
        timestep=0,
        guidance="Bootstrap execution without guidance.",
        execution_thinking="Use a robust shelf baseline.",
        solution="#include <bits/stdc++.h>\nint main() { return 0; }",
        verifier_reward=74.5,
        verifier_raw_score=74.5,
        verifier_status="valid",
        verifier_message="accepted",
        summary="Enumerate orientations and use a robust shelf baseline.",
        reusable_idea="Robust shelf baseline.",
        failure_mode=None,
        metadata={"bootstrap": True},
    )
    root.entry_id = entry.id
    path.write_text(
        json.dumps(
            {
                "nodes": {root.id: root.to_dict()},
                "entries": {entry.id: entry.to_dict()},
                "groups": {},
                "best_node_id": root.id,
            }
        )
    )
    return root, entry


def _config(tmp_path: Path, *, num_steps: int = 1) -> dict:
    seed_path = tmp_path / "seed.json"
    _write_seed(seed_path)
    return {
        "run": {
            "output_dir": str(tmp_path / "output"),
            "num_steps": num_steps,
            "seed_library_path": str(seed_path),
        },
        "task": {
            "id": "polyomino_packing",
            "frontiercs": {"problem_id": "0", "n_cases": 70},
        },
        "search": {
            "group_size": 16,
            "generation_concurrency": 8,
            "evaluation_concurrency": 16,
            "eval_timeout": 10,
            "puct_c": 1.0,
            "topk_children": 2,
        },
        "llm": {
            "execution": {
                "provider": "openai_compatible",
                "model": "openai/gpt-oss-120b",
                "temperature": 0.0,
                "max_tokens": 512,
                "max_model_len": 4096,
            }
        },
    }


class ConcurrentFakeClient:
    def __init__(self, *, fail_index: int | None = None):
        self.fail_index = fail_index
        self.calls: list[int] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, request):
        assert request.temperature == 0.0
        index = int(request.metadata["candidate_index"])
        self.calls.append(index)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            if index == self.fail_index:
                raise RuntimeError("synthetic API failure")
            text = f"""<think>Improve candidate {index}.</think>
<summary>Candidate {index} uses a distinct packing refinement.</summary>
<solution>
```cpp
#include <bits/stdc++.h>
// candidate-index: {index}
int main() {{ return 0; }}
```
</solution>"""
            return LLMResponse(
                text=text,
                model="openai/gpt-oss-120b",
                finish_reason="stop",
                usage={"completion_tokens": 100},
            )
        finally:
            self.active -= 1


class ConcurrentVerifier:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[int] = []

    def __call__(self, text, *, timeout_s, config):
        match = re.search(r"candidate-index: (\d+)", text)
        assert match is not None
        index = int(match.group(1))
        with self.lock:
            self.calls.append(index)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            return VerificationResult(
                reward=75.0 + index,
                raw_score=75.0 + index,
                valid=True,
                status="valid",
                message="accepted",
                artifacts={"candidate_index": index},
            )
        finally:
            with self.lock:
                self.active -= 1


def _runner(tmp_path: Path, client, verifier) -> PolyominoInferenceAblationRunner:
    return PolyominoInferenceAblationRunner(
        _config(tmp_path),
        client=client,
        verifier=verifier,
        prompt_token_counter=lambda _system, _user: 256,
    )


def test_fixed_prompt_attaches_only_selected_parent_summary_and_full_code(tmp_path):
    root, entry = _write_seed(tmp_path / "seed.json")
    prompt = build_polyomino_inference_ablation_prompt(
        problem_prompt=get_task_spec("polyomino_packing").problem_prompt,
        selected_node=root,
        selected_entry=entry,
    )

    assert entry.summary in prompt.user
    assert entry.solution in prompt.user
    assert "FrontierCS score: 74.5" in prompt.user
    assert "<global_best>" not in prompt.user
    assert "<local_failures>" not in prompt.user
    assert "<guidance>" not in prompt.user
    assert prompt.user.index("<summary>") < prompt.user.index("<solution>", prompt.user.index("Return exactly"))


def test_one_step_uses_eight_generation_and_sixteen_evaluation_workers(tmp_path):
    client = ConcurrentFakeClient()
    verifier = ConcurrentVerifier()
    runner = _runner(tmp_path, client, verifier)

    summary = asyncio.run(runner.run())
    snapshot = GuidanceLibrary(runner.library_path, rollout_n=16).snapshot()
    ablation_entries = [
        entry
        for entry in snapshot["entries"].values()
        if (entry.get("metadata") or {}).get("inference_ablation")
    ]

    assert sorted(client.calls) == list(range(16))
    assert client.max_active == 8
    assert verifier.max_active == 16
    assert len(verifier.calls) == 16
    assert len(ablation_entries) == 16
    assert {entry["parent_id"] for entry in ablation_entries} == {"seed-root"}
    assert {entry["guidance"] for entry in ablation_entries} == {""}
    assert {entry["verifier_status"] for entry in ablation_entries} == {"valid"}
    assert snapshot["groups"]["inference:1:group0"]["finalized"] is True
    assert snapshot["puct_T"] == 1
    assert summary["run"]["training_enabled"] is False
    assert summary["steps"][0]["candidate_count"] == 16
    assert (runner.output_dir / "run_summary.md").exists()


def test_generation_failure_is_written_as_invalid_child_without_cancelling_group(tmp_path):
    client = ConcurrentFakeClient(fail_index=3)
    verifier = ConcurrentVerifier()
    runner = _runner(tmp_path, client, verifier)

    asyncio.run(runner.run())
    snapshot = GuidanceLibrary(runner.library_path, rollout_n=16).snapshot()
    entries = [
        entry
        for entry in snapshot["entries"].values()
        if (entry.get("metadata") or {}).get("inference_ablation")
    ]

    assert len(client.calls) == 16
    assert len(verifier.calls) == 15
    assert sum(entry["verifier_status"] == "execution_error" for entry in entries) == 1
    assert snapshot["groups"]["inference:1:group0"]["finalized"] is True


def test_resume_only_generates_missing_candidate_indices(tmp_path):
    client = ConcurrentFakeClient()
    verifier = ConcurrentVerifier()
    runner = _runner(tmp_path, client, verifier)
    library = runner.prepare()
    selected = library.acquire_group("inference:1:group0", visible_timestep_exclusive=1)
    for index in range(3):
        library.submit_child(
            "inference:1:group0",
            LibraryEntry(
                id=f"partial-{index}",
                parent_id=selected.id,
                problem_id="polyomino_packing",
                timestep=1,
                guidance="",
                execution_thinking="partial",
                solution=f"// partial {index}",
                verifier_reward=70.0 + index,
                verifier_raw_score=70.0 + index,
                verifier_status="valid",
                verifier_message="accepted",
                summary=f"partial summary {index}",
                reusable_idea=f"partial {index}",
                failure_mode=None,
                metadata={
                    "inference_ablation": True,
                    "group_uid": "inference:1:group0",
                    "candidate_index": index,
                    "selected_node_id": selected.id,
                },
            ),
        )
    assert library.snapshot()["puct_T"] == 0

    asyncio.run(runner.run())
    snapshot = GuidanceLibrary(runner.library_path, rollout_n=16).snapshot()

    assert sorted(client.calls) == list(range(3, 16))
    assert snapshot["groups"]["inference:1:group0"]["submitted"] == 16
    assert snapshot["groups"]["inference:1:group0"]["finalized"] is True
    assert snapshot["puct_T"] == 1


def test_modal_inference_ablation_config_and_launcher_are_inference_only():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_1gpu_gpt_oss_120b_inference_puct.yaml")
    config = yaml.safe_load(config_path.read_text())
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()
    launcher = Path("scripts/run_modal_polyomino_inference_puct.sh").read_text()

    assert config["run"]["num_steps"] == 50
    assert config["search"]["group_size"] == 16
    assert config["search"]["generation_concurrency"] == 8
    assert config["search"]["evaluation_concurrency"] == 16
    assert config["llm"]["execution"]["model"] == "openai/gpt-oss-120b"
    assert config["llm"]["execution"]["temperature"] == 0.0
    assert "gpu=GPT_OSS_120B_SINGLE_GPU_CONFIG" in script
    assert "judge = _start_judge(workers=16)" in script
    assert "guidance_ttt.inference_ablation" in script
    assert "run_gpt_oss_120b_inference_puct" in launcher
