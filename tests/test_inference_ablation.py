import asyncio
import json
import re
import threading
import time
from collections import Counter
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
            "discover_compat": True,
            "groups_per_batch": 8,
            "group_size": 16,
            "generation_concurrency": 8,
            "evaluation_concurrency": 16,
            "eval_timeout": 10,
            "puct_c": 1.0,
            "puct_q_mode": "best_child",
            "max_buffer_size": 1000,
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
        self.calls: list[tuple[int, int]] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, request):
        assert request.temperature == 0.0
        group_index = int(request.metadata["group_index"])
        index = int(request.metadata["candidate_index"])
        self.calls.append((group_index, index))
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
// group-index: {group_index}
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
        self.calls: list[tuple[int, int]] = []

    def __call__(self, text, *, timeout_s, config):
        group_match = re.search(r"group-index: (\d+)", text)
        candidate_match = re.search(r"candidate-index: (\d+)", text)
        assert group_match is not None and candidate_match is not None
        group_index = int(group_match.group(1))
        index = int(candidate_match.group(1))
        with self.lock:
            self.calls.append((group_index, index))
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
                artifacts={"group_index": group_index, "candidate_index": index},
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

    assert sorted(client.calls) == [
        (group_index, candidate_index)
        for group_index in range(8)
        for candidate_index in range(16)
    ]
    assert client.max_active == 8
    assert verifier.max_active == 16
    assert len(verifier.calls) == 128
    assert len(ablation_entries) == 128
    assert len({entry["parent_id"] for entry in ablation_entries}) == 8
    assert {entry["guidance"] for entry in ablation_entries} == {""}
    assert {entry["verifier_status"] for entry in ablation_entries} == {"valid"}
    assert all(snapshot["groups"][f"inference:1:group{index}"]["finalized"] for index in range(8))
    assert snapshot["puct_T"] == 128
    assert summary["run"]["training_enabled"] is False
    assert summary["run"]["groups_per_batch"] == 8
    assert summary["steps"][0]["candidate_count"] == 128
    assert len(summary["steps"][0]["selected_node_ids"]) == 8
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

    assert len(client.calls) == 128
    assert len(verifier.calls) == 120
    assert sum(entry["verifier_status"] == "execution_error" for entry in entries) == 8
    assert all(snapshot["groups"][f"inference:1:group{index}"]["finalized"] for index in range(8))
    failed_entry_ids = {
        entry["id"] for entry in entries if entry["verifier_status"] == "execution_error"
    }
    assert all(node["entry_id"] not in failed_entry_ids for node in snapshot["nodes"].values())


def test_resume_only_generates_missing_candidate_indices(tmp_path):
    client = ConcurrentFakeClient()
    verifier = ConcurrentVerifier()
    runner = _runner(tmp_path, client, verifier)
    library = runner.prepare()
    selected = library.acquire_group(
        "inference:1:group0",
        visible_timestep_exclusive=1,
        require_solution=True,
    )
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
                    "group_index": 0,
                    "candidate_index": index,
                    "selected_node_id": selected.id,
                },
            ),
        )
    assert library.snapshot()["puct_T"] == 3

    asyncio.run(runner.run())
    snapshot = GuidanceLibrary(runner.library_path, rollout_n=16).snapshot()

    assert len(client.calls) == 125
    counts = Counter(group_index for group_index, _candidate_index in client.calls)
    assert counts == {0: 13, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16}
    assert all(snapshot["groups"][f"inference:1:group{index}"]["submitted"] == 16 for index in range(8))
    assert all(snapshot["groups"][f"inference:1:group{index}"]["finalized"] for index in range(8))
    assert snapshot["puct_T"] == 128


def test_modal_inference_ablation_config_and_launcher_are_inference_only():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_1gpu_gpt_oss_120b_inference_puct.yaml")
    config = yaml.safe_load(config_path.read_text())
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()
    launcher = Path("scripts/run_modal_polyomino_inference_puct.sh").read_text()

    assert config["run"]["num_steps"] == 50
    assert config["search"]["discover_compat"] is True
    assert config["search"]["groups_per_batch"] == 8
    assert config["search"]["group_size"] == 16
    assert config["search"]["puct_q_mode"] == "best_child"
    assert config["search"]["generation_concurrency"] == 8
    assert config["search"]["evaluation_concurrency"] == 16
    assert config["llm"]["execution"]["model"] == "openai/gpt-oss-120b"
    assert config["llm"]["execution"]["temperature"] == 0.0
    assert "gpu=GPT_OSS_120B_SINGLE_GPU_CONFIG" in script
    assert "judge = _start_judge(workers=16)" in script
    assert "guidance_ttt.inference_ablation" in script
    assert "run_gpt_oss_120b_inference_puct" in launcher
