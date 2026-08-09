import asyncio
import json

import pytest
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import truncate_prompt_ids

from guidance_ttt.agent_loop import (
    EXECUTION_SUMMARY_SECTIONS,
    AgentLoopBase,
    GuidanceExecutionAgentLoop,
    _EXECUTION_CACHE_WARM_STATES,
    _GUIDANCE_LIBRARIES,
    _execution_cache_warm_key,
    _shared_guidance_library,
    _normalize_task_config,
    _verifier_config_from_task_config,
    _verify_execution_without_fallback,
    build_agent_loop_output,
    build_execution_summary,
)
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.state import LLMRequest, LLMResponse, LibraryEntry, LibraryNode, VerificationResult
from guidance_ttt.tasks import get_task_spec
from guidance_ttt.verifier.frontiercs_adapter import FrontierCSResult


@pytest.mark.parametrize(
    ("truncation", "expected"),
    [
        ("left", [4, 5, 6, 7]),
        ("right", [0, 1, 2, 3]),
        ("middle", [0, 1, 6, 7]),
    ],
)
def test_dynamic_agent_loop_prompt_is_truncated_before_generation(truncation, expected):
    assert truncate_prompt_ids(list(range(8)), max_length=4, truncation=truncation) == expected


def test_dynamic_agent_loop_prompt_error_is_actionable():
    with pytest.raises(ValueError, match=r"6 tokens.*prompt_length=4.*before generating"):
        truncate_prompt_ids(list(range(6)), max_length=4, truncation="error")


def test_dynamic_agent_loop_prompt_within_budget_is_unchanged():
    prompt_ids = [1, 2, 3]

    assert truncate_prompt_ids(prompt_ids, max_length=4, truncation="middle") is prompt_ids


@pytest.mark.anyio
async def test_apply_chat_template_truncates_dynamic_prompt_before_it_is_returned(monkeypatch):
    class StubAgentLoop(AgentLoopBase):
        async def run(self, sampling_params, **kwargs):
            raise NotImplementedError

    loop = StubAgentLoop.__new__(StubAgentLoop)
    loop.processor = None
    loop.tokenizer = object()
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []
    loop.rollout_config = OmegaConf.create({"prompt_length": 4})
    loop.data_config = OmegaConf.create({"truncation": "middle"})
    loop.loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        "verl.experimental.agent_loop.agent_loop.apply_chat_template",
        lambda *args, **kwargs: list(range(8)),
    )

    prompt_ids = await loop.apply_chat_template([{"role": "user", "content": "ignored"}])

    assert prompt_ids == [0, 1, 6, 7]


def test_agent_loop_output_trains_only_guidance_tokens():
    output = build_agent_loop_output(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_logprobs=[-0.1, -0.2],
        reward=3.0,
        extra_fields={"solution": "def run(): pass"},
    )

    assert output.prompt_ids == [1, 2, 3]
    assert output.response_ids == [4, 5]
    assert output.response_mask == [1, 1]
    assert output.reward_score == 3.0
    assert output.extra_fields["solution"] == "def run(): pass"


def test_execution_cache_warm_key_ignores_guidance_suffix_only():
    first = _execution_cache_warm_key(
        system="system",
        user="<problem>same</problem>\n<selected_parent>code</selected_parent>\n<guidance>first</guidance>",
    )
    second = _execution_cache_warm_key(
        system="system",
        user="<problem>same</problem>\n<selected_parent>code</selected_parent>\n<guidance>second</guidance>",
    )
    different_parent = _execution_cache_warm_key(
        system="system",
        user="<problem>same</problem>\n<selected_parent>other</selected_parent>\n<guidance>first</guidance>",
    )

    assert first == second
    assert first != different_parent


@pytest.mark.anyio
async def test_cache_warm_first_holds_followers_until_leader_finishes():
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()
    calls = []

    class ControlledClient:
        async def complete(self, request):
            calls.append(dict(request.metadata))
            if request.metadata["cache_warm_role"] == "leader":
                leader_started.set()
                await release_leader.wait()
            return LLMResponse(text="ok", model="test", finish_reason="stop")

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.execution_llm_config = {
        "provider": "test",
        "model": "test",
        "concurrency": 4,
        "cache_warm_first": True,
    }
    loop.execution_concurrency = 4
    loop.execution_client = ControlledClient()
    _EXECUTION_CACHE_WARM_STATES.clear()

    leader_request = LLMRequest("s", "u1", "test", 0, None, {"cache_warm_key": "shared"})
    follower_request = LLMRequest("s", "u2", "test", 0, None, {"cache_warm_key": "shared"})
    leader = asyncio.create_task(loop._complete_execution(leader_request))
    await leader_started.wait()
    follower = asyncio.create_task(loop._complete_execution(follower_request))
    await asyncio.sleep(0.01)

    assert [call["cache_warm_role"] for call in calls] == ["leader"]

    release_leader.set()
    await asyncio.gather(leader, follower)

    assert [call["cache_warm_role"] for call in calls] == ["leader", "follower"]
    assert calls[1]["cache_warm_leader_succeeded"] is True
    assert calls[1]["cache_warm_wait_s"] >= 0


@pytest.mark.anyio
async def test_cache_warm_first_waits_for_configured_propagation_delay():
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()
    calls = []

    class ControlledClient:
        async def complete(self, request):
            calls.append(dict(request.metadata))
            if request.metadata["cache_warm_role"] == "leader":
                leader_started.set()
                await release_leader.wait()
            return LLMResponse(text="ok", model="test", finish_reason="stop")

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.execution_llm_config = {
        "provider": "test",
        "model": "test",
        "concurrency": 4,
        "cache_warm_first": True,
        "cache_warm_delay_s": 0.05,
    }
    loop.execution_concurrency = 4
    loop.execution_client = ControlledClient()
    _EXECUTION_CACHE_WARM_STATES.clear()

    leader_request = LLMRequest("s", "u1", "test", 0, None, {"cache_warm_key": "shared"})
    follower_request = LLMRequest("s", "u2", "test", 0, None, {"cache_warm_key": "shared"})
    leader = asyncio.create_task(loop._complete_execution(leader_request))
    await leader_started.wait()
    follower = asyncio.create_task(loop._complete_execution(follower_request))
    release_leader.set()
    await leader
    await asyncio.sleep(0.01)

    assert [call["cache_warm_role"] for call in calls] == ["leader"]

    await asyncio.wait_for(follower, timeout=1)

    assert [call["cache_warm_role"] for call in calls] == ["leader", "follower"]
    assert follower_request.metadata["cache_warm_propagation_wait_s"] >= 0.03


@pytest.mark.anyio
async def test_cache_warm_first_allows_different_prefix_leaders_to_run_concurrently():
    both_started = asyncio.Event()
    release = asyncio.Event()
    started_keys = set()

    class ControlledClient:
        async def complete(self, request):
            started_keys.add(request.metadata["cache_warm_key"])
            if len(started_keys) == 2:
                both_started.set()
            await release.wait()
            return LLMResponse(text="ok", model="test", finish_reason="stop")

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.execution_llm_config = {
        "provider": "test",
        "model": "test",
        "concurrency": 4,
        "cache_warm_first": True,
    }
    loop.execution_concurrency = 4
    loop.execution_client = ControlledClient()
    _EXECUTION_CACHE_WARM_STATES.clear()

    first = asyncio.create_task(
        loop._complete_execution(LLMRequest("s", "u1", "test", 0, None, {"cache_warm_key": "first"}))
    )
    second = asyncio.create_task(
        loop._complete_execution(LLMRequest("s", "u2", "test", 0, None, {"cache_warm_key": "second"}))
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(first, second)

    assert started_keys == {"first", "second"}


@pytest.mark.anyio
async def test_cache_warm_first_releases_follower_when_leader_fails():
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()

    class FailingLeaderClient:
        async def complete(self, request):
            if request.metadata["cache_warm_role"] == "leader":
                leader_started.set()
                await release_leader.wait()
                raise RuntimeError("leader failed")
            return LLMResponse(text="ok", model="test", finish_reason="stop")

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.execution_llm_config = {
        "provider": "test",
        "model": "test",
        "concurrency": 4,
        "cache_warm_first": True,
    }
    loop.execution_concurrency = 4
    loop.execution_client = FailingLeaderClient()
    _EXECUTION_CACHE_WARM_STATES.clear()

    leader_request = LLMRequest("s", "u1", "test", 0, None, {"cache_warm_key": "shared"})
    follower_request = LLMRequest("s", "u2", "test", 0, None, {"cache_warm_key": "shared"})
    leader = asyncio.create_task(loop._complete_execution(leader_request))
    await leader_started.wait()
    follower = asyncio.create_task(loop._complete_execution(follower_request))
    release_leader.set()

    with pytest.raises(RuntimeError, match="leader failed"):
        await leader
    response = await asyncio.wait_for(follower, timeout=1)

    assert response.text == "ok"
    assert follower_request.metadata["cache_warm_leader_succeeded"] is False


@pytest.mark.anyio
async def test_cache_prime_releases_all_full_quality_candidates_after_short_primer():
    primer_started = asyncio.Event()
    release_primer = asyncio.Event()
    actual_started = asyncio.Event()
    calls = []

    class ControlledClient:
        async def complete(self, request):
            calls.append(request)
            if request.metadata["cache_warm_role"] == "primer":
                primer_started.set()
                await release_primer.wait()
                return LLMResponse(
                    text="",
                    reasoning="p",
                    model="test",
                    finish_reason="length",
                    usage={"prompt_tokens": 100, "completion_tokens": 1},
                    metadata={"api_stream_partial_accepted": True},
                )
            if sum(call.metadata["cache_warm_role"] != "primer" for call in calls) == 2:
                actual_started.set()
            return LLMResponse(text="ok", model="test", finish_reason="stop")

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.execution_llm_config = {
        "provider": "test",
        "model": "test",
        "concurrency": 4,
        "cache_prime_first": True,
        "cache_prime_max_tokens": 1,
    }
    loop.execution_concurrency = 4
    loop.execution_client = ControlledClient()
    _EXECUTION_CACHE_WARM_STATES.clear()

    owner_request = LLMRequest("s", "u1", "test", 1.0, 131072, {"cache_warm_key": "shared"})
    follower_request = LLMRequest("s", "u2", "test", 1.0, 131072, {"cache_warm_key": "shared"})
    owner = asyncio.create_task(loop._complete_execution(owner_request))
    await primer_started.wait()
    follower = asyncio.create_task(loop._complete_execution(follower_request))
    await asyncio.sleep(0.01)

    assert len(calls) == 1
    assert calls[0].max_tokens == 1
    assert calls[0].metadata["accept_partial_stream"] is True

    release_primer.set()
    await asyncio.wait_for(actual_started.wait(), timeout=1)
    responses = await asyncio.gather(owner, follower)

    assert [response.text for response in responses] == ["ok", "ok"]
    assert [call.max_tokens for call in calls[1:]] == [131072, 131072]
    assert {call.metadata["cache_warm_role"] for call in calls[1:]} == {
        "primer_owner",
        "follower",
    }
    assert owner_request.metadata["cache_prime_usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 1,
    }
    assert follower_request.metadata["cache_prime_succeeded"] is True


def test_shared_guidance_library_reuses_one_archive_instance(tmp_path):
    path = tmp_path / "library.json"
    root = LibraryNode(
        id="root",
        problem_id="test",
        timestep=0,
        entry_id="seed",
        value=1.0,
        raw_score=1.0,
        visits=0,
        parent_id=None,
    )
    entry = LibraryEntry(
        id="seed",
        parent_id=None,
        problem_id="test",
        timestep=0,
        guidance="seed",
        execution_thinking="",
        solution="int main() {}",
        verifier_reward=1.0,
        verifier_raw_score=1.0,
        verifier_status="valid",
        verifier_message="ok",
        summary="seed",
        reusable_idea="seed",
        failure_mode=None,
    )
    config = {
        "rollout_n": 8,
        "puct_c": 1.0,
        "puct_q_mode": "best_child",
        "max_buffer_size": 1000,
        "topk_children": 2,
    }
    GuidanceLibrary(path, initial_nodes=[root], **config)
    # Attach the seed entry through the on-disk representation used by tests.
    data = json.loads(path.read_text())
    data["entries"] = {entry.id: entry.to_dict()}
    path.write_text(json.dumps(data))
    _GUIDANCE_LIBRARIES.pop(str(path.resolve()), None)

    first = _shared_guidance_library(path, config)
    second = _shared_guidance_library(path, config)

    assert first is second


def test_verification_result_for_execution_error_has_zero_reward():
    result = VerificationResult.execution_error("api failed")

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "execution_error"
    assert result.message == "api failed"


def test_invalid_execution_is_not_replaced_by_minimax_fallback():
    result = _verify_execution_without_fallback(
        execution_text="",
        guidance="Use deterministic minimax search.",
        timeout_s=20,
    )

    assert result.fallback_used is False
    assert result.original_execution_text == ""
    assert result.verification.valid is False
    assert result.verification.status in {"parse_error", "execution_error"}
    assert result.verification.raw_score is None
    assert result.solution == ""
    assert "project_to_box_sum" not in result.solution


def test_valid_execution_is_kept_without_fallback():
    valid_text = """<execution_thinking>
Use the baseline to confirm verifier plumbing.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>
<summary>
Execution Interpretation
Baseline execution.

Implemented Algorithm
Return the constant profile.

New Ideas Introduced
valid baseline

Empirical Outcome
pending

Failure / Bottleneck Analysis
weak score

Next Guidance Delta
improve score
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=valid_text,
        guidance="Keep valid code.",
        timeout_s=20,
    )

    assert result.fallback_used is False
    assert result.execution_text == valid_text
    assert result.execution_thinking == "Use the baseline to confirm verifier plumbing."
    assert result.verification.valid is True
    assert result.verification.raw_score == 0.5
    assert "Execution Interpretation" in result.summary
    assert "Use the baseline to confirm verifier plumbing." in result.summary
    assert "Implemented Algorithm" in result.summary
    assert "```python\ndef run(seed=42, budget_s=1, **kwargs):" in result.summary
    assert "Empirical Outcome\nVerifier status: valid\nRaw C5: 0.5" in result.summary
    assert "Verified returned profile: n_points=2, c5_bound=0.5" in result.summary
    assert "head=[0.5, 0.5]" in result.summary
    assert "Next Guidance Delta\nimprove score" in result.summary


def test_native_qwen_reasoning_is_preserved_without_execution_thinking_block():
    execution_text = """<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>
<summary>Use the requested bounded mutation.</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        execution_reasoning="Compare the parent against two bounded mutations.",
        guidance="Use bounded mutation.",
        timeout_s=20,
        prompt_mode="code_delta",
    )

    assert result.verification.valid is True
    assert result.execution_thinking == "Compare the parent against two bounded mutations."
    assert result.summary == "Use the requested bounded mutation."


def test_raw_qwen_think_tag_is_used_when_reasoning_content_is_not_separate():
    execution_text = """<think>Inspect the skyline gaps before modifying the parent.</think>
<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>
<summary>Use the requested skyline mutation.</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Use skyline mutation.",
        timeout_s=20,
        prompt_mode="code_delta",
    )

    assert result.verification.valid is True
    assert result.execution_thinking == "Inspect the skyline gaps before modifying the parent."


def test_solution_tag_is_preferred_when_summary_contains_python_block():
    execution_text = """<execution_thinking>
Use tagged solution.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>

<summary>
Execution Interpretation
The summary includes a non-solution code example.

Implemented Algorithm
```python
def helper_only():
    return None
```

New Ideas Introduced
tagged extraction

Empirical Outcome
pending

Failure / Bottleneck Analysis
none

Next Guidance Delta
continue
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Use tagged solution.",
        timeout_s=20,
    )

    assert result.verification.valid is True
    assert "def run(seed=42" in result.solution
    assert "helper_only" not in result.solution
    assert result.model_summary is not None
    assert "The summary includes a non-solution code example." in result.model_summary


def test_terminal_unclosed_summary_is_extracted_for_model_summary():
    execution_text = """<execution_thinking>
Use tagged solution.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>

<summary>
This final summary reaches EOF without a closing tag."""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Use tagged solution.",
        timeout_s=20,
    )

    assert result.verification.valid is True
    assert result.model_summary == "This final summary reaches EOF without a closing tag."
    assert "This final summary reaches EOF without a closing tag." in result.summary


def test_polyomino_execution_verification_uses_cpp_task_spec(monkeypatch):
    def fake_evaluate_cpp_solution(code, *, problem_id, config=None):
        assert problem_id == "0"
        assert config == {"n_cases": 70}
        assert "int main()" in code
        return FrontierCSResult(valid=True, score=12.5, message="Score: 12.50/100", artifacts={})

    monkeypatch.setattr("guidance_ttt.verifier.polyomino.evaluate_cpp_solution", fake_evaluate_cpp_solution)

    execution_text = """<execution_thinking>
Use a simple shelf placement baseline.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() { return 0; }
```
</solution>

<summary>
Use shelf packing with normalized offsets.
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Try a shelf packing baseline.",
        timeout_s=340,
        task_spec=get_task_spec("polyomino_packing"),
        verifier_config={"problem_id": "0", "n_cases": 70},
    )

    assert result.verification.valid is True
    assert result.verification.raw_score == 12.5
    assert result.solution.startswith("#include <bits/stdc++.h>")
    assert "```cpp\n#include <bits/stdc++.h>" in result.summary
    assert "Empirical Outcome\nVerifier status: valid\nFrontierCS score: 12.5\nReward: 12.5" in result.summary


def test_code_delta_verification_keeps_raw_model_summary_without_canonical_wrapper(monkeypatch):
    def fake_evaluate_cpp_solution(code, *, problem_id, config=None):
        return FrontierCSResult(valid=True, score=14.0, message="Score: 14.00/100", artifacts={})

    monkeypatch.setattr("guidance_ttt.verifier.polyomino.evaluate_cpp_solution", fake_evaluate_cpp_solution)
    delta_summary = (
        "Replaced the single skyline choice with a bounded beam over feasible gaps and retained "
        "the parent's orientation normalization."
    )
    execution_text = f"""<execution_thinking>
Apply the requested bounded search to the parent implementation.
</execution_thinking>
<solution>
```cpp
int main() {{ return 0; }}
```
</solution>
<summary>
{delta_summary}
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Use bounded beam search.",
        timeout_s=340,
        task_spec=get_task_spec("polyomino_packing"),
        verifier_config={"problem_id": "0"},
        prompt_mode="code_delta",
    )

    assert result.model_summary == delta_summary
    assert result.summary == delta_summary
    assert "Implemented Algorithm" not in result.summary
    assert "```cpp" not in result.summary


def test_task_config_normalization_converts_nested_omegaconf_to_plain_dict():
    task_config = _normalize_task_config(
        OmegaConf.create(
            {
                "id": "polyomino_packing",
                "frontiercs": {
                    "problem_id": "0",
                    "n_cases": 70,
                },
            }
        )
    )

    assert type(task_config) is dict
    assert type(task_config["frontiercs"]) is dict
    assert _verifier_config_from_task_config(task_config) == {
        "problem_id": "0",
        "n_cases": 70,
    }


def test_build_execution_summary_normalizes_all_sections_and_verifier_outcome():
    verification = VerificationResult(
        reward=2.5,
        raw_score=0.4,
        valid=True,
        status="valid",
        message="C5 bound: 0.400000",
        artifacts={"h_values": [0.25, 0.75], "c5_bound": 0.4, "n_points": 2},
    )

    summary = build_execution_summary(
        model_summary="""Execution Interpretation
model interpretation

Implemented Algorithm
model algorithm

New Ideas Introduced
coordinate repair

Empirical Outcome
model guessed success

Failure / Bottleneck Analysis
weak local basin

Next Guidance Delta
preserve repair and shrink steps""",
        execution_thinking="execution thought",
        solution="def run(seed=42, budget_s=1, **kwargs):\n    return ([0.5, 0.5], 0.5, 2)",
        guidance="try repair",
        verification=verification,
    )

    for section in EXECUTION_SUMMARY_SECTIONS:
        assert section in summary
    assert "Execution Interpretation\nexecution thought" in summary
    assert "model interpretation" in summary
    assert "Implemented Algorithm\nmodel algorithm" in summary
    assert "```python\ndef run(seed=42, budget_s=1, **kwargs):" in summary
    assert "Empirical Outcome\nVerifier status: valid\nRaw C5: 0.4\nReward: 2.5" in summary
    assert "Verified returned profile: n_points=2, c5_bound=0.4" in summary
    assert "model guessed success" not in summary


def test_build_execution_summary_synthesizes_missing_summary_with_code():
    verification = VerificationResult.execution_error("missing run")

    summary = build_execution_summary(
        model_summary=None,
        execution_thinking="I attempted a search.",
        solution="def helper():\n    pass",
        guidance="Use deterministic search.",
        verification=verification,
    )

    for section in EXECUTION_SUMMARY_SECTIONS:
        assert section in summary
    assert "Execution Interpretation\nI attempted a search." in summary
    assert "```python\ndef helper():" in summary
    assert "Empirical Outcome\nVerifier status: execution_error\nRaw C5: None\nReward: 0.0" in summary
    assert "Verifier reported execution_error: missing run" in summary
    assert "Next Guidance Delta\nContinue from the submitted guidance: Use deterministic search." in summary


def test_agent_loop_loads_execution_llm_from_rollout_config_path(tmp_path):
    config_path = tmp_path / "agent_loop.yaml"
    config_path.write_text(
        """
- name: guidance_execution_erdos
  _target_: guidance_ttt.agent_loop.GuidanceExecutionAgentLoop
  execution_llm:
    provider: local
    model: models/gpt-oss-20b
    device_map: auto
"""
    )
    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.rollout_config = OmegaConf.create(
        {
            "agent": {
                "agent_loop_config_path": str(config_path),
                "default_agent_loop": "guidance_execution_erdos",
            }
        }
    )

    execution_llm = loop._execution_llm_from_rollout_config()

    assert execution_llm == {
        "provider": "local",
        "model": "models/gpt-oss-20b",
        "device_map": "auto",
    }


def test_agent_loop_loads_execution_llm_from_legacy_erdos_config_without_default_name(tmp_path):
    config_path = tmp_path / "agent_loop.yaml"
    config_path.write_text(
        """
- name: guidance_execution_erdos
  _target_: guidance_ttt.agent_loop.GuidanceExecutionAgentLoop
  execution_llm:
    provider: local
    model: legacy-exec
"""
    )
    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.rollout_config = OmegaConf.create(
        {
            "agent": {
                "agent_loop_config_path": str(config_path),
            }
        }
    )

    execution_llm = loop._execution_llm_from_rollout_config()

    assert execution_llm == {
        "provider": "local",
        "model": "legacy-exec",
    }


def test_agent_loop_loads_task_config_from_rollout_config_path(tmp_path):
    config_path = tmp_path / "agent_loop.yaml"
    config_path.write_text(
        """
- name: guidance_execution_task
  _target_: guidance_ttt.agent_loop.GuidanceExecutionAgentLoop
  task:
    id: polyomino_packing
    frontiercs:
      base_dir: /opt/Frontier-CS
      judge_url: http://127.0.0.1:8081
      problem_id: "0"
"""
    )
    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.rollout_config = OmegaConf.create(
        {
            "agent": {
                "agent_loop_config_path": str(config_path),
                "default_agent_loop": "guidance_execution_task",
            }
        }
    )

    task_config = loop._task_config_from_rollout_config()

    assert task_config == {
        "id": "polyomino_packing",
        "frontiercs": {
            "base_dir": "/opt/Frontier-CS",
            "judge_url": "http://127.0.0.1:8081",
            "problem_id": "0",
        },
    }


def test_agent_loop_recovers_full_config_after_registry_target_overwrite(tmp_path, monkeypatch):
    config_path = tmp_path / "agent_loop.yaml"
    config_path.write_text(
        """
- name: guidance_execution_task
  _target_: guidance_ttt.agent_loop.GuidanceExecutionAgentLoop
  task:
    id: polyomino_packing
  prompt_mode: code_delta
  verifier_timeout_s: 340
  problem_prompt: recovered problem prompt
  execution_llm:
    provider: mock
    model: recovered-executor
    concurrency: 16
"""
    )
    rollout_config = OmegaConf.create(
        {
            "response_length": 8192,
            "agent": {
                "agent_loop_config_path": str(config_path),
                "default_agent_loop": "guidance_execution_task",
            },
        }
    )

    def fake_base_init(self, *args, **kwargs):
        self.rollout_config = rollout_config

    monkeypatch.setattr(AgentLoopBase, "__init__", fake_base_init)

    # A bare Hydra target passes none of the YAML fields after the registry is
    # overwritten by the class decorators. The loop must recover them itself.
    loop = GuidanceExecutionAgentLoop()

    assert loop.task_config == {"id": "polyomino_packing"}
    assert loop.prompt_mode == "code_delta"
    assert loop.verifier_timeout_s == 340
    assert loop.problem_prompt_override == "recovered problem prompt"
    assert loop.execution_llm_config == {
        "provider": "mock",
        "model": "recovered-executor",
        "concurrency": 16,
    }
    assert loop.execution_concurrency == 16
    assert loop.response_length == 8192


def test_agent_loop_uses_extra_info_task_config_for_verifier_config():
    full_task_config = {
        "id": "polyomino_packing",
        "frontiercs": {
            "base_dir": "/opt/Frontier-CS",
            "judge_url": "http://127.0.0.1:8081",
            "problem_id": "0",
        },
    }
    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.task_config = {"id": "erdos_min_overlap"}

    task_config = loop._task_config_for_extra_info(
        {"task": "polyomino_packing", "task_config": full_task_config},
        get_task_spec("polyomino_packing"),
    )

    assert task_config == full_task_config
    assert _verifier_config_from_task_config(task_config) == {
        "base_dir": "/opt/Frontier-CS",
        "judge_url": "http://127.0.0.1:8081",
        "problem_id": "0",
    }


@pytest.mark.anyio
async def test_empty_guidance_generation_retries_with_min_tokens():
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            if token_ids == [0]:
                return "" if skip_special_tokens else "<|im_end|>"
            return "<guidance>try coordinate descent</guidance>"

    class FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, *, request_id, prompt_ids, sampling_params):
            self.calls.append(dict(sampling_params))
            if len(self.calls) == 1:
                return type("Output", (), {"token_ids": [0], "log_probs": [-0.1], "stop_reason": "stop"})()
            return type("Output", (), {"token_ids": [10, 11, 12], "log_probs": [-0.2, -0.3, -0.4], "stop_reason": "length"})()

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.server_manager = FakeServerManager()
    loop.tokenizer = FakeTokenizer()
    loop.response_length = 128

    generation = await loop._generate_guidance_response([1, 2, 3], {"temperature": 1.0})

    assert generation.text == "<guidance>try coordinate descent</guidance>"
    assert generation.response_ids == [10, 11, 12]
    assert generation.attempts == 2
    assert loop.server_manager.calls[0] == {"temperature": 1.0}
    assert loop.server_manager.calls[1]["min_tokens"] >= 16
    assert loop.server_manager.calls[1]["max_tokens"] <= 128


@pytest.mark.anyio
async def test_execution_llm_concurrency_limit_is_shared_across_loop_instances():
    class SlowExecutionClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def complete(self, request):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return LLMResponse(text="ok", model=request.model, finish_reason="stop")

    client = SlowExecutionClient()
    loops = []
    for _ in range(2):
        loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
        loop.execution_client = client
        loop.execution_concurrency = 1
        loop.execution_llm_config = {
            "provider": "openai_compatible",
            "model": "openai/gpt-oss-120b",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        loops.append(loop)

    request = LLMRequest(
        system="system",
        user="user",
        model="openai/gpt-oss-120b",
        temperature=0.0,
        max_tokens=None,
    )

    await asyncio.gather(*(loop._complete_execution(request) for loop in loops))

    assert client.max_active == 1
