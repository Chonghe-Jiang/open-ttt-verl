from __future__ import annotations

from guidance_ttt.bootstrap import _validate_required_blocks, build_bootstrap_execution_prompt
from guidance_ttt.state import VerificationResult
from guidance_ttt.tasks import get_task_spec


def test_vliw_bootstrap_prompt_attaches_verified_original_solution():
    spec = get_task_spec("vliw_kernel_optimization")
    verification = VerificationResult(
        reward=0.0,
        raw_score=147734.0,
        valid=True,
        status="valid",
        message="cycles=147734",
        artifacts={},
    )

    prompt = build_bootstrap_execution_prompt(
        task_spec=spec,
        execution_prompt_style="gpt_api_brief_thinking",
        baseline_verification=verification,
        prompt_mode="code_delta",
    )

    assert "safe initial seed candidate" in prompt.user
    assert "<baseline_candidate>" in prompt.user
    assert "Simulator cycles: 147734.0" in prompt.user
    assert spec.bootstrap_solution in prompt.user
    assert "complete file, not a patch or diff" in prompt.user
    assert "changes from the supplied baseline" in prompt.user
    assert "reproduce the verified baseline exactly" in prompt.user
    assert "no code change was made" in prompt.user
    assert "exactly two top-level XML blocks" in prompt.user
    assert "<execution_thinking>\n" not in prompt.user


def test_summary_only_bootstrap_requests_a_self_contained_full_candidate_summary():
    spec = get_task_spec("vliw_kernel_optimization")

    prompt = build_bootstrap_execution_prompt(
        task_spec=spec,
        execution_prompt_style="gpt_api_brief_thinking",
        prompt_mode="summary_only",
    )

    assert "self-contained algorithm summary" in prompt.system
    assert "Describe the complete algorithm implemented by the submitted candidate" in prompt.user
    assert "guidance model will not receive source code in summary_only mode" in prompt.user
    assert "Describe only the concrete algorithmic changes" not in prompt.user


def test_gpt_api_brief_bootstrap_validation_accepts_two_required_blocks():
    text = """<solution>
```python
class KernelBuilder:
    def build_kernel(self, *args):
        self.instrs = []
```
</solution>
<summary>
Use one empty instruction stream for this format-only test.
</summary>"""

    _validate_required_blocks(text, execution_prompt_style="gpt_api_brief_thinking")
