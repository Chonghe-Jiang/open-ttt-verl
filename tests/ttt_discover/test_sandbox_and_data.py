# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from verl_ttt_discover.agent_loop import make_group_uid
from verl_ttt_discover.data import build_slot_records
from verl_ttt_discover.erdos_env import verify_c5_solution
from verl_ttt_discover.sandbox import evaluate_python_code, extract_python_code
from verl_ttt_discover.state import DiscoveryState


def test_make_group_uid_uses_step_and_verl_uid():
    assert make_group_uid(global_step=7, uid="abc") == "7:abc"


def test_build_slot_records_produces_static_verl_rows():
    records = build_slot_records(num_slots=2, archive_path="outputs/run/archive.json")

    assert records[0]["data_source"] == "ttt_erdos"
    assert records[0]["prompt"] == [{"role": "user", "content": ""}]
    assert records[0]["extra_info"]["slot_id"] == "slot_0"
    assert records[1]["extra_info"]["archive_path"] == "outputs/run/archive.json"


def test_evaluate_python_code_runs_erdos_solution(tmp_path):
    h_values = np.full(4, 0.5)
    c5_bound = verify_c5_solution(
        h_values,
        float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * 0.5)),
        4,
    )
    state = DiscoveryState(timestep=-1, value=-c5_bound, raw_score=c5_bound, code="", construction=h_values.tolist())
    code = """
def run(seed=42, budget_s=10, **kwargs):
    return initial_h_values, evaluate_erdos_solution((initial_h_values, 0.5, len(initial_h_values))), len(initial_h_values)
"""

    result = evaluate_python_code(code, state=state, timeout_s=5, work_dir=tmp_path)

    assert result.error is None
    assert result.output[1] == c5_bound


def test_evaluate_python_code_reports_invalid_program(tmp_path):
    state = DiscoveryState(timestep=-1, value=-0.5, raw_score=0.5, code="", construction=[0.5, 0.5])

    result = evaluate_python_code("def run():\n    raise RuntimeError('bad')\n", state=state, timeout_s=5, work_dir=tmp_path)

    assert "bad" in result.error


def test_evaluate_python_code_allows_top_level_helpers(tmp_path):
    state = DiscoveryState(timestep=-1, value=-0.5, raw_score=0.5, code="", construction=[0.5, 0.5])
    code = """
import numpy as np

def helper():
    return np.asarray(initial_h_values, dtype=float)

def run(seed=42, budget_s=10, **kwargs):
    h = helper()
    return h, evaluate_erdos_solution((h, 0.5, len(h))), len(h)
"""

    result = evaluate_python_code(code, state=state, timeout_s=5, work_dir=tmp_path)

    assert result.error is None
    assert result.output[1] == 0.5


def test_extract_python_code_uses_last_python_block():
    response = """```python
def helper():
    pass
```

Final answer:
```python
def run(seed=42, budget_s=10, **kwargs):
    return [0.5, 0.5], 0.5, 2
```"""

    assert extract_python_code(response).startswith("def run")
