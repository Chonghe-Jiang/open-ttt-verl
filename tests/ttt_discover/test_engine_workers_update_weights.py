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

from pathlib import Path


def test_naive_update_weights_does_not_unconditionally_offload_actor_to_cpu():
    source = Path("verl/workers/engine_workers.py").read_text()

    assert "if self.actor.engine.is_param_offload_enabled:" in source
    assert "self.actor.engine.to(\"cpu\", model=True, optimizer=False, grad=False)" in source
