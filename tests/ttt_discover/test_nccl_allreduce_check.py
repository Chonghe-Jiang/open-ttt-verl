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

from scripts.ttt_discover.nccl_allreduce_check import expected_allreduce_sum


def test_expected_allreduce_sum_matches_rank_sum():
    assert expected_allreduce_sum(world_size=1) == 1.0
    assert expected_allreduce_sum(world_size=2) == 3.0
    assert expected_allreduce_sum(world_size=4) == 10.0
