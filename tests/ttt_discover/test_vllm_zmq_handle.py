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

import pytest

pytest.importorskip("vllm", exc_type=ImportError)

from verl.workers.rollout.vllm_rollout.utils import get_zmq_handle_for_device_uuid


def test_zmq_handle_for_device_uuid_preserves_default_without_suffix(monkeypatch):
    monkeypatch.delenv("VERL_VLLM_ZMQ_SUFFIX", raising=False)

    assert get_zmq_handle_for_device_uuid("GPU-test") == "ipc:///tmp/rl-colocate-zmq-GPU-test.sock"


def test_zmq_handle_for_device_uuid_includes_run_suffix(monkeypatch):
    monkeypatch.setenv("VERL_VLLM_ZMQ_SUFFIX", "ttt-run-123")

    assert get_zmq_handle_for_device_uuid("GPU-test") == "ipc:///tmp/rl-colocate-zmq-GPU-test-ttt-run-123.sock"


def test_zmq_handle_for_device_uuid_hashes_long_suffix(monkeypatch):
    monkeypatch.setenv("VERL_VLLM_ZMQ_SUFFIX", "ttt-2gpu_scale_gptoss20b_flash_g4_n16-3712198")

    handle = get_zmq_handle_for_device_uuid("GPU-610d2053-81e2-7ed3-97b6-67c7a438b0f9")

    assert len(handle.removeprefix("ipc://")) <= 107
    assert "2gpu_scale_gptoss20b_flash" not in handle
