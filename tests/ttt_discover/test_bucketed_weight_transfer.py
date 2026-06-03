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

import pytest

pytest.importorskip("vllm", exc_type=ImportError)

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    _ipc_path_from_zmq_handle,
    _unlink_stale_ipc_socket,
)


def test_ipc_path_from_zmq_handle_only_accepts_ipc_handles():
    assert _ipc_path_from_zmq_handle("ipc:///tmp/verl.sock") == "/tmp/verl.sock"
    assert _ipc_path_from_zmq_handle("tcp://127.0.0.1:1234") is None


def test_unlink_stale_ipc_socket_removes_existing_ipc_file(tmp_path):
    socket_path = tmp_path / "stale.sock"
    socket_path.write_text("stale")

    _unlink_stale_ipc_socket(f"ipc://{socket_path}")

    assert not socket_path.exists()


def test_unlink_stale_ipc_socket_ignores_non_ipc_file(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("keep")

    _unlink_stale_ipc_socket(f"tcp://{Path(marker)}")

    assert marker.exists()
