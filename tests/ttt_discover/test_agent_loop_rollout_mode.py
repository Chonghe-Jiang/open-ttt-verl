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

import asyncio

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopManager


class _FakeWorkerGroup:
    world_size = 2


class _FakeReplica:
    def __init__(self, *args, **kwargs):
        self.mode = None
        self._server_handle = object()
        self._server_address = "127.0.0.1:0"

    async def init_hybrid(self, worker_group):
        self.mode = "hybrid"

    async def init_colocated(self, resource_pool):
        self.mode = "colocated"

    async def init_standalone(self):
        self.mode = "standalone"


class _TestAgentLoopManager(AgentLoopManager):
    rollout_replica_class = _FakeReplica


def _config(checkpoint_backend: str):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "name": "vllm",
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "checkpoint_engine": {"backend": checkpoint_backend},
                    "prometheus": {"enable": False},
                },
            }
        }
    )


def test_vllm_non_naive_checkpoint_backend_uses_colocated_checkpoint_workers():
    manager = _TestAgentLoopManager(
        config=_config("nccl"),
        worker_group=_FakeWorkerGroup(),
        rollout_resource_pool=object(),
    )

    asyncio.run(manager._initialize_llm_servers())

    assert manager.rollout_replicas[0].mode == "colocated"


def test_vllm_naive_checkpoint_backend_keeps_hybrid_rollout_workers():
    manager = _TestAgentLoopManager(
        config=_config("naive"),
        worker_group=_FakeWorkerGroup(),
        rollout_resource_pool=object(),
    )

    asyncio.run(manager._initialize_llm_servers())

    assert manager.rollout_replicas[0].mode == "hybrid"
