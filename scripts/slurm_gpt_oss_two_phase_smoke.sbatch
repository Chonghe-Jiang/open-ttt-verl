#!/usr/bin/env bash
#SBATCH --job-name=oss-2phase-smoke
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/oss-2phase-smoke-%j.out
#SBATCH --error=outputs/slurm/oss-2phase-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm .hf_cache .tmp .apptainer_home/.cache

APPTAINER_BASE=(
  apptainer exec --nv --cleanenv
  --home "${REPO_ROOT}/.apptainer_home:/container_home"
  --bind "${REPO_ROOT}:/workspace/guidance"
  --pwd /workspace/guidance
  --env PYTHONPATH=/workspace/guidance
  --env HF_HOME=/workspace/guidance/.hf_cache
  --env TMPDIR=/workspace/guidance/.tmp
  --env VLLM_NO_USAGE_STATS=1
  "${SIF_PATH}"
)

SERVER_LOG="outputs/slurm/oss-2phase-smoke-vllm-${SLURM_JOB_ID}.log"
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  "${APPTAINER_BASE[@]}" vllm serve /workspace/guidance/models/gpt-oss-120b \
    --served-model-name openai/gpt-oss-120b \
    --host 127.0.0.1 --port 8002 --dtype auto --trust-remote-code \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
    --max-model-len 32768 --max-num-seqs 2 \
    --download-dir /workspace/guidance/.hf_cache/hub --enforce-eager \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8002/v1/models >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -200 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8002/v1/models >/dev/null

"${APPTAINER_BASE[@]}" python - <<'PY'
import asyncio
import json

from guidance_ttt.llm_client import GPTOSSTwoPhaseOpenAICompatibleLLMClient
from guidance_ttt.state import LLMRequest


async def main():
    client = GPTOSSTwoPhaseOpenAICompatibleLLMClient(
        {
            "base_url": "http://127.0.0.1:8002/v1",
            "api_key": "local-vllm",
            "reasoning_effort": "high",
            "phase1_max_tokens": 512,
            "context_window": 32768,
            "context_buffer": 50,
            "timeout_s": 900,
        }
    )
    response = await client.complete(
        LLMRequest(
            system="You are a careful C++ execution model.",
            user=(
                "Return a final answer containing exactly this block and no other final text: "
                "<solution>```cpp\\nint main(){return 0;}\\n```</solution>"
            ),
            model="openai/gpt-oss-120b",
            temperature=0.0,
            max_tokens=None,
            metadata={"purpose": "two_phase_smoke"},
        )
    )
    report = {
        "finish_reason": response.finish_reason,
        "metadata": response.metadata,
        "usage": response.usage,
        "reasoning_chars": len(response.reasoning),
        "text": response.text,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    assert response.metadata["forced_final"] is True, report
    assert response.usage["phase2_completion_tokens"] > 0, report
    assert "<solution>" in response.text, report


asyncio.run(main())
PY
