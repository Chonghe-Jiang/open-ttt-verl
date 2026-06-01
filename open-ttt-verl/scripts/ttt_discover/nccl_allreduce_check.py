from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def expected_allreduce_sum(world_size: int) -> float:
    return float(world_size * (world_size + 1) // 2)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def main() -> None:
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    world_size = _env_int("WORLD_SIZE", 1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL preflight")
    if world_size < 1:
        raise RuntimeError(f"Invalid WORLD_SIZE={world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = expected_allreduce_sum(world_size)
    if not torch.isclose(value.cpu(), torch.tensor([expected])).item():
        raise RuntimeError(f"Rank {rank}: all_reduce got {value.item()}, expected {expected}")

    dist.barrier()
    size_mb = 256
    numel = size_mb * 1024 * 1024 // 4
    payload = torch.ones(numel, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)
    bus_gbps = (size_mb / 1024) / elapsed

    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={torch.cuda.get_device_name(local_rank)!r} "
        f"allreduce_sum={value.item():.1f} bandwidth_gib_s={bus_gbps:.2f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
