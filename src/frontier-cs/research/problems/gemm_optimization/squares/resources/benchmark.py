import torch
import math
import triton
from typing import Optional
from itertools import product


# Ensure CUDA is available and properly initialize device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)

triton.set_allocator(alloc_fn)
torch.manual_seed(0)
try:
    torch.cuda.manual_seed_all(0)
except Exception:
    pass
assert triton.runtime.driver.active.get_current_target().backend == "cuda", "This benchmark only supports CUDA backend."

def _bench_ms(fn):
    out = triton.testing.do_bench(fn, quantiles=[0.5])
    if isinstance(out, (tuple, list)):
        return float(out[0])
    return float(out)
def pt_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
def _reference_mm(a: torch.Tensor, b: torch.Tensor, out_dtype=torch.float16):
    ref32 = pt_gelu(a.float() @ b.float())
    return ref32.to(out_dtype)

def _is_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-2, atol=5e-3):
    return torch.allclose(x, y, rtol=rtol, atol=atol)
    
def _bench_pair(M, N, K, answer_matmul, baseline_matmul=torch.matmul):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    baseline_ms = _bench_ms(lambda: pt_gelu(baseline_matmul(a, b)))
    answer_ms = _bench_ms(lambda: answer_matmul(a, b))
    to_tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    baseline_tflops = to_tflops(baseline_ms) if baseline_ms is not None else None
    answer_tflops = to_tflops(answer_ms)
    c_ref = _reference_mm(a, b, out_dtype=torch.float16)
    c_tri = answer_matmul(a, b)
    passed = _is_close(c_tri, c_ref, rtol=1e-2, atol=5e-3)
    return {
        "M": M, "N": N, "K": K,
        "baseline_ms": baseline_ms, "answer_ms": answer_ms,
        "baseline_tflops": baseline_tflops, "answer_tflops": answer_tflops,
        "close_passed": passed, 
        "rtol": 1e-2, "atol": 5e-3, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        m = 1024
        a = torch.randn((m, m), device=DEVICE, dtype=torch.float16)
        b = torch.randn((m, m), device=DEVICE, dtype=torch.float16)
        for _ in range(max(1, int(iters))):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_matmul, baseline_matmul=torch.matmul, print_output=False):
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    # Only squares: 512..8192 by 1024
    squares = [s for s in range(512, 8192 + 1, 1024)]
    shapes = [(s, s, s) for s in squares]
    rows = []
    for (M, N, K) in shapes:
        r = _bench_pair(M, N, K, answer_matmul, baseline_matmul)
        rows.append(r)
    print("\n=== Answer vs Baseline: Speedup for each shape (based on median time) ===")
    speedups = []
    for r in rows:
        tm, cm = r["answer_ms"], r["baseline_ms"]
        sp = cm / tm
        speedups.append(sp)
        status = "OK" if r["close_passed"] else "FAIL"
        if print_output:
            print(
                f"M={r['M']:4d} N={r['N']:4d} K={r['K']:4d}  "
                f"baseline={cm:7.3f} ms  answer={tm:7.3f} ms  speedup={sp:5.2f}x  "
                f"[Passed: {status}  "
                f"rtol={r['rtol']:.1e} atol={r['atol']:.1e}]"
            )
    if speedups:
        arith_mean = sum(speedups) / len(speedups)
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        median = sorted(speedups)[len(speedups)//2]
        if print_output:
            print("\n--- Summary ---")
            print(f"Sample size: {len(speedups)}")
            print(f"Arithmetic mean speedup: {arith_mean:.3f}x")
            print(f"Geometric mean speedup: {geo_mean:.3f}x")
            print(f"Median speedup: {median:.3f}x")
    return rows, arith_mean, geo_mean, median

def run_benchmark(answer_matmul, baseline_matmul=torch.matmul, print_output=False):
    rows, arith_mean, geo_mean, median = summarize_speedup(answer_matmul, baseline_matmul, print_output=print_output)
    return {
        "rows": rows,
        "arithmetic_mean_speedup": arith_mean,
        "geometric_mean_speedup": geo_mean,
        "median_speedup": median,
        "pass_all": all(r["close_passed"] for r in rows),
    }