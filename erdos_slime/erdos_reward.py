import numpy as np
import subprocess
import tempfile
import os
import sys
import json



def verify_c5_solution(h_values, c5_achieved, n_points):
    h = np.asarray(h_values, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"h must be 1D, got {h.shape}")
    if len(h) != int(n_points):
        raise ValueError(f"Expected len {n_points}, got {len(h)}")
    if not np.all(np.isfinite(h)):
        raise ValueError("h contains NaN/inf")
    if np.any(h < 0) or np.any(h > 1):
        raise ValueError(f"h out of [0,1]: [{h.min():.4f}, {h.max():.4f}]")
    target = int(n_points) / 2.0
    s = float(np.sum(h))
    if s == 0:
        raise ValueError("h sums to zero")
    if not np.isclose(s, target, atol=1e-8):
        h = h * (target / s)
        if np.any(h < 0) or np.any(h > 1):
            raise ValueError("After normalization h out of [0,1]")
    dx = 2.0 / int(n_points)
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    if not np.isfinite(c5):
        raise ValueError(f"C5 not finite: {c5}")
    if not np.isclose(c5, float(c5_achieved), atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {c5:.6f}")
    return c5


def evaluate_erdos_solution(result):
    h_values, c5_bound, n_points = result
    return verify_c5_solution(h_values, c5_bound, n_points)


def _get_verifier_source():
    import inspect
    return (
        inspect.getsource(verify_c5_solution)
        + "\n\n"
        + inspect.getsource(evaluate_erdos_solution)
    )


def compute_reward_from_code(code: str, initial_h_values=None) -> dict:
    prelude = "import numpy as np\n\n"
    prelude += _get_verifier_source() + "\n\n"
    if initial_h_values is not None:
        arr = list(np.asarray(initial_h_values, dtype=float))
        prelude += f"initial_h_values = np.array({arr!r})\n\n"
    else:
        prelude += "initial_h_values = None\n\n"

    runner = """
import json as _json
try:
    result = run(seed=42, budget_s=60)
    h_values, c5_bound, n_points = result
    c5 = evaluate_erdos_solution(result)
    reward = float(1.0 / (1e-8 + c5))
    print(_json.dumps({
        "success": True,
        "reward": reward,
        "c5_bound": float(c5),
        "n_points": int(n_points),
        "h_values": list(map(float, h_values)),
    }))
except Exception as e:
    print(_json.dumps({"success": False, "reward": 0.0, "error": str(e)}))
"""
    full_code = prelude + code + "\n" + runner
    fname = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            fname = f.name
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=90
        )
        if fname:
            os.unlink(fname)
        last_line = proc.stdout.strip().split('\n')[-1] if proc.stdout.strip() else ""
        if not last_line:
            return {"success": False, "reward": 0.0, "error": proc.stderr[:300]}
        return json.loads(last_line)
    except subprocess.TimeoutExpired:
        if fname:
            try: os.unlink(fname)
            except: pass
        return {"success": False, "reward": 0.0, "error": "timeout"}
    except Exception as e:
        if fname:
            try: os.unlink(fname)
            except: pass
        return {"success": False, "reward": 0.0, "error": str(e)}
