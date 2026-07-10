# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import os
import math
import fcntl
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from verl_ttt_discover.cpp_sandbox import STDCPP_INCLUDE_BUNDLE, cleanup_cpp_sandbox, compile_cpp_code, run_cpp_binary
from verl_ttt_discover.discover_compat import BaseRewardEvaluator, Environment
from verl_ttt_discover.state import DiscoveryState


OFFICIAL_SCORE_SCALE = 100000.0
REWARD_NORMALIZER = 10000.0
FRONTIER_CS_DATASET_BASE_URL = "https://huggingface.co/datasets/FrontierCS/Frontier-CS/resolve/main"
FRONTIER_CS_POLYOMINO_PROBLEM_ID = 0
FRONTIER_CS_POLYOMINO_NUM_CASES = 70
FRONTIER_CS_POLYOMINO_CASE_TIMEOUT_S = 2
FRONTIER_CS_POLYOMINO_MEMORY_MB = 256
FRONTIER_CS_POLYOMINO_CONCURRENCY = 16


POLYOMINO_PROBLEM_STATEMENT = """# Pack the Polyominoes (Reflections Allowed)

You are given n polyominoes on a unit square grid. Each polyomino has 1 to 10 cells and is connected edge-to-edge. Place every polyomino into one axis-aligned integer grid rectangle, allowing reflection across the y-axis, rotations by multiples of 90 degrees, and integer translations. Cells may not overlap and must stay inside the rectangle.

Input:
- Line 1: n, with 100 <= n <= 10000.
- For each polyomino i:
  - one line k_i, 1 <= k_i <= 10.
  - k_i lines x y giving local cell coordinates. Coordinates may be negative.

Output:
- Line 1: two integers W H.
- Then n lines, one per input polyomino: X_i Y_i R_i F_i.
- R_i is in {0,1,2,3}, the number of 90 degree clockwise rotations.
- F_i is in {0,1}; if F_i=1, reflect across the y-axis before rotation.
- Transform order is reflection, then rotation, then translation.

Every transformed cell must satisfy 0 <= x < W and 0 <= y < H. No two cells may overlap.

Score per case is 100000 * total_cells / (W * H). Higher is better. Invalid output receives zero on that case.
"""


SAMPLE_INPUT = """3
1
0 0
3
0 0
1 0
0 1
4
0 0
1 0
0 1
1 1
"""


SMALL_VALIDATION_INPUT = """100
""" + "\n".join("1\n0 0" for _ in range(100)) + "\n"


BASELINE_CPP_CODE = r"""
#include <bits/stdc++.h>
using namespace std;

struct Cell { long long x, y; };
struct Orient {
    int w, h, r, f;
    long long minx, miny;
};

static pair<long long,long long> rot90cw(long long x, long long y, int r) {
    switch (r & 3) {
        case 0: return {x, y};
        case 1: return {y, -x};
        case 2: return {-x, -y};
        default: return {-y, x};
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<Cell>> pieces(n);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        pieces[i].resize(k);
        total += k;
        for (int j = 0; j < k; ++j) cin >> pieces[i][j].x >> pieces[i][j].y;
    }

    vector<vector<Orient>> oris(n);
    int min_global_w = 1;
    for (int i = 0; i < n; ++i) {
        set<vector<pair<long long,long long>>> seen;
        for (int f = 0; f <= 1; ++f) {
            for (int r = 0; r < 4; ++r) {
                vector<pair<long long,long long>> cells;
                long long minx = LLONG_MAX, miny = LLONG_MAX, maxx = LLONG_MIN, maxy = LLONG_MIN;
                for (auto c : pieces[i]) {
                    long long x = f ? -c.x : c.x;
                    long long y = c.y;
                    auto q = rot90cw(x, y, r);
                    cells.push_back(q);
                    minx = min(minx, q.first);
                    miny = min(miny, q.second);
                    maxx = max(maxx, q.first);
                    maxy = max(maxy, q.second);
                }
                vector<pair<long long,long long>> norm = cells;
                for (auto &p : norm) {
                    p.first -= minx;
                    p.second -= miny;
                }
                sort(norm.begin(), norm.end());
                if (seen.insert(norm).second) {
                    int w = int(maxx - minx + 1);
                    int h = int(maxy - miny + 1);
                    oris[i].push_back({w, h, r, f, minx, miny});
                    min_global_w = max(min_global_w, w);
                }
            }
        }
    }

    int W = max(min_global_w, (int)ceil(sqrt((double)max(1LL, total))));
    struct Place { long long x, y; int r, f; };
    vector<Place> ans(n);
    int x = 0, y = 0, shelf_h = 0;

    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int a, int b) {
        int besta = 1000000, bestb = 1000000;
        for (auto &o : oris[a]) besta = min(besta, o.w * o.h);
        for (auto &o : oris[b]) bestb = min(bestb, o.w * o.h);
        if (besta != bestb) return besta > bestb;
        return a < b;
    });

    for (int id : order) {
        int best = -1;
        for (int t = 0; t < (int)oris[id].size(); ++t) {
            auto &o = oris[id][t];
            if (o.w <= W - x) {
                if (best == -1 || o.h < oris[id][best].h || (o.h == oris[id][best].h && o.w > oris[id][best].w)) best = t;
            }
        }
        if (best == -1) {
            y += shelf_h;
            x = 0;
            shelf_h = 0;
            for (int t = 0; t < (int)oris[id].size(); ++t) {
                auto &o = oris[id][t];
                if (o.w <= W) {
                    if (best == -1 || o.h < oris[id][best].h || (o.h == oris[id][best].h && o.w > oris[id][best].w)) best = t;
                }
            }
        }
        if (best == -1) best = 0;
        auto &o = oris[id][best];
        ans[id] = {x - o.minx, y - o.miny, o.r, o.f};
        x += o.w;
        shelf_h = max(shelf_h, o.h);
    }
    int H = max(1, y + shelf_h);
    cout << W << " " << H << "\n";
    for (int i = 0; i < n; ++i) {
        cout << ans[i].x << " " << ans[i].y << " " << ans[i].r << " " << ans[i].f << "\n";
    }
    return 0;
}
""".strip()


@dataclass
class PolyominoCase:
    input_text: str
    n: int
    shapes: list[list[tuple[int, int]]]
    total_cells: int


@dataclass
class PolyominoScore:
    reward: float
    raw_score: float
    state: DiscoveryState
    message: str
    stdout: str = ""


class PolyominoEvaluationError(ValueError):
    pass


def _frontier_cache_dir() -> str:
    configured = os.environ.get("FRONTIER_CS_CACHE_DIR")
    if configured:
        return configured
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "frontier_cs")
    return os.path.expanduser("~/.cache/frontier_cs")


def _official_case_limit() -> int | None:
    value = os.environ.get("FRONTIER_CS_POLYOMINO_CASE_LIMIT")
    if not value:
        return None
    limit = int(value)
    return limit if limit > 0 else None


def _official_concurrency(default: int = FRONTIER_CS_POLYOMINO_CONCURRENCY) -> int:
    return int(os.environ.get("FRONTIER_CS_POLYOMINO_CONCURRENCY", str(default)))


def _download_text(url: str, *, timeout_s: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def _download_bytes(url: str, *, timeout_s: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read()


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as file:
        file.write(data)
    os.replace(tmp_path, path)


def _ensure_official_file(cache_dir: str, rel_path: str, *, text_rewrite: bool = False) -> str:
    output_path = os.path.join(cache_dir, rel_path)
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lock_path = output_path + ".lock"
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(output_path):
            return output_path
        url = f"{FRONTIER_CS_DATASET_BASE_URL}/{rel_path}"
        if text_rewrite:
            text = _download_text(url)
            text = text.replace("#include <bits/stdc++.h>", STDCPP_INCLUDE_BUNDLE)
            _atomic_write(output_path, text.encode("utf-8"))
        else:
            _atomic_write(output_path, _download_bytes(url))
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return output_path


def ensure_frontier_cs_polyomino_assets(cache_dir: str | None = None, *, case_limit: int | None = None) -> dict[str, Any]:
    cache_dir = cache_dir or _frontier_cache_dir()
    include_path = _ensure_official_file(cache_dir, "algorithmic/judge/include/testlib.h")
    checker_src = _ensure_official_file(
        cache_dir,
        f"algorithmic/problems/{FRONTIER_CS_POLYOMINO_PROBLEM_ID}/chk.cc",
        text_rewrite=True,
    )
    case_paths = []
    num_cases = FRONTIER_CS_POLYOMINO_NUM_CASES if case_limit is None else case_limit
    for case_id in range(1, num_cases + 1):
        case_paths.append(
            (
                _ensure_official_file(
                    cache_dir,
                    f"algorithmic/problems/{FRONTIER_CS_POLYOMINO_PROBLEM_ID}/testdata/{case_id}.in",
                ),
                _ensure_official_file(
                    cache_dir,
                    f"algorithmic/problems/{FRONTIER_CS_POLYOMINO_PROBLEM_ID}/testdata/{case_id}.ans",
                ),
            )
        )
    return {"cache_dir": cache_dir, "include_path": include_path, "checker_src": checker_src, "case_paths": case_paths}


def compile_official_polyomino_checker(*, cache_dir: str | None = None, timeout_s: int = 20) -> str:
    assets = ensure_frontier_cs_polyomino_assets(cache_dir, case_limit=0)
    checker_src = assets["checker_src"]
    checker_bin = checker_src + ".bin"
    if os.path.exists(checker_bin) and os.path.getmtime(checker_bin) >= os.path.getmtime(checker_src):
        return checker_bin
    lock_path = checker_bin + ".lock"
    os.makedirs(os.path.dirname(checker_bin), exist_ok=True)
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(checker_bin) and os.path.getmtime(checker_bin) >= os.path.getmtime(checker_src):
            return checker_bin
        include_dir = os.path.dirname(assets["include_path"])
        proc = subprocess.run(
            ["g++", "-O2", "-std=gnu++17", f"-I{include_dir}", checker_src, "-o", checker_bin],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if proc.returncode != 0:
        raise PolyominoEvaluationError(f"Official checker compile failed: {proc.stderr[-2000:]}")
    return checker_bin


def parse_polyomino_input(input_text: str) -> PolyominoCase:
    tokens = input_text.split()
    if not tokens:
        raise PolyominoEvaluationError("Empty input")
    pos = 0
    n = int(tokens[pos])
    pos += 1
    shapes: list[list[tuple[int, int]]] = []
    total = 0
    for _ in range(n):
        k = int(tokens[pos])
        pos += 1
        shape = []
        for _ in range(k):
            x = int(tokens[pos])
            y = int(tokens[pos + 1])
            pos += 2
            shape.append((x, y))
        shapes.append(shape)
        total += k
    return PolyominoCase(input_text=input_text, n=n, shapes=shapes, total_cells=total)


def _rot90cw(x: int, y: int, r: int) -> tuple[int, int]:
    r &= 3
    if r == 0:
        return x, y
    if r == 1:
        return y, -x
    if r == 2:
        return -x, -y
    return -y, x


def verify_polyomino_output(case: PolyominoCase, output_text: str) -> float:
    tokens = output_text.split()
    if len(tokens) < 2 + 4 * case.n:
        raise PolyominoEvaluationError(f"Output has too few tokens: {len(tokens)}")
    pos = 0
    try:
        width = int(tokens[pos])
        height = int(tokens[pos + 1])
        pos += 2
    except ValueError as exc:
        raise PolyominoEvaluationError("W/H must be integers") from exc
    if width <= 0 or height <= 0:
        raise PolyominoEvaluationError("W/H must be positive")

    occupied: set[tuple[int, int]] = set()
    for i, shape in enumerate(case.shapes):
        try:
            tx = int(tokens[pos])
            ty = int(tokens[pos + 1])
            r = int(tokens[pos + 2])
            f = int(tokens[pos + 3])
            pos += 4
        except ValueError as exc:
            raise PolyominoEvaluationError(f"Placement {i + 1} is not integer") from exc
        if r not in (0, 1, 2, 3) or f not in (0, 1):
            raise PolyominoEvaluationError(f"Placement {i + 1} has invalid R/F")
        for sx, sy in shape:
            rx = -sx if f else sx
            ry = sy
            qx, qy = _rot90cw(rx, ry, r)
            gx, gy = qx + tx, qy + ty
            if gx < 0 or gy < 0 or gx >= width or gy >= height:
                raise PolyominoEvaluationError(f"Piece {i + 1} cell out of bounds at ({gx},{gy})")
            if (gx, gy) in occupied:
                raise PolyominoEvaluationError(f"Overlap at ({gx},{gy})")
            occupied.add((gx, gy))
    if pos != len(tokens):
        raise PolyominoEvaluationError("Extra tokens after expected placements")
    return float(case.total_cells) / float(width * height)


def default_validation_cases() -> list[PolyominoCase]:
    return [parse_polyomino_input(SAMPLE_INPUT), parse_polyomino_input(SMALL_VALIDATION_INPUT)]


def _run_official_checker(checker_bin: str, input_path: str, output_path: str, answer_path: str, result_path: str) -> tuple[float, str]:
    proc = subprocess.run(
        [checker_bin, input_path, output_path, answer_path, result_path],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )
    result_text = ""
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8", errors="replace") as file:
            result_text = file.read().strip()
    if proc.returncode != 7:
        message = result_text or proc.stderr.strip() or proc.stdout.strip() or f"checker exited {proc.returncode}"
        return 0.0, message
    try:
        ratio = float(result_text.split(maxsplit=1)[0])
    except (IndexError, ValueError):
        return 0.0, f"Cannot parse official checker result: {result_text}"
    return ratio * OFFICIAL_SCORE_SCALE, result_text


def evaluate_polyomino_cpp_official(
    code: str,
    *,
    timeout_s: int = 8,
    work_dir: str | None = None,
    cache_dir: str | None = None,
    concurrency: int = FRONTIER_CS_POLYOMINO_CONCURRENCY,
    case_limit: int | None = None,
) -> tuple[float, str]:
    assets = ensure_frontier_cs_polyomino_assets(cache_dir, case_limit=case_limit)
    checker_bin = compile_official_polyomino_checker(cache_dir=assets["cache_dir"], timeout_s=max(1, min(timeout_s, 20)))
    sandbox = compile_cpp_code(code, timeout_s=max(1, min(timeout_s, 20)), work_dir=work_dir)
    if sandbox.error is not None or sandbox.binary_path is None:
        msg = sandbox.error or "Compilation failed"
        if sandbox.compile_stderr:
            msg += "\n" + sandbox.compile_stderr[-2000:]
        return 0.0, msg

    case_paths = assets["case_paths"]
    max_workers = max(1, min(int(concurrency), len(case_paths)))

    def run_case(case_index: int, input_path: str, answer_path: str) -> tuple[int, float, str]:
        with open(input_path, encoding="utf-8") as file:
            input_text = file.read()
        run = run_cpp_binary(
            sandbox.binary_path,
            input_text,
            timeout_s=FRONTIER_CS_POLYOMINO_CASE_TIMEOUT_S,
        )
        if run.timed_out:
            return case_index, 0.0, f"case{case_index}=timeout"
        if run.returncode != 0:
            return case_index, 0.0, f"case{case_index}=exit:{run.returncode}:{run.stderr[-500:]}"
        output_path = os.path.join(str(sandbox.work_dir), f"case_{case_index}.out")
        result_path = os.path.join(str(sandbox.work_dir), f"case_{case_index}.result")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(run.stdout)
        score, message = _run_official_checker(checker_bin, input_path, output_path, answer_path, result_path)
        return case_index, score, f"case{case_index}=score:{score:.2f}:{message}"

    try:
        scores: list[float] = []
        messages: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_case, idx, input_path, answer_path)
                for idx, (input_path, answer_path) in enumerate(case_paths, start=1)
            ]
            for future in as_completed(futures):
                idx, score, message = future.result()
                scores.append(score)
                messages.append((idx, message))
        messages.sort(key=lambda item: item[0])
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        valid_cases = sum(1 for score in scores if score > 0)
        summary = f"official_cases={len(scores)},valid={valid_cases},concurrency={max_workers},mean_score={mean_score:.2f}"
        detail = "; ".join(message for _, message in messages[:5])
        if len(messages) > 5:
            detail += "; ..."
        return mean_score, f"{summary}; {detail}"
    except Exception as exc:
        return 0.0, str(exc)
    finally:
        cleanup_cpp_sandbox(sandbox)


def evaluate_polyomino_cpp(code: str, *, timeout_s: int = 8, work_dir: str | None = None) -> tuple[float, str]:
    sandbox = compile_cpp_code(code, timeout_s=max(1, min(timeout_s, 20)), work_dir=work_dir)
    if sandbox.error is not None or sandbox.binary_path is None:
        msg = sandbox.error or "Compilation failed"
        if sandbox.compile_stderr:
            msg += "\n" + sandbox.compile_stderr[-2000:]
        return 0.0, msg
    try:
        ratios = []
        messages = []
        per_case_timeout = max(1, int(math.ceil(timeout_s / 2)))
        for idx, case in enumerate(default_validation_cases(), start=1):
            run = run_cpp_binary(sandbox.binary_path, case.input_text, timeout_s=per_case_timeout)
            if run.timed_out:
                return 0.0, f"Case {idx} timed out"
            if run.returncode != 0:
                return 0.0, f"Case {idx} exited {run.returncode}: {run.stderr[-1000:]}"
            ratio = verify_polyomino_output(case, run.stdout)
            score = OFFICIAL_SCORE_SCALE * ratio
            ratios.append(score)
            messages.append(f"case{idx}=ratio:{ratio:.6f},score:{score:.2f}")
        return float(sum(ratios) / len(ratios)), ", ".join(messages)
    except Exception as exc:
        return 0.0, str(exc)
    finally:
        cleanup_cpp_sandbox(sandbox)


def create_polyomino_initial_state() -> DiscoveryState:
    ratio, message = evaluate_polyomino_cpp_official(
        BASELINE_CPP_CODE,
        timeout_s=8,
        concurrency=_official_concurrency(),
        case_limit=_official_case_limit(),
    )
    return DiscoveryState(
        timestep=-1,
        value=ratio,
        raw_score=ratio,
        code=BASELINE_CPP_CODE,
        construction=None,
        observation=message,
    )


def score_polyomino_result(*, raw_score: float, code: str, timestep: int, message: str) -> PolyominoScore:
    reward = raw_score / REWARD_NORMALIZER
    state = DiscoveryState(
        timestep=timestep,
        value=raw_score,
        raw_score=raw_score,
        code=code,
        construction=None,
        observation=message,
    )
    return PolyominoScore(reward=reward, raw_score=raw_score, state=state, message=message)


def build_polyomino_prompt(state: DiscoveryState, *, budget_s: int) -> str:
    current = "No previous valid C++ solution is available."
    if state.code.strip():
        current = (
            f"Current best public validation ratio: {state.raw_score:.6f}.\n"
            f"Previous evaluation notes: {state.observation or 'none'}.\n"
            "Here is the previous C++17 solution to improve:\n"
            "```cpp\n"
            f"{state.code.strip()}\n"
            "```"
        )
    return f"""You are solving Frontier-CS algorithmic problem 0: Pack the Polyominoes.

{POLYOMINO_PROBLEM_STATEMENT}

You must output a complete C++17 program. The program reads from stdin and writes the placement to stdout.
The local evaluator compiles your code with g++ -O2 -std=gnu++17 and runs it with a {budget_s}s total budget on validation cases.

{current}

Improve the packing algorithm while preserving validity. Return only one final ```cpp code block. Do not include Python, explanations, markdown outside the code block, or tool calls.
"""


class PolyominoRewardEvaluator(BaseRewardEvaluator):
    def get_reward(self, code: str, state: DiscoveryState) -> dict[str, Any]:
        raw_score, message = evaluate_polyomino_cpp_official(
            code,
            timeout_s=self.eval_timeout,
            concurrency=_official_concurrency(),
            case_limit=_official_case_limit(),
        )
        if raw_score <= 0:
            return self.failure_entry(message)
        return {
            "reward": raw_score / REWARD_NORMALIZER,
            "msg": message,
            "correctness": 1.0,
            "raw_score": raw_score,
            "result_construction": None,
            "stdout": message,
        }


class PolyominoEnv(Environment):
    reward_function = PolyominoRewardEvaluator
    state_type = DiscoveryState

    @classmethod
    def create_initial_state(cls, problem_type: str) -> DiscoveryState:
        return create_polyomino_initial_state()

    def get_question(self) -> str:
        state = self.initial_state
        current = state.to_prompt(target=90000.0, metric_name="performance", maximize=True, language="cpp")
        return f"""{POLYOMINO_PROBLEM_STATEMENT}

{current}

Rules:
- You must use C++17 to solve the problem.
- Define all of your code in one final ```cpp ``` block.
- In your final response, you should only output the code of your program. Do not include any other text.
- Your program reads the polyomino instance from stdin and writes W H followed by one X Y R F placement per piece.
- The evaluator compiles with g++ -O2 -std=gnu++17 and runs your program within {self.eval_timeout}s total budget.

Try diverse approaches to improve the packing score while preserving validity.
"""

    def _get_code_languages(self) -> list[str]:
        return ["cpp", "c++"]

    def _should_keep_code_separators(self) -> bool:
        return False
