from __future__ import annotations

import os
from pathlib import Path

from guidance_ttt.verifier.polyomino import verify_polyomino_solution_text


BASELINE_RESPONSE = r"""
<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<array<int, 4>> ans(n);
    int cursor = 0;
    int max_h = 1;

    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        int minx = INT_MAX, maxx = INT_MIN, miny = INT_MAX, maxy = INT_MIN;
        for (int j = 0; j < k; ++j) {
            int x, y;
            cin >> x >> y;
            minx = min(minx, x);
            maxx = max(maxx, x);
            miny = min(miny, y);
            maxy = max(maxy, y);
        }
        int w = maxx - minx + 1;
        int h = maxy - miny + 1;
        ans[i] = {cursor - minx, -miny, 0, 0};
        cursor += w;
        max_h = max(max_h, h);
    }

    cout << max(1, cursor) << ' ' << max_h << '\n';
    for (auto &a : ans) {
        cout << a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << '\n';
    }
    return 0;
}
```
</solution>
"""


def main() -> None:
    default_frontier_dir = Path(__file__).resolve().parents[1] / "reference" / "Frontier-CS"
    frontier_dir = Path(os.environ.get("FRONTIERCS_DIR", default_frontier_dir)).expanduser().resolve()
    result = verify_polyomino_solution_text(
        BASELINE_RESPONSE,
        problem_id="0",
        config={
            "base_dir": str(frontier_dir),
            "n_cases": 70,
        },
    )
    print(f"valid={result.valid}")
    print(f"status={result.status}")
    print(f"reward={result.reward}")
    print(f"raw_score={result.raw_score}")
    print(f"message={result.message[:2000]}")
    print(f"artifacts_keys={sorted(result.artifacts.keys())}")
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
