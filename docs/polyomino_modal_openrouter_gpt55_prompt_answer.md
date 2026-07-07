# Polyomino Modal Prompt/Answer Dump

- Output dir: `/runs/guidance_ttt/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary`
- Library: `/runs/guidance_ttt/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary/library.json`
- Entry count: `3`
- Best node id: `794c4038-5fd8-476e-a487-e814655d21d3`

## Entry `9c30fcd6-fb06-4725-b961-401b9534b25e`

- Timestep: `0`
- Status: `valid`
- Reward: `74.55953630142857`
- Raw score: `74.55953630142857`
- Guidance format ok: `False`
- Execution model: `openai/gpt-5.5`
- Execution finish reason: `stop`

### Guidance Prompt System

```text

```

### Guidance Prompt User

```text

```

### Guidance Answer

```text
Bootstrap execution without guidance.
```

### Raw Guidance Answer

```text

```

### Raw Guidance Answer With Specials

```text

```

### Execution Prompt System

```text
You are the bootstrap execution model. Produce one concrete runnable C++17 baseline candidate and a concise summary for seeding a Guidance-TTT library.
```

### Execution Prompt User

```text
<problem>
You are solving FrontierCS algorithmic problem 0: Polyomino Packing.

Write a self-contained C++17 program that reads one instance from stdin and writes
one placement to stdout.

Input:
- The first line contains n, the number of polyominoes.
- For each polyomino i, one line contains k_i, followed by k_i lines of integer
  cell coordinates x y in the polyomino local frame.
- Each polyomino has 1 to 10 cells, is 4-connected, and coordinates may be negative.

Output:
- First line: two integers W H for the chosen board.
- Then exactly n lines, one per input polyomino, each with X Y R F.
- X Y is the integer translation.
- R is one of 0, 1, 2, 3 and means clockwise rotation by R * 90 degrees.
- F is 0 or 1. If F=1, reflect across the y-axis before applying the rotation.

Validity:
- Apply transforms in this order: optional reflection, rotation, translation.
- Every transformed cell must satisfy 0 <= x < W and 0 <= y < H.
- No two transformed cells may overlap.
- Invalid output, crashes, or timeouts receive zero score.

Objective:
- Maximize the FrontierCS score by minimizing packing area W * H across the benchmark cases.
- Ties favor smaller H, then smaller W.
- Solutions are compiled with g++ -std=c++17 -O2 and evaluated by FrontierCS/go-judge.

</problem>

You are generating the initial bootstrap candidate for a Guidance-TTT run.

There is no previous library summary and no guidance yet. Produce one concrete baseline solution that satisfies the problem statement and can seed the future search history.

The <solution> block must contain one complete C++17 program in a ```cpp fenced block. It must read the Polyomino Packing instance from stdin and write the placement to stdout.

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.

Required output format:

<execution_thinking>
A short explanation of the baseline strategy used to produce the initial candidate.
Do not include code.
</execution_thinking>

<solution>
```cpp
A complete self-contained C++17 program.
```
</solution>

<summary>
A concise natural-language summary of the candidate.

This summary must describe the implemented baseline algorithm, the main construction/search/refinement mechanism, and what future guidance could improve. If the solution intentionally uses a simple heuristic rather than a full optimization method, state that clearly.

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.
</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.

```

### Execution Answer

```text
<execution_thinking>
This bootstrap candidate uses a safe constructive heuristic. It enumerates all unique rotations/reflections of each polyomino, then tries multiple board-width limits. A fast shelf packer provides reliable valid layouts, and for smaller instances a denser bottom-left grid placement heuristic is attempted to improve area. The best valid layout found is emitted in the required transform format.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;

struct Orient {
    int R = 0, F = 0;
    int minx = 0, miny = 0;
    int w = 0, h = 0;
    vector<pair<int,int>> cells; // normalized cells
};

struct Piece {
    vector<pair<int,int>> orig;
    vector<Orient> ori;
    int area = 0;
    int minW = 0;
    int minBox = 0;
};

struct OutPlace {
    long long X = 0, Y = 0;
    int R = 0, F = 0;
};

struct Result {
    bool valid = false;
    int W = 0, H = 0;
    int limitW = 0;
    vector<OutPlace> out;
};

static pair<int,int> transformCell(int x, int y, int F, int R) {
    if (F) x = -x; // reflect across y-axis first
    int tx = x, ty = y;
    if (R == 0) {
        tx = x; ty = y;
    } else if (R == 1) { // clockwise 90
        tx = y; ty = -x;
    } else if (R == 2) {
        tx = -x; ty = -y;
    } else { // R == 3
        tx = -y; ty = x;
    }
    return {tx, ty};
}

static vector<Orient> makeOrientations(const vector<pair<int,int>>& cells) {
    vector<Orient> res;
    set<vector<pair<int,int>>> seen;
    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<pair<int,int>> raw;
            raw.reserve(cells.size());
            int mnx = INT_MAX, mny = INT_MAX, mxx = INT_MIN, mxy = INT_MIN;
            for (auto [x,y] : cells) {
                auto p = transformCell(x, y, F, R);
                raw.push_back(p);
                mnx = min(mnx, p.first);
                mny = min(mny, p.second);
                mxx = max(mxx, p.first);
                mxy = max(mxy, p.second);
            }
            vector<pair<int,int>> norm;
            norm.reserve(raw.size());
            for (auto [x,y] : raw) norm.push_back({x - mnx, y - mny});
            sort(norm.begin(), norm.end());
            if (seen.insert(norm).second) {
                Orient o;
                o.R = R; o.F = F;
                o.minx = mnx; o.miny = mny;
                o.w = mxx - mnx + 1;
                o.h = mxy - mny + 1;
                o.cells = norm;
                res.push_back(o);
            }
        }
    }
    sort(res.begin(), res.end(), [](const Orient& a, const Orient& b) {
        if (a.h != b.h) return a.h < b.h;
        if (a.w != b.w) return a.w < b.w;
        if (a.F != b.F) return a.F < b.F;
        return a.R < b.R;
    });
    return res;
}

static bool betterResult(const Result& a, const Result& b) {
    if (!a.valid) return false;
    if (!b.valid) return true;
    long long aa = 1LL * a.W * a.H;
    long long bb = 1LL * b.W * b.H;
    if (aa != bb) return aa < bb;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static vector<int> makeOrder(const vector<Piece>& pieces) {
    int n = (int)pieces.size();
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].minBox != pieces[b].minBox) return pieces[a].minBox > pieces[b].minBox;
        return a < b;
    });
    return order;
}

static Result shelfPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {
    int n = (int)pieces.size();
    Result res;
    res.valid = false;
    res.limitW = limitW;
    res.out.assign(n, {});
    if (n == 0) {
        res.valid = true;
        res.W = res.H = 0;
        return res;
    }

    int x = 0, y = 0, rowH = 0;
    int usedW = 0;

    for (int id : order) {
        const Piece& p = pieces[id];

        int bestIdx = -1;
        tuple<int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

        // Try to fit in current row.
        for (int i = 0; i < (int)p.ori.size(); ++i) {
            const Orient& o = p.ori[i];
            if (o.w <= limitW - x) {
                int newRowH = max(rowH, o.h);
                auto key = make_tuple(newRowH, o.h, o.w, i);
                if (key < bestKey) {
                    bestKey = key;
                    bestIdx = i;
                }
            }
        }

        if (bestIdx < 0) {
            y += rowH;
            x = 0;
            rowH = 0;
            bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
            for (int i = 0; i < (int)p.ori.size(); ++i) {
                const Orient& o = p.ori[i];
                if (o.w <= limitW) {
                    auto key = make_tuple(o.h, o.w, o.R, i);
                    if (key < bestKey) {
                        bestKey = key;
                        bestIdx = i;
                    }
                }
            }
        }

        if (bestIdx < 0) return res;

        const Orient& o = p.ori[bestIdx];
        res.out[id].X = (long long)x - o.minx;
        res.out[id].Y = (long long)y - o.miny;
        res.out[id].R = o.R;
        res.out[id].F = o.F;

        x += o.w;
        rowH = max(rowH, o.h);
        usedW = max(usedW, x);
    }

    res.valid = true;
    res.W = usedW;
    res.H = y + rowH;
    return res;
}

static bool canPlaceAt(const vector<string>& grid, int W, const Orient& o, int px, int py) {
    if (px < 0 || px + o.w > W) return false;
    for (auto [cx, cy] : o.cells) {
        int x = px + cx;
        int y = py + cy;
        if (x < 0 || x >= W || y < 0) return false;
        if (y < (int)grid.size() && grid[y][x]) return false;
    }
    return true;
}

static Result bottomLeftPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {
    int n = (int)pieces.size();
    Result res;
    res.valid = false;
    res.limitW = limitW;
    res.out.assign(n, {});
    if (n == 0) {
        res.valid = true;
        res.W = res.H = 0;
        return res;
    }

    vector<string> grid;
    int curH = 0;
    int usedW = 0;

    for (int id : order) {
        const Piece& p = pieces[id];

        bool found = false;
        int bestO = -1, bestX = 0, bestY = 0;
        tuple<int,int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};

        for (int oi = 0; oi < (int)p.ori.size(); ++oi) {
            const Orient& o = p.ori[oi];
            if (o.w > limitW) continue;
            int maxX = limitW - o.w;
            int yLimit = curH;
            for (int y = 0; y <= yLimit; ++y) {
                if (y >= get<0>(bestKey)) break;
                for (int x = 0; x <= maxX; ++x) {
                    if (canPlaceAt(grid, limitW, o, x, y)) {
                        int nh = max(curH, y + o.h);
                        auto key = make_tuple(nh, y, x, o.h, o.w);
                        if (key < bestKey) {
                            bestKey = key;
                            bestO = oi;
                            bestX = x;
                            bestY = y;
                            found = true;
                        }
                    }
                }
            }
        }

        if (!found) return res;

        const Orient& o = p.ori[bestO];
        int newH = max(curH, bestY + o.h);
        if ((int)grid.size() < newH) grid.resize(newH, string(limitW, 0));

        for (auto [cx, cy] : o.cells) {
            int gx = bestX + cx;
            int gy = bestY + cy;
            grid[gy][gx] = 1;
            usedW = max(usedW, gx + 1);
        }

        curH = newH;
        res.out[id].X = (long long)bestX - o.minx;
        res.out[id].Y = (long long)bestY - o.miny;
        res.out[id].R = o.R;
        res.out[id].F = o.F;
    }

    res.valid = true;
    res.W = usedW;
    res.H = curH;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<Piece> pieces(n);
    long long totalArea = 0;

    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        pieces[i].orig.resize(k);
        for (int j = 0; j < k; ++j) {
            cin >> pieces[i].orig[j].first >> pieces[i].orig[j].second;
        }
        pieces[i].area = k;
        totalArea += k;
        pieces[i].ori = makeOrientations(pieces[i].orig);

        pieces[i].minW = INT_MAX;
        pieces[i].minBox = INT_MAX;
        for (const auto& o : pieces[i].ori) {
            pieces[i].minW = min(pieces[i].minW, o.w);
            pieces[i].minBox = min(pieces[i].minBox, o.w * o.h);
        }
    }

    if (n == 0) {
        cout << "0 0\n";
        return 0;
    }

    vector<int> order = makeOrder(pieces);

    int globalMinW = 1;
    long long rowWll = 0;
    for (const auto& p : pieces) {
        globalMinW = max(globalMinW, p.minW);
        rowWll += p.minW;
    }

    int root = (int)ceil(sqrt((long double)max(1LL, totalArea)));
    int rowWcap = (int)min<long long>(rowWll, INT_MAX / 4);
    int upper = min(rowWcap, max(globalMinW, 5 * root + 100));

    set<int> candSet;
    auto addCand = [&](int w) {
        if (w >= globalMinW && w <= upper) candSet.insert(w);
    };

    addCand(globalMinW);
    for (int w = globalMinW; w <= min(upper, 30); ++w) addCand(w);

    if (upper - globalMinW <= 180) {
        for (int w = globalMinW; w <= upper; ++w) addCand(w);
    } else {
        vector<long double> ratios = {0.55L, 0.70L, 0.85L, 1.00L, 1.15L, 1.30L,
                                      1.50L, 1.75L, 2.05L, 2.45L, 3.00L, 3.70L, 4.50L};
        for (long double r : ratios) {
            int base = (int)llround(root * r);
            for (int d = -4; d <= 4; ++d) addCand(base + d);
        }
        int samples = 120;
        for (int i = 0; i < samples; ++i) {
            int w = globalMinW + (int)((long long)(upper - globalMinW) * i / max(1, samples - 1));
            addCand(w);
        }
    }

    vector<int> candidates(candSet.begin(), candSet.end());

    Result best;
    // Always establish a valid baseline.
    best = shelfPack(pieces, order, globalMinW);

    vector<int> promising;
    for (int w : candidates) {
        Result r = shelfPack(pieces, order, w);
        if (betterResult(r, best)) best = r;
    }

    promising.push_back(best.limitW);
    for (int d = -12; d <= 12; ++d) promising.push_back(root + d);
    for (long double rr : {0.75L, 1.0L, 1.25L, 1.5L, 2.0L, 2.75L}) {
        int base = (int)llround(root * rr);
        for (int d = -3; d <= 3; ++d) promising.push_back(base + d);
    }
    for (int w : candidates) {
        if ((int)promising.size() > 80) break;
        promising.push_back(w);
    }

    sort(promising.begin(), promising.end());
    promising.erase(unique(promising.begin(), promising.end()), promising.end());

    auto start = chrono::steady_clock::now();
    const double timeBudget = 2.2;

    if (totalArea <= 12000) {
        for (int w : promising) {
            if (w < globalMinW || w > upper) continue;
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            if (elapsed > timeBudget) break;

            Result r = bottomLeftPack(pieces, order, w);
            if (betterResult(r, best)) best = r;
        }
    }

    if (!best.valid) {
        // Extremely defensive fallback; should not be reached.
        best = shelfPack(pieces, order, globalMinW);
    }

    cout << best.W << ' ' << best.H << '\n';
    for (int i = 0; i < n; ++i) {
        cout << best.out[i].X << ' ' << best.out[i].Y << ' '
             << best.out[i].R << ' ' << best.out[i].F << '\n';
    }

    return 0;
}
```
</solution>

<summary>
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.
</summary>
```

### Raw Model Summary

```text
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.
```

### Canonical Library Summary

```text
Execution Interpretation
This bootstrap candidate uses a safe constructive heuristic. It enumerates all unique rotations/reflections of each polyomino, then tries multiple board-width limits. A fast shelf packer provides reliable valid layouts, and for smaller instances a denser bottom-left grid placement heuristic is attempted to improve area. The best valid layout found is emitted in the required transform format.

Implemented Algorithm
```cpp
#include <bits/stdc++.h>
using namespace std;

struct Orient {
    int R = 0, F = 0;
    int minx = 0, miny = 0;
    int w = 0, h = 0;
    vector<pair<int,int>> cells; // normalized cells
};

struct Piece {
    vector<pair<int,int>> orig;
    vector<Orient> ori;
    int area = 0;
    int minW = 0;
    int minBox = 0;
};

struct OutPlace {
    long long X = 0, Y = 0;
    int R = 0, F = 0;
};

struct Result {
    bool valid = false;
    int W = 0, H = 0;
    int limitW = 0;
    vector<OutPlace> out;
};

static pair<int,int> transformCell(int x, int y, int F, int R) {
    if (F) x = -x; // reflect across y-axis first
    int tx = x, ty = y;
    if (R == 0) {
        tx = x; ty = y;
    } else if (R == 1) { // clockwise 90
        tx = y; ty = -x;
    } else if (R == 2) {
        tx = -x; ty = -y;
    } else { // R == 3
        tx = -y; ty = x;
    }
    return {tx, ty};
}

static vector<Orient> makeOrientations(const vector<pair<int,int>>& cells) {
    vector<Orient> res;
    set<vector<pair<int,int>>> seen;
    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<pair<int,int>> raw;
            raw.reserve(cells.size());
            int mnx = INT_MAX, mny = INT_MAX, mxx = INT_MIN, mxy = INT_MIN;
            for (auto [x,y] : cells) {
                auto p = transformCell(x, y, F, R);
                raw.push_back(p);
                mnx = min(mnx, p.first);
                mny = min(mny, p.second);
                mxx = max(mxx, p.first);
                mxy = max(mxy, p.second);
            }
            vector<pair<int,int>> norm;
            norm.reserve(raw.size());
            for (auto [x,y] : raw) norm.push_back({x - mnx, y - mny});
            sort(norm.begin(), norm.end());
            if (seen.insert(norm).second) {
                Orient o;
                o.R = R; o.F = F;
                o.minx = mnx; o.miny = mny;
                o.w = mxx - mnx + 1;
                o.h = mxy - mny + 1;
                o.cells = norm;
                res.push_back(o);
            }
        }
    }
    sort(res.begin(), res.end(), [](const Orient& a, const Orient& b) {
        if (a.h != b.h) return a.h < b.h;
        if (a.w != b.w) return a.w < b.w;
        if (a.F != b.F) return a.F < b.F;
        return a.R < b.R;
    });
    return res;
}

static bool betterResult(const Result& a, const Result& b) {
    if (!a.valid) return false;
    if (!b.valid) return true;
    long long aa = 1LL * a.W * a.H;
    long long bb = 1LL * b.W * b.H;
    if (aa != bb) return aa < bb;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static vector<int> makeOrder(const vector<Piece>& pieces) {
    int n = (int)pieces.size();
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].minBox != pieces[b].minBox) return pieces[a].minBox > pieces[b].minBox;
        return a < b;
    });
    return order;
}

static Result shelfPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {
    int n = (int)pieces.size();
    Result res;
    res.valid = false;
    res.limitW = limitW;
    res.out.assign(n, {});
    if (n == 0) {
        res.valid = true;
        res.W = res.H = 0;
        return res;
    }

    int x = 0, y = 0, rowH = 0;
    int usedW = 0;

    for (int id : order) {
        const Piece& p = pieces[id];

        int bestIdx = -1;
        tuple<int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

        // Try to fit in current row.
        for (int i = 0; i < (int)p.ori.size(); ++i) {
            const Orient& o = p.ori[i];
            if (o.w <= limitW - x) {
                int newRowH = max(rowH, o.h);
                auto key = make_tuple(newRowH, o.h, o.w, i);
                if (key < bestKey) {
                    bestKey = key;
                    bestIdx = i;
                }
            }
        }

        if (bestIdx < 0) {
            y += rowH;
            x = 0;
            rowH = 0;
            bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
            for (int i = 0; i < (int)p.ori.size(); ++i) {
                const Orient& o = p.ori[i];
                if (o.w <= limitW) {
                    auto key = make_tuple(o.h, o.w, o.R, i);
                    if (key < bestKey) {
                        bestKey = key;
                        bestIdx = i;
                    }
                }
            }
        }

        if (bestIdx < 0) return res;

        const Orient& o = p.ori[bestIdx];
        res.out[id].X = (long long)x - o.minx;
        res.out[id].Y = (long long)y - o.miny;
        res.out[id].R = o.R;
        res.out[id].F = o.F;

        x += o.w;
        rowH = max(rowH, o.h);
        usedW = max(usedW, x);
    }

    res.valid = true;
    res.W = usedW;
    res.H = y + rowH;
    return res;
}

static bool canPlaceAt(const vector<string>& grid, int W, const Orient& o, int px, int py) {
    if (px < 0 || px + o.w > W) return false;
    for (auto [cx, cy] : o.cells) {
        int x = px + cx;
        int y = py + cy;
        if (x < 0 || x >= W || y < 0) return false;
        if (y < (int)grid.size() && grid[y][x]) return false;
    }
    return true;
}

static Result bottomLeftPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {
    int n = (int)pieces.size();
    Result res;
    res.valid = false;
    res.limitW = limitW;
    res.out.assign(n, {});
    if (n == 0) {
        res.valid = true;
        res.W = res.H = 0;
        return res;
    }

    vector<string> grid;
    int curH = 0;
    int usedW = 0;

    for (int id : order) {
        const Piece& p = pieces[id];

        bool found = false;
        int bestO = -1, bestX = 0, bestY = 0;
        tuple<int,int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};

        for (int oi = 0; oi < (int)p.ori.size(); ++oi) {
            const Orient& o = p.ori[oi];
            if (o.w > limitW) continue;
            int maxX = limitW - o.w;
            int yLimit = curH;
            for (int y = 0; y <= yLimit; ++y) {
                if (y >= get<0>(bestKey)) break;
                for (int x = 0; x <= maxX; ++x) {
                    if (canPlaceAt(grid, limitW, o, x, y)) {
                        int nh = max(curH, y + o.h);
                        auto key = make_tuple(nh, y, x, o.h, o.w);
                        if (key < bestKey) {
                            bestKey = key;
                            bestO = oi;
                            bestX = x;
                            bestY = y;
                            found = true;
                        }
                    }
                }
            }
        }

        if (!found) return res;

        const Orient& o = p.ori[bestO];
        int newH = max(curH, bestY + o.h);
        if ((int)grid.size() < newH) grid.resize(newH, string(limitW, 0));

        for (auto [cx, cy] : o.cells) {
            int gx = bestX + cx;
            int gy = bestY + cy;
            grid[gy][gx] = 1;
            usedW = max(usedW, gx + 1);
        }

        curH = newH;
        res.out[id].X = (long long)bestX - o.minx;
        res.out[id].Y = (long long)bestY - o.miny;
        res.out[id].R = o.R;
        res.out[id].F = o.F;
    }

    res.valid = true;
    res.W = usedW;
    res.H = curH;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<Piece> pieces(n);
    long long totalArea = 0;

    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        pieces[i].orig.resize(k);
        for (int j = 0; j < k; ++j) {
            cin >> pieces[i].orig[j].first >> pieces[i].orig[j].second;
        }
        pieces[i].area = k;
        totalArea += k;
        pieces[i].ori = makeOrientations(pieces[i].orig);

        pieces[i].minW = INT_MAX;
        pieces[i].minBox = INT_MAX;
        for (const auto& o : pieces[i].ori) {
            pieces[i].minW = min(pieces[i].minW, o.w);
            pieces[i].minBox = min(pieces[i].minBox, o.w * o.h);
        }
    }

    if (n == 0) {
        cout << "0 0\n";
        return 0;
    }

    vector<int> order = makeOrder(pieces);

    int globalMinW = 1;
    long long rowWll = 0;
    for (const auto& p : pieces) {
        globalMinW = max(globalMinW, p.minW);
        rowWll += p.minW;
    }

    int root = (int)ceil(sqrt((long double)max(1LL, totalArea)));
    int rowWcap = (int)min<long long>(rowWll, INT_MAX / 4);
    int upper = min(rowWcap, max(globalMinW, 5 * root + 100));

    set<int> candSet;
    auto addCand = [&](int w) {
        if (w >= globalMinW && w <= upper) candSet.insert(w);
    };

    addCand(globalMinW);
    for (int w = globalMinW; w <= min(upper, 30); ++w) addCand(w);

    if (upper - globalMinW <= 180) {
        for (int w = globalMinW; w <= upper; ++w) addCand(w);
    } else {
        vector<long double> ratios = {0.55L, 0.70L, 0.85L, 1.00L, 1.15L, 1.30L,
                                      1.50L, 1.75L, 2.05L, 2.45L, 3.00L, 3.70L, 4.50L};
        for (long double r : ratios) {
            int base = (int)llround(root * r);
            for (int d = -4; d <= 4; ++d) addCand(base + d);
        }
        int samples = 120;
        for (int i = 0; i < samples; ++i) {
            int w = globalMinW + (int)((long long)(upper - globalMinW) * i / max(1, samples - 1));
            addCand(w);
        }
    }

    vector<int> candidates(candSet.begin(), candSet.end());

    Result best;
    // Always establish a valid baseline.
    best = shelfPack(pieces, order, globalMinW);

    vector<int> promising;
    for (int w : candidates) {
        Result r = shelfPack(pieces, order, w);
        if (betterResult(r, best)) best = r;
    }

    promising.push_back(best.limitW);
    for (int d = -12; d <= 12; ++d) promising.push_back(root + d);
    for (long double rr : {0.75L, 1.0L, 1.25L, 1.5L, 2.0L, 2.75L}) {
        int base = (int)llround(root * rr);
        for (int d = -3; d <= 3; ++d) promising.push_back(base + d);
    }
    for (int w : candidates) {
        if ((int)promising.size() > 80) break;
        promising.push_back(w);
    }

    sort(promising.begin(), promising.end());
    promising.erase(unique(promising.begin(), promising.end()), promising.end());

    auto start = chrono::steady_clock::now();
    const double timeBudget = 2.2;

    if (totalArea <= 12000) {
        for (int w : promising) {
            if (w < globalMinW || w > upper) continue;
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            if (elapsed > timeBudget) break;

            Result r = bottomLeftPack(pieces, order, w);
            if (betterResult(r, best)) best = r;
        }
    }

    if (!best.valid) {
        // Extremely defensive fallback; should not be reached.
        best = shelfPack(pieces, order, globalMinW);
    }

    cout << best.W << ' ' << best.H << '\n';
    for (int i = 0; i < n; ++i) {
        cout << best.out[i].X << ' ' << best.out[i].Y << ' '
             << best.out[i].R << ' ' << best.out[i].F << '\n';
    }

    return 0;
}
```

New Ideas Introduced
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.

Empirical Outcome
Verifier status: valid
FrontierCS score: 74.55953630142857
Reward: 74.55953630142857
Verifier message: accepted

Failure / Bottleneck Analysis
No verifier failure reported.

Next Guidance Delta
Continue from the submitted guidance: Bootstrap execution without guidance.
```

### Verification Artifacts

```json
{
  "code": "#include <bits/stdc++.h>\nusing namespace std;\n\nstruct Orient {\n    int R = 0, F = 0;\n    int minx = 0, miny = 0;\n    int w = 0, h = 0;\n    vector<pair<int,int>> cells; // normalized cells\n};\n\nstruct Piece {\n    vector<pair<int,int>> orig;\n    vector<Orient> ori;\n    int area = 0;\n    int minW = 0;\n    int minBox = 0;\n};\n\nstruct OutPlace {\n    long long X = 0, Y = 0;\n    int R = 0, F = 0;\n};\n\nstruct Result {\n    bool valid = false;\n    int W = 0, H = 0;\n    int limitW = 0;\n    vector<OutPlace> out;\n};\n\nstatic pair<int,int> transformCell(int x, int y, int F, int R) {\n    if (F) x = -x; // reflect across y-axis first\n    int tx = x, ty = y;\n    if (R == 0) {\n        tx = x; ty = y;\n    } else if (R == 1) { // clockwise 90\n        tx = y; ty = -x;\n    } else if (R == 2) {\n        tx = -x; ty = -y;\n    } else { // R == 3\n        tx = -y; ty = x;\n    }\n    return {tx, ty};\n}\n\nstatic vector<Orient> makeOrientations(const vector<pair<int,int>>& cells) {\n    vector<Orient> res;\n    set<vector<pair<int,int>>> seen;\n    for (int F = 0; F <= 1; ++F) {\n        for (int R = 0; R < 4; ++R) {\n            vector<pair<int,int>> raw;\n            raw.reserve(cells.size());\n            int mnx = INT_MAX, mny = INT_MAX, mxx = INT_MIN, mxy = INT_MIN;\n            for (auto [x,y] : cells) {\n                auto p = transformCell(x, y, F, R);\n                raw.push_back(p);\n                mnx = min(mnx, p.first);\n                mny = min(mny, p.second);\n                mxx = max(mxx, p.first);\n                mxy = max(mxy, p.second);\n            }\n            vector<pair<int,int>> norm;\n            norm.reserve(raw.size());\n            for (auto [x,y] : raw) norm.push_back({x - mnx, y - mny});\n            sort(norm.begin(), norm.end());\n            if (seen.insert(norm).second) {\n                Orient o;\n                o.R = R; o.F = F;\n                o.minx = mnx; o.miny = mny;\n                o.w = mxx - mnx + 1;\n                o.h = mxy - mny + 1;\n                o.cells = norm;\n                res.push_back(o);\n            }\n        }\n    }\n    sort(res.begin(), res.end(), [](const Orient& a, const Orient& b) {\n        if (a.h != b.h) return a.h < b.h;\n        if (a.w != b.w) return a.w < b.w;\n        if (a.F != b.F) return a.F < b.F;\n        return a.R < b.R;\n    });\n    return res;\n}\n\nstatic bool betterResult(const Result& a, const Result& b) {\n    if (!a.valid) return false;\n    if (!b.valid) return true;\n    long long aa = 1LL * a.W * a.H;\n    long long bb = 1LL * b.W * b.H;\n    if (aa != bb) return aa < bb;\n    if (a.H != b.H) return a.H < b.H;\n    return a.W < b.W;\n}\n\nstatic vector<int> makeOrder(const vector<Piece>& pieces) {\n    int n = (int)pieces.size();\n    vector<int> order(n);\n    iota(order.begin(), order.end(), 0);\n    sort(order.begin(), order.end(), [&](int a, int b) {\n        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;\n        if (pieces[a].minBox != pieces[b].minBox) return pieces[a].minBox > pieces[b].minBox;\n        return a < b;\n    });\n    return order;\n}\n\nstatic Result shelfPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {\n    int n = (int)pieces.size();\n    Result res;\n    res.valid = false;\n    res.limitW = limitW;\n    res.out.assign(n, {});\n    if (n == 0) {\n        res.valid = true;\n        res.W = res.H = 0;\n        return res;\n    }\n\n    int x = 0, y = 0, rowH = 0;\n    int usedW = 0;\n\n    for (int id : order) {\n        const Piece& p = pieces[id];\n\n        int bestIdx = -1;\n        tuple<int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};\n\n        // Try to fit in current row.\n        for (int i = 0; i < (int)p.ori.size(); ++i) {\n            const Orient& o = p.ori[i];\n            if (o.w <= limitW - x) {\n                int newRowH = max(rowH, o.h);\n                auto key = make_tuple(newRowH, o.h, o.w, i);\n                if (key < bestKey) {\n                    bestKey = key;\n                    bestIdx = i;\n                }\n            }\n        }\n\n        if (bestIdx < 0) {\n            y += rowH;\n            x = 0;\n            rowH = 0;\n            bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};\n            for (int i = 0; i < (int)p.ori.size(); ++i) {\n                const Orient& o = p.ori[i];\n                if (o.w <= limitW) {\n                    auto key = make_tuple(o.h, o.w, o.R, i);\n                    if (key < bestKey) {\n                        bestKey = key;\n                        bestIdx = i;\n                    }\n                }\n            }\n        }\n\n        if (bestIdx < 0) return res;\n\n        const Orient& o = p.ori[bestIdx];\n        res.out[id].X = (long long)x - o.minx;\n        res.out[id].Y = (long long)y - o.miny;\n        res.out[id].R = o.R;\n        res.out[id].F = o.F;\n\n        x += o.w;\n        rowH = max(rowH, o.h);\n        usedW = max(usedW, x);\n    }\n\n    res.valid = true;\n    res.W = usedW;\n    res.H = y + rowH;\n    return res;\n}\n\nstatic bool canPlaceAt(const vector<string>& grid, int W, const Orient& o, int px, int py) {\n    if (px < 0 || px + o.w > W) return false;\n    for (auto [cx, cy] : o.cells) {\n        int x = px + cx;\n        int y = py + cy;\n        if (x < 0 || x >= W || y < 0) return false;\n        if (y < (int)grid.size() && grid[y][x]) return false;\n    }\n    return true;\n}\n\nstatic Result bottomLeftPack(const vector<Piece>& pieces, const vector<int>& order, int limitW) {\n    int n = (int)pieces.size();\n    Result res;\n    res.valid = false;\n    res.limitW = limitW;\n    res.out.assign(n, {});\n    if (n == 0) {\n        res.valid = true;\n        res.W = res.H = 0;\n        return res;\n    }\n\n    vector<string> grid;\n    int curH = 0;\n    int usedW = 0;\n\n    for (int id : order) {\n        const Piece& p = pieces[id];\n\n        bool found = false;\n        int bestO = -1, bestX = 0, bestY = 0;\n        tuple<int,int,int,int,int> bestKey = {INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};\n\n        for (int oi = 0; oi < (int)p.ori.size(); ++oi) {\n            const Orient& o = p.ori[oi];\n            if (o.w > limitW) continue;\n            int maxX = limitW - o.w;\n            int yLimit = curH;\n            for (int y = 0; y <= yLimit; ++y) {\n                if (y >= get<0>(bestKey)) break;\n                for (int x = 0; x <= maxX; ++x) {\n                    if (canPlaceAt(grid, limitW, o, x, y)) {\n                        int nh = max(curH, y + o.h);\n                        auto key = make_tuple(nh, y, x, o.h, o.w);\n                        if (key < bestKey) {\n                            bestKey = key;\n                            bestO = oi;\n                            bestX = x;\n                            bestY = y;\n                            found = true;\n                        }\n                    }\n                }\n            }\n        }\n\n        if (!found) return res;\n\n        const Orient& o = p.ori[bestO];\n        int newH = max(curH, bestY + o.h);\n        if ((int)grid.size() < newH) grid.resize(newH, string(limitW, 0));\n\n        for (auto [cx, cy] : o.cells) {\n            int gx = bestX + cx;\n            int gy = bestY + cy;\n            grid[gy][gx] = 1;\n            usedW = max(usedW, gx + 1);\n        }\n\n        curH = newH;\n        res.out[id].X = (long long)bestX - o.minx;\n        res.out[id].Y = (long long)bestY - o.miny;\n        res.out[id].R = o.R;\n        res.out[id].F = o.F;\n    }\n\n    res.valid = true;\n    res.W = usedW;\n    res.H = curH;\n    return res;\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    int n;\n    if (!(cin >> n)) return 0;\n\n    vector<Piece> pieces(n);\n    long long totalArea = 0;\n\n    for (int i = 0; i < n; ++i) {\n        int k;\n        cin >> k;\n        pieces[i].orig.resize(k);\n        for (int j = 0; j < k; ++j) {\n            cin >> pieces[i].orig[j].first >> pieces[i].orig[j].second;\n        }\n        pieces[i].area = k;\n        totalArea += k;\n        pieces[i].ori = makeOrientations(pieces[i].orig);\n\n        pieces[i].minW = INT_MAX;\n        pieces[i].minBox = INT_MAX;\n        for (const auto& o : pieces[i].ori) {\n            pieces[i].minW = min(pieces[i].minW, o.w);\n            pieces[i].minBox = min(pieces[i].minBox, o.w * o.h);\n        }\n    }\n\n    if (n == 0) {\n        cout << \"0 0\\n\";\n        return 0;\n    }\n\n    vector<int> order = makeOrder(pieces);\n\n    int globalMinW = 1;\n    long long rowWll = 0;\n    for (const auto& p : pieces) {\n        globalMinW = max(globalMinW, p.minW);\n        rowWll += p.minW;\n    }\n\n    int root = (int)ceil(sqrt((long double)max(1LL, totalArea)));\n    int rowWcap = (int)min<long long>(rowWll, INT_MAX / 4);\n    int upper = min(rowWcap, max(globalMinW, 5 * root + 100));\n\n    set<int> candSet;\n    auto addCand = [&](int w) {\n        if (w >= globalMinW && w <= upper) candSet.insert(w);\n    };\n\n    addCand(globalMinW);\n    for (int w = globalMinW; w <= min(upper, 30); ++w) addCand(w);\n\n    if (upper - globalMinW <= 180) {\n        for (int w = globalMinW; w <= upper; ++w) addCand(w);\n    } else {\n        vector<long double> ratios = {0.55L, 0.70L, 0.85L, 1.00L, 1.15L, 1.30L,\n                                      1.50L, 1.75L, 2.05L, 2.45L, 3.00L, 3.70L, 4.50L};\n        for (long double r : ratios) {\n            int base = (int)llround(root * r);\n            for (int d = -4; d <= 4; ++d) addCand(base + d);\n        }\n        int samples = 120;\n        for (int i = 0; i < samples; ++i) {\n            int w = globalMinW + (int)((long long)(upper - globalMinW) * i / max(1, samples - 1));\n            addCand(w);\n        }\n    }\n\n    vector<int> candidates(candSet.begin(), candSet.end());\n\n    Result best;\n    // Always establish a valid baseline.\n    best = shelfPack(pieces, order, globalMinW);\n\n    vector<int> promising;\n    for (int w : candidates) {\n        Result r = shelfPack(pieces, order, w);\n        if (betterResult(r, best)) best = r;\n    }\n\n    promising.push_back(best.limitW);\n    for (int d = -12; d <= 12; ++d) promising.push_back(root + d);\n    for (long double rr : {0.75L, 1.0L, 1.25L, 1.5L, 2.0L, 2.75L}) {\n        int base = (int)llround(root * rr);\n        for (int d = -3; d <= 3; ++d) promising.push_back(base + d);\n    }\n    for (int w : candidates) {\n        if ((int)promising.size() > 80) break;\n        promising.push_back(w);\n    }\n\n    sort(promising.begin(), promising.end());\n    promising.erase(unique(promising.begin(), promising.end()), promising.end());\n\n    auto start = chrono::steady_clock::now();\n    const double timeBudget = 2.2;\n\n    if (totalArea <= 12000) {\n        for (int w : promising) {\n            if (w < globalMinW || w > upper) continue;\n            auto now = chrono::steady_clock::now();\n            double elapsed = chrono::duration<double>(now - start).count();\n            if (elapsed > timeBudget) break;\n\n            Result r = bottomLeftPack(pieces, order, w);\n            if (betterResult(r, best)) best = r;\n        }\n    }\n\n    if (!best.valid) {\n        // Extremely defensive fallback; should not be reached.\n        best = shelfPack(pieces, order, globalMinW);\n    }\n\n    cout << best.W << ' ' << best.H << '\\n';\n    for (int i = 0; i < n; ++i) {\n        cout << best.out[i].X << ' ' << best.out[i].Y << ' '\n             << best.out[i].R << ' ' << best.out[i].F << '\\n';\n    }\n\n    return 0;\n}",
  "frontiercs_base_dir": "/opt/Frontier-CS",
  "frontiercs_config": {
    "memory_mb": 256,
    "n_cases": 70,
    "time_limit_s": 2,
    "total_timeout_s": 340
  },
  "problem_id": "0"
}
```

## Entry `457127b4-eb6b-440f-89bd-7c85d33490fa`

- Timestep: `1`
- Status: `valid`
- Reward: `5.582855064285715`
- Raw score: `5.582855064285715`
- Guidance format ok: `True`
- Execution model: `openai/gpt-5.5`
- Execution finish reason: `None`

### Guidance Prompt System

```text
You are the Guidance Model, acting as a strategic navigator for an open-ended scientific discovery process.

Your primary objective is to provide **evolutionary guidance**. Do not write final code or focus on low-level implementation details. Instead, your task is to propose high-level directional shifts, conceptual mutations, and novel pathways to explore the search space.

Focus on how the current ideas can *evolve* to escape local optima and discover fundamentally new mechanisms.
```

### Guidance Prompt User

```text
<problem>
You are solving FrontierCS algorithmic problem 0: Polyomino Packing.

Write a self-contained C++17 program that reads one instance from stdin and writes
one placement to stdout.

Input:
- The first line contains n, the number of polyominoes.
- For each polyomino i, one line contains k_i, followed by k_i lines of integer
  cell coordinates x y in the polyomino local frame.
- Each polyomino has 1 to 10 cells, is 4-connected, and coordinates may be negative.

Output:
- First line: two integers W H for the chosen board.
- Then exactly n lines, one per input polyomino, each with X Y R F.
- X Y is the integer translation.
- R is one of 0, 1, 2, 3 and means clockwise rotation by R * 90 degrees.
- F is 0 or 1. If F=1, reflect across the y-axis before applying the rotation.

Validity:
- Apply transforms in this order: optional reflection, rotation, translation.
- Every transformed cell must satisfy 0 <= x < W and 0 <= y < H.
- No two transformed cells may overlap.
- Invalid output, crashes, or timeouts receive zero score.

Objective:
- Maximize the FrontierCS score by minimizing packing area W * H across the benchmark cases.
- Ties favor smaller H, then smaller W.
- Solutions are compiled with g++ -std=c++17 -O2 and evaluated by FrontierCS/go-judge.

</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.

Verifier status: valid
FrontierCS score: 74.55953630142857
Reward: 74.55953630142857
Verifier message: accepted
</selected_summary>

<local_failures>
No local failure summaries yet.
</local_failures>

# Objective
Your task is to provide the next **evolutionary guidance** to beat the current visible FrontierCS score target (74.55953630142857). Higher FrontierCS score is better.

# Evolutionary Guidelines
1. Analyze the search history.
Use `<selected_summary>` and `<local_failures>` to identify what has already been tried, what worked, what failed, and what bottleneck the next attempt should address.

2. Use internal reasoning, but only expose guidance.
Think through the current search state before answering. Do not create a custom reasoning XML block. Only the final guidance should be exposed to the downstream execution model.

3. Stay at the algorithmic-strategy level.
Propose high-level algorithmic directions, search mechanisms, objective changes, relaxation ideas, or hybrid optimization strategies. Do not write code, pseudocode, implementation details, or parameter schedules.

4. Produce exactly the required XML structure.
Return exactly one `<guidance>` block and no other custom XML blocks, commentary, code, or markdown.

Think step by step internally. Do not include the internal reasoning in the final XML block. Provide your response exactly in the following format:

<guidance>
Provide the final evolutionary guidance for the next execution attempt. Describe the main algorithmic direction to explore and why it is promising. Keep the guidance conceptual and actionable.
</guidance>

```

### Guidance Answer

```text
Introduce a hybrid approach combining adaptive width selection with simulated annealing for local optimization. Dynamically adjust width candidates based on polyomino aspect ratios and use a probabilistic swap-based local search to iteratively refine placements. Prioritize polyomino ordering by both area and "shape compactness" metrics to guide the search toward configurations with lower bounding box ratios. This addresses the current limitations of fixed-width heuristics and greedy placement, enabling escape from suboptimal arrangements while maintaining validity constraints.
```

### Raw Guidance Answer

```text
<think>
Okay, I need to figure out how to improve the current FrontierCS score for the Polyomino Packing problem. The current approach uses a heuristic method with shelf-based packing and greedy placement, achieving a score of around 74.56. To beat this, I should think about what's missing or could be optimized.

First, let's review the selected summary. The algorithm tries different rotations and reflections for each polyomino, then uses a shelf-based approach for width limits. For smaller instances, it uses a greedy bottom-left placement with larger pieces first. The verifier says it's valid, but maybe there's room for better area optimization.

The local failures section is empty, so no specific issues there. The main problem might be that the current approach is too simplistic and doesn't explore more complex packing strategies. The current method might not efficiently handle cases where a more optimal arrangement requires non-greedy choices or better handling of varying polyomino shapes.

What could be the next evolutionary step? The current approach is heuristic and doesn't use advanced optimization techniques like simulated annealing, beam search, or more sophisticated placement strategies. Maybe introducing a more dynamic or adaptive method could help. For example, instead of fixed width limits, maybe using a variable width selection based on the polyomino sizes. Also, considering the order of placing polyominoes isn't optimal—perhaps a better heuristic for ordering, like sorting by area or shape complexity.

Another idea is to incorporate local search or post-processing steps. After placing the polyominoes, maybe there's a way to compact the arrangement by shifting pieces to fill gaps. Additionally, the current approach doesn't handle reflections and rotations as thoroughly as possible. Maybe considering more transformations or using a more exhaustive search for orientations.

Alternatively, using a hybrid approach that combines the current greedy method with some form of simulated annealing could allow escaping local optima. Or maybe a beam search that explores multiple candidate placements and keeps the best ones. Also, considering the aspect ratio of the board—prioritizing H over W when possible might help in some cases.

Another angle is to analyze the polyominoes' shapes more deeply. For instance, some shapes might fit better in certain orientations or positions. Maybe grouping polyominoes by their bounding box dimensions or other features to guide the packing.

The key is to move beyond the current heuristics and introduce more sophisticated strategies that can explore a larger solution space. Perhaps integrating a more advanced search algorithm while maintaining the robustness of the existing method. This could involve more intelligent ordering of pieces, better width selection, and mechanisms for adjusting placements after initial packing.

So, the next evolutionary guidance should focus on enhancing the search strategy. Maybe implementing a beam search or simulated annealing to explore better placements, or improving the ordering of polyominoes. Also, considering more advanced placement techniques that account for the entire board's layout rather than just local greedy choices. This should help in reducing the overall area W*H more effectively, thus improving the FrontierCS score.
</think>

<guidance>
Introduce a hybrid approach combining adaptive width selection with simulated annealing for local optimization. Dynamically adjust width candidates based on polyomino aspect ratios and use a probabilistic swap-based local search to iteratively refine placements. Prioritize polyomino ordering by both area and "shape compactness" metrics to guide the search toward configurations with lower bounding box ratios. This addresses the current limitations of fixed-width heuristics and greedy placement, enabling escape from suboptimal arrangements while maintaining validity constraints.
</guidance>
```

### Raw Guidance Answer With Specials

```text
<think>
Okay, I need to figure out how to improve the current FrontierCS score for the Polyomino Packing problem. The current approach uses a heuristic method with shelf-based packing and greedy placement, achieving a score of around 74.56. To beat this, I should think about what's missing or could be optimized.

First, let's review the selected summary. The algorithm tries different rotations and reflections for each polyomino, then uses a shelf-based approach for width limits. For smaller instances, it uses a greedy bottom-left placement with larger pieces first. The verifier says it's valid, but maybe there's room for better area optimization.

The local failures section is empty, so no specific issues there. The main problem might be that the current approach is too simplistic and doesn't explore more complex packing strategies. The current method might not efficiently handle cases where a more optimal arrangement requires non-greedy choices or better handling of varying polyomino shapes.

What could be the next evolutionary step? The current approach is heuristic and doesn't use advanced optimization techniques like simulated annealing, beam search, or more sophisticated placement strategies. Maybe introducing a more dynamic or adaptive method could help. For example, instead of fixed width limits, maybe using a variable width selection based on the polyomino sizes. Also, considering the order of placing polyominoes isn't optimal—perhaps a better heuristic for ordering, like sorting by area or shape complexity.

Another idea is to incorporate local search or post-processing steps. After placing the polyominoes, maybe there's a way to compact the arrangement by shifting pieces to fill gaps. Additionally, the current approach doesn't handle reflections and rotations as thoroughly as possible. Maybe considering more transformations or using a more exhaustive search for orientations.

Alternatively, using a hybrid approach that combines the current greedy method with some form of simulated annealing could allow escaping local optima. Or maybe a beam search that explores multiple candidate placements and keeps the best ones. Also, considering the aspect ratio of the board—prioritizing H over W when possible might help in some cases.

Another angle is to analyze the polyominoes' shapes more deeply. For instance, some shapes might fit better in certain orientations or positions. Maybe grouping polyominoes by their bounding box dimensions or other features to guide the packing.

The key is to move beyond the current heuristics and introduce more sophisticated strategies that can explore a larger solution space. Perhaps integrating a more advanced search algorithm while maintaining the robustness of the existing method. This could involve more intelligent ordering of pieces, better width selection, and mechanisms for adjusting placements after initial packing.

So, the next evolutionary guidance should focus on enhancing the search strategy. Maybe implementing a beam search or simulated annealing to explore better placements, or improving the ordering of polyominoes. Also, considering more advanced placement techniques that account for the entire board's layout rather than just local greedy choices. This should help in reducing the overall area W*H more effectively, thus improving the FrontierCS score.
</think>

<guidance>
Introduce a hybrid approach combining adaptive width selection with simulated annealing for local optimization. Dynamically adjust width candidates based on polyomino aspect ratios and use a probabilistic swap-based local search to iteratively refine placements. Prioritize polyomino ordering by both area and "shape compactness" metrics to guide the search toward configurations with lower bounding box ratios. This addresses the current limitations of fixed-width heuristics and greedy placement, enabling escape from suboptimal arrangements while maintaining validity constraints.
</guidance><|im_end|>
```

### Execution Prompt System

```text
You are the execution model. Turn guidance into one concrete runnable C++17 candidate. Output execution thinking first, then the code block, then the summary.
```

### Execution Prompt User

```text
<problem>
You are solving FrontierCS algorithmic problem 0: Polyomino Packing.

Write a self-contained C++17 program that reads one instance from stdin and writes
one placement to stdout.

Input:
- The first line contains n, the number of polyominoes.
- For each polyomino i, one line contains k_i, followed by k_i lines of integer
  cell coordinates x y in the polyomino local frame.
- Each polyomino has 1 to 10 cells, is 4-connected, and coordinates may be negative.

Output:
- First line: two integers W H for the chosen board.
- Then exactly n lines, one per input polyomino, each with X Y R F.
- X Y is the integer translation.
- R is one of 0, 1, 2, 3 and means clockwise rotation by R * 90 degrees.
- F is 0 or 1. If F=1, reflect across the y-axis before applying the rotation.

Validity:
- Apply transforms in this order: optional reflection, rotation, translation.
- Every transformed cell must satisfy 0 <= x < W and 0 <= y < H.
- No two transformed cells may overlap.
- Invalid output, crashes, or timeouts receive zero score.

Objective:
- Maximize the FrontierCS score by minimizing packing area W * H across the benchmark cases.
- Ties favor smaller H, then smaller W.
- Solutions are compiled with g++ -std=c++17 -O2 and evaluated by FrontierCS/go-judge.

</problem>

The next sections describe the current search state for this problem.

<selected_summary>
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.

Verifier status: valid
FrontierCS score: 74.55953630142857
Reward: 74.55953630142857
Verifier message: accepted
</selected_summary>

<guidance>
Introduce a hybrid approach combining adaptive width selection with simulated annealing for local optimization. Dynamically adjust width candidates based on polyomino aspect ratios and use a probabilistic swap-based local search to iteratively refine placements. Prioritize polyomino ordering by both area and "shape compactness" metrics to guide the search toward configurations with lower bounding box ratios. This addresses the current limitations of fixed-width heuristics and greedy placement, enabling escape from suboptimal arrangements while maintaining validity constraints.
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Score direction: max.
Implement one concrete solution that follows the guidance while satisfying the problem specification.
The <solution> block must contain one complete C++17 program in a ```cpp fenced block. It must read the Polyomino Packing instance from stdin and write the placement to stdout.

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.
You must output all three XML blocks exactly as shown below.
The <execution_thinking>...</execution_thinking> block is mandatory and must use angle brackets.
The <solution> block is mandatory and must contain a fenced ```cpp code block.
The <summary>...</summary> block is mandatory and must be closed.
Do not output only execution_thinking, only a summary, or plain natural language.
Do not omit angle brackets from XML tags.

Required output format:

<execution_thinking>
A short explanation of how the guidance was converted into the submitted algorithm.
Do not include code.
</execution_thinking>

<solution>
```cpp
// complete executable solution required by the problem
```
</solution>

<summary>
A concise natural-language summary of the candidate.

This summary must describe the implemented algorithm, the guidance-driven change from the prior idea, and the main search/refinement/optimization mechanisms used. If a suggested guidance component was not actually implemented, explicitly state that it was simplified or omitted. Mention implementation details only when they are conceptually important, such as placement ordering, orientation normalization, feasibility checks, local search, restart strategy, board-size selection, or constraint handling.

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.

</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.

```

### Execution Answer

```text
<execution_thinking>
I converted the guidance into a concrete heuristic packer: it generates all unique orientations, builds adaptive board-width candidates from total area and shape aspect data, evaluates shelf and bottom-left greedy placements with area/compactness-aware piece orders, and then performs a time-limited simulated-annealing style local search over piece-order swaps and width changes. Every candidate placement is constructed by explicit non-overlap checks, and the best valid result is emitted using the original transform parameters.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;

struct OrigPiece {
    vector<pair<long long,long long>> cells;
};

struct Variant {
    vector<pair<int,int>> cells;          // normalized occupied cells
    vector<vector<int>> rows;             // row -> xs, for faster fit
    int w = 0, h = 0;
    int R = 0, F = 0;
    long long minx = 0, miny = 0;         // raw transformed minimum before normalization
};

struct Piece {
    vector<Variant> vars;
    int area = 0;
    double compactness = 1.0;
    int minBBoxArea = 1;
    int maxDim = 1;
};

struct OutPlace {
    int x = 0, y = 0, var = 0;            // normalized placement and chosen variant
};

struct Result {
    bool ok = false;
    int W = 0, H = 0;                     // actual bounding box
    int fixedW = 0;                       // width limit used by greedy/shelf
    vector<OutPlace> place;
};

static chrono::steady_clock::time_point START_TIME;
static const double TIME_LIMIT_SEC = 1.85;

static double elapsedSec() {
    return chrono::duration<double>(chrono::steady_clock::now() - START_TIME).count();
}

static pair<long long,long long> transformCell(long long x, long long y, int R, int F) {
    if (F) x = -x; // reflection across y-axis
    long long nx = x, ny = y;
    if (R == 0) {
        nx = x; ny = y;
    } else if (R == 1) { // clockwise 90: (x,y)->(y,-x)
        nx = y; ny = -x;
    } else if (R == 2) {
        nx = -x; ny = -y;
    } else {
        nx = -y; ny = x;
    }
    return {nx, ny};
}

static void generateVariants(const OrigPiece &op, Piece &p) {
    p.area = (int)op.cells.size();
    map<string, Variant> seen;

    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<pair<long long,long long>> raw;
            raw.reserve(op.cells.size());
            long long mnx = LLONG_MAX, mny = LLONG_MAX;
            for (auto &c : op.cells) {
                auto t = transformCell(c.first, c.second, R, F);
                raw.push_back(t);
                mnx = min(mnx, t.first);
                mny = min(mny, t.second);
            }

            vector<pair<int,int>> norm;
            norm.reserve(raw.size());
            int mx = 0, my = 0;
            for (auto &q : raw) {
                int x = (int)(q.first - mnx);
                int y = (int)(q.second - mny);
                norm.push_back({x,y});
                mx = max(mx, x);
                my = max(my, y);
            }
            sort(norm.begin(), norm.end());

            string key;
            for (auto &q : norm) {
                key += to_string(q.first);
                key += ",";
                key += to_string(q.second);
                key += ";";
            }
            if (seen.count(key)) continue;

            Variant v;
            v.cells = norm;
            v.w = mx + 1;
            v.h = my + 1;
            v.R = R;
            v.F = F;
            v.minx = mnx;
            v.miny = mny;
            v.rows.assign(v.h, {});
            for (auto &q : norm) v.rows[q.second].push_back(q.first);
            for (auto &row : v.rows) sort(row.begin(), row.end());
            seen[key] = v;
        }
    }

    p.vars.clear();
    for (auto &kv : seen) p.vars.push_back(kv.second);

    int bestBBox = INT_MAX, maxDim = 1;
    for (auto &v : p.vars) {
        bestBBox = min(bestBBox, v.w * v.h);
        maxDim = max(maxDim, max(v.w, v.h));
    }
    p.minBBoxArea = max(1, bestBBox);
    p.maxDim = maxDim;
    p.compactness = (double)p.area / (double)p.minBBoxArea;
}

static bool betterResult(const Result &a, const Result &b) {
    if (!a.ok) return false;
    if (!b.ok) return true;
    long long aa = 1LL * a.W * a.H;
    long long ba = 1LL * b.W * b.H;
    if (aa != ba) return aa < ba;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static Result makeHorizontalStrip(const vector<Piece> &pieces, const vector<int> &order) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.place.assign(n, {});
    int x = 0, H = 0;
    for (int id : order) {
        int bestVar = 0;
        for (int vi = 1; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &a = pieces[id].vars[vi];
            const Variant &b = pieces[id].vars[bestVar];
            if (a.h != b.h ? a.h < b.h : a.w < b.w) bestVar = vi;
        }
        const Variant &v = pieces[id].vars[bestVar];
        res.place[id] = {x, 0, bestVar};
        x += v.w;
        H = max(H, v.h);
    }
    res.W = max(1, x);
    res.H = max(1, H);
    res.fixedW = res.W;
    return res;
}

static Result shelfPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.fixedW = Wlimit;
    res.place.assign(n, {});

    int rowX = 0, rowY = 0, rowH = 0;
    int usedW = 0;

    for (int id : order) {
        int bestVar = -1;
        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &v = pieces[id].vars[vi];
            if (v.w > Wlimit) continue;
            if (bestVar < 0) bestVar = vi;
            else {
                const Variant &b = pieces[id].vars[bestVar];
                if (v.h != b.h ? v.h < b.h : v.w < b.w) bestVar = vi;
            }
        }
        if (bestVar < 0) {
            res.ok = false;
            return res;
        }
        const Variant &v = pieces[id].vars[bestVar];
        if (rowX > 0 && rowX + v.w > Wlimit) {
            rowY += rowH;
            rowX = 0;
            rowH = 0;
        }
        res.place[id] = {rowX, rowY, bestVar};
        rowX += v.w;
        rowH = max(rowH, v.h);
        usedW = max(usedW, rowX);
    }

    res.W = max(1, usedW);
    res.H = max(1, rowY + rowH);
    return res;
}

static inline bool canPlaceAt(const vector<string> &grid, const Variant &v, int x, int y) {
    int gh = (int)grid.size();
    for (int dy = 0; dy < v.h; ++dy) {
        int yy = y + dy;
        if (yy >= gh) continue;
        const string &row = grid[yy];
        for (int dx : v.rows[dy]) {
            if (row[x + dx]) return false;
        }
    }
    return true;
}

static Result greedyPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.fixedW = Wlimit;
    res.place.assign(n, {});

    vector<string> grid;
    int curH = 0, usedW = 0;

    for (int idx = 0; idx < n; ++idx) {
        int id = order[idx];

        int bestVar = -1, bestX = 0, bestY = 0;
        int bestNewH = INT_MAX;
        int bestUsedW = INT_MAX;
        int bestTie = INT_MAX;

        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &v = pieces[id].vars[vi];
            if (v.w > Wlimit) continue;

            for (int y = 0; y <= curH; ++y) {
                int newHAtY = max(curH, y + v.h);
                if (newHAtY > bestNewH) {
                    if (y + v.h > bestNewH) break;
                }
                for (int x = 0; x <= Wlimit - v.w; ++x) {
                    int newH = max(curH, y + v.h);
                    if (newH > bestNewH) continue;
                    if (!canPlaceAt(grid, v, x, y)) continue;

                    int nUsedW = max(usedW, x + v.w);
                    int tie = y * 100000 + x + v.h * 17 + v.w;
                    bool take = false;
                    if (newH < bestNewH) take = true;
                    else if (newH == bestNewH && nUsedW < bestUsedW) take = true;
                    else if (newH == bestNewH && nUsedW == bestUsedW && tie < bestTie) take = true;

                    if (take) {
                        bestNewH = newH;
                        bestUsedW = nUsedW;
                        bestTie = tie;
                        bestVar = vi;
                        bestX = x;
                        bestY = y;
                    }
                }
            }
        }

        if (bestVar < 0) {
            res.ok = false;
            return res;
        }

        const Variant &v = pieces[id].vars[bestVar];
        if ((int)grid.size() < bestY + v.h) {
            grid.resize(bestY + v.h, string(Wlimit, '\0'));
        }
        for (auto &c : v.cells) {
            grid[bestY + c.second][bestX + c.first] = 1;
        }

        res.place[id] = {bestX, bestY, bestVar};
        curH = max(curH, bestY + v.h);
        usedW = max(usedW, bestX + v.w);

        // Very large instances should not get stuck in expensive exhaustive scans.
        if ((idx & 15) == 0 && elapsedSec() > TIME_LIMIT_SEC * 0.98) {
            // Still a valid partial algorithm only if completed; continue is unsafe.
            // Do nothing here; outer callers avoid starting late.
        }
    }

    res.W = max(1, usedW);
    res.H = max(1, curH);
    return res;
}

static vector<int> buildAdaptiveWidths(const vector<Piece> &pieces) {
    long long A = 0;
    int lower = 1;
    int maxDim = 1;
    double aspectSum = 0.0;

    for (const Piece &p : pieces) {
        A += p.area;
        int minW = INT_MAX;
        double bestAspect = 1.0;
        int bestBox = INT_MAX;
        for (const Variant &v : p.vars) {
            minW = min(minW, v.w);
            maxDim = max(maxDim, max(v.w, v.h));
            int box = v.w * v.h;
            if (box < bestBox) {
                bestBox = box;
                bestAspect = max((double)v.w / v.h, (double)v.h / v.w);
            }
        }
        lower = max(lower, minW);
        aspectSum += bestAspect;
    }

    int n = max(1, (int)pieces.size());
    double avgAspect = aspectSum / n;
    int sqrtA = max(1, (int)ceil(sqrt((double)max(1LL, A))));
    int upper = (int)min<long long>(A, max<long long>(lower, A));

    set<int> s;
    auto add = [&](long long w) {
        if (w < lower) w = lower;
        if (w > upper) w = upper;
        if (w >= 1) s.insert((int)w);
    };

    for (int w = lower; w <= min(upper, lower + 25); ++w) add(w);

    vector<double> ratios = {0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.45, 1.75, 2.10, 2.60, 3.20, 4.00};
    if (avgAspect > 1.8) {
        ratios.push_back(4.8);
        ratios.push_back(5.8);
    }
    if (avgAspect > 2.8) ratios.push_back(7.0);

    vector<int> deltas = {-8,-5,-3,-2,-1,0,1,2,3,5,8};
    for (double r : ratios) {
        int base = (int)llround(sqrt((double)A * r));
        for (int d : deltas) add(base + d);
        add((long long)llround(base * 0.85));
        add((long long)llround(base * 1.15));
        add((long long)llround(base * 1.35));
    }

    if (A <= 900) {
        for (int w = lower; w <= upper; ++w) add(w);
    } else {
        int lo = max(lower, sqrtA / 2);
        int hi = min<long long>(upper, 3LL * sqrtA + 3LL * maxDim + 80);
        int range = max(1, hi - lo + 1);
        int step = max(1, range / 260);
        for (int w = lo; w <= hi; w += step) add(w);
        for (int w = max(lower, sqrtA - 40); w <= min(upper, sqrtA + 80); ++w) add(w);
    }

    add(maxDim);
    add(sqrtA);
    add((long long)sqrtA + maxDim);
    add((long long)2 * sqrtA);
    add(upper);

    vector<int> out(s.begin(), s.end());
    return out;
}

static double resultScore(const Result &r, long long totalArea) {
    long long area = 1LL * r.W * r.H;
    double denom = (double)max(1LL, totalArea);
    return (double)area + (double)r.H / (denom + 1.0) + (double)r.W / ((denom + 1.0) * (denom + 1.0));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    START_TIME = chrono::steady_clock::now();

    int n;
    if (!(cin >> n)) return 0;

    vector<OrigPiece> orig(n);
    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        orig[i].cells.resize(k);
        for (int j = 0; j < k; ++j) {
            long long x, y;
            cin >> x >> y;
            orig[i].cells[j] = {x, y};
        }
    }

    vector<Piece> pieces(n);
    long long totalArea = 0;
    for (int i = 0; i < n; ++i) {
        generateVariants(orig[i], pieces[i]);
        totalArea += pieces[i].area;
    }

    vector<int> baseOrder(n);
    iota(baseOrder.begin(), baseOrder.end(), 0);

    vector<int> compactOrder = baseOrder;
    sort(compactOrder.begin(), compactOrder.end(), [&](int a, int b) {
        const Piece &pa = pieces[a], &pb = pieces[b];
        double sa = pa.area * 1000.0 + (1.0 - pa.compactness) * 130.0 + pa.maxDim * 7.0 + pa.minBBoxArea * 0.35;
        double sb = pb.area * 1000.0 + (1.0 - pb.compactness) * 130.0 + pb.maxDim * 7.0 + pb.minBBoxArea * 0.35;
        if (fabs(sa - sb) > 1e-9) return sa > sb;
        return a < b;
    });

    vector<int> bboxOrder = baseOrder;
    sort(bboxOrder.begin(), bboxOrder.end(), [&](int a, int b) {
        const Piece &pa = pieces[a], &pb = pieces[b];
        if (pa.minBBoxArea != pb.minBBoxArea) return pa.minBBoxArea > pb.minBBoxArea;
        if (pa.area != pb.area) return pa.area > pb.area;
        if (pa.compactness != pb.compactness) return pa.compactness < pb.compactness;
        return a < b;
    });

    vector<int> areaOrder = baseOrder;
    sort(areaOrder.begin(), areaOrder.end(), [&](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].maxDim != pieces[b].maxDim) return pieces[a].maxDim > pieces[b].maxDim;
        return a < b;
    });

    vector<vector<int>> orders;
    orders.push_back(compactOrder);
    orders.push_back(bboxOrder);
    orders.push_back(areaOrder);
    orders.push_back(baseOrder);

    mt19937 rng(1234567);
    for (int t = 0; t < 4 && n > 1; ++t) {
        vector<int> ord = compactOrder;
        stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
            double na = ((int)(rng() % 1000) - 500) / 1000.0;
            double nb = ((int)(rng() % 1000) - 500) / 1000.0;
            double sa = pieces[a].area * 50.0 + (1.0 - pieces[a].compactness) * 8.0 + pieces[a].maxDim + na;
            double sb = pieces[b].area * 50.0 + (1.0 - pieces[b].compactness) * 8.0 + pieces[b].maxDim + nb;
            return sa > sb;
        });
        orders.push_back(ord);
    }

    vector<int> widths = buildAdaptiveWidths(pieces);

    Result best = makeHorizontalStrip(pieces, compactOrder);
    vector<int> bestOrder = compactOrder;

    // Fast shelf pass over adaptive widths.
    for (const auto &ord : orders) {
        for (int W : widths) {
            Result r = shelfPack(pieces, ord, W);
            if (betterResult(r, best)) {
                best = r;
                bestOrder = ord;
            }
        }
    }

    // Bottom-left greedy pass.  Width candidates are adaptive, but limited by time.
    for (const auto &ord : orders) {
        for (int W : widths) {
            if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;
            Result r = greedyPack(pieces, ord, W);
            if (betterResult(r, best)) {
                best = r;
                bestOrder = ord;
            }
        }
        if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;
    }

    // Add local width neighborhood around current best actual and fixed widths.
    set<int> wset(widths.begin(), widths.end());
    int lower = 1;
    for (const Piece &p : pieces) {
        int mw = INT_MAX;
        for (const Variant &v : p.vars) mw = min(mw, v.w);
        lower = max(lower, mw);
    }
    int upper = max(lower, (int)totalArea);
    auto addW = [&](int w) {
        w = max(lower, min(upper, w));
        wset.insert(w);
    };
    for (int d = -25; d <= 35; ++d) {
        addW(best.W + d);
        addW(best.fixedW + d);
    }
    widths.assign(wset.begin(), wset.end());

    // Simulated-annealing-like local search over ordering swaps and width changes.
    if (n > 1) {
        vector<int> curOrder = bestOrder;
        int curW = max(lower, best.fixedW);
        Result cur = greedyPack(pieces, curOrder, curW);
        if (!cur.ok) cur = best;
        if (betterResult(cur, best)) {
            best = cur;
            bestOrder = curOrder;
        }

        double curScore = resultScore(cur, totalArea);
        double bestScore = resultScore(best, totalArea);
        int iter = 0;

        while (elapsedSec() < TIME_LIMIT_SEC * 0.97) {
            ++iter;
            vector<int> nextOrder = curOrder;
            int nextW = curW;

            int moveType = (int)(rng() % 100);
            if (moveType < 68) {
                int i = (int)(rng() % n);
                int j;
                if ((rng() % 100) < 55) {
                    int span = 1 + (int)(rng() % min(n, 8));
                    j = max(0, min(n - 1, i + ((rng() & 1) ? span : -span)));
                } else {
                    j = (int)(rng() % n);
                }
                if (i != j) swap(nextOrder[i], nextOrder[j]);
            } else if (moveType < 86) {
                int i = (int)(rng() % n);
                int len = 2 + (int)(rng() % min(n, 10));
                int j = min(n, i + len);
                if (i < j) reverse(nextOrder.begin() + i, nextOrder.begin() + j);
            } else {
                // Shape-guided perturbation: pull a bulky/less-compact random piece earlier.
                int i = (int)(rng() % n);
                int j = (int)(rng() % n);
                if (pieces[nextOrder[j]].area > pieces[nextOrder[i]].area ||
                    pieces[nextOrder[j]].compactness < pieces[nextOrder[i]].compactness) {
                    int val = nextOrder[j];
                    nextOrder.erase(nextOrder.begin() + j);
                    nextOrder.insert(nextOrder.begin() + i, val);
                }
            }

            if ((rng() % 100) < 30) {
                if ((rng() % 100) < 55 && !widths.empty()) {
                    nextW = widths[rng() % widths.size()];
                } else {
                    int delta = (int)(rng() % 17) - 8;
                    if (delta == 0) delta = 1;
                    nextW = max(lower, min(upper, curW + delta));
                }
            }

            Result nr = greedyPack(pieces, nextOrder, nextW);
            if (!nr.ok) continue;

            double ns = resultScore(nr, totalArea);
            double progress = min(1.0, elapsedSec() / (TIME_LIMIT_SEC * 0.97));
            double temp = max(0.0005, (double)max(1LL, totalArea) * 0.15 * (1.0 - progress) + 0.02);
            bool accept = false;
            if (ns <= curScore) accept = true;
            else {
                double prob = exp((curScore - ns) / temp);
                double u = (double)(rng() & 0xFFFFFF) / (double)0x1000000;
                if (u < prob) accept = true;
            }

            if (accept) {
                curOrder.swap(nextOrder);
                cur = nr;
                curW = nextW;
                curScore = ns;
            }

            if (betterResult(nr, best)) {
                best = nr;
                bestOrder = nextOrder;
                bestScore = ns;
                (void)bestScore;
                // Explore close widths of the new incumbent.
                addW(best.W);
                addW(best.fixedW);
            }
        }
    }

    // As a final safety net, ensure the chosen result is valid-looking.
    if (!best.ok || (int)best.place.size() != n) {
        best = makeHorizontalStrip(pieces, compactOrder);
    }

    cout << best.W << " " << best.H << "\n";
    for (int i = 0; i < n; ++i) {
        const OutPlace &pl = best.place[i];
        const Variant &v = pieces[i].vars[pl.var];
        long long X = (long long)pl.x - v.minx;
        long long Y = (long long)pl.y - v.miny;
        cout << X << " " << Y << " " << v.R << " " << v.F << "\n";
    }

    return 0;
}
```
</solution>

<summary>
This candidate implements a hybrid heuristic for Polyomino Packing. It enumerates all unique reflected and rotated orientations for each polyomino while preserving the transform parameters needed for output. It computes area, bounding-box, aspect, and compactness metrics, then uses those metrics to create shape-aware piece orders and adaptive width candidates instead of relying only on fixed-width trials.

The solver first guarantees validity with a horizontal strip placement, then evaluates shelf packings and bottom-left greedy grid packings across adaptive widths. The greedy placer scans feasible positions and orientations under a width limit, choosing placements that minimize resulting height and bounding width while explicitly checking cell overlap. After initial construction, it performs a time-limited simulated-annealing-style local search: neighbors swap or reorder pieces and sometimes change the width candidate, accepting worsening moves probabilistically to escape local minima. The best valid placement found is emitted, with board dimensions tightened to the actual occupied bounding box.

The guidance components are implemented in simplified practical form: adaptive width selection uses total area and aspect/compactness statistics, and simulated annealing is swap/reversal/width based rather than a full geometric perturbation optimizer.
</summary>
```

### Raw Model Summary

```text
This candidate implements a hybrid heuristic for Polyomino Packing. It enumerates all unique reflected and rotated orientations for each polyomino while preserving the transform parameters needed for output. It computes area, bounding-box, aspect, and compactness metrics, then uses those metrics to create shape-aware piece orders and adaptive width candidates instead of relying only on fixed-width trials.

The solver first guarantees validity with a horizontal strip placement, then evaluates shelf packings and bottom-left greedy grid packings across adaptive widths. The greedy placer scans feasible positions and orientations under a width limit, choosing placements that minimize resulting height and bounding width while explicitly checking cell overlap. After initial construction, it performs a time-limited simulated-annealing-style local search: neighbors swap or reorder pieces and sometimes change the width candidate, accepting worsening moves probabilistically to escape local minima. The best valid placement found is emitted, with board dimensions tightened to the actual occupied bounding box.

The guidance components are implemented in simplified practical form: adaptive width selection uses total area and aspect/compactness statistics, and simulated annealing is swap/reversal/width based rather than a full geometric perturbation optimizer.
```

### Canonical Library Summary

```text
Execution Interpretation
I converted the guidance into a concrete heuristic packer: it generates all unique orientations, builds adaptive board-width candidates from total area and shape aspect data, evaluates shelf and bottom-left greedy placements with area/compactness-aware piece orders, and then performs a time-limited simulated-annealing style local search over piece-order swaps and width changes. Every candidate placement is constructed by explicit non-overlap checks, and the best valid result is emitted using the original transform parameters.

Implemented Algorithm
```cpp
#include <bits/stdc++.h>
using namespace std;

struct OrigPiece {
    vector<pair<long long,long long>> cells;
};

struct Variant {
    vector<pair<int,int>> cells;          // normalized occupied cells
    vector<vector<int>> rows;             // row -> xs, for faster fit
    int w = 0, h = 0;
    int R = 0, F = 0;
    long long minx = 0, miny = 0;         // raw transformed minimum before normalization
};

struct Piece {
    vector<Variant> vars;
    int area = 0;
    double compactness = 1.0;
    int minBBoxArea = 1;
    int maxDim = 1;
};

struct OutPlace {
    int x = 0, y = 0, var = 0;            // normalized placement and chosen variant
};

struct Result {
    bool ok = false;
    int W = 0, H = 0;                     // actual bounding box
    int fixedW = 0;                       // width limit used by greedy/shelf
    vector<OutPlace> place;
};

static chrono::steady_clock::time_point START_TIME;
static const double TIME_LIMIT_SEC = 1.85;

static double elapsedSec() {
    return chrono::duration<double>(chrono::steady_clock::now() - START_TIME).count();
}

static pair<long long,long long> transformCell(long long x, long long y, int R, int F) {
    if (F) x = -x; // reflection across y-axis
    long long nx = x, ny = y;
    if (R == 0) {
        nx = x; ny = y;
    } else if (R == 1) { // clockwise 90: (x,y)->(y,-x)
        nx = y; ny = -x;
    } else if (R == 2) {
        nx = -x; ny = -y;
    } else {
        nx = -y; ny = x;
    }
    return {nx, ny};
}

static void generateVariants(const OrigPiece &op, Piece &p) {
    p.area = (int)op.cells.size();
    map<string, Variant> seen;

    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<pair<long long,long long>> raw;
            raw.reserve(op.cells.size());
            long long mnx = LLONG_MAX, mny = LLONG_MAX;
            for (auto &c : op.cells) {
                auto t = transformCell(c.first, c.second, R, F);
                raw.push_back(t);
                mnx = min(mnx, t.first);
                mny = min(mny, t.second);
            }

            vector<pair<int,int>> norm;
            norm.reserve(raw.size());
            int mx = 0, my = 0;
            for (auto &q : raw) {
                int x = (int)(q.first - mnx);
                int y = (int)(q.second - mny);
                norm.push_back({x,y});
                mx = max(mx, x);
                my = max(my, y);
            }
            sort(norm.begin(), norm.end());

            string key;
            for (auto &q : norm) {
                key += to_string(q.first);
                key += ",";
                key += to_string(q.second);
                key += ";";
            }
            if (seen.count(key)) continue;

            Variant v;
            v.cells = norm;
            v.w = mx + 1;
            v.h = my + 1;
            v.R = R;
            v.F = F;
            v.minx = mnx;
            v.miny = mny;
            v.rows.assign(v.h, {});
            for (auto &q : norm) v.rows[q.second].push_back(q.first);
            for (auto &row : v.rows) sort(row.begin(), row.end());
            seen[key] = v;
        }
    }

    p.vars.clear();
    for (auto &kv : seen) p.vars.push_back(kv.second);

    int bestBBox = INT_MAX, maxDim = 1;
    for (auto &v : p.vars) {
        bestBBox = min(bestBBox, v.w * v.h);
        maxDim = max(maxDim, max(v.w, v.h));
    }
    p.minBBoxArea = max(1, bestBBox);
    p.maxDim = maxDim;
    p.compactness = (double)p.area / (double)p.minBBoxArea;
}

static bool betterResult(const Result &a, const Result &b) {
    if (!a.ok) return false;
    if (!b.ok) return true;
    long long aa = 1LL * a.W * a.H;
    long long ba = 1LL * b.W * b.H;
    if (aa != ba) return aa < ba;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static Result makeHorizontalStrip(const vector<Piece> &pieces, const vector<int> &order) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.place.assign(n, {});
    int x = 0, H = 0;
    for (int id : order) {
        int bestVar = 0;
        for (int vi = 1; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &a = pieces[id].vars[vi];
            const Variant &b = pieces[id].vars[bestVar];
            if (a.h != b.h ? a.h < b.h : a.w < b.w) bestVar = vi;
        }
        const Variant &v = pieces[id].vars[bestVar];
        res.place[id] = {x, 0, bestVar};
        x += v.w;
        H = max(H, v.h);
    }
    res.W = max(1, x);
    res.H = max(1, H);
    res.fixedW = res.W;
    return res;
}

static Result shelfPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.fixedW = Wlimit;
    res.place.assign(n, {});

    int rowX = 0, rowY = 0, rowH = 0;
    int usedW = 0;

    for (int id : order) {
        int bestVar = -1;
        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &v = pieces[id].vars[vi];
            if (v.w > Wlimit) continue;
            if (bestVar < 0) bestVar = vi;
            else {
                const Variant &b = pieces[id].vars[bestVar];
                if (v.h != b.h ? v.h < b.h : v.w < b.w) bestVar = vi;
            }
        }
        if (bestVar < 0) {
            res.ok = false;
            return res;
        }
        const Variant &v = pieces[id].vars[bestVar];
        if (rowX > 0 && rowX + v.w > Wlimit) {
            rowY += rowH;
            rowX = 0;
            rowH = 0;
        }
        res.place[id] = {rowX, rowY, bestVar};
        rowX += v.w;
        rowH = max(rowH, v.h);
        usedW = max(usedW, rowX);
    }

    res.W = max(1, usedW);
    res.H = max(1, rowY + rowH);
    return res;
}

static inline bool canPlaceAt(const vector<string> &grid, const Variant &v, int x, int y) {
    int gh = (int)grid.size();
    for (int dy = 0; dy < v.h; ++dy) {
        int yy = y + dy;
        if (yy >= gh) continue;
        const string &row = grid[yy];
        for (int dx : v.rows[dy]) {
            if (row[x + dx]) return false;
        }
    }
    return true;
}

static Result greedyPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {
    int n = pieces.size();
    Result res;
    res.ok = true;
    res.fixedW = Wlimit;
    res.place.assign(n, {});

    vector<string> grid;
    int curH = 0, usedW = 0;

    for (int idx = 0; idx < n; ++idx) {
        int id = order[idx];

        int bestVar = -1, bestX = 0, bestY = 0;
        int bestNewH = INT_MAX;
        int bestUsedW = INT_MAX;
        int bestTie = INT_MAX;

        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {
            const Variant &v = pieces[id].vars[vi];
            if (v.w > Wlimit) continue;

            for (int y = 0; y <= curH; ++y) {
                int newHAtY = max(curH, y + v.h);
                if (newHAtY > bestNewH) {
                    if (y + v.h > bestNewH) break;
                }
                for (int x = 0; x <= Wlimit - v.w; ++x) {
                    int newH = max(curH, y + v.h);
                    if (newH > bestNewH) continue;
                    if (!canPlaceAt(grid, v, x, y)) continue;

                    int nUsedW = max(usedW, x + v.w);
                    int tie = y * 100000 + x + v.h * 17 + v.w;
                    bool take = false;
                    if (newH < bestNewH) take = true;
                    else if (newH == bestNewH && nUsedW < bestUsedW) take = true;
                    else if (newH == bestNewH && nUsedW == bestUsedW && tie < bestTie) take = true;

                    if (take) {
                        bestNewH = newH;
                        bestUsedW = nUsedW;
                        bestTie = tie;
                        bestVar = vi;
                        bestX = x;
                        bestY = y;
                    }
                }
            }
        }

        if (bestVar < 0) {
            res.ok = false;
            return res;
        }

        const Variant &v = pieces[id].vars[bestVar];
        if ((int)grid.size() < bestY + v.h) {
            grid.resize(bestY + v.h, string(Wlimit, '\0'));
        }
        for (auto &c : v.cells) {
            grid[bestY + c.second][bestX + c.first] = 1;
        }

        res.place[id] = {bestX, bestY, bestVar};
        curH = max(curH, bestY + v.h);
        usedW = max(usedW, bestX + v.w);

        // Very large instances should not get stuck in expensive exhaustive scans.
        if ((idx & 15) == 0 && elapsedSec() > TIME_LIMIT_SEC * 0.98) {
            // Still a valid partial algorithm only if completed; continue is unsafe.
            // Do nothing here; outer callers avoid starting late.
        }
    }

    res.W = max(1, usedW);
    res.H = max(1, curH);
    return res;
}

static vector<int> buildAdaptiveWidths(const vector<Piece> &pieces) {
    long long A = 0;
    int lower = 1;
    int maxDim = 1;
    double aspectSum = 0.0;

    for (const Piece &p : pieces) {
        A += p.area;
        int minW = INT_MAX;
        double bestAspect = 1.0;
        int bestBox = INT_MAX;
        for (const Variant &v : p.vars) {
            minW = min(minW, v.w);
            maxDim = max(maxDim, max(v.w, v.h));
            int box = v.w * v.h;
            if (box < bestBox) {
                bestBox = box;
                bestAspect = max((double)v.w / v.h, (double)v.h / v.w);
            }
        }
        lower = max(lower, minW);
        aspectSum += bestAspect;
    }

    int n = max(1, (int)pieces.size());
    double avgAspect = aspectSum / n;
    int sqrtA = max(1, (int)ceil(sqrt((double)max(1LL, A))));
    int upper = (int)min<long long>(A, max<long long>(lower, A));

    set<int> s;
    auto add = [&](long long w) {
        if (w < lower) w = lower;
        if (w > upper) w = upper;
        if (w >= 1) s.insert((int)w);
    };

    for (int w = lower; w <= min(upper, lower + 25); ++w) add(w);

    vector<double> ratios = {0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.45, 1.75, 2.10, 2.60, 3.20, 4.00};
    if (avgAspect > 1.8) {
        ratios.push_back(4.8);
        ratios.push_back(5.8);
    }
    if (avgAspect > 2.8) ratios.push_back(7.0);

    vector<int> deltas = {-8,-5,-3,-2,-1,0,1,2,3,5,8};
    for (double r : ratios) {
        int base = (int)llround(sqrt((double)A * r));
        for (int d : deltas) add(base + d);
        add((long long)llround(base * 0.85));
        add((long long)llround(base * 1.15));
        add((long long)llround(base * 1.35));
    }

    if (A <= 900) {
        for (int w = lower; w <= upper; ++w) add(w);
    } else {
        int lo = max(lower, sqrtA / 2);
        int hi = min<long long>(upper, 3LL * sqrtA + 3LL * maxDim + 80);
        int range = max(1, hi - lo + 1);
        int step = max(1, range / 260);
        for (int w = lo; w <= hi; w += step) add(w);
        for (int w = max(lower, sqrtA - 40); w <= min(upper, sqrtA + 80); ++w) add(w);
    }

    add(maxDim);
    add(sqrtA);
    add((long long)sqrtA + maxDim);
    add((long long)2 * sqrtA);
    add(upper);

    vector<int> out(s.begin(), s.end());
    return out;
}

static double resultScore(const Result &r, long long totalArea) {
    long long area = 1LL * r.W * r.H;
    double denom = (double)max(1LL, totalArea);
    return (double)area + (double)r.H / (denom + 1.0) + (double)r.W / ((denom + 1.0) * (denom + 1.0));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    START_TIME = chrono::steady_clock::now();

    int n;
    if (!(cin >> n)) return 0;

    vector<OrigPiece> orig(n);
    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        orig[i].cells.resize(k);
        for (int j = 0; j < k; ++j) {
            long long x, y;
            cin >> x >> y;
            orig[i].cells[j] = {x, y};
        }
    }

    vector<Piece> pieces(n);
    long long totalArea = 0;
    for (int i = 0; i < n; ++i) {
        generateVariants(orig[i], pieces[i]);
        totalArea += pieces[i].area;
    }

    vector<int> baseOrder(n);
    iota(baseOrder.begin(), baseOrder.end(), 0);

    vector<int> compactOrder = baseOrder;
    sort(compactOrder.begin(), compactOrder.end(), [&](int a, int b) {
        const Piece &pa = pieces[a], &pb = pieces[b];
        double sa = pa.area * 1000.0 + (1.0 - pa.compactness) * 130.0 + pa.maxDim * 7.0 + pa.minBBoxArea * 0.35;
        double sb = pb.area * 1000.0 + (1.0 - pb.compactness) * 130.0 + pb.maxDim * 7.0 + pb.minBBoxArea * 0.35;
        if (fabs(sa - sb) > 1e-9) return sa > sb;
        return a < b;
    });

    vector<int> bboxOrder = baseOrder;
    sort(bboxOrder.begin(), bboxOrder.end(), [&](int a, int b) {
        const Piece &pa = pieces[a], &pb = pieces[b];
        if (pa.minBBoxArea != pb.minBBoxArea) return pa.minBBoxArea > pb.minBBoxArea;
        if (pa.area != pb.area) return pa.area > pb.area;
        if (pa.compactness != pb.compactness) return pa.compactness < pb.compactness;
        return a < b;
    });

    vector<int> areaOrder = baseOrder;
    sort(areaOrder.begin(), areaOrder.end(), [&](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].maxDim != pieces[b].maxDim) return pieces[a].maxDim > pieces[b].maxDim;
        return a < b;
    });

    vector<vector<int>> orders;
    orders.push_back(compactOrder);
    orders.push_back(bboxOrder);
    orders.push_back(areaOrder);
    orders.push_back(baseOrder);

    mt19937 rng(1234567);
    for (int t = 0; t < 4 && n > 1; ++t) {
        vector<int> ord = compactOrder;
        stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
            double na = ((int)(rng() % 1000) - 500) / 1000.0;
            double nb = ((int)(rng() % 1000) - 500) / 1000.0;
            double sa = pieces[a].area * 50.0 + (1.0 - pieces[a].compactness) * 8.0 + pieces[a].maxDim + na;
            double sb = pieces[b].area * 50.0 + (1.0 - pieces[b].compactness) * 8.0 + pieces[b].maxDim + nb;
            return sa > sb;
        });
        orders.push_back(ord);
    }

    vector<int> widths = buildAdaptiveWidths(pieces);

    Result best = makeHorizontalStrip(pieces, compactOrder);
    vector<int> bestOrder = compactOrder;

    // Fast shelf pass over adaptive widths.
    for (const auto &ord : orders) {
        for (int W : widths) {
            Result r = shelfPack(pieces, ord, W);
            if (betterResult(r, best)) {
                best = r;
                bestOrder = ord;
            }
        }
    }

    // Bottom-left greedy pass.  Width candidates are adaptive, but limited by time.
    for (const auto &ord : orders) {
        for (int W : widths) {
            if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;
            Result r = greedyPack(pieces, ord, W);
            if (betterResult(r, best)) {
                best = r;
                bestOrder = ord;
            }
        }
        if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;
    }

    // Add local width neighborhood around current best actual and fixed widths.
    set<int> wset(widths.begin(), widths.end());
    int lower = 1;
    for (const Piece &p : pieces) {
        int mw = INT_MAX;
        for (const Variant &v : p.vars) mw = min(mw, v.w);
        lower = max(lower, mw);
    }
    int upper = max(lower, (int)totalArea);
    auto addW = [&](int w) {
        w = max(lower, min(upper, w));
        wset.insert(w);
    };
    for (int d = -25; d <= 35; ++d) {
        addW(best.W + d);
        addW(best.fixedW + d);
    }
    widths.assign(wset.begin(), wset.end());

    // Simulated-annealing-like local search over ordering swaps and width changes.
    if (n > 1) {
        vector<int> curOrder = bestOrder;
        int curW = max(lower, best.fixedW);
        Result cur = greedyPack(pieces, curOrder, curW);
        if (!cur.ok) cur = best;
        if (betterResult(cur, best)) {
            best = cur;
            bestOrder = curOrder;
        }

        double curScore = resultScore(cur, totalArea);
        double bestScore = resultScore(best, totalArea);
        int iter = 0;

        while (elapsedSec() < TIME_LIMIT_SEC * 0.97) {
            ++iter;
            vector<int> nextOrder = curOrder;
            int nextW = curW;

            int moveType = (int)(rng() % 100);
            if (moveType < 68) {
                int i = (int)(rng() % n);
                int j;
                if ((rng() % 100) < 55) {
                    int span = 1 + (int)(rng() % min(n, 8));
                    j = max(0, min(n - 1, i + ((rng() & 1) ? span : -span)));
                } else {
                    j = (int)(rng() % n);
                }
                if (i != j) swap(nextOrder[i], nextOrder[j]);
            } else if (moveType < 86) {
                int i = (int)(rng() % n);
                int len = 2 + (int)(rng() % min(n, 10));
                int j = min(n, i + len);
                if (i < j) reverse(nextOrder.begin() + i, nextOrder.begin() + j);
            } else {
                // Shape-guided perturbation: pull a bulky/less-compact random piece earlier.
                int i = (int)(rng() % n);
                int j = (int)(rng() % n);
                if (pieces[nextOrder[j]].area > pieces[nextOrder[i]].area ||
                    pieces[nextOrder[j]].compactness < pieces[nextOrder[i]].compactness) {
                    int val = nextOrder[j];
                    nextOrder.erase(nextOrder.begin() + j);
                    nextOrder.insert(nextOrder.begin() + i, val);
                }
            }

            if ((rng() % 100) < 30) {
                if ((rng() % 100) < 55 && !widths.empty()) {
                    nextW = widths[rng() % widths.size()];
                } else {
                    int delta = (int)(rng() % 17) - 8;
                    if (delta == 0) delta = 1;
                    nextW = max(lower, min(upper, curW + delta));
                }
            }

            Result nr = greedyPack(pieces, nextOrder, nextW);
            if (!nr.ok) continue;

            double ns = resultScore(nr, totalArea);
            double progress = min(1.0, elapsedSec() / (TIME_LIMIT_SEC * 0.97));
            double temp = max(0.0005, (double)max(1LL, totalArea) * 0.15 * (1.0 - progress) + 0.02);
            bool accept = false;
            if (ns <= curScore) accept = true;
            else {
                double prob = exp((curScore - ns) / temp);
                double u = (double)(rng() & 0xFFFFFF) / (double)0x1000000;
                if (u < prob) accept = true;
            }

            if (accept) {
                curOrder.swap(nextOrder);
                cur = nr;
                curW = nextW;
                curScore = ns;
            }

            if (betterResult(nr, best)) {
                best = nr;
                bestOrder = nextOrder;
                bestScore = ns;
                (void)bestScore;
                // Explore close widths of the new incumbent.
                addW(best.W);
                addW(best.fixedW);
            }
        }
    }

    // As a final safety net, ensure the chosen result is valid-looking.
    if (!best.ok || (int)best.place.size() != n) {
        best = makeHorizontalStrip(pieces, compactOrder);
    }

    cout << best.W << " " << best.H << "\n";
    for (int i = 0; i < n; ++i) {
        const OutPlace &pl = best.place[i];
        const Variant &v = pieces[i].vars[pl.var];
        long long X = (long long)pl.x - v.minx;
        long long Y = (long long)pl.y - v.miny;
        cout << X << " " << Y << " " << v.R << " " << v.F << "\n";
    }

    return 0;
}
```

New Ideas Introduced
This candidate implements a hybrid heuristic for Polyomino Packing. It enumerates all unique reflected and rotated orientations for each polyomino while preserving the transform parameters needed for output. It computes area, bounding-box, aspect, and compactness metrics, then uses those metrics to create shape-aware piece orders and adaptive width candidates instead of relying only on fixed-width trials.

The solver first guarantees validity with a horizontal strip placement, then evaluates shelf packings and bottom-left greedy grid packings across adaptive widths. The greedy placer scans feasible positions and orientations under a width limit, choosing placements that minimize resulting height and bounding width while explicitly checking cell overlap. After initial construction, it performs a time-limited simulated-annealing-style local search: neighbors swap or reorder pieces and sometimes change the width candidate, accepting worsening moves probabilistically to escape local minima. The best valid placement found is emitted, with board dimensions tightened to the actual occupied bounding box.

The guidance components are implemented in simplified practical form: adaptive width selection uses total area and aspect/compactness statistics, and simulated annealing is swap/reversal/width based rather than a full geometric perturbation optimizer.

Empirical Outcome
Verifier status: valid
FrontierCS score: 5.582855064285715
Reward: 5.582855064285715
Verifier message: accepted

Failure / Bottleneck Analysis
No verifier failure reported.

Next Guidance Delta
Continue from the submitted guidance: Introduce a hybrid approach combining adaptive width selection with simulated annealing for local optimization. Dynamically adjust width candidates based on polyomino aspect ratios and use a probabilistic swap-based local search to iteratively refine placements. Prioritize polyomino ordering by both area and "shape compactness" metrics to guide the search toward configurations with lower bounding box ratios. This addresses the current limitations of fixed-width heuristics and greedy placement, e
```

### Verification Artifacts

```json
{
  "code": "#include <bits/stdc++.h>\nusing namespace std;\n\nstruct OrigPiece {\n    vector<pair<long long,long long>> cells;\n};\n\nstruct Variant {\n    vector<pair<int,int>> cells;          // normalized occupied cells\n    vector<vector<int>> rows;             // row -> xs, for faster fit\n    int w = 0, h = 0;\n    int R = 0, F = 0;\n    long long minx = 0, miny = 0;         // raw transformed minimum before normalization\n};\n\nstruct Piece {\n    vector<Variant> vars;\n    int area = 0;\n    double compactness = 1.0;\n    int minBBoxArea = 1;\n    int maxDim = 1;\n};\n\nstruct OutPlace {\n    int x = 0, y = 0, var = 0;            // normalized placement and chosen variant\n};\n\nstruct Result {\n    bool ok = false;\n    int W = 0, H = 0;                     // actual bounding box\n    int fixedW = 0;                       // width limit used by greedy/shelf\n    vector<OutPlace> place;\n};\n\nstatic chrono::steady_clock::time_point START_TIME;\nstatic const double TIME_LIMIT_SEC = 1.85;\n\nstatic double elapsedSec() {\n    return chrono::duration<double>(chrono::steady_clock::now() - START_TIME).count();\n}\n\nstatic pair<long long,long long> transformCell(long long x, long long y, int R, int F) {\n    if (F) x = -x; // reflection across y-axis\n    long long nx = x, ny = y;\n    if (R == 0) {\n        nx = x; ny = y;\n    } else if (R == 1) { // clockwise 90: (x,y)->(y,-x)\n        nx = y; ny = -x;\n    } else if (R == 2) {\n        nx = -x; ny = -y;\n    } else {\n        nx = -y; ny = x;\n    }\n    return {nx, ny};\n}\n\nstatic void generateVariants(const OrigPiece &op, Piece &p) {\n    p.area = (int)op.cells.size();\n    map<string, Variant> seen;\n\n    for (int F = 0; F <= 1; ++F) {\n        for (int R = 0; R < 4; ++R) {\n            vector<pair<long long,long long>> raw;\n            raw.reserve(op.cells.size());\n            long long mnx = LLONG_MAX, mny = LLONG_MAX;\n            for (auto &c : op.cells) {\n                auto t = transformCell(c.first, c.second, R, F);\n                raw.push_back(t);\n                mnx = min(mnx, t.first);\n                mny = min(mny, t.second);\n            }\n\n            vector<pair<int,int>> norm;\n            norm.reserve(raw.size());\n            int mx = 0, my = 0;\n            for (auto &q : raw) {\n                int x = (int)(q.first - mnx);\n                int y = (int)(q.second - mny);\n                norm.push_back({x,y});\n                mx = max(mx, x);\n                my = max(my, y);\n            }\n            sort(norm.begin(), norm.end());\n\n            string key;\n            for (auto &q : norm) {\n                key += to_string(q.first);\n                key += \",\";\n                key += to_string(q.second);\n                key += \";\";\n            }\n            if (seen.count(key)) continue;\n\n            Variant v;\n            v.cells = norm;\n            v.w = mx + 1;\n            v.h = my + 1;\n            v.R = R;\n            v.F = F;\n            v.minx = mnx;\n            v.miny = mny;\n            v.rows.assign(v.h, {});\n            for (auto &q : norm) v.rows[q.second].push_back(q.first);\n            for (auto &row : v.rows) sort(row.begin(), row.end());\n            seen[key] = v;\n        }\n    }\n\n    p.vars.clear();\n    for (auto &kv : seen) p.vars.push_back(kv.second);\n\n    int bestBBox = INT_MAX, maxDim = 1;\n    for (auto &v : p.vars) {\n        bestBBox = min(bestBBox, v.w * v.h);\n        maxDim = max(maxDim, max(v.w, v.h));\n    }\n    p.minBBoxArea = max(1, bestBBox);\n    p.maxDim = maxDim;\n    p.compactness = (double)p.area / (double)p.minBBoxArea;\n}\n\nstatic bool betterResult(const Result &a, const Result &b) {\n    if (!a.ok) return false;\n    if (!b.ok) return true;\n    long long aa = 1LL * a.W * a.H;\n    long long ba = 1LL * b.W * b.H;\n    if (aa != ba) return aa < ba;\n    if (a.H != b.H) return a.H < b.H;\n    return a.W < b.W;\n}\n\nstatic Result makeHorizontalStrip(const vector<Piece> &pieces, const vector<int> &order) {\n    int n = pieces.size();\n    Result res;\n    res.ok = true;\n    res.place.assign(n, {});\n    int x = 0, H = 0;\n    for (int id : order) {\n        int bestVar = 0;\n        for (int vi = 1; vi < (int)pieces[id].vars.size(); ++vi) {\n            const Variant &a = pieces[id].vars[vi];\n            const Variant &b = pieces[id].vars[bestVar];\n            if (a.h != b.h ? a.h < b.h : a.w < b.w) bestVar = vi;\n        }\n        const Variant &v = pieces[id].vars[bestVar];\n        res.place[id] = {x, 0, bestVar};\n        x += v.w;\n        H = max(H, v.h);\n    }\n    res.W = max(1, x);\n    res.H = max(1, H);\n    res.fixedW = res.W;\n    return res;\n}\n\nstatic Result shelfPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {\n    int n = pieces.size();\n    Result res;\n    res.ok = true;\n    res.fixedW = Wlimit;\n    res.place.assign(n, {});\n\n    int rowX = 0, rowY = 0, rowH = 0;\n    int usedW = 0;\n\n    for (int id : order) {\n        int bestVar = -1;\n        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {\n            const Variant &v = pieces[id].vars[vi];\n            if (v.w > Wlimit) continue;\n            if (bestVar < 0) bestVar = vi;\n            else {\n                const Variant &b = pieces[id].vars[bestVar];\n                if (v.h != b.h ? v.h < b.h : v.w < b.w) bestVar = vi;\n            }\n        }\n        if (bestVar < 0) {\n            res.ok = false;\n            return res;\n        }\n        const Variant &v = pieces[id].vars[bestVar];\n        if (rowX > 0 && rowX + v.w > Wlimit) {\n            rowY += rowH;\n            rowX = 0;\n            rowH = 0;\n        }\n        res.place[id] = {rowX, rowY, bestVar};\n        rowX += v.w;\n        rowH = max(rowH, v.h);\n        usedW = max(usedW, rowX);\n    }\n\n    res.W = max(1, usedW);\n    res.H = max(1, rowY + rowH);\n    return res;\n}\n\nstatic inline bool canPlaceAt(const vector<string> &grid, const Variant &v, int x, int y) {\n    int gh = (int)grid.size();\n    for (int dy = 0; dy < v.h; ++dy) {\n        int yy = y + dy;\n        if (yy >= gh) continue;\n        const string &row = grid[yy];\n        for (int dx : v.rows[dy]) {\n            if (row[x + dx]) return false;\n        }\n    }\n    return true;\n}\n\nstatic Result greedyPack(const vector<Piece> &pieces, const vector<int> &order, int Wlimit) {\n    int n = pieces.size();\n    Result res;\n    res.ok = true;\n    res.fixedW = Wlimit;\n    res.place.assign(n, {});\n\n    vector<string> grid;\n    int curH = 0, usedW = 0;\n\n    for (int idx = 0; idx < n; ++idx) {\n        int id = order[idx];\n\n        int bestVar = -1, bestX = 0, bestY = 0;\n        int bestNewH = INT_MAX;\n        int bestUsedW = INT_MAX;\n        int bestTie = INT_MAX;\n\n        for (int vi = 0; vi < (int)pieces[id].vars.size(); ++vi) {\n            const Variant &v = pieces[id].vars[vi];\n            if (v.w > Wlimit) continue;\n\n            for (int y = 0; y <= curH; ++y) {\n                int newHAtY = max(curH, y + v.h);\n                if (newHAtY > bestNewH) {\n                    if (y + v.h > bestNewH) break;\n                }\n                for (int x = 0; x <= Wlimit - v.w; ++x) {\n                    int newH = max(curH, y + v.h);\n                    if (newH > bestNewH) continue;\n                    if (!canPlaceAt(grid, v, x, y)) continue;\n\n                    int nUsedW = max(usedW, x + v.w);\n                    int tie = y * 100000 + x + v.h * 17 + v.w;\n                    bool take = false;\n                    if (newH < bestNewH) take = true;\n                    else if (newH == bestNewH && nUsedW < bestUsedW) take = true;\n                    else if (newH == bestNewH && nUsedW == bestUsedW && tie < bestTie) take = true;\n\n                    if (take) {\n                        bestNewH = newH;\n                        bestUsedW = nUsedW;\n                        bestTie = tie;\n                        bestVar = vi;\n                        bestX = x;\n                        bestY = y;\n                    }\n                }\n            }\n        }\n\n        if (bestVar < 0) {\n            res.ok = false;\n            return res;\n        }\n\n        const Variant &v = pieces[id].vars[bestVar];\n        if ((int)grid.size() < bestY + v.h) {\n            grid.resize(bestY + v.h, string(Wlimit, '\\0'));\n        }\n        for (auto &c : v.cells) {\n            grid[bestY + c.second][bestX + c.first] = 1;\n        }\n\n        res.place[id] = {bestX, bestY, bestVar};\n        curH = max(curH, bestY + v.h);\n        usedW = max(usedW, bestX + v.w);\n\n        // Very large instances should not get stuck in expensive exhaustive scans.\n        if ((idx & 15) == 0 && elapsedSec() > TIME_LIMIT_SEC * 0.98) {\n            // Still a valid partial algorithm only if completed; continue is unsafe.\n            // Do nothing here; outer callers avoid starting late.\n        }\n    }\n\n    res.W = max(1, usedW);\n    res.H = max(1, curH);\n    return res;\n}\n\nstatic vector<int> buildAdaptiveWidths(const vector<Piece> &pieces) {\n    long long A = 0;\n    int lower = 1;\n    int maxDim = 1;\n    double aspectSum = 0.0;\n\n    for (const Piece &p : pieces) {\n        A += p.area;\n        int minW = INT_MAX;\n        double bestAspect = 1.0;\n        int bestBox = INT_MAX;\n        for (const Variant &v : p.vars) {\n            minW = min(minW, v.w);\n            maxDim = max(maxDim, max(v.w, v.h));\n            int box = v.w * v.h;\n            if (box < bestBox) {\n                bestBox = box;\n                bestAspect = max((double)v.w / v.h, (double)v.h / v.w);\n            }\n        }\n        lower = max(lower, minW);\n        aspectSum += bestAspect;\n    }\n\n    int n = max(1, (int)pieces.size());\n    double avgAspect = aspectSum / n;\n    int sqrtA = max(1, (int)ceil(sqrt((double)max(1LL, A))));\n    int upper = (int)min<long long>(A, max<long long>(lower, A));\n\n    set<int> s;\n    auto add = [&](long long w) {\n        if (w < lower) w = lower;\n        if (w > upper) w = upper;\n        if (w >= 1) s.insert((int)w);\n    };\n\n    for (int w = lower; w <= min(upper, lower + 25); ++w) add(w);\n\n    vector<double> ratios = {0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.45, 1.75, 2.10, 2.60, 3.20, 4.00};\n    if (avgAspect > 1.8) {\n        ratios.push_back(4.8);\n        ratios.push_back(5.8);\n    }\n    if (avgAspect > 2.8) ratios.push_back(7.0);\n\n    vector<int> deltas = {-8,-5,-3,-2,-1,0,1,2,3,5,8};\n    for (double r : ratios) {\n        int base = (int)llround(sqrt((double)A * r));\n        for (int d : deltas) add(base + d);\n        add((long long)llround(base * 0.85));\n        add((long long)llround(base * 1.15));\n        add((long long)llround(base * 1.35));\n    }\n\n    if (A <= 900) {\n        for (int w = lower; w <= upper; ++w) add(w);\n    } else {\n        int lo = max(lower, sqrtA / 2);\n        int hi = min<long long>(upper, 3LL * sqrtA + 3LL * maxDim + 80);\n        int range = max(1, hi - lo + 1);\n        int step = max(1, range / 260);\n        for (int w = lo; w <= hi; w += step) add(w);\n        for (int w = max(lower, sqrtA - 40); w <= min(upper, sqrtA + 80); ++w) add(w);\n    }\n\n    add(maxDim);\n    add(sqrtA);\n    add((long long)sqrtA + maxDim);\n    add((long long)2 * sqrtA);\n    add(upper);\n\n    vector<int> out(s.begin(), s.end());\n    return out;\n}\n\nstatic double resultScore(const Result &r, long long totalArea) {\n    long long area = 1LL * r.W * r.H;\n    double denom = (double)max(1LL, totalArea);\n    return (double)area + (double)r.H / (denom + 1.0) + (double)r.W / ((denom + 1.0) * (denom + 1.0));\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    START_TIME = chrono::steady_clock::now();\n\n    int n;\n    if (!(cin >> n)) return 0;\n\n    vector<OrigPiece> orig(n);\n    for (int i = 0; i < n; ++i) {\n        int k;\n        cin >> k;\n        orig[i].cells.resize(k);\n        for (int j = 0; j < k; ++j) {\n            long long x, y;\n            cin >> x >> y;\n            orig[i].cells[j] = {x, y};\n        }\n    }\n\n    vector<Piece> pieces(n);\n    long long totalArea = 0;\n    for (int i = 0; i < n; ++i) {\n        generateVariants(orig[i], pieces[i]);\n        totalArea += pieces[i].area;\n    }\n\n    vector<int> baseOrder(n);\n    iota(baseOrder.begin(), baseOrder.end(), 0);\n\n    vector<int> compactOrder = baseOrder;\n    sort(compactOrder.begin(), compactOrder.end(), [&](int a, int b) {\n        const Piece &pa = pieces[a], &pb = pieces[b];\n        double sa = pa.area * 1000.0 + (1.0 - pa.compactness) * 130.0 + pa.maxDim * 7.0 + pa.minBBoxArea * 0.35;\n        double sb = pb.area * 1000.0 + (1.0 - pb.compactness) * 130.0 + pb.maxDim * 7.0 + pb.minBBoxArea * 0.35;\n        if (fabs(sa - sb) > 1e-9) return sa > sb;\n        return a < b;\n    });\n\n    vector<int> bboxOrder = baseOrder;\n    sort(bboxOrder.begin(), bboxOrder.end(), [&](int a, int b) {\n        const Piece &pa = pieces[a], &pb = pieces[b];\n        if (pa.minBBoxArea != pb.minBBoxArea) return pa.minBBoxArea > pb.minBBoxArea;\n        if (pa.area != pb.area) return pa.area > pb.area;\n        if (pa.compactness != pb.compactness) return pa.compactness < pb.compactness;\n        return a < b;\n    });\n\n    vector<int> areaOrder = baseOrder;\n    sort(areaOrder.begin(), areaOrder.end(), [&](int a, int b) {\n        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;\n        if (pieces[a].maxDim != pieces[b].maxDim) return pieces[a].maxDim > pieces[b].maxDim;\n        return a < b;\n    });\n\n    vector<vector<int>> orders;\n    orders.push_back(compactOrder);\n    orders.push_back(bboxOrder);\n    orders.push_back(areaOrder);\n    orders.push_back(baseOrder);\n\n    mt19937 rng(1234567);\n    for (int t = 0; t < 4 && n > 1; ++t) {\n        vector<int> ord = compactOrder;\n        stable_sort(ord.begin(), ord.end(), [&](int a, int b) {\n            double na = ((int)(rng() % 1000) - 500) / 1000.0;\n            double nb = ((int)(rng() % 1000) - 500) / 1000.0;\n            double sa = pieces[a].area * 50.0 + (1.0 - pieces[a].compactness) * 8.0 + pieces[a].maxDim + na;\n            double sb = pieces[b].area * 50.0 + (1.0 - pieces[b].compactness) * 8.0 + pieces[b].maxDim + nb;\n            return sa > sb;\n        });\n        orders.push_back(ord);\n    }\n\n    vector<int> widths = buildAdaptiveWidths(pieces);\n\n    Result best = makeHorizontalStrip(pieces, compactOrder);\n    vector<int> bestOrder = compactOrder;\n\n    // Fast shelf pass over adaptive widths.\n    for (const auto &ord : orders) {\n        for (int W : widths) {\n            Result r = shelfPack(pieces, ord, W);\n            if (betterResult(r, best)) {\n                best = r;\n                bestOrder = ord;\n            }\n        }\n    }\n\n    // Bottom-left greedy pass.  Width candidates are adaptive, but limited by time.\n    for (const auto &ord : orders) {\n        for (int W : widths) {\n            if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;\n            Result r = greedyPack(pieces, ord, W);\n            if (betterResult(r, best)) {\n                best = r;\n                bestOrder = ord;\n            }\n        }\n        if (elapsedSec() > TIME_LIMIT_SEC * 0.62) break;\n    }\n\n    // Add local width neighborhood around current best actual and fixed widths.\n    set<int> wset(widths.begin(), widths.end());\n    int lower = 1;\n    for (const Piece &p : pieces) {\n        int mw = INT_MAX;\n        for (const Variant &v : p.vars) mw = min(mw, v.w);\n        lower = max(lower, mw);\n    }\n    int upper = max(lower, (int)totalArea);\n    auto addW = [&](int w) {\n        w = max(lower, min(upper, w));\n        wset.insert(w);\n    };\n    for (int d = -25; d <= 35; ++d) {\n        addW(best.W + d);\n        addW(best.fixedW + d);\n    }\n    widths.assign(wset.begin(), wset.end());\n\n    // Simulated-annealing-like local search over ordering swaps and width changes.\n    if (n > 1) {\n        vector<int> curOrder = bestOrder;\n        int curW = max(lower, best.fixedW);\n        Result cur = greedyPack(pieces, curOrder, curW);\n        if (!cur.ok) cur = best;\n        if (betterResult(cur, best)) {\n            best = cur;\n            bestOrder = curOrder;\n        }\n\n        double curScore = resultScore(cur, totalArea);\n        double bestScore = resultScore(best, totalArea);\n        int iter = 0;\n\n        while (elapsedSec() < TIME_LIMIT_SEC * 0.97) {\n            ++iter;\n            vector<int> nextOrder = curOrder;\n            int nextW = curW;\n\n            int moveType = (int)(rng() % 100);\n            if (moveType < 68) {\n                int i = (int)(rng() % n);\n                int j;\n                if ((rng() % 100) < 55) {\n                    int span = 1 + (int)(rng() % min(n, 8));\n                    j = max(0, min(n - 1, i + ((rng() & 1) ? span : -span)));\n                } else {\n                    j = (int)(rng() % n);\n                }\n                if (i != j) swap(nextOrder[i], nextOrder[j]);\n            } else if (moveType < 86) {\n                int i = (int)(rng() % n);\n                int len = 2 + (int)(rng() % min(n, 10));\n                int j = min(n, i + len);\n                if (i < j) reverse(nextOrder.begin() + i, nextOrder.begin() + j);\n            } else {\n                // Shape-guided perturbation: pull a bulky/less-compact random piece earlier.\n                int i = (int)(rng() % n);\n                int j = (int)(rng() % n);\n                if (pieces[nextOrder[j]].area > pieces[nextOrder[i]].area ||\n                    pieces[nextOrder[j]].compactness < pieces[nextOrder[i]].compactness) {\n                    int val = nextOrder[j];\n                    nextOrder.erase(nextOrder.begin() + j);\n                    nextOrder.insert(nextOrder.begin() + i, val);\n                }\n            }\n\n            if ((rng() % 100) < 30) {\n                if ((rng() % 100) < 55 && !widths.empty()) {\n                    nextW = widths[rng() % widths.size()];\n                } else {\n                    int delta = (int)(rng() % 17) - 8;\n                    if (delta == 0) delta = 1;\n                    nextW = max(lower, min(upper, curW + delta));\n                }\n            }\n\n            Result nr = greedyPack(pieces, nextOrder, nextW);\n            if (!nr.ok) continue;\n\n            double ns = resultScore(nr, totalArea);\n            double progress = min(1.0, elapsedSec() / (TIME_LIMIT_SEC * 0.97));\n            double temp = max(0.0005, (double)max(1LL, totalArea) * 0.15 * (1.0 - progress) + 0.02);\n            bool accept = false;\n            if (ns <= curScore) accept = true;\n            else {\n                double prob = exp((curScore - ns) / temp);\n                double u = (double)(rng() & 0xFFFFFF) / (double)0x1000000;\n                if (u < prob) accept = true;\n            }\n\n            if (accept) {\n                curOrder.swap(nextOrder);\n                cur = nr;\n                curW = nextW;\n                curScore = ns;\n            }\n\n            if (betterResult(nr, best)) {\n                best = nr;\n                bestOrder = nextOrder;\n                bestScore = ns;\n                (void)bestScore;\n                // Explore close widths of the new incumbent.\n                addW(best.W);\n                addW(best.fixedW);\n            }\n        }\n    }\n\n    // As a final safety net, ensure the chosen result is valid-looking.\n    if (!best.ok || (int)best.place.size() != n) {\n        best = makeHorizontalStrip(pieces, compactOrder);\n    }\n\n    cout << best.W << \" \" << best.H << \"\\n\";\n    for (int i = 0; i < n; ++i) {\n        const OutPlace &pl = best.place[i];\n        const Variant &v = pieces[i].vars[pl.var];\n        long long X = (long long)pl.x - v.minx;\n        long long Y = (long long)pl.y - v.miny;\n        cout << X << \" \" << Y << \" \" << v.R << \" \" << v.F << \"\\n\";\n    }\n\n    return 0;\n}",
  "frontiercs_base_dir": "/opt/Frontier-CS",
  "frontiercs_config": {
    "memory_mb": 256,
    "n_cases": 70,
    "time_limit_s": 2,
    "total_timeout_s": 340
  },
  "problem_id": "0"
}
```

## Entry `55787a31-762a-4767-8e43-6d346631cfc0`

- Timestep: `1`
- Status: `valid`
- Reward: `70.71252564714288`
- Raw score: `70.71252564714288`
- Guidance format ok: `True`
- Execution model: `openai/gpt-5.5`
- Execution finish reason: `None`

### Guidance Prompt System

```text
You are the Guidance Model, acting as a strategic navigator for an open-ended scientific discovery process.

Your primary objective is to provide **evolutionary guidance**. Do not write final code or focus on low-level implementation details. Instead, your task is to propose high-level directional shifts, conceptual mutations, and novel pathways to explore the search space.

Focus on how the current ideas can *evolve* to escape local optima and discover fundamentally new mechanisms.
```

### Guidance Prompt User

```text
<problem>
You are solving FrontierCS algorithmic problem 0: Polyomino Packing.

Write a self-contained C++17 program that reads one instance from stdin and writes
one placement to stdout.

Input:
- The first line contains n, the number of polyominoes.
- For each polyomino i, one line contains k_i, followed by k_i lines of integer
  cell coordinates x y in the polyomino local frame.
- Each polyomino has 1 to 10 cells, is 4-connected, and coordinates may be negative.

Output:
- First line: two integers W H for the chosen board.
- Then exactly n lines, one per input polyomino, each with X Y R F.
- X Y is the integer translation.
- R is one of 0, 1, 2, 3 and means clockwise rotation by R * 90 degrees.
- F is 0 or 1. If F=1, reflect across the y-axis before applying the rotation.

Validity:
- Apply transforms in this order: optional reflection, rotation, translation.
- Every transformed cell must satisfy 0 <= x < W and 0 <= y < H.
- No two transformed cells may overlap.
- Invalid output, crashes, or timeouts receive zero score.

Objective:
- Maximize the FrontierCS score by minimizing packing area W * H across the benchmark cases.
- Ties favor smaller H, then smaller W.
- Solutions are compiled with g++ -std=c++17 -O2 and evaluated by FrontierCS/go-judge.

</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.

Verifier status: valid
FrontierCS score: 74.55953630142857
Reward: 74.55953630142857
Verifier message: accepted
</selected_summary>

<local_failures>
No local failure summaries yet.
</local_failures>

# Objective
Your task is to provide the next **evolutionary guidance** to beat the current visible FrontierCS score target (74.55953630142857). Higher FrontierCS score is better.

# Evolutionary Guidelines
1. Analyze the search history.
Use `<selected_summary>` and `<local_failures>` to identify what has already been tried, what worked, what failed, and what bottleneck the next attempt should address.

2. Use internal reasoning, but only expose guidance.
Think through the current search state before answering. Do not create a custom reasoning XML block. Only the final guidance should be exposed to the downstream execution model.

3. Stay at the algorithmic-strategy level.
Propose high-level algorithmic directions, search mechanisms, objective changes, relaxation ideas, or hybrid optimization strategies. Do not write code, pseudocode, implementation details, or parameter schedules.

4. Produce exactly the required XML structure.
Return exactly one `<guidance>` block and no other custom XML blocks, commentary, code, or markdown.

Think step by step internally. Do not include the internal reasoning in the final XML block. Provide your response exactly in the following format:

<guidance>
Provide the final evolutionary guidance for the next execution attempt. Describe the main algorithmic direction to explore and why it is promising. Keep the guidance conceptual and actionable.
</guidance>

```

### Guidance Answer

```text
Introduce a hybrid approach combining greedy placement with a dynamic local search phase. After initial placement, apply a simulated annealing-inspired perturbation strategy to iteratively reposition pieces, prioritizing moves that reduce the total area while maintaining validity. This allows escaping local optima by exploring non-intuitive configurations, particularly for complex polyomino shapes. Pair this with an adaptive width selection mechanism that uses geometric heuristics to predict optimal dimensions based on polyomino aspect ratios and bounding box distributions.
```

### Raw Guidance Answer

```text
<think>
Okay, let's see. The current algorithm is a heuristic approach that uses shelf-based packing and greedy placement. It's getting a score of around 74.56, but the goal is to beat that. So, what can be improved?

First, the existing method tries different widths and uses a greedy approach, placing larger pieces first. But maybe there's a better way to handle the ordering of pieces. Like, maybe the current ordering isn't optimal. What if we consider more sophisticated ordering strategies, like based on some heuristic that considers both area and shape? For example, using a measure that combines the area with the aspect ratio or the bounding box dimensions. That could help in arranging pieces more efficiently.

Another thought: the current method uses a simple shelf-based approach. But what if we introduce a more dynamic way of managing the packing space? Like, instead of fixed shelves, use a bin-packing approach where each piece is placed in the best possible spot considering the remaining space. This could involve checking multiple positions and orientations for each piece, which might lead to better utilization of the space.

Also, the algorithm might be missing some advanced packing techniques. For example, simulated annealing or beam search could explore a wider range of possibilities by allowing temporary worse placements to escape local optima. These methods are more computationally intensive but might lead to better overall packing efficiency.

What about the reflection and rotation options? The current approach considers all possible transformations, but maybe there's a way to prioritize certain transformations based on the shape of the polyomino. For instance, some shapes might fit better in certain orientations, so pre-selecting those could save time and improve the packing.

Another angle is the selection of the board dimensions. The current method tries several width limits, but perhaps a more intelligent way to determine the optimal width could help. Maybe using a binary search approach on the possible widths, combined with some heuristic to estimate the minimum required height for each width. This could reduce the number of width options to check and speed up the process.

Also, post-processing compaction might be beneficial. After placing all pieces, there might be gaps that can be filled by adjusting the positions of some pieces. Implementing a compaction phase that shifts pieces to fill empty spaces could reduce the overall area.

Wait, the current algorithm is heuristic and might not explore all possibilities. Introducing a hybrid approach that combines the existing greedy method with a more advanced optimization technique, like genetic algorithms or local search, might yield better results. These methods can explore a larger solution space and might find more optimal packings.

Moreover, the selection of initial widths could be improved. Instead of trying a fixed set of widths, maybe using a more adaptive approach based on the sizes of the polyominoes. For example, calculating a lower bound on the width based on the largest piece's bounding box and then expanding from there.

Another idea is to consider the aspect ratio of the board. The current scoring prefers smaller H first, then W. Maybe optimizing for a more square-like board (balanced W and H) could sometimes yield a better overall area, even if it's not the absolute minimum. This might require adjusting the objective function slightly to consider both dimensions more equally.

So, putting this all together, the next step should focus on enhancing the piece ordering strategy and incorporating a more dynamic packing approach, possibly with a combination of greedy placement and advanced optimization techniques like simulated annealing or beam search. This would allow the algorithm to explore a broader range of possibilities and potentially find more efficient packings, leading to a higher FrontierCS score.
</think>

<guidance>
Introduce a hybrid approach combining greedy placement with a dynamic local search phase. After initial placement, apply a simulated annealing-inspired perturbation strategy to iteratively reposition pieces, prioritizing moves that reduce the total area while maintaining validity. This allows escaping local optima by exploring non-intuitive configurations, particularly for complex polyomino shapes. Pair this with an adaptive width selection mechanism that uses geometric heuristics to predict optimal dimensions based on polyomino aspect ratios and bounding box distributions.
</guidance>
```

### Raw Guidance Answer With Specials

```text
<think>
Okay, let's see. The current algorithm is a heuristic approach that uses shelf-based packing and greedy placement. It's getting a score of around 74.56, but the goal is to beat that. So, what can be improved?

First, the existing method tries different widths and uses a greedy approach, placing larger pieces first. But maybe there's a better way to handle the ordering of pieces. Like, maybe the current ordering isn't optimal. What if we consider more sophisticated ordering strategies, like based on some heuristic that considers both area and shape? For example, using a measure that combines the area with the aspect ratio or the bounding box dimensions. That could help in arranging pieces more efficiently.

Another thought: the current method uses a simple shelf-based approach. But what if we introduce a more dynamic way of managing the packing space? Like, instead of fixed shelves, use a bin-packing approach where each piece is placed in the best possible spot considering the remaining space. This could involve checking multiple positions and orientations for each piece, which might lead to better utilization of the space.

Also, the algorithm might be missing some advanced packing techniques. For example, simulated annealing or beam search could explore a wider range of possibilities by allowing temporary worse placements to escape local optima. These methods are more computationally intensive but might lead to better overall packing efficiency.

What about the reflection and rotation options? The current approach considers all possible transformations, but maybe there's a way to prioritize certain transformations based on the shape of the polyomino. For instance, some shapes might fit better in certain orientations, so pre-selecting those could save time and improve the packing.

Another angle is the selection of the board dimensions. The current method tries several width limits, but perhaps a more intelligent way to determine the optimal width could help. Maybe using a binary search approach on the possible widths, combined with some heuristic to estimate the minimum required height for each width. This could reduce the number of width options to check and speed up the process.

Also, post-processing compaction might be beneficial. After placing all pieces, there might be gaps that can be filled by adjusting the positions of some pieces. Implementing a compaction phase that shifts pieces to fill empty spaces could reduce the overall area.

Wait, the current algorithm is heuristic and might not explore all possibilities. Introducing a hybrid approach that combines the existing greedy method with a more advanced optimization technique, like genetic algorithms or local search, might yield better results. These methods can explore a larger solution space and might find more optimal packings.

Moreover, the selection of initial widths could be improved. Instead of trying a fixed set of widths, maybe using a more adaptive approach based on the sizes of the polyominoes. For example, calculating a lower bound on the width based on the largest piece's bounding box and then expanding from there.

Another idea is to consider the aspect ratio of the board. The current scoring prefers smaller H first, then W. Maybe optimizing for a more square-like board (balanced W and H) could sometimes yield a better overall area, even if it's not the absolute minimum. This might require adjusting the objective function slightly to consider both dimensions more equally.

So, putting this all together, the next step should focus on enhancing the piece ordering strategy and incorporating a more dynamic packing approach, possibly with a combination of greedy placement and advanced optimization techniques like simulated annealing or beam search. This would allow the algorithm to explore a broader range of possibilities and potentially find more efficient packings, leading to a higher FrontierCS score.
</think>

<guidance>
Introduce a hybrid approach combining greedy placement with a dynamic local search phase. After initial placement, apply a simulated annealing-inspired perturbation strategy to iteratively reposition pieces, prioritizing moves that reduce the total area while maintaining validity. This allows escaping local optima by exploring non-intuitive configurations, particularly for complex polyomino shapes. Pair this with an adaptive width selection mechanism that uses geometric heuristics to predict optimal dimensions based on polyomino aspect ratios and bounding box distributions.
</guidance><|im_end|>
```

### Execution Prompt System

```text
You are the execution model. Turn guidance into one concrete runnable C++17 candidate. Output execution thinking first, then the code block, then the summary.
```

### Execution Prompt User

```text
<problem>
You are solving FrontierCS algorithmic problem 0: Polyomino Packing.

Write a self-contained C++17 program that reads one instance from stdin and writes
one placement to stdout.

Input:
- The first line contains n, the number of polyominoes.
- For each polyomino i, one line contains k_i, followed by k_i lines of integer
  cell coordinates x y in the polyomino local frame.
- Each polyomino has 1 to 10 cells, is 4-connected, and coordinates may be negative.

Output:
- First line: two integers W H for the chosen board.
- Then exactly n lines, one per input polyomino, each with X Y R F.
- X Y is the integer translation.
- R is one of 0, 1, 2, 3 and means clockwise rotation by R * 90 degrees.
- F is 0 or 1. If F=1, reflect across the y-axis before applying the rotation.

Validity:
- Apply transforms in this order: optional reflection, rotation, translation.
- Every transformed cell must satisfy 0 <= x < W and 0 <= y < H.
- No two transformed cells may overlap.
- Invalid output, crashes, or timeouts receive zero score.

Objective:
- Maximize the FrontierCS score by minimizing packing area W * H across the benchmark cases.
- Ties favor smaller H, then smaller W.
- Solutions are compiled with g++ -std=c++17 -O2 and evaluated by FrontierCS/go-judge.

</problem>

The next sections describe the current search state for this problem.

<selected_summary>
This candidate enumerates each polyomino’s unique rotations and reflected rotations, preserving enough information to output the required original transform parameters. It first evaluates a simple shelf-based packing over several width limits to guarantee a valid placement quickly. For smaller instances, it then applies a greedy bottom-left grid placement on selected widths, placing larger pieces first and choosing the position/orientation that minimizes the resulting height.

The algorithm is intentionally heuristic rather than a full optimizer. It prioritizes robustness and validity while still attempting basic area reduction. Future guidance could improve width selection, piece ordering, local repacking, hole filling, simulated annealing or beam search over placements, and post-processing compaction.

Verifier status: valid
FrontierCS score: 74.55953630142857
Reward: 74.55953630142857
Verifier message: accepted
</selected_summary>

<guidance>
Introduce a hybrid approach combining greedy placement with a dynamic local search phase. After initial placement, apply a simulated annealing-inspired perturbation strategy to iteratively reposition pieces, prioritizing moves that reduce the total area while maintaining validity. This allows escaping local optima by exploring non-intuitive configurations, particularly for complex polyomino shapes. Pair this with an adaptive width selection mechanism that uses geometric heuristics to predict optimal dimensions based on polyomino aspect ratios and bounding box distributions.
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Score direction: max.
Implement one concrete solution that follows the guidance while satisfying the problem specification.
The <solution> block must contain one complete C++17 program in a ```cpp fenced block. It must read the Polyomino Packing instance from stdin and write the placement to stdout.

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.
You must output all three XML blocks exactly as shown below.
The <execution_thinking>...</execution_thinking> block is mandatory and must use angle brackets.
The <solution> block is mandatory and must contain a fenced ```cpp code block.
The <summary>...</summary> block is mandatory and must be closed.
Do not output only execution_thinking, only a summary, or plain natural language.
Do not omit angle brackets from XML tags.

Required output format:

<execution_thinking>
A short explanation of how the guidance was converted into the submitted algorithm.
Do not include code.
</execution_thinking>

<solution>
```cpp
// complete executable solution required by the problem
```
</solution>

<summary>
A concise natural-language summary of the candidate.

This summary must describe the implemented algorithm, the guidance-driven change from the prior idea, and the main search/refinement/optimization mechanisms used. If a suggested guidance component was not actually implemented, explicitly state that it was simplified or omitted. Mention implementation details only when they are conceptually important, such as placement ordering, orientation normalization, feasibility checks, local search, restart strategy, board-size selection, or constraint handling.

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.

</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.

```

### Execution Answer

```text
<execution_thinking>
I converted the guidance into a concrete heuristic packer that still guarantees validity. The program generates all unique reflected/rotated orientations, builds adaptive candidate board widths from total area and piece bounding-box aspect ratios, and creates initial solutions using deterministic and randomized bottom-left greedy placement. It then adds a simulated-annealing-inspired local search that removes and reinserts boundary-biased pieces inside the current board, accepting improving moves and occasional perturbations while preserving non-overlap. Finally, it attempts fixed-rectangle repacking into smaller-area boards before outputting the best verified placement.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;

struct Cell {
    int x, y;
    bool operator<(const Cell& o) const {
        if (x != o.x) return x < o.x;
        return y < o.y;
    }
    bool operator==(const Cell& o) const { return x == o.x && y == o.y; }
};

struct Orient {
    int R = 0, F = 0;
    int minx = 0, miny = 0;
    int w = 0, h = 0;
    vector<Cell> cells; // normalized cells
};

struct Piece {
    vector<Cell> orig;
    vector<Orient> ori;
    int area = 0;
    int minW = 0, minH = 0, minBoxArea = 0;
};

struct Placement {
    int oi = 0;
    int x = 0, y = 0; // position of normalized orientation
};

struct Result {
    int W = 0, H = 0;
    vector<Placement> p;
    bool ok = false;
};

static vector<Piece> pieces;
static int n;
static long long totalArea = 0;
static mt19937 rng;

static inline long long objectiveWH(int W, int H) {
    return 1000000000LL * (long long)W * (long long)H + 1000000LL * H + W;
}

static bool betterResult(const Result& a, const Result& b) {
    if (!a.ok) return false;
    if (!b.ok) return true;
    long long aa = 1LL * a.W * a.H, bb = 1LL * b.W * b.H;
    if (aa != bb) return aa < bb;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static pair<Cell, Cell> transformCell(Cell c, int F, int R) {
    int x = c.x, y = c.y;
    if (F) x = -x; // reflect across y-axis first
    int nx = x, ny = y;
    if (R == 0) { nx = x; ny = y; }
    else if (R == 1) { nx = y; ny = -x; }      // clockwise 90
    else if (R == 2) { nx = -x; ny = -y; }
    else { nx = -y; ny = x; }
    return {{nx, ny}, {0, 0}};
}

static void generateOrientations(Piece& pc) {
    map<vector<Cell>, Orient> seen;
    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<Cell> raw;
            raw.reserve(pc.orig.size());
            int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
            for (auto c : pc.orig) {
                auto t = transformCell(c, F, R).first;
                raw.push_back(t);
                minx = min(minx, t.x);
                miny = min(miny, t.y);
                maxx = max(maxx, t.x);
                maxy = max(maxy, t.y);
            }
            vector<Cell> norm;
            norm.reserve(raw.size());
            for (auto c : raw) norm.push_back({c.x - minx, c.y - miny});
            sort(norm.begin(), norm.end());
            norm.erase(unique(norm.begin(), norm.end()), norm.end());
            Orient o;
            o.R = R; o.F = F; o.minx = minx; o.miny = miny;
            o.w = maxx - minx + 1;
            o.h = maxy - miny + 1;
            o.cells = norm;
            if (!seen.count(norm)) seen[norm] = o;
        }
    }
    pc.ori.clear();
    for (auto& kv : seen) pc.ori.push_back(kv.second);
    pc.minW = INT_MAX;
    pc.minH = INT_MAX;
    pc.minBoxArea = INT_MAX;
    for (auto& o : pc.ori) {
        pc.minW = min(pc.minW, o.w);
        pc.minH = min(pc.minH, o.h);
        pc.minBoxArea = min(pc.minBoxArea, o.w * o.h);
    }
}

static void computeUsed(Result& res) {
    int mx = 0, my = 0;
    for (int i = 0; i < n; ++i) {
        const auto& pl = res.p[i];
        const auto& o = pieces[i].ori[pl.oi];
        for (auto c : o.cells) {
            mx = max(mx, pl.x + c.x + 1);
            my = max(my, pl.y + c.y + 1);
        }
    }
    res.W = mx;
    res.H = my;
    res.ok = true;
}

static vector<int> baseOrder() {
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].minBoxArea != pieces[b].minBoxArea) return pieces[a].minBoxArea > pieces[b].minBoxArea;
        int da = pieces[a].minW + pieces[a].minH;
        int db = pieces[b].minW + pieces[b].minH;
        if (da != db) return da > db;
        return a < b;
    });
    return ord;
}

static vector<int> randomizedOrder(const vector<int>& base) {
    vector<pair<double,int>> v;
    v.reserve(n);
    uniform_real_distribution<double> U(0.0, 1.0);
    for (int id : base) {
        double score = pieces[id].area * 100.0 + pieces[id].minBoxArea * 3.0
                     + (pieces[id].minW + pieces[id].minH) + U(rng) * 80.0;
        v.push_back({-score, id});
    }
    sort(v.begin(), v.end());
    vector<int> ord;
    for (auto& e : v) ord.push_back(e.second);
    return ord;
}

struct Cand {
    long long key;
    int x, y, oi;
    int usedW, usedH;
};

static Result greedyPlace(int W, int Hlimit, const vector<int>& order, bool randomized) {
    Result res;
    res.ok = false;
    res.W = W;
    res.H = 0;
    res.p.assign(n, Placement());

    vector<vector<unsigned char>> occ;
    if (Hlimit > 0) occ.assign(Hlimit, vector<unsigned char>(W, 0));

    int curH = 0, curW = 0;
    const int K = randomized ? 8 : 1;

    auto canPlace = [&](const Orient& o, int px, int py) -> bool {
        if (px < 0 || py < 0 || px + o.w > W) return false;
        if (Hlimit > 0 && py + o.h > Hlimit) return false;
        for (auto c : o.cells) {
            int xx = px + c.x, yy = py + c.y;
            if (yy < (int)occ.size() && occ[yy][xx]) return false;
        }
        return true;
    };

    auto contactScore = [&](const Orient& o, int px, int py) -> int {
        int s = 0;
        static const int dx[4] = {1, -1, 0, 0};
        static const int dy[4] = {0, 0, 1, -1};
        for (auto c : o.cells) {
            int x = px + c.x, y = py + c.y;
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= W || ny < 0) s++;
                else if (Hlimit > 0 && ny >= Hlimit) s++;
                else if (ny < (int)occ.size() && occ[ny][nx]) s++;
            }
        }
        return s;
    };

    auto addTop = [&](vector<Cand>& top, const Cand& c) {
        if ((int)top.size() < K) {
            top.push_back(c);
        } else {
            int worst = 0;
            for (int i = 1; i < (int)top.size(); ++i)
                if (top[i].key > top[worst].key) worst = i;
            if (c.key < top[worst].key) top[worst] = c;
        }
    };

    for (int id : order) {
        vector<Cand> top;
        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {
            const auto& o = pieces[id].ori[oi];
            if (o.w > W) continue;
            if (Hlimit > 0 && o.h > Hlimit) continue;
            int maxY = (Hlimit > 0) ? (Hlimit - o.h) : curH;
            for (int y = 0; y <= maxY; ++y) {
                for (int x = 0; x + o.w <= W; ++x) {
                    if (!canPlace(o, x, y)) continue;
                    int nH = max(curH, y + o.h);
                    int nW = max(curW, x + o.w);
                    int cont = contactScore(o, x, y);
                    long long key;
                    if (Hlimit > 0) {
                        key = 1000000000000LL * nH + 1000000000LL * nW
                            + 10000LL * y + x - 2000LL * cont;
                    } else {
                        key = 1000000000000LL * nH + 1000000000LL * nW
                            + 10000LL * y + x - 2000LL * cont;
                    }
                    addTop(top, {key, x, y, oi, nW, nH});
                }
            }
        }
        if (top.empty()) return res;
        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });
        int pick = 0;
        if (randomized && top.size() > 1) {
            uniform_real_distribution<double> U(0.0, 1.0);
            int lim = min<int>(top.size(), 6);
            pick = min(lim - 1, (int)(U(rng) * U(rng) * lim));
        }
        Cand c = top[pick];
        const auto& o = pieces[id].ori[c.oi];
        if (Hlimit == 0 && c.y + o.h > (int)occ.size())
            occ.resize(c.y + o.h, vector<unsigned char>(W, 0));
        for (auto cc : o.cells) occ[c.y + cc.y][c.x + cc.x] = 1;
        res.p[id] = {c.oi, c.x, c.y};
        curH = max(curH, c.y + o.h);
        curW = max(curW, c.x + o.w);
    }
    res.W = curW;
    res.H = curH;
    res.ok = true;
    return res;
}

static bool verifyResult(const Result& res) {
    if (!res.ok || res.W <= 0 || res.H <= 0 || (int)res.p.size() != n) return false;
    long long area = 1LL * res.W * res.H;
    if (area <= 20000000LL) {
        vector<unsigned char> occ((size_t)area, 0);
        for (int i = 0; i < n; ++i) {
            if (res.p[i].oi < 0 || res.p[i].oi >= (int)pieces[i].ori.size()) return false;
            const auto& pl = res.p[i];
            const auto& o = pieces[i].ori[pl.oi];
            for (auto c : o.cells) {
                int x = pl.x + c.x, y = pl.y + c.y;
                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;
                size_t idx = (size_t)y * res.W + x;
                if (occ[idx]) return false;
                occ[idx] = 1;
            }
        }
    } else {
        unordered_set<long long> occ;
        occ.reserve((size_t)totalArea * 2 + 10);
        for (int i = 0; i < n; ++i) {
            const auto& pl = res.p[i];
            const auto& o = pieces[i].ori[pl.oi];
            for (auto c : o.cells) {
                int x = pl.x + c.x, y = pl.y + c.y;
                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;
                long long key = ((long long)y << 32) ^ (unsigned int)x;
                if (!occ.insert(key).second) return false;
            }
        }
    }
    return true;
}

static vector<int> candidateWidths() {
    int minFeasibleW = 1;
    long long sumW = 0, sumH = 0;
    for (auto& pc : pieces) {
        minFeasibleW = max(minFeasibleW, pc.minW);
        int best = INT_MAX, bw = pc.minW, bh = pc.minH;
        for (auto& o : pc.ori) {
            if (o.w * o.h < best) {
                best = o.w * o.h; bw = o.w; bh = o.h;
            }
        }
        sumW += bw;
        sumH += bh;
    }

    double aspect = (sumH > 0) ? (double)sumW / (double)sumH : 1.0;
    aspect = max(0.25, min(4.0, aspect));
    double sq = sqrt((double)max(1LL, totalArea));
    int predicted = max(minFeasibleW, (int)llround(sqrt((double)totalArea * aspect)));

    vector<int> ws;
    auto add = [&](int w) {
        if (w >= minFeasibleW && w <= max<long long>(min((long long)INT_MAX, totalArea), minFeasibleW))
            ws.push_back(w);
    };

    add(minFeasibleW);
    add((int)totalArea);
    vector<double> factors = {0.45,0.55,0.65,0.75,0.85,0.95,1.0,1.1,1.25,1.4,1.6,1.85,2.15,2.5,3.0};
    for (double f : factors) {
        add((int)ceil(sq * f));
        add((int)ceil(predicted * f));
    }
    for (int d = -12; d <= 12; ++d) add(predicted + d);
    for (auto& pc : pieces) {
        for (auto& o : pc.ori) {
            add(o.w);
            add(o.w + 1);
            add(o.w + 2);
        }
    }
    if (totalArea <= 350) {
        for (int w = minFeasibleW; w <= (int)totalArea; ++w) add(w);
    }
    sort(ws.begin(), ws.end());
    ws.erase(unique(ws.begin(), ws.end()), ws.end());

    sort(ws.begin(), ws.end(), [&](int a, int b) {
        int da = abs(a - predicted), db = abs(b - predicted);
        if (da != db) return da < db;
        return a < b;
    });
    if (ws.size() > 90) ws.resize(90);
    return ws;
}

static Result localSearch(Result start, double deadlineSeconds, chrono::steady_clock::time_point t0) {
    if (!start.ok) return start;
    int capW = start.W, capH = start.H;
    if (capW <= 0 || capH <= 0) return start;
    long long capArea = 1LL * capW * capH;
    if (capArea > 600000LL || n == 0) return start;

    vector<vector<unsigned char>> occ(capH, vector<unsigned char>(capW, 0));
    vector<int> rowCnt(capH, 0), colCnt(capW, 0);
    auto addPiece = [&](int id, const Placement& pl, int delta) {
        const auto& o = pieces[id].ori[pl.oi];
        for (auto c : o.cells) {
            int x = pl.x + c.x, y = pl.y + c.y;
            if (delta > 0) {
                occ[y][x] = 1; rowCnt[y]++; colCnt[x]++;
            } else {
                occ[y][x] = 0; rowCnt[y]--; colCnt[x]--;
            }
        }
    };
    for (int i = 0; i < n; ++i) addPiece(i, start.p[i], +1);

    auto usedW = [&]() {
        for (int x = capW - 1; x >= 0; --x) if (colCnt[x]) return x + 1;
        return 0;
    };
    auto usedH = [&]() {
        for (int y = capH - 1; y >= 0; --y) if (rowCnt[y]) return y + 1;
        return 0;
    };
    auto can = [&](const Orient& o, int px, int py) {
        if (px < 0 || py < 0 || px + o.w > capW || py + o.h > capH) return false;
        for (auto c : o.cells) if (occ[py + c.y][px + c.x]) return false;
        return true;
    };
    auto addTop = [&](vector<Cand>& top, const Cand& c) {
        const int K = 10;
        if ((int)top.size() < K) top.push_back(c);
        else {
            int worst = 0;
            for (int i = 1; i < (int)top.size(); ++i)
                if (top[i].key > top[worst].key) worst = i;
            if (c.key < top[worst].key) top[worst] = c;
        }
    };

    Result cur = start, best = start;
    int curW = usedW(), curH = usedH();
    cur.W = curW; cur.H = curH;
    if (betterResult(cur, best)) best = cur;

    uniform_real_distribution<double> U(0.0, 1.0);
    int iter = 0;
    while (true) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (elapsed >= deadlineSeconds) break;
        ++iter;

        int id;
        if (U(rng) < 0.70) {
            vector<int> boundary;
            boundary.reserve(n);
            for (int i = 0; i < n; ++i) {
                const auto& pl = cur.p[i];
                const auto& o = pieces[i].ori[pl.oi];
                if (pl.x + o.w >= curW || pl.y + o.h >= curH) boundary.push_back(i);
            }
            if (!boundary.empty()) id = boundary[rng() % boundary.size()];
            else id = rng() % n;
        } else id = rng() % n;

        Placement old = cur.p[id];
        addPiece(id, old, -1);
        int baseW = usedW(), baseH = usedH();

        vector<Cand> top;
        bool fullScan = (long long)capW * capH * (int)pieces[id].ori.size() <= 180000LL;

        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {
            const auto& o = pieces[id].ori[oi];
            if (o.w > capW || o.h > capH) continue;
            if (fullScan) {
                for (int y = 0; y + o.h <= capH; ++y) {
                    for (int x = 0; x + o.w <= capW; ++x) {
                        if (!can(o, x, y)) continue;
                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                        addTop(top, {key, x, y, oi, nw, nh});
                    }
                }
            } else {
                int sx = max(1, capW / 55);
                int sy = max(1, capH / 55);
                for (int y = 0; y + o.h <= capH; y += sy) {
                    for (int x = 0; x + o.w <= capW; x += sx) {
                        if (!can(o, x, y)) continue;
                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                        addTop(top, {key, x, y, oi, nw, nh});
                    }
                }
                for (int s = 0; s < 260; ++s) {
                    int x = (capW == o.w) ? 0 : (int)(rng() % (capW - o.w + 1));
                    int y = (capH == o.h) ? 0 : (int)(rng() % (capH - o.h + 1));
                    if (!can(o, x, y)) continue;
                    int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                    long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                    addTop(top, {key, x, y, oi, nw, nh});
                }
            }
        }

        if (top.empty()) {
            addPiece(id, old, +1);
            continue;
        }
        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });

        int pick = 0;
        double temp = max(0.05, 2.0 * (1.0 - min(1.0, elapsed / max(0.001, deadlineSeconds))));
        if (top.size() > 1 && U(rng) < 0.20 * temp) {
            int lim = min<int>(top.size(), 6);
            pick = rng() % lim;
        }
        Cand chosen = top[pick];

        long long oldArea = 1LL * curW * curH;
        long long newArea = 1LL * chosen.usedW * chosen.usedH;
        bool accept = false;
        if (objectiveWH(chosen.usedW, chosen.usedH) <= objectiveWH(curW, curH)) accept = true;
        else {
            double delta = (double)(newArea - oldArea);
            if (delta <= 0) accept = (U(rng) < 0.25);
            else accept = (U(rng) < exp(-delta / temp));
        }

        if (accept) {
            cur.p[id] = {chosen.oi, chosen.x, chosen.y};
            addPiece(id, cur.p[id], +1);
            curW = usedW(); curH = usedH();
            cur.W = curW; cur.H = curH; cur.ok = true;
            if (betterResult(cur, best)) best = cur;
        } else {
            addPiece(id, old, +1);
        }
    }
    return best;
}

static vector<pair<int,int>> fixedRectanglesFromBest(const Result& best) {
    vector<int> ws = candidateWidths();
    for (int d = -30; d <= 30; ++d) ws.push_back(best.W + d);
    long long bestArea = 1LL * best.W * best.H;
    double sq = sqrt((double)bestArea);
    vector<double> fs = {0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.3,1.5,1.75,2.0};
    for (double f : fs) ws.push_back((int)ceil(sq * f));

    int minFeasibleW = 1;
    for (auto& pc : pieces) minFeasibleW = max(minFeasibleW, pc.minW);
    sort(ws.begin(), ws.end());
    ws.erase(unique(ws.begin(), ws.end()), ws.end());

    vector<pair<int,int>> rects;
    for (int W : ws) {
        if (W < minFeasibleW || W <= 0) continue;
        int H = (int)((bestArea - 1) / W);
        if (H <= 0) continue;
        if (1LL * W * H >= bestArea) continue;
        rects.push_back({W, H});
    }
    sort(rects.begin(), rects.end(), [](auto a, auto b) {
        long long aa = 1LL * a.first * a.second, bb = 1LL * b.first * b.second;
        if (aa != bb) return aa < bb;
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
    if (rects.size() > 120) rects.resize(120);
    return rects;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    pieces.assign(n, Piece());
    uint64_t seed = 1469598103934665603ULL ^ (uint64_t)n;
    for (int i = 0; i < n; ++i) {
        int k; cin >> k;
        pieces[i].area = k;
        pieces[i].orig.resize(k);
        totalArea += k;
        for (int j = 0; j < k; ++j) {
            int x, y; cin >> x >> y;
            pieces[i].orig[j] = {x, y};
            seed ^= (uint64_t)(x + 1000003LL * y + 911382323LL * i + 97266353LL * j);
            seed *= 1099511628211ULL;
        }
        generateOrientations(pieces[i]);
    }
    rng.seed((uint32_t)(seed ^ (seed >> 32)));

    auto t0 = chrono::steady_clock::now();
    double timeBudget = 3.4;
    if (totalArea <= 250) timeBudget = 4.2;
    if (totalArea >= 6000) timeBudget = 2.4;

    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    };

    vector<int> widths = candidateWidths();
    vector<int> ord0 = baseOrder();

    Result best;
    // Guaranteed simple fallback: use a feasible adaptive width greedily.
    for (int W : widths) {
        if (elapsed() > timeBudget * 0.38) break;
        Result r = greedyPlace(W, 0, ord0, false);
        if (r.ok) {
            computeUsed(r);
            if (betterResult(r, best)) best = r;
        }
    }

    if (!best.ok) {
        int W = 1;
        for (auto& pc : pieces) W = max(W, pc.minW);
        best = greedyPlace(W, 0, ord0, false);
        computeUsed(best);
    }

    // Randomized greedy restarts on promising widths.
    vector<int> prom = widths;
    prom.push_back(best.W);
    for (int d = -8; d <= 8; ++d) prom.push_back(best.W + d);
    sort(prom.begin(), prom.end());
    prom.erase(unique(prom.begin(), prom.end()), prom.end());
    prom.erase(remove_if(prom.begin(), prom.end(), [](int x){ return x <= 0; }), prom.end());

    int restart = 0;
    while (elapsed() < timeBudget * 0.58 && !prom.empty()) {
        int W = prom[restart % prom.size()];
        vector<int> ord = randomizedOrder(ord0);
        Result r = greedyPlace(W, 0, ord, true);
        if (r.ok) {
            computeUsed(r);
            if (betterResult(r, best)) best = r;
        }
        restart++;
    }

    // Simulated-annealing-inspired local reinsertion search.
    Result ls = localSearch(best, timeBudget * 0.78, t0);
    if (betterResult(ls, best)) best = ls;

    // Fixed smaller-rectangle repacking attempts.
    int fixedAttempts = 0;
    while (elapsed() < timeBudget * 0.96) {
        auto rects = fixedRectanglesFromBest(best);
        if (rects.empty()) break;
        bool improved = false;
        for (auto [W, H] : rects) {
            if (elapsed() >= timeBudget * 0.96) break;
            vector<int> ord = (fixedAttempts % 4 == 0) ? ord0 : randomizedOrder(ord0);
            Result r = greedyPlace(W, H, ord, fixedAttempts % 4 != 0);
            fixedAttempts++;
            if (r.ok) {
                computeUsed(r);
                if (betterResult(r, best)) {
                    best = r;
                    Result ls2 = localSearch(best, min(timeBudget * 0.96, elapsed() + 0.25), t0);
                    if (betterResult(ls2, best)) best = ls2;
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    if (!verifyResult(best)) {
        int W = 1;
        for (auto& pc : pieces) W = max(W, pc.minW);
        best = greedyPlace(W, 0, ord0, false);
        computeUsed(best);
    }

    cout << best.W << ' ' << best.H << "\n";
    for (int i = 0; i < n; ++i) {
        const Placement& pl = best.p[i];
        const Orient& o = pieces[i].ori[pl.oi];
        int X = pl.x - o.minx;
        int Y = pl.y - o.miny;
        cout << X << ' ' << Y << ' ' << o.R << ' ' << o.F << "\n";
    }
    return 0;
}
```
</solution>

<summary>
This candidate implements a hybrid polyomino packing heuristic. It enumerates all unique rotations and reflected rotations for each input polyomino while retaining the transform parameters needed for valid output.

Compared with the prior greedy-focused idea, this version adds guidance-driven adaptive width selection and a dynamic local refinement phase. Candidate widths are predicted from total area, square-root scaling, and aggregate bounding-box aspect ratios, then explored with deterministic and randomized bottom-left greedy placement. Pieces are ordered mostly by size and bounding-box difficulty, with randomized restarts adding perturbation.

After an initial packing is found, the program runs a simulated-annealing-inspired local search. It repeatedly removes a piece, usually one touching the current top or right boundary, tries alternative orientations and positions inside the current board, and accepts improving moves plus occasional non-improving perturbations. This preserves validity at every accepted state and can reduce the used bounding rectangle. A final fixed-rectangle repacking phase tries to fit all pieces into smaller-area candidate boards using randomized greedy orders.

The local search is simplified rather than a full annealing optimizer: it uses single-piece remove-and-reinsert moves and a bounded time budget, not multi-piece neighborhood reconstruction. The solution verifies the final placement and falls back to a guaranteed greedy packing if needed.
</summary>
```

### Raw Model Summary

```text
This candidate implements a hybrid polyomino packing heuristic. It enumerates all unique rotations and reflected rotations for each input polyomino while retaining the transform parameters needed for valid output.

Compared with the prior greedy-focused idea, this version adds guidance-driven adaptive width selection and a dynamic local refinement phase. Candidate widths are predicted from total area, square-root scaling, and aggregate bounding-box aspect ratios, then explored with deterministic and randomized bottom-left greedy placement. Pieces are ordered mostly by size and bounding-box difficulty, with randomized restarts adding perturbation.

After an initial packing is found, the program runs a simulated-annealing-inspired local search. It repeatedly removes a piece, usually one touching the current top or right boundary, tries alternative orientations and positions inside the current board, and accepts improving moves plus occasional non-improving perturbations. This preserves validity at every accepted state and can reduce the used bounding rectangle. A final fixed-rectangle repacking phase tries to fit all pieces into smaller-area candidate boards using randomized greedy orders.

The local search is simplified rather than a full annealing optimizer: it uses single-piece remove-and-reinsert moves and a bounded time budget, not multi-piece neighborhood reconstruction. The solution verifies the final placement and falls back to a guaranteed greedy packing if needed.
```

### Canonical Library Summary

```text
Execution Interpretation
I converted the guidance into a concrete heuristic packer that still guarantees validity. The program generates all unique reflected/rotated orientations, builds adaptive candidate board widths from total area and piece bounding-box aspect ratios, and creates initial solutions using deterministic and randomized bottom-left greedy placement. It then adds a simulated-annealing-inspired local search that removes and reinserts boundary-biased pieces inside the current board, accepting improving moves and occasional perturbations while preserving non-overlap. Finally, it attempts fixed-rectangle repacking into smaller-area boards before outputting the best verified placement.

Implemented Algorithm
```cpp
#include <bits/stdc++.h>
using namespace std;

struct Cell {
    int x, y;
    bool operator<(const Cell& o) const {
        if (x != o.x) return x < o.x;
        return y < o.y;
    }
    bool operator==(const Cell& o) const { return x == o.x && y == o.y; }
};

struct Orient {
    int R = 0, F = 0;
    int minx = 0, miny = 0;
    int w = 0, h = 0;
    vector<Cell> cells; // normalized cells
};

struct Piece {
    vector<Cell> orig;
    vector<Orient> ori;
    int area = 0;
    int minW = 0, minH = 0, minBoxArea = 0;
};

struct Placement {
    int oi = 0;
    int x = 0, y = 0; // position of normalized orientation
};

struct Result {
    int W = 0, H = 0;
    vector<Placement> p;
    bool ok = false;
};

static vector<Piece> pieces;
static int n;
static long long totalArea = 0;
static mt19937 rng;

static inline long long objectiveWH(int W, int H) {
    return 1000000000LL * (long long)W * (long long)H + 1000000LL * H + W;
}

static bool betterResult(const Result& a, const Result& b) {
    if (!a.ok) return false;
    if (!b.ok) return true;
    long long aa = 1LL * a.W * a.H, bb = 1LL * b.W * b.H;
    if (aa != bb) return aa < bb;
    if (a.H != b.H) return a.H < b.H;
    return a.W < b.W;
}

static pair<Cell, Cell> transformCell(Cell c, int F, int R) {
    int x = c.x, y = c.y;
    if (F) x = -x; // reflect across y-axis first
    int nx = x, ny = y;
    if (R == 0) { nx = x; ny = y; }
    else if (R == 1) { nx = y; ny = -x; }      // clockwise 90
    else if (R == 2) { nx = -x; ny = -y; }
    else { nx = -y; ny = x; }
    return {{nx, ny}, {0, 0}};
}

static void generateOrientations(Piece& pc) {
    map<vector<Cell>, Orient> seen;
    for (int F = 0; F <= 1; ++F) {
        for (int R = 0; R < 4; ++R) {
            vector<Cell> raw;
            raw.reserve(pc.orig.size());
            int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
            for (auto c : pc.orig) {
                auto t = transformCell(c, F, R).first;
                raw.push_back(t);
                minx = min(minx, t.x);
                miny = min(miny, t.y);
                maxx = max(maxx, t.x);
                maxy = max(maxy, t.y);
            }
            vector<Cell> norm;
            norm.reserve(raw.size());
            for (auto c : raw) norm.push_back({c.x - minx, c.y - miny});
            sort(norm.begin(), norm.end());
            norm.erase(unique(norm.begin(), norm.end()), norm.end());
            Orient o;
            o.R = R; o.F = F; o.minx = minx; o.miny = miny;
            o.w = maxx - minx + 1;
            o.h = maxy - miny + 1;
            o.cells = norm;
            if (!seen.count(norm)) seen[norm] = o;
        }
    }
    pc.ori.clear();
    for (auto& kv : seen) pc.ori.push_back(kv.second);
    pc.minW = INT_MAX;
    pc.minH = INT_MAX;
    pc.minBoxArea = INT_MAX;
    for (auto& o : pc.ori) {
        pc.minW = min(pc.minW, o.w);
        pc.minH = min(pc.minH, o.h);
        pc.minBoxArea = min(pc.minBoxArea, o.w * o.h);
    }
}

static void computeUsed(Result& res) {
    int mx = 0, my = 0;
    for (int i = 0; i < n; ++i) {
        const auto& pl = res.p[i];
        const auto& o = pieces[i].ori[pl.oi];
        for (auto c : o.cells) {
            mx = max(mx, pl.x + c.x + 1);
            my = max(my, pl.y + c.y + 1);
        }
    }
    res.W = mx;
    res.H = my;
    res.ok = true;
}

static vector<int> baseOrder() {
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [](int a, int b) {
        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;
        if (pieces[a].minBoxArea != pieces[b].minBoxArea) return pieces[a].minBoxArea > pieces[b].minBoxArea;
        int da = pieces[a].minW + pieces[a].minH;
        int db = pieces[b].minW + pieces[b].minH;
        if (da != db) return da > db;
        return a < b;
    });
    return ord;
}

static vector<int> randomizedOrder(const vector<int>& base) {
    vector<pair<double,int>> v;
    v.reserve(n);
    uniform_real_distribution<double> U(0.0, 1.0);
    for (int id : base) {
        double score = pieces[id].area * 100.0 + pieces[id].minBoxArea * 3.0
                     + (pieces[id].minW + pieces[id].minH) + U(rng) * 80.0;
        v.push_back({-score, id});
    }
    sort(v.begin(), v.end());
    vector<int> ord;
    for (auto& e : v) ord.push_back(e.second);
    return ord;
}

struct Cand {
    long long key;
    int x, y, oi;
    int usedW, usedH;
};

static Result greedyPlace(int W, int Hlimit, const vector<int>& order, bool randomized) {
    Result res;
    res.ok = false;
    res.W = W;
    res.H = 0;
    res.p.assign(n, Placement());

    vector<vector<unsigned char>> occ;
    if (Hlimit > 0) occ.assign(Hlimit, vector<unsigned char>(W, 0));

    int curH = 0, curW = 0;
    const int K = randomized ? 8 : 1;

    auto canPlace = [&](const Orient& o, int px, int py) -> bool {
        if (px < 0 || py < 0 || px + o.w > W) return false;
        if (Hlimit > 0 && py + o.h > Hlimit) return false;
        for (auto c : o.cells) {
            int xx = px + c.x, yy = py + c.y;
            if (yy < (int)occ.size() && occ[yy][xx]) return false;
        }
        return true;
    };

    auto contactScore = [&](const Orient& o, int px, int py) -> int {
        int s = 0;
        static const int dx[4] = {1, -1, 0, 0};
        static const int dy[4] = {0, 0, 1, -1};
        for (auto c : o.cells) {
            int x = px + c.x, y = py + c.y;
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= W || ny < 0) s++;
                else if (Hlimit > 0 && ny >= Hlimit) s++;
                else if (ny < (int)occ.size() && occ[ny][nx]) s++;
            }
        }
        return s;
    };

    auto addTop = [&](vector<Cand>& top, const Cand& c) {
        if ((int)top.size() < K) {
            top.push_back(c);
        } else {
            int worst = 0;
            for (int i = 1; i < (int)top.size(); ++i)
                if (top[i].key > top[worst].key) worst = i;
            if (c.key < top[worst].key) top[worst] = c;
        }
    };

    for (int id : order) {
        vector<Cand> top;
        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {
            const auto& o = pieces[id].ori[oi];
            if (o.w > W) continue;
            if (Hlimit > 0 && o.h > Hlimit) continue;
            int maxY = (Hlimit > 0) ? (Hlimit - o.h) : curH;
            for (int y = 0; y <= maxY; ++y) {
                for (int x = 0; x + o.w <= W; ++x) {
                    if (!canPlace(o, x, y)) continue;
                    int nH = max(curH, y + o.h);
                    int nW = max(curW, x + o.w);
                    int cont = contactScore(o, x, y);
                    long long key;
                    if (Hlimit > 0) {
                        key = 1000000000000LL * nH + 1000000000LL * nW
                            + 10000LL * y + x - 2000LL * cont;
                    } else {
                        key = 1000000000000LL * nH + 1000000000LL * nW
                            + 10000LL * y + x - 2000LL * cont;
                    }
                    addTop(top, {key, x, y, oi, nW, nH});
                }
            }
        }
        if (top.empty()) return res;
        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });
        int pick = 0;
        if (randomized && top.size() > 1) {
            uniform_real_distribution<double> U(0.0, 1.0);
            int lim = min<int>(top.size(), 6);
            pick = min(lim - 1, (int)(U(rng) * U(rng) * lim));
        }
        Cand c = top[pick];
        const auto& o = pieces[id].ori[c.oi];
        if (Hlimit == 0 && c.y + o.h > (int)occ.size())
            occ.resize(c.y + o.h, vector<unsigned char>(W, 0));
        for (auto cc : o.cells) occ[c.y + cc.y][c.x + cc.x] = 1;
        res.p[id] = {c.oi, c.x, c.y};
        curH = max(curH, c.y + o.h);
        curW = max(curW, c.x + o.w);
    }
    res.W = curW;
    res.H = curH;
    res.ok = true;
    return res;
}

static bool verifyResult(const Result& res) {
    if (!res.ok || res.W <= 0 || res.H <= 0 || (int)res.p.size() != n) return false;
    long long area = 1LL * res.W * res.H;
    if (area <= 20000000LL) {
        vector<unsigned char> occ((size_t)area, 0);
        for (int i = 0; i < n; ++i) {
            if (res.p[i].oi < 0 || res.p[i].oi >= (int)pieces[i].ori.size()) return false;
            const auto& pl = res.p[i];
            const auto& o = pieces[i].ori[pl.oi];
            for (auto c : o.cells) {
                int x = pl.x + c.x, y = pl.y + c.y;
                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;
                size_t idx = (size_t)y * res.W + x;
                if (occ[idx]) return false;
                occ[idx] = 1;
            }
        }
    } else {
        unordered_set<long long> occ;
        occ.reserve((size_t)totalArea * 2 + 10);
        for (int i = 0; i < n; ++i) {
            const auto& pl = res.p[i];
            const auto& o = pieces[i].ori[pl.oi];
            for (auto c : o.cells) {
                int x = pl.x + c.x, y = pl.y + c.y;
                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;
                long long key = ((long long)y << 32) ^ (unsigned int)x;
                if (!occ.insert(key).second) return false;
            }
        }
    }
    return true;
}

static vector<int> candidateWidths() {
    int minFeasibleW = 1;
    long long sumW = 0, sumH = 0;
    for (auto& pc : pieces) {
        minFeasibleW = max(minFeasibleW, pc.minW);
        int best = INT_MAX, bw = pc.minW, bh = pc.minH;
        for (auto& o : pc.ori) {
            if (o.w * o.h < best) {
                best = o.w * o.h; bw = o.w; bh = o.h;
            }
        }
        sumW += bw;
        sumH += bh;
    }

    double aspect = (sumH > 0) ? (double)sumW / (double)sumH : 1.0;
    aspect = max(0.25, min(4.0, aspect));
    double sq = sqrt((double)max(1LL, totalArea));
    int predicted = max(minFeasibleW, (int)llround(sqrt((double)totalArea * aspect)));

    vector<int> ws;
    auto add = [&](int w) {
        if (w >= minFeasibleW && w <= max<long long>(min((long long)INT_MAX, totalArea), minFeasibleW))
            ws.push_back(w);
    };

    add(minFeasibleW);
    add((int)totalArea);
    vector<double> factors = {0.45,0.55,0.65,0.75,0.85,0.95,1.0,1.1,1.25,1.4,1.6,1.85,2.15,2.5,3.0};
    for (double f : factors) {
        add((int)ceil(sq * f));
        add((int)ceil(predicted * f));
    }
    for (int d = -12; d <= 12; ++d) add(predicted + d);
    for (auto& pc : pieces) {
        for (auto& o : pc.ori) {
            add(o.w);
            add(o.w + 1);
            add(o.w + 2);
        }
    }
    if (totalArea <= 350) {
        for (int w = minFeasibleW; w <= (int)totalArea; ++w) add(w);
    }
    sort(ws.begin(), ws.end());
    ws.erase(unique(ws.begin(), ws.end()), ws.end());

    sort(ws.begin(), ws.end(), [&](int a, int b) {
        int da = abs(a - predicted), db = abs(b - predicted);
        if (da != db) return da < db;
        return a < b;
    });
    if (ws.size() > 90) ws.resize(90);
    return ws;
}

static Result localSearch(Result start, double deadlineSeconds, chrono::steady_clock::time_point t0) {
    if (!start.ok) return start;
    int capW = start.W, capH = start.H;
    if (capW <= 0 || capH <= 0) return start;
    long long capArea = 1LL * capW * capH;
    if (capArea > 600000LL || n == 0) return start;

    vector<vector<unsigned char>> occ(capH, vector<unsigned char>(capW, 0));
    vector<int> rowCnt(capH, 0), colCnt(capW, 0);
    auto addPiece = [&](int id, const Placement& pl, int delta) {
        const auto& o = pieces[id].ori[pl.oi];
        for (auto c : o.cells) {
            int x = pl.x + c.x, y = pl.y + c.y;
            if (delta > 0) {
                occ[y][x] = 1; rowCnt[y]++; colCnt[x]++;
            } else {
                occ[y][x] = 0; rowCnt[y]--; colCnt[x]--;
            }
        }
    };
    for (int i = 0; i < n; ++i) addPiece(i, start.p[i], +1);

    auto usedW = [&]() {
        for (int x = capW - 1; x >= 0; --x) if (colCnt[x]) return x + 1;
        return 0;
    };
    auto usedH = [&]() {
        for (int y = capH - 1; y >= 0; --y) if (rowCnt[y]) return y + 1;
        return 0;
    };
    auto can = [&](const Orient& o, int px, int py) {
        if (px < 0 || py < 0 || px + o.w > capW || py + o.h > capH) return false;
        for (auto c : o.cells) if (occ[py + c.y][px + c.x]) return false;
        return true;
    };
    auto addTop = [&](vector<Cand>& top, const Cand& c) {
        const int K = 10;
        if ((int)top.size() < K) top.push_back(c);
        else {
            int worst = 0;
            for (int i = 1; i < (int)top.size(); ++i)
                if (top[i].key > top[worst].key) worst = i;
            if (c.key < top[worst].key) top[worst] = c;
        }
    };

    Result cur = start, best = start;
    int curW = usedW(), curH = usedH();
    cur.W = curW; cur.H = curH;
    if (betterResult(cur, best)) best = cur;

    uniform_real_distribution<double> U(0.0, 1.0);
    int iter = 0;
    while (true) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (elapsed >= deadlineSeconds) break;
        ++iter;

        int id;
        if (U(rng) < 0.70) {
            vector<int> boundary;
            boundary.reserve(n);
            for (int i = 0; i < n; ++i) {
                const auto& pl = cur.p[i];
                const auto& o = pieces[i].ori[pl.oi];
                if (pl.x + o.w >= curW || pl.y + o.h >= curH) boundary.push_back(i);
            }
            if (!boundary.empty()) id = boundary[rng() % boundary.size()];
            else id = rng() % n;
        } else id = rng() % n;

        Placement old = cur.p[id];
        addPiece(id, old, -1);
        int baseW = usedW(), baseH = usedH();

        vector<Cand> top;
        bool fullScan = (long long)capW * capH * (int)pieces[id].ori.size() <= 180000LL;

        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {
            const auto& o = pieces[id].ori[oi];
            if (o.w > capW || o.h > capH) continue;
            if (fullScan) {
                for (int y = 0; y + o.h <= capH; ++y) {
                    for (int x = 0; x + o.w <= capW; ++x) {
                        if (!can(o, x, y)) continue;
                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                        addTop(top, {key, x, y, oi, nw, nh});
                    }
                }
            } else {
                int sx = max(1, capW / 55);
                int sy = max(1, capH / 55);
                for (int y = 0; y + o.h <= capH; y += sy) {
                    for (int x = 0; x + o.w <= capW; x += sx) {
                        if (!can(o, x, y)) continue;
                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                        addTop(top, {key, x, y, oi, nw, nh});
                    }
                }
                for (int s = 0; s < 260; ++s) {
                    int x = (capW == o.w) ? 0 : (int)(rng() % (capW - o.w + 1));
                    int y = (capH == o.h) ? 0 : (int)(rng() % (capH - o.h + 1));
                    if (!can(o, x, y)) continue;
                    int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);
                    long long key = objectiveWH(nw, nh) + 1000LL * y + x;
                    addTop(top, {key, x, y, oi, nw, nh});
                }
            }
        }

        if (top.empty()) {
            addPiece(id, old, +1);
            continue;
        }
        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });

        int pick = 0;
        double temp = max(0.05, 2.0 * (1.0 - min(1.0, elapsed / max(0.001, deadlineSeconds))));
        if (top.size() > 1 && U(rng) < 0.20 * temp) {
            int lim = min<int>(top.size(), 6);
            pick = rng() % lim;
        }
        Cand chosen = top[pick];

        long long oldArea = 1LL * curW * curH;
        long long newArea = 1LL * chosen.usedW * chosen.usedH;
        bool accept = false;
        if (objectiveWH(chosen.usedW, chosen.usedH) <= objectiveWH(curW, curH)) accept = true;
        else {
            double delta = (double)(newArea - oldArea);
            if (delta <= 0) accept = (U(rng) < 0.25);
            else accept = (U(rng) < exp(-delta / temp));
        }

        if (accept) {
            cur.p[id] = {chosen.oi, chosen.x, chosen.y};
            addPiece(id, cur.p[id], +1);
            curW = usedW(); curH = usedH();
            cur.W = curW; cur.H = curH; cur.ok = true;
            if (betterResult(cur, best)) best = cur;
        } else {
            addPiece(id, old, +1);
        }
    }
    return best;
}

static vector<pair<int,int>> fixedRectanglesFromBest(const Result& best) {
    vector<int> ws = candidateWidths();
    for (int d = -30; d <= 30; ++d) ws.push_back(best.W + d);
    long long bestArea = 1LL * best.W * best.H;
    double sq = sqrt((double)bestArea);
    vector<double> fs = {0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.3,1.5,1.75,2.0};
    for (double f : fs) ws.push_back((int)ceil(sq * f));

    int minFeasibleW = 1;
    for (auto& pc : pieces) minFeasibleW = max(minFeasibleW, pc.minW);
    sort(ws.begin(), ws.end());
    ws.erase(unique(ws.begin(), ws.end()), ws.end());

    vector<pair<int,int>> rects;
    for (int W : ws) {
        if (W < minFeasibleW || W <= 0) continue;
        int H = (int)((bestArea - 1) / W);
        if (H <= 0) continue;
        if (1LL * W * H >= bestArea) continue;
        rects.push_back({W, H});
    }
    sort(rects.begin(), rects.end(), [](auto a, auto b) {
        long long aa = 1LL * a.first * a.second, bb = 1LL * b.first * b.second;
        if (aa != bb) return aa < bb;
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
    if (rects.size() > 120) rects.resize(120);
    return rects;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    pieces.assign(n, Piece());
    uint64_t seed = 1469598103934665603ULL ^ (uint64_t)n;
    for (int i = 0; i < n; ++i) {
        int k; cin >> k;
        pieces[i].area = k;
        pieces[i].orig.resize(k);
        totalArea += k;
        for (int j = 0; j < k; ++j) {
            int x, y; cin >> x >> y;
            pieces[i].orig[j] = {x, y};
            seed ^= (uint64_t)(x + 1000003LL * y + 911382323LL * i + 97266353LL * j);
            seed *= 1099511628211ULL;
        }
        generateOrientations(pieces[i]);
    }
    rng.seed((uint32_t)(seed ^ (seed >> 32)));

    auto t0 = chrono::steady_clock::now();
    double timeBudget = 3.4;
    if (totalArea <= 250) timeBudget = 4.2;
    if (totalArea >= 6000) timeBudget = 2.4;

    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    };

    vector<int> widths = candidateWidths();
    vector<int> ord0 = baseOrder();

    Result best;
    // Guaranteed simple fallback: use a feasible adaptive width greedily.
    for (int W : widths) {
        if (elapsed() > timeBudget * 0.38) break;
        Result r = greedyPlace(W, 0, ord0, false);
        if (r.ok) {
            computeUsed(r);
            if (betterResult(r, best)) best = r;
        }
    }

    if (!best.ok) {
        int W = 1;
        for (auto& pc : pieces) W = max(W, pc.minW);
        best = greedyPlace(W, 0, ord0, false);
        computeUsed(best);
    }

    // Randomized greedy restarts on promising widths.
    vector<int> prom = widths;
    prom.push_back(best.W);
    for (int d = -8; d <= 8; ++d) prom.push_back(best.W + d);
    sort(prom.begin(), prom.end());
    prom.erase(unique(prom.begin(), prom.end()), prom.end());
    prom.erase(remove_if(prom.begin(), prom.end(), [](int x){ return x <= 0; }), prom.end());

    int restart = 0;
    while (elapsed() < timeBudget * 0.58 && !prom.empty()) {
        int W = prom[restart % prom.size()];
        vector<int> ord = randomizedOrder(ord0);
        Result r = greedyPlace(W, 0, ord, true);
        if (r.ok) {
            computeUsed(r);
            if (betterResult(r, best)) best = r;
        }
        restart++;
    }

    // Simulated-annealing-inspired local reinsertion search.
    Result ls = localSearch(best, timeBudget * 0.78, t0);
    if (betterResult(ls, best)) best = ls;

    // Fixed smaller-rectangle repacking attempts.
    int fixedAttempts = 0;
    while (elapsed() < timeBudget * 0.96) {
        auto rects = fixedRectanglesFromBest(best);
        if (rects.empty()) break;
        bool improved = false;
        for (auto [W, H] : rects) {
            if (elapsed() >= timeBudget * 0.96) break;
            vector<int> ord = (fixedAttempts % 4 == 0) ? ord0 : randomizedOrder(ord0);
            Result r = greedyPlace(W, H, ord, fixedAttempts % 4 != 0);
            fixedAttempts++;
            if (r.ok) {
                computeUsed(r);
                if (betterResult(r, best)) {
                    best = r;
                    Result ls2 = localSearch(best, min(timeBudget * 0.96, elapsed() + 0.25), t0);
                    if (betterResult(ls2, best)) best = ls2;
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    if (!verifyResult(best)) {
        int W = 1;
        for (auto& pc : pieces) W = max(W, pc.minW);
        best = greedyPlace(W, 0, ord0, false);
        computeUsed(best);
    }

    cout << best.W << ' ' << best.H << "\n";
    for (int i = 0; i < n; ++i) {
        const Placement& pl = best.p[i];
        const Orient& o = pieces[i].ori[pl.oi];
        int X = pl.x - o.minx;
        int Y = pl.y - o.miny;
        cout << X << ' ' << Y << ' ' << o.R << ' ' << o.F << "\n";
    }
    return 0;
}
```

New Ideas Introduced
This candidate implements a hybrid polyomino packing heuristic. It enumerates all unique rotations and reflected rotations for each input polyomino while retaining the transform parameters needed for valid output.

Compared with the prior greedy-focused idea, this version adds guidance-driven adaptive width selection and a dynamic local refinement phase. Candidate widths are predicted from total area, square-root scaling, and aggregate bounding-box aspect ratios, then explored with deterministic and randomized bottom-left greedy placement. Pieces are ordered mostly by size and bounding-box difficulty, with randomized restarts adding perturbation.

After an initial packing is found, the program runs a simulated-annealing-inspired local search. It repeatedly removes a piece, usually one touching the current top or right boundary, tries alternative orientations and positions inside the current board, and accepts improving moves plus occasional non-improving perturbations. This preserves validity at every accepted state and can reduce the used bounding rectangle. A final fixed-rectangle repacking phase tries to fit all pieces into smaller-area candidate boards using randomized greedy orders.

The local search is simplified rather than a full annealing optimizer: it uses single-piece remove-and-reinsert moves and a bounded time budget, not multi-piece neighborhood reconstruction. The solution verifies the final placement and falls back to a guaranteed greedy packing if needed.

Empirical Outcome
Verifier status: valid
FrontierCS score: 70.71252564714288
Reward: 70.71252564714288
Verifier message: accepted

Failure / Bottleneck Analysis
No verifier failure reported.

Next Guidance Delta
Continue from the submitted guidance: Introduce a hybrid approach combining greedy placement with a dynamic local search phase. After initial placement, apply a simulated annealing-inspired perturbation strategy to iteratively reposition pieces, prioritizing moves that reduce the total area while maintaining validity. This allows escaping local optima by exploring non-intuitive configurations, particularly for complex polyomino shapes. Pair this with an adaptive width selection mechanism that uses geometric heuristics to predict opt
```

### Verification Artifacts

```json
{
  "code": "#include <bits/stdc++.h>\nusing namespace std;\n\nstruct Cell {\n    int x, y;\n    bool operator<(const Cell& o) const {\n        if (x != o.x) return x < o.x;\n        return y < o.y;\n    }\n    bool operator==(const Cell& o) const { return x == o.x && y == o.y; }\n};\n\nstruct Orient {\n    int R = 0, F = 0;\n    int minx = 0, miny = 0;\n    int w = 0, h = 0;\n    vector<Cell> cells; // normalized cells\n};\n\nstruct Piece {\n    vector<Cell> orig;\n    vector<Orient> ori;\n    int area = 0;\n    int minW = 0, minH = 0, minBoxArea = 0;\n};\n\nstruct Placement {\n    int oi = 0;\n    int x = 0, y = 0; // position of normalized orientation\n};\n\nstruct Result {\n    int W = 0, H = 0;\n    vector<Placement> p;\n    bool ok = false;\n};\n\nstatic vector<Piece> pieces;\nstatic int n;\nstatic long long totalArea = 0;\nstatic mt19937 rng;\n\nstatic inline long long objectiveWH(int W, int H) {\n    return 1000000000LL * (long long)W * (long long)H + 1000000LL * H + W;\n}\n\nstatic bool betterResult(const Result& a, const Result& b) {\n    if (!a.ok) return false;\n    if (!b.ok) return true;\n    long long aa = 1LL * a.W * a.H, bb = 1LL * b.W * b.H;\n    if (aa != bb) return aa < bb;\n    if (a.H != b.H) return a.H < b.H;\n    return a.W < b.W;\n}\n\nstatic pair<Cell, Cell> transformCell(Cell c, int F, int R) {\n    int x = c.x, y = c.y;\n    if (F) x = -x; // reflect across y-axis first\n    int nx = x, ny = y;\n    if (R == 0) { nx = x; ny = y; }\n    else if (R == 1) { nx = y; ny = -x; }      // clockwise 90\n    else if (R == 2) { nx = -x; ny = -y; }\n    else { nx = -y; ny = x; }\n    return {{nx, ny}, {0, 0}};\n}\n\nstatic void generateOrientations(Piece& pc) {\n    map<vector<Cell>, Orient> seen;\n    for (int F = 0; F <= 1; ++F) {\n        for (int R = 0; R < 4; ++R) {\n            vector<Cell> raw;\n            raw.reserve(pc.orig.size());\n            int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;\n            for (auto c : pc.orig) {\n                auto t = transformCell(c, F, R).first;\n                raw.push_back(t);\n                minx = min(minx, t.x);\n                miny = min(miny, t.y);\n                maxx = max(maxx, t.x);\n                maxy = max(maxy, t.y);\n            }\n            vector<Cell> norm;\n            norm.reserve(raw.size());\n            for (auto c : raw) norm.push_back({c.x - minx, c.y - miny});\n            sort(norm.begin(), norm.end());\n            norm.erase(unique(norm.begin(), norm.end()), norm.end());\n            Orient o;\n            o.R = R; o.F = F; o.minx = minx; o.miny = miny;\n            o.w = maxx - minx + 1;\n            o.h = maxy - miny + 1;\n            o.cells = norm;\n            if (!seen.count(norm)) seen[norm] = o;\n        }\n    }\n    pc.ori.clear();\n    for (auto& kv : seen) pc.ori.push_back(kv.second);\n    pc.minW = INT_MAX;\n    pc.minH = INT_MAX;\n    pc.minBoxArea = INT_MAX;\n    for (auto& o : pc.ori) {\n        pc.minW = min(pc.minW, o.w);\n        pc.minH = min(pc.minH, o.h);\n        pc.minBoxArea = min(pc.minBoxArea, o.w * o.h);\n    }\n}\n\nstatic void computeUsed(Result& res) {\n    int mx = 0, my = 0;\n    for (int i = 0; i < n; ++i) {\n        const auto& pl = res.p[i];\n        const auto& o = pieces[i].ori[pl.oi];\n        for (auto c : o.cells) {\n            mx = max(mx, pl.x + c.x + 1);\n            my = max(my, pl.y + c.y + 1);\n        }\n    }\n    res.W = mx;\n    res.H = my;\n    res.ok = true;\n}\n\nstatic vector<int> baseOrder() {\n    vector<int> ord(n);\n    iota(ord.begin(), ord.end(), 0);\n    sort(ord.begin(), ord.end(), [](int a, int b) {\n        if (pieces[a].area != pieces[b].area) return pieces[a].area > pieces[b].area;\n        if (pieces[a].minBoxArea != pieces[b].minBoxArea) return pieces[a].minBoxArea > pieces[b].minBoxArea;\n        int da = pieces[a].minW + pieces[a].minH;\n        int db = pieces[b].minW + pieces[b].minH;\n        if (da != db) return da > db;\n        return a < b;\n    });\n    return ord;\n}\n\nstatic vector<int> randomizedOrder(const vector<int>& base) {\n    vector<pair<double,int>> v;\n    v.reserve(n);\n    uniform_real_distribution<double> U(0.0, 1.0);\n    for (int id : base) {\n        double score = pieces[id].area * 100.0 + pieces[id].minBoxArea * 3.0\n                     + (pieces[id].minW + pieces[id].minH) + U(rng) * 80.0;\n        v.push_back({-score, id});\n    }\n    sort(v.begin(), v.end());\n    vector<int> ord;\n    for (auto& e : v) ord.push_back(e.second);\n    return ord;\n}\n\nstruct Cand {\n    long long key;\n    int x, y, oi;\n    int usedW, usedH;\n};\n\nstatic Result greedyPlace(int W, int Hlimit, const vector<int>& order, bool randomized) {\n    Result res;\n    res.ok = false;\n    res.W = W;\n    res.H = 0;\n    res.p.assign(n, Placement());\n\n    vector<vector<unsigned char>> occ;\n    if (Hlimit > 0) occ.assign(Hlimit, vector<unsigned char>(W, 0));\n\n    int curH = 0, curW = 0;\n    const int K = randomized ? 8 : 1;\n\n    auto canPlace = [&](const Orient& o, int px, int py) -> bool {\n        if (px < 0 || py < 0 || px + o.w > W) return false;\n        if (Hlimit > 0 && py + o.h > Hlimit) return false;\n        for (auto c : o.cells) {\n            int xx = px + c.x, yy = py + c.y;\n            if (yy < (int)occ.size() && occ[yy][xx]) return false;\n        }\n        return true;\n    };\n\n    auto contactScore = [&](const Orient& o, int px, int py) -> int {\n        int s = 0;\n        static const int dx[4] = {1, -1, 0, 0};\n        static const int dy[4] = {0, 0, 1, -1};\n        for (auto c : o.cells) {\n            int x = px + c.x, y = py + c.y;\n            for (int d = 0; d < 4; ++d) {\n                int nx = x + dx[d], ny = y + dy[d];\n                if (nx < 0 || nx >= W || ny < 0) s++;\n                else if (Hlimit > 0 && ny >= Hlimit) s++;\n                else if (ny < (int)occ.size() && occ[ny][nx]) s++;\n            }\n        }\n        return s;\n    };\n\n    auto addTop = [&](vector<Cand>& top, const Cand& c) {\n        if ((int)top.size() < K) {\n            top.push_back(c);\n        } else {\n            int worst = 0;\n            for (int i = 1; i < (int)top.size(); ++i)\n                if (top[i].key > top[worst].key) worst = i;\n            if (c.key < top[worst].key) top[worst] = c;\n        }\n    };\n\n    for (int id : order) {\n        vector<Cand> top;\n        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {\n            const auto& o = pieces[id].ori[oi];\n            if (o.w > W) continue;\n            if (Hlimit > 0 && o.h > Hlimit) continue;\n            int maxY = (Hlimit > 0) ? (Hlimit - o.h) : curH;\n            for (int y = 0; y <= maxY; ++y) {\n                for (int x = 0; x + o.w <= W; ++x) {\n                    if (!canPlace(o, x, y)) continue;\n                    int nH = max(curH, y + o.h);\n                    int nW = max(curW, x + o.w);\n                    int cont = contactScore(o, x, y);\n                    long long key;\n                    if (Hlimit > 0) {\n                        key = 1000000000000LL * nH + 1000000000LL * nW\n                            + 10000LL * y + x - 2000LL * cont;\n                    } else {\n                        key = 1000000000000LL * nH + 1000000000LL * nW\n                            + 10000LL * y + x - 2000LL * cont;\n                    }\n                    addTop(top, {key, x, y, oi, nW, nH});\n                }\n            }\n        }\n        if (top.empty()) return res;\n        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });\n        int pick = 0;\n        if (randomized && top.size() > 1) {\n            uniform_real_distribution<double> U(0.0, 1.0);\n            int lim = min<int>(top.size(), 6);\n            pick = min(lim - 1, (int)(U(rng) * U(rng) * lim));\n        }\n        Cand c = top[pick];\n        const auto& o = pieces[id].ori[c.oi];\n        if (Hlimit == 0 && c.y + o.h > (int)occ.size())\n            occ.resize(c.y + o.h, vector<unsigned char>(W, 0));\n        for (auto cc : o.cells) occ[c.y + cc.y][c.x + cc.x] = 1;\n        res.p[id] = {c.oi, c.x, c.y};\n        curH = max(curH, c.y + o.h);\n        curW = max(curW, c.x + o.w);\n    }\n    res.W = curW;\n    res.H = curH;\n    res.ok = true;\n    return res;\n}\n\nstatic bool verifyResult(const Result& res) {\n    if (!res.ok || res.W <= 0 || res.H <= 0 || (int)res.p.size() != n) return false;\n    long long area = 1LL * res.W * res.H;\n    if (area <= 20000000LL) {\n        vector<unsigned char> occ((size_t)area, 0);\n        for (int i = 0; i < n; ++i) {\n            if (res.p[i].oi < 0 || res.p[i].oi >= (int)pieces[i].ori.size()) return false;\n            const auto& pl = res.p[i];\n            const auto& o = pieces[i].ori[pl.oi];\n            for (auto c : o.cells) {\n                int x = pl.x + c.x, y = pl.y + c.y;\n                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;\n                size_t idx = (size_t)y * res.W + x;\n                if (occ[idx]) return false;\n                occ[idx] = 1;\n            }\n        }\n    } else {\n        unordered_set<long long> occ;\n        occ.reserve((size_t)totalArea * 2 + 10);\n        for (int i = 0; i < n; ++i) {\n            const auto& pl = res.p[i];\n            const auto& o = pieces[i].ori[pl.oi];\n            for (auto c : o.cells) {\n                int x = pl.x + c.x, y = pl.y + c.y;\n                if (x < 0 || y < 0 || x >= res.W || y >= res.H) return false;\n                long long key = ((long long)y << 32) ^ (unsigned int)x;\n                if (!occ.insert(key).second) return false;\n            }\n        }\n    }\n    return true;\n}\n\nstatic vector<int> candidateWidths() {\n    int minFeasibleW = 1;\n    long long sumW = 0, sumH = 0;\n    for (auto& pc : pieces) {\n        minFeasibleW = max(minFeasibleW, pc.minW);\n        int best = INT_MAX, bw = pc.minW, bh = pc.minH;\n        for (auto& o : pc.ori) {\n            if (o.w * o.h < best) {\n                best = o.w * o.h; bw = o.w; bh = o.h;\n            }\n        }\n        sumW += bw;\n        sumH += bh;\n    }\n\n    double aspect = (sumH > 0) ? (double)sumW / (double)sumH : 1.0;\n    aspect = max(0.25, min(4.0, aspect));\n    double sq = sqrt((double)max(1LL, totalArea));\n    int predicted = max(minFeasibleW, (int)llround(sqrt((double)totalArea * aspect)));\n\n    vector<int> ws;\n    auto add = [&](int w) {\n        if (w >= minFeasibleW && w <= max<long long>(min((long long)INT_MAX, totalArea), minFeasibleW))\n            ws.push_back(w);\n    };\n\n    add(minFeasibleW);\n    add((int)totalArea);\n    vector<double> factors = {0.45,0.55,0.65,0.75,0.85,0.95,1.0,1.1,1.25,1.4,1.6,1.85,2.15,2.5,3.0};\n    for (double f : factors) {\n        add((int)ceil(sq * f));\n        add((int)ceil(predicted * f));\n    }\n    for (int d = -12; d <= 12; ++d) add(predicted + d);\n    for (auto& pc : pieces) {\n        for (auto& o : pc.ori) {\n            add(o.w);\n            add(o.w + 1);\n            add(o.w + 2);\n        }\n    }\n    if (totalArea <= 350) {\n        for (int w = minFeasibleW; w <= (int)totalArea; ++w) add(w);\n    }\n    sort(ws.begin(), ws.end());\n    ws.erase(unique(ws.begin(), ws.end()), ws.end());\n\n    sort(ws.begin(), ws.end(), [&](int a, int b) {\n        int da = abs(a - predicted), db = abs(b - predicted);\n        if (da != db) return da < db;\n        return a < b;\n    });\n    if (ws.size() > 90) ws.resize(90);\n    return ws;\n}\n\nstatic Result localSearch(Result start, double deadlineSeconds, chrono::steady_clock::time_point t0) {\n    if (!start.ok) return start;\n    int capW = start.W, capH = start.H;\n    if (capW <= 0 || capH <= 0) return start;\n    long long capArea = 1LL * capW * capH;\n    if (capArea > 600000LL || n == 0) return start;\n\n    vector<vector<unsigned char>> occ(capH, vector<unsigned char>(capW, 0));\n    vector<int> rowCnt(capH, 0), colCnt(capW, 0);\n    auto addPiece = [&](int id, const Placement& pl, int delta) {\n        const auto& o = pieces[id].ori[pl.oi];\n        for (auto c : o.cells) {\n            int x = pl.x + c.x, y = pl.y + c.y;\n            if (delta > 0) {\n                occ[y][x] = 1; rowCnt[y]++; colCnt[x]++;\n            } else {\n                occ[y][x] = 0; rowCnt[y]--; colCnt[x]--;\n            }\n        }\n    };\n    for (int i = 0; i < n; ++i) addPiece(i, start.p[i], +1);\n\n    auto usedW = [&]() {\n        for (int x = capW - 1; x >= 0; --x) if (colCnt[x]) return x + 1;\n        return 0;\n    };\n    auto usedH = [&]() {\n        for (int y = capH - 1; y >= 0; --y) if (rowCnt[y]) return y + 1;\n        return 0;\n    };\n    auto can = [&](const Orient& o, int px, int py) {\n        if (px < 0 || py < 0 || px + o.w > capW || py + o.h > capH) return false;\n        for (auto c : o.cells) if (occ[py + c.y][px + c.x]) return false;\n        return true;\n    };\n    auto addTop = [&](vector<Cand>& top, const Cand& c) {\n        const int K = 10;\n        if ((int)top.size() < K) top.push_back(c);\n        else {\n            int worst = 0;\n            for (int i = 1; i < (int)top.size(); ++i)\n                if (top[i].key > top[worst].key) worst = i;\n            if (c.key < top[worst].key) top[worst] = c;\n        }\n    };\n\n    Result cur = start, best = start;\n    int curW = usedW(), curH = usedH();\n    cur.W = curW; cur.H = curH;\n    if (betterResult(cur, best)) best = cur;\n\n    uniform_real_distribution<double> U(0.0, 1.0);\n    int iter = 0;\n    while (true) {\n        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();\n        if (elapsed >= deadlineSeconds) break;\n        ++iter;\n\n        int id;\n        if (U(rng) < 0.70) {\n            vector<int> boundary;\n            boundary.reserve(n);\n            for (int i = 0; i < n; ++i) {\n                const auto& pl = cur.p[i];\n                const auto& o = pieces[i].ori[pl.oi];\n                if (pl.x + o.w >= curW || pl.y + o.h >= curH) boundary.push_back(i);\n            }\n            if (!boundary.empty()) id = boundary[rng() % boundary.size()];\n            else id = rng() % n;\n        } else id = rng() % n;\n\n        Placement old = cur.p[id];\n        addPiece(id, old, -1);\n        int baseW = usedW(), baseH = usedH();\n\n        vector<Cand> top;\n        bool fullScan = (long long)capW * capH * (int)pieces[id].ori.size() <= 180000LL;\n\n        for (int oi = 0; oi < (int)pieces[id].ori.size(); ++oi) {\n            const auto& o = pieces[id].ori[oi];\n            if (o.w > capW || o.h > capH) continue;\n            if (fullScan) {\n                for (int y = 0; y + o.h <= capH; ++y) {\n                    for (int x = 0; x + o.w <= capW; ++x) {\n                        if (!can(o, x, y)) continue;\n                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);\n                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;\n                        addTop(top, {key, x, y, oi, nw, nh});\n                    }\n                }\n            } else {\n                int sx = max(1, capW / 55);\n                int sy = max(1, capH / 55);\n                for (int y = 0; y + o.h <= capH; y += sy) {\n                    for (int x = 0; x + o.w <= capW; x += sx) {\n                        if (!can(o, x, y)) continue;\n                        int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);\n                        long long key = objectiveWH(nw, nh) + 1000LL * y + x;\n                        addTop(top, {key, x, y, oi, nw, nh});\n                    }\n                }\n                for (int s = 0; s < 260; ++s) {\n                    int x = (capW == o.w) ? 0 : (int)(rng() % (capW - o.w + 1));\n                    int y = (capH == o.h) ? 0 : (int)(rng() % (capH - o.h + 1));\n                    if (!can(o, x, y)) continue;\n                    int nw = max(baseW, x + o.w), nh = max(baseH, y + o.h);\n                    long long key = objectiveWH(nw, nh) + 1000LL * y + x;\n                    addTop(top, {key, x, y, oi, nw, nh});\n                }\n            }\n        }\n\n        if (top.empty()) {\n            addPiece(id, old, +1);\n            continue;\n        }\n        sort(top.begin(), top.end(), [](const Cand& a, const Cand& b){ return a.key < b.key; });\n\n        int pick = 0;\n        double temp = max(0.05, 2.0 * (1.0 - min(1.0, elapsed / max(0.001, deadlineSeconds))));\n        if (top.size() > 1 && U(rng) < 0.20 * temp) {\n            int lim = min<int>(top.size(), 6);\n            pick = rng() % lim;\n        }\n        Cand chosen = top[pick];\n\n        long long oldArea = 1LL * curW * curH;\n        long long newArea = 1LL * chosen.usedW * chosen.usedH;\n        bool accept = false;\n        if (objectiveWH(chosen.usedW, chosen.usedH) <= objectiveWH(curW, curH)) accept = true;\n        else {\n            double delta = (double)(newArea - oldArea);\n            if (delta <= 0) accept = (U(rng) < 0.25);\n            else accept = (U(rng) < exp(-delta / temp));\n        }\n\n        if (accept) {\n            cur.p[id] = {chosen.oi, chosen.x, chosen.y};\n            addPiece(id, cur.p[id], +1);\n            curW = usedW(); curH = usedH();\n            cur.W = curW; cur.H = curH; cur.ok = true;\n            if (betterResult(cur, best)) best = cur;\n        } else {\n            addPiece(id, old, +1);\n        }\n    }\n    return best;\n}\n\nstatic vector<pair<int,int>> fixedRectanglesFromBest(const Result& best) {\n    vector<int> ws = candidateWidths();\n    for (int d = -30; d <= 30; ++d) ws.push_back(best.W + d);\n    long long bestArea = 1LL * best.W * best.H;\n    double sq = sqrt((double)bestArea);\n    vector<double> fs = {0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.3,1.5,1.75,2.0};\n    for (double f : fs) ws.push_back((int)ceil(sq * f));\n\n    int minFeasibleW = 1;\n    for (auto& pc : pieces) minFeasibleW = max(minFeasibleW, pc.minW);\n    sort(ws.begin(), ws.end());\n    ws.erase(unique(ws.begin(), ws.end()), ws.end());\n\n    vector<pair<int,int>> rects;\n    for (int W : ws) {\n        if (W < minFeasibleW || W <= 0) continue;\n        int H = (int)((bestArea - 1) / W);\n        if (H <= 0) continue;\n        if (1LL * W * H >= bestArea) continue;\n        rects.push_back({W, H});\n    }\n    sort(rects.begin(), rects.end(), [](auto a, auto b) {\n        long long aa = 1LL * a.first * a.second, bb = 1LL * b.first * b.second;\n        if (aa != bb) return aa < bb;\n        if (a.second != b.second) return a.second < b.second;\n        return a.first < b.first;\n    });\n    if (rects.size() > 120) rects.resize(120);\n    return rects;\n}\n\nint main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n\n    if (!(cin >> n)) return 0;\n    pieces.assign(n, Piece());\n    uint64_t seed = 1469598103934665603ULL ^ (uint64_t)n;\n    for (int i = 0; i < n; ++i) {\n        int k; cin >> k;\n        pieces[i].area = k;\n        pieces[i].orig.resize(k);\n        totalArea += k;\n        for (int j = 0; j < k; ++j) {\n            int x, y; cin >> x >> y;\n            pieces[i].orig[j] = {x, y};\n            seed ^= (uint64_t)(x + 1000003LL * y + 911382323LL * i + 97266353LL * j);\n            seed *= 1099511628211ULL;\n        }\n        generateOrientations(pieces[i]);\n    }\n    rng.seed((uint32_t)(seed ^ (seed >> 32)));\n\n    auto t0 = chrono::steady_clock::now();\n    double timeBudget = 3.4;\n    if (totalArea <= 250) timeBudget = 4.2;\n    if (totalArea >= 6000) timeBudget = 2.4;\n\n    auto elapsed = [&]() {\n        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();\n    };\n\n    vector<int> widths = candidateWidths();\n    vector<int> ord0 = baseOrder();\n\n    Result best;\n    // Guaranteed simple fallback: use a feasible adaptive width greedily.\n    for (int W : widths) {\n        if (elapsed() > timeBudget * 0.38) break;\n        Result r = greedyPlace(W, 0, ord0, false);\n        if (r.ok) {\n            computeUsed(r);\n            if (betterResult(r, best)) best = r;\n        }\n    }\n\n    if (!best.ok) {\n        int W = 1;\n        for (auto& pc : pieces) W = max(W, pc.minW);\n        best = greedyPlace(W, 0, ord0, false);\n        computeUsed(best);\n    }\n\n    // Randomized greedy restarts on promising widths.\n    vector<int> prom = widths;\n    prom.push_back(best.W);\n    for (int d = -8; d <= 8; ++d) prom.push_back(best.W + d);\n    sort(prom.begin(), prom.end());\n    prom.erase(unique(prom.begin(), prom.end()), prom.end());\n    prom.erase(remove_if(prom.begin(), prom.end(), [](int x){ return x <= 0; }), prom.end());\n\n    int restart = 0;\n    while (elapsed() < timeBudget * 0.58 && !prom.empty()) {\n        int W = prom[restart % prom.size()];\n        vector<int> ord = randomizedOrder(ord0);\n        Result r = greedyPlace(W, 0, ord, true);\n        if (r.ok) {\n            computeUsed(r);\n            if (betterResult(r, best)) best = r;\n        }\n        restart++;\n    }\n\n    // Simulated-annealing-inspired local reinsertion search.\n    Result ls = localSearch(best, timeBudget * 0.78, t0);\n    if (betterResult(ls, best)) best = ls;\n\n    // Fixed smaller-rectangle repacking attempts.\n    int fixedAttempts = 0;\n    while (elapsed() < timeBudget * 0.96) {\n        auto rects = fixedRectanglesFromBest(best);\n        if (rects.empty()) break;\n        bool improved = false;\n        for (auto [W, H] : rects) {\n            if (elapsed() >= timeBudget * 0.96) break;\n            vector<int> ord = (fixedAttempts % 4 == 0) ? ord0 : randomizedOrder(ord0);\n            Result r = greedyPlace(W, H, ord, fixedAttempts % 4 != 0);\n            fixedAttempts++;\n            if (r.ok) {\n                computeUsed(r);\n                if (betterResult(r, best)) {\n                    best = r;\n                    Result ls2 = localSearch(best, min(timeBudget * 0.96, elapsed() + 0.25), t0);\n                    if (betterResult(ls2, best)) best = ls2;\n                    improved = true;\n                    break;\n                }\n            }\n        }\n        if (!improved) break;\n    }\n\n    if (!verifyResult(best)) {\n        int W = 1;\n        for (auto& pc : pieces) W = max(W, pc.minW);\n        best = greedyPlace(W, 0, ord0, false);\n        computeUsed(best);\n    }\n\n    cout << best.W << ' ' << best.H << \"\\n\";\n    for (int i = 0; i < n; ++i) {\n        const Placement& pl = best.p[i];\n        const Orient& o = pieces[i].ori[pl.oi];\n        int X = pl.x - o.minx;\n        int Y = pl.y - o.miny;\n        cout << X << ' ' << Y << ' ' << o.R << ' ' << o.F << \"\\n\";\n    }\n    return 0;\n}",
  "frontiercs_base_dir": "/opt/Frontier-CS",
  "frontiercs_config": {
    "memory_mb": 256,
    "n_cases": 70,
    "time_limit_s": 2,
    "total_timeout_s": 340
  },
  "problem_id": "0"
}
```
