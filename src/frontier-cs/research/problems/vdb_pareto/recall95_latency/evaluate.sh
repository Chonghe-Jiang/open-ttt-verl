#!/usr/bin/env bash
set -e

PROBLEM_DIR=$(pwd)
echo "[evaluate] Running VDB Design evaluation (recall95_latency)..." >&2
echo "[evaluate] Current dir: $(pwd)" >&2

./run_evaluator.sh 2>&1 | tee /tmp/eval_output.log
RUN_EXIT_CODE=${PIPESTATUS[0]}

if grep -q "TimeoutError" /tmp/eval_output.log; then
    echo "[evaluate] TIMEOUT FAILURE detected - returning score 0.0" >&2
    if [[ -f "results.json" ]]; then
        echo "[evaluate] Found results.json for timeout case" >&2
        cat results.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('score', 0.0))"
        exit 0
    else
        echo "[evaluate] No results.json found for timeout, returning 0.0" >&2
        echo "0.0"
        exit 0
    fi
elif [[ $RUN_EXIT_CODE -ne 0 ]]; then
    echo "[evaluate] run_evaluator failed with exit code $RUN_EXIT_CODE" >&2
    
    if [[ -f "results.json" ]]; then
        echo "[evaluate] Found results.json despite failure" >&2
        cat results.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('score', 0.0))"
        exit 0
    else
        echo "[evaluate] No results.json found, returning 0.0" >&2
        echo "0.0"
        exit 0
    fi
else
    echo "[evaluate] run_evaluator succeeded" >&2
fi

if [[ -f "results.json" ]]; then
    echo "[evaluate] Found results.json" >&2
    cat results.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('score', 50.0))"
else
    echo "[evaluate] WARNING: results.json not found, returning mock score" >&2
    echo "0.0"
fi


