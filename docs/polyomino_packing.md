# Polyomino Packing Task

`polyomino_packing` adds FrontierCS algorithmic problem 0 to the Guidance-TTT
loop. The guidance model is still the trainable actor; the execution model now
submits a complete C++17 program instead of a Python `run()` function.

## Evaluation

Polyomino evaluation always uses the external FrontierCS/go-judge/Docker stack.
There is no local smoke evaluator and no approximate fallback score.

Runtime requirements:

```bash
python -c "import frontier_cs"
docker --version
```

If FrontierCS or Docker/go-judge is unavailable, the verifier returns
`environment_error`, `reward=0.0`, and writes the environment failure into the
library entry metadata. FrontierCS should be installed outside this repo, for
example from a separate `reference/` checkout or editable install; it is not
vendored into `guidance/`.

The pip package does not include the benchmark data tree. Set
`task.frontiercs.base_dir` to a Frontier-CS checkout that contains
`algorithmic/docker-compose.yml`.

## Prompt Contract

The task prompt describes the polyomino input/output format, transform order,
validity rules, and score direction. The execution prompt requires:

````text
<solution>
```cpp
// complete C++17 program
```
</solution>

<summary>
Natural language method summary only.
</summary>
````

The program must read stdin and write the placement to stdout. The library attach
policy is unchanged: guidance and execution both receive raw summaries selected
by PUCT, and execution additionally receives the parsed guidance.

## Config

Use:

```bash
python -m guidance_ttt.main_erdos --config guidance_ttt/config/polyomino_frontiercs.yaml --prepare-only
```

The generated agent loop config uses `guidance_execution_task` with:

```yaml
task:
  id: polyomino_packing
  frontiercs:
    base_dir: ../reference/Frontier-CS
    problem_id: "0"
    n_cases: 70
    time_limit_s: 2
    memory_mb: 256
    total_timeout_s: 340
```
