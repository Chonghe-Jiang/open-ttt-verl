# Guidance-TTT History Archive Design

## Goal

Preserve all meaningful Guidance-TTT experiment history in a private Hugging
Face dataset while keeping the source repository small, reviewable, and free
of credentials or model checkpoints.

## Scope

The archive covers every run under `outputs/guidance_ttt/` that satisfies at
least one of these rules:

- The run completed at least 10 training steps.
- The run is a formal `1day`, `two_day`, `50step`, or Discover run and produced
  substantive library history, even if it stopped before step 10.

Directories identified as `smoke`, `prepare`, or `preflight` are excluded, as
are pure setup jobs. Relevant Slurm logs are selected from included run names
and job identifiers rather than copied wholesale.

The archive includes:

- Current and preserved `library*.json` history files for every selected run.
- `agent_loop.yaml` and `ttt_slots.parquet`.
- Source recipes, launch scripts, resume scripts, and validators used by the
  selected experiments.
- Relevant Slurm stdout, stderr, judge, and service logs.
- A machine-readable manifest with source paths, sizes, SHA-256 checksums, and
  archive paths.
- A dataset card and a human-readable run overview.

The archive excludes:

- `checkpoints/` and all model, optimizer, adapter, or training-state files.
- Smoke-test, prepare, preflight, and setup run directories and logs.
- `.env`, `.secrets`, tokens, authorization headers, and other credentials.
- Lock files, temporary files, caches, downloaded models, and runtime images.

## Destination Layout

The private Hugging Face dataset will be:

`LeoJiangOR/guidance-ttt-history`

The uploaded layout will be:

```text
README.md
manifest.json
experiments/
  <family>/
    <run-name>/
      config/
      history/
      metadata/
      logs/
source/
  configs/
  scripts/
reports/
  overview.md
  runs.jsonl
```

Staging is built under the ignored repository path
`.tmp/guidance_history_hf/`. Source experiment outputs remain unchanged.

## Security And Integrity

Before upload, every staged text file is scanned for Hugging Face/OpenRouter
token patterns, authorization headers, and common secret assignments. Files
that contain credentials are not uploaded until a sanitized copy is produced.
The manifest records SHA-256 checksums so the upload can be verified against
the staged source. Repository visibility is checked through the Hugging Face
API before and after upload.

The selection report records why each run was included or excluded, its
detected maximum step, formal-run tags, total bytes, and associated jobs.

## Code Repository

The existing uncommitted implementation is treated as user-owned work. It will
be reviewed without resetting or discarding files, tested with focused and
full test suites where practical, scanned for credentials, committed in
logical groups, and pushed to the existing GitHub branch
`guidance-ttt-modal`.

The large history archive is not added to the GitHub repository. GitHub only
receives source code, recipes, scripts, tests, and this documentation.

## Verification

Completion requires:

- The Hugging Face repository exists and reports `private: true`.
- No checkpoint, smoke output, cache, lock, or credential is present remotely.
- Remote file sizes and checksums match the staged manifest.
- Every selected run's configuration, history snapshots, metadata, and relevant
  logs are represented in the manifest.
- The selection report contains no included smoke/prepare/preflight/setup run,
  and every included run satisfies the approved selection rule.
- Focused tests for the modified code pass.
- The GitHub branch is pushed and its remote commit matches local `HEAD`.
