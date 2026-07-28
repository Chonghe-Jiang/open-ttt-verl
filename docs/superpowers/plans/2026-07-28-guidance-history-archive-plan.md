# Guidance-TTT History Archive Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive every meaningful Guidance-TTT experiment in a private Hugging Face dataset and publish the validated source changes to the existing GitHub branch.

**Architecture:** A tested Python inventory tool scans run metadata without loading multi-gigabyte histories into memory, applies the approved selection policy, and builds a deterministic staging tree under `.tmp/`. The staging tree preserves selected raw histories and associated artifacts, adds manifests and reports, and is uploaded to a private Hugging Face dataset through the Hub HTTP client. Source code changes remain separate and are committed to GitHub only after tests and credential scans pass.

**Tech Stack:** Python 3, pytest, Git, Slurm logs, Hugging Face Hub/Xet, SHA-256 manifests.

---

## Chunk 1: Deterministic Archive Builder

### Task 1: Add Selection And Inventory Tests

**Files:**
- Create: `tests/test_archive_guidance_history.py`
- Create: `scripts/archive_guidance_history.py`

- [ ] **Step 1: Write failing tests for approved run selection**

Create fixtures for:

- A run with a detected maximum step of 10.
- A run below 10 steps with a `1day` or `discover` tag and substantive history.
- A run below 10 steps without a formal-run tag.
- Runs containing `smoke`, `prepare`, `preflight`, or `setup`.

Assert that only the first two categories are included and that every decision
contains a machine-readable reason.

- [ ] **Step 2: Run the selection tests and confirm failure**

Run:

```bash
python -m pytest tests/test_archive_guidance_history.py -q
```

Expected: failure because the archive module is not implemented.

- [ ] **Step 3: Implement bounded-memory step detection and selection**

Implement:

- Chunked byte scanning for `"step"`, `"global_step"`, and equivalent integer
  fields without `json.load()` on the full history.
- Case-insensitive excluded-name detection.
- Formal-run tag detection for `1day`, `two_day`, `50step`, and `discover`.
- Substantive-history detection based on a nonempty valid library file and
  detected entry/step evidence.
- Structured `included`, `reason`, `max_step`, `formal_tags`, and byte-count
  fields.

- [ ] **Step 4: Run the selection tests**

Run:

```bash
python -m pytest tests/test_archive_guidance_history.py -q
```

Expected: all archive selection tests pass.

### Task 2: Add Staging, Manifest, And Security Tests

**Files:**
- Modify: `tests/test_archive_guidance_history.py`
- Modify: `scripts/archive_guidance_history.py`

- [ ] **Step 1: Write failing staging tests**

Test that the builder:

- Preserves `library*.json`, `agent_loop.yaml`, `ttt_slots.parquet`, and
  `run_summary.json` when present.
- Excludes `checkpoints/`, `*.lock`, temporary files, caches, and model state.
- Places runs into stable experiment-family directories.
- Writes `manifest.json`, `reports/runs.jsonl`, and `reports/overview.md`.
- Records file size and SHA-256.
- Rejects staged text containing Hugging Face/OpenRouter token patterns,
  authorization headers, or secret assignments.

- [ ] **Step 2: Run the staging tests and confirm failure**

Run:

```bash
python -m pytest tests/test_archive_guidance_history.py -q
```

Expected: staging/security assertions fail.

- [ ] **Step 3: Implement staging and validation**

Use hard links when source and staging are on the same filesystem, falling back
to streamed copy. Never modify source artifacts. Generate the dataset card,
selection report, overview, and manifest after staging. Scan text artifacts
before they are accepted into the manifest.

- [ ] **Step 4: Run the complete archive-tool test file**

Run:

```bash
python -m pytest tests/test_archive_guidance_history.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit archive tooling**

```bash
git add scripts/archive_guidance_history.py tests/test_archive_guidance_history.py
git commit -m "tools: add Guidance-TTT history archiver"
```

## Chunk 2: Build And Publish The Private Dataset

### Task 3: Inventory And Stage Real Runs

**Files:**
- Generated: `.tmp/guidance_history_hf/**`

- [ ] **Step 1: Run a dry inventory**

```bash
python scripts/archive_guidance_history.py \
  --repo-root . \
  --outputs outputs \
  --staging .tmp/guidance_history_hf \
  --inventory-only
```

Review every included/excluded run and verify the approved rule.

- [ ] **Step 2: Build the staging tree**

```bash
python scripts/archive_guidance_history.py \
  --repo-root . \
  --outputs outputs \
  --staging .tmp/guidance_history_hf
```

- [ ] **Step 3: Verify contents and size**

Confirm:

- No path contains checkpoint/smoke/prepare/preflight/setup output.
- Every selected run has at least one history file.
- No staged file contains a detected credential.
- No source output changed.
- Manifest file count, byte totals, and checksums match staging.

### Task 4: Create And Upload Hugging Face Dataset

**Files:**
- Remote private dataset: `LeoJiangOR/guidance-ttt-history`

- [ ] **Step 1: Install the current Hugging Face Hub client outside the repo**

Use a user-scoped or temporary environment. Do not add it to project
requirements.

- [ ] **Step 2: Create the private dataset**

Create `LeoJiangOR/guidance-ttt-history` with repository type `dataset` and
`private=true`. Query the API immediately to verify visibility before upload.

- [ ] **Step 3: Upload the staging directory**

Use the resumable Hugging Face folder upload API with Xet high-performance mode.
The token is supplied only through the process environment and is never written
to the repository, staging tree, shell profile, or logs.

- [ ] **Step 4: Verify the remote dataset**

Compare the remote file list and reported sizes against `manifest.json`.
Download and verify the small metadata files, confirm private visibility again,
and verify excluded path patterns return no matches.

## Chunk 3: Validate And Publish Source Repository

### Task 5: Review Existing Uncommitted Source Work

**Files:**
- Existing modified and untracked files reported by `git status`

- [ ] **Step 1: Inspect every diff and classify it**

Group changes into:

- Discover/library and OpenRouter client behavior.
- Verl integration changes.
- GLM/27B configurations and Slurm launchers.
- Tests and validation tooling.

Do not reset, discard, or overwrite user-owned changes.

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest \
  tests/test_library_puct.py \
  tests/test_llm_client_mock.py \
  tests/test_verl_ext.py \
  tests/test_openrouter_glm52_discover_config.py \
  tests/test_archive_guidance_history.py -q
```

Fix failures with test-driven changes.

- [ ] **Step 3: Run broader verification**

Run the complete feasible test suite. Record environmental skips or
infrastructure-only failures rather than weakening tests.

- [ ] **Step 4: Scan tracked candidates for credentials and oversized files**

Block the commit if any token, private key, `.env`, secret assignment, generated
history, checkpoint, or file above GitHub's normal size limit is staged.

### Task 6: Document, Commit, And Push

**Files:**
- Modify: `README.md`
- Create: `docs/guidance_history.md`
- Existing source/config/script/test changes

- [ ] **Step 1: Document the private history location**

Add a short history document describing selection rules, excluded artifacts,
dataset layout, and access URL. Do not include credentials.

- [ ] **Step 2: Commit existing source work in logical groups**

Stage only reviewed files. Keep archive tooling/docs separate from the core
Discover/GLM implementation when practical.

- [ ] **Step 3: Verify the final branch**

Run focused tests again, inspect `git status`, inspect commits since the original
remote head, and verify no secret appears in staged or committed content.

- [ ] **Step 4: Push GitHub branch**

```bash
git push origin guidance-ttt-modal
```

- [ ] **Step 5: Verify remote commit**

Use `git ls-remote` to confirm the GitHub branch hash matches local `HEAD`.
