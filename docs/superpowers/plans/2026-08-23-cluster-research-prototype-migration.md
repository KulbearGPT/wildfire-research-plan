# Cluster Research Prototype Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current repository into a small, cluster-ready Slurm research prototype while preserving the completed baseline evidence and publishing a single authoritative next-experiment roadmap.

**Architecture:** Keep the existing website, Phase 0 package, and reproduction controls in place. Add a root research entry point, an explicit roadmap, one small standard-library cluster control CLI, five thin shell wrappers, and a hash manifest for the 999 active event-level HDF5 files. Site-specific accounts and paths live only in an ignored profile; scientific commands remain ordinary commands passed through Slurm.

**Tech Stack:** Python 3.13 standard library, pytest, POSIX shell, Slurm CLI, Git, CSV/JSON, existing `wildfire_phase0` package.

**Spec:** `docs/superpowers/specs/2026-08-23-cluster-research-prototype-design.md`

## Global Constraints

- This is a fast research prototype, not a deployment, serving, workflow-platform, database, or Kubernetes project.
- Do not move or reinterpret existing website, Phase 0, reproduction, or sealed baseline evidence paths.
- Do not run training, inference, Slurm, `sbatch`, or any scientific child while implementing this plan.
- Do not commit data, official weights, checkpoints, environments, caches, real cluster profiles, credentials, local absolute paths, or run artifacts.
- Keep cluster code standard-library-only and small; no new runtime dependency is allowed.
- One Slurm task requests one GPU. Parallel folds, seeds, or corruptions use a bounded array; no DDP or multi-node support is added.
- The active migration dataset contract is exactly eight years `2016` through `2023`, 999 direct event-level HDF5 files, and 49,816,826,985 bytes before hashing.
- Candidate methods from the shared planning conversation remain labelled candidates or hypotheses unless a primary paper or official source is verified.
- Every implementation task follows RED -> GREEN TDD and ends with a focused test run and a reviewable commit.
- Use `D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe` for local pytest commands because bare `python` is not on the current PATH.

---

## File Structure

### New tracked files

- `README.md` — root project entry point and completed/current status.
- `docs/research-roadmap.md` — single ordered experiment roadmap.
- `docs/cluster-migration.md` — site-independent Slurm migration guide.
- `configs/cluster/profile.example.env` — dummy, non-secret site profile.
- `environments/README.md` — audit/training environment boundary and cluster qualification procedure.
- `manifests/data/wstsplus-hdf5.csv` — 999 path/size/SHA-256 records generated from the active fixed HDF5 tree.
- `manifests/data/wstsplus-hdf5.summary.json` — count, bytes, years, and manifest hash.
- `manifests/weights/README.md` — canonical pointer to the existing official-weight manifest, avoiding duplication.
- `scripts/cluster/clusterctl.py` — profile parsing, `sbatch` rendering, manifest creation/verification, and compact run-record helpers.
- `scripts/cluster/bootstrap.sh` — module/virtualenv bootstrap wrapper for an explicit requirements file.
- `scripts/cluster/preflight.sh` — profile and manifest preflight wrapper.
- `scripts/cluster/submit.sh` — submission wrapper with mandatory dry-run support.
- `scripts/cluster/job.sh` — generic one-command Slurm worker with no retry.
- `scripts/cluster/collect.sh` — compact result collector wrapper.
- `tests/test_project_research_organization.py` — README/roadmap/link/claim-boundary contract.
- `tests/test_cluster_profile.py` — profile schema and `sbatch` rendering contract.
- `tests/test_cluster_manifest.py` — HDF5 manifest contract.
- `tests/test_cluster_scripts.py` — thin-wrapper, no-retry, ignore, and evidence contract.

### Modified tracked files

- `.gitignore` — ignore real cluster profiles and all cluster-local heavy state while preserving examples and manifests.

### Explicitly unchanged

- `index.html`, `baseline-reproduction/`, and `related-work/`.
- `src/wildfire_phase0/` scientific/data behavior.
- `reproductions/wsts_res18_unet_t1/` controls, locks, patches, and manifests.
- Existing experiment evidence documents and numerical claims.

---

### Task 1: Add the project entry point and authoritative research roadmap

**Files:**
- Create: `README.md`
- Create: `docs/research-roadmap.md`
- Create: `tests/test_project_research_organization.py`

**Interfaces:**
- Consumes: existing `docs/experiments/phase0.md`, `docs/experiments/res18_unet_t1_reproduction.md`, `related-work/index.html`, and the approved spec.
- Produces: stable relative links and ordered stage headings consumed by the cluster migration guide.

- [ ] **Step 1: Write the failing organization tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_links_to_current_evidence_and_cluster_guides() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in (
        "docs/experiments/phase0.md",
        "docs/experiments/res18_unet_t1_reproduction.md",
        "docs/research-roadmap.md",
        "docs/cluster-migration.md",
        "baseline-reproduction/",
        "related-work/",
    ):
        assert target in text


def test_roadmap_orders_completed_gates_before_new_methods() -> None:
    text = (ROOT / "docs" / "research-roadmap.md").read_text(encoding="utf-8")
    headings = [
        "Stage 0 — Cluster migration and equivalence",
        "Stage 1 — Training-contract sensitivity",
        "Stage 2 — WSTS+ learned controls",
        "Stage 3 — Controlled-missingness diagnosis",
        "Stage 4 — Matched robust baselines and main method",
        "Stage 5 — Conditional extensions",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_roadmap_preserves_claim_boundaries() -> None:
    text = (ROOT / "docs" / "research-roadmap.md").read_text(encoding="utf-8")
    assert "controlled missingness" in text
    assert "does not establish natural-missingness or operational-deployment performance" in text
    assert "released-weight executable reproducibility" in text
    assert "does not prove paper-table provenance identity" in text
    assert "Candidate, primary-source verification required" in text
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_project_research_organization.py -q
```

Expected: FAIL because `README.md` and `docs/research-roadmap.md` do not exist.

- [ ] **Step 3: Write `README.md`**

The README must contain these sections, in this order:

```markdown
# Reliable Next-Day Wildfire Spread Forecasting

## Current status
## Scientific boundary
## Repository map
## Fast local verification
## Slurm cluster quick start
## Next experiment
```

State the completed 999-event audit, fixed split, deterministic rules, official
Fold-2 training reproduction, and twelve released-weight evaluation. Link rather
than duplicate long evidence tables. State that the immediate work is cluster
equivalence, the paired positive-weight sensitivity experiment, then WSTS+
learned controls.

- [ ] **Step 4: Write `docs/research-roadmap.md`**

Use exactly the six stage headings asserted by the test. Each stage contains:

```markdown
**Question:** one scientific question
**Required runs:** the smallest run set that answers it
**Gate:** an observable continue/stop condition
**Output:** one compact evidence artifact or reviewed result table
```

Stage 4 lists zero-fill/LOCF, concatenate-plus-mask, uniform fusion, lightweight
direct gate, MaskUNet-to-frozen-forecaster, FireEx-style experts with a
capacity-matched generalist ensemble, and residual prototype-reliability gating.
Stage 5 labels MaskCVAE, STARS, arbitrary-modal attention, hybrid routing, and
timestamp-centric methods as `Candidate, primary-source verification required`
unless a primary source was actually checked during the task. Do not copy
unverified performance numbers from the shared ChatGPT conversation.

- [ ] **Step 5: Run focused GREEN tests**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_project_research_organization.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Check links and commit**

```powershell
$repo = 'D:\WildFire Project\wildfire-research-plan'
$links = Select-String -Path "$repo\README.md" -Pattern '\]\(([^)]+)\)' -AllMatches
git diff --check
git add README.md docs/research-roadmap.md tests/test_project_research_organization.py
git commit -m "docs: add cluster research roadmap"
```

Expected: every relative target exists except `docs/cluster-migration.md`, which
is deliberately supplied by Task 5; the test should check the literal link now
and Task 5 will close existence in the final integration gate.

---

### Task 2: Add the cluster profile contract and dry-run Slurm renderer

**Files:**
- Create: `configs/cluster/profile.example.env`
- Create: `scripts/cluster/clusterctl.py`
- Create: `tests/test_cluster_profile.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `parse_profile(path, *, allow_placeholders=False) -> dict[str, str]`
- Produces: `render_submit_command(profile, *, profile_path, job_name, time_class, command, array=None) -> list[str]`
- Produces CLI: `clusterctl.py profile validate PROFILE [--allow-placeholders]`
- Produces CLI: `clusterctl.py submit PROFILE --job-name NAME --time-class {smoke,calibration,full} [--array START-END] [--dry-run] -- COMMAND...`

- [ ] **Step 1: Write failing profile tests**

```python
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cluster" / "clusterctl.py"


def load_clusterctl():
    spec = importlib.util.spec_from_file_location("clusterctl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_profile_is_complete_but_contains_only_placeholders() -> None:
    ctl = load_clusterctl()
    profile = ctl.parse_profile(
        ROOT / "configs" / "cluster" / "profile.example.env",
        allow_placeholders=True,
    )
    assert profile["CLUSTER_PROFILE_SCHEMA"] == "1"
    assert profile["SLURM_ARRAY_CONCURRENCY"] == "4"
    assert profile["SLURM_GPU_REQUEST"] == "--gpus=h100:1"


@pytest.mark.parametrize("bad_value", ["", "replace-me", "<GPU_ACCOUNT>"])
def test_formal_profile_rejects_placeholders(tmp_path: Path, bad_value: str) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    profile["SLURM_GPU_ACCOUNT"] = bad_value
    path = write_profile(tmp_path, profile)
    with pytest.raises(ValueError, match="SLURM_GPU_ACCOUNT"):
        ctl.parse_profile(path)


def test_render_uses_one_gpu_and_bounded_array_without_running_sbatch(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    profile = formal_profile_values()
    path = write_profile(tmp_path, profile)
    command = ctl.render_submit_command(
        profile,
        profile_path=path,
        job_name="folds",
        time_class="full",
        command=["python", "train.py", "--seed", "0"],
        array="0-11",
    )
    assert command[0] == "sbatch"
    assert "--gpus=h100:1" in command
    assert "--array=0-11%4" in command
    assert not any("gpus=h100:2" in token for token in command)
```

Include helpers that write a complete temporary profile. Add cases for duplicate
keys, unknown keys, malformed lines, invalid integer concurrency, invalid time
class, newline/control characters, and missing command after `--`.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_cluster_profile.py -q
```

Expected: FAIL because the profile and CLI do not exist.

- [ ] **Step 3: Implement the exact profile schema**

Use this required key tuple in `clusterctl.py`:

```python
REQUIRED_PROFILE_KEYS = (
    "CLUSTER_PROFILE_SCHEMA",
    "SLURM_CPU_ACCOUNT",
    "SLURM_GPU_ACCOUNT",
    "SLURM_CPU_PARTITION",
    "SLURM_GPU_PARTITION",
    "SLURM_GPU_REQUEST",
    "SLURM_ARRAY_CONCURRENCY",
    "MODULES",
    "ENV_ACTIVATE",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RUNS_ROOT",
    "CACHE_ROOT",
    "LOG_ROOT",
    "SCRATCH_ENV",
    "SMOKE_TIME",
    "CALIBRATION_TIME",
    "FULL_TIME",
)
```

Parse only `KEY=VALUE`, blank lines, and `#` comments. Do not `eval`, expand
variables, execute command substitution, or accept duplicate/unknown keys.
Allow dummy values only when `allow_placeholders=True`.

- [ ] **Step 4: Create the example profile**

```dotenv
CLUSTER_PROFILE_SCHEMA=1
SLURM_CPU_ACCOUNT=replace-me
SLURM_GPU_ACCOUNT=replace-me
SLURM_CPU_PARTITION=replace-me
SLURM_GPU_PARTITION=replace-me
SLURM_GPU_REQUEST=--gpus=h100:1
SLURM_ARRAY_CONCURRENCY=4
MODULES=StdEnv/2023,gcc/12.3,cuda/12.2,python/3.11.5
ENV_ACTIVATE=/project/GROUP/wildfire/envs/training/bin/activate
PROJECT_ROOT=/project/GROUP/wildfire
DATA_ROOT=/project/GROUP/wildfire/data/hdf5
RUNS_ROOT=/project/GROUP/wildfire/runs
CACHE_ROOT=/project/GROUP/wildfire/caches
LOG_ROOT=/project/GROUP/wildfire/logs
SCRATCH_ENV=SLURM_TMPDIR
SMOKE_TIME=00:30:00
CALIBRATION_TIME=03:00:00
FULL_TIME=12:00:00
```

- [ ] **Step 5: Implement deterministic `sbatch` rendering**

The command contains, in fixed order:

```python
[
    "sbatch",
    "--parsable",
    f"--job-name={job_name}",
    f"--account={profile['SLURM_GPU_ACCOUNT']}",
    f"--partition={profile['SLURM_GPU_PARTITION']}",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=8",
    "--mem=96G",
    profile["SLURM_GPU_REQUEST"],
    f"--time={profile[time_key]}",
    f"--output={profile['LOG_ROOT']}/%x-%A_%a.out",
    f"--error={profile['LOG_ROOT']}/%x-%A_%a.err",
    # optional --array=START-END%CONCURRENCY
    str(ROOT / "scripts" / "cluster" / "job.sh"),
    str(profile_path.resolve()),
    "--",
    *scientific_command,
]
```

`--dry-run` prints `shlex.join(command) + "\n"` and never invokes
`subprocess.run`. Non-dry execution calls exactly one `subprocess.run(command,
check=False)` and returns its exit code; it never retries.

- [ ] **Step 6: Extend `.gitignore`**

Add exact patterns:

```gitignore
/configs/cluster/profile.env
/data/
/checkpoints/
/runs/
/logs/
/caches/
/tmp/
/envs/
*.ckpt
```

Do not ignore `profile.example.env`, `manifests/`, `README.md`, or reviewed
summary documents.

- [ ] **Step 7: Run GREEN and commit**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_cluster_profile.py -q
git diff --check
git add .gitignore configs/cluster/profile.example.env scripts/cluster/clusterctl.py tests/test_cluster_profile.py
git commit -m "feat: add portable Slurm profile"
```

---

### Task 3: Add the inode-safe HDF5 migration manifest

**Files:**
- Modify: `scripts/cluster/clusterctl.py`
- Create: `manifests/data/wstsplus-hdf5.csv`
- Create: `manifests/data/wstsplus-hdf5.summary.json`
- Create: `manifests/weights/README.md`
- Create: `tests/test_cluster_manifest.py`

**Interfaces:**
- Produces: `build_hdf5_manifest(data_root: Path) -> list[ManifestEntry]`
- Produces: `write_hdf5_manifest(data_root: Path, csv_path: Path, summary_path: Path) -> dict[str, object]`
- Produces: `verify_hdf5_manifest(data_root: Path, csv_path: Path, summary_path: Path) -> dict[str, object]`
- Adds CLI: `clusterctl.py manifest create DATA_ROOT CSV SUMMARY`
- Adds CLI: `clusterctl.py manifest verify DATA_ROOT CSV SUMMARY`

- [ ] **Step 1: Write manifest RED tests**

```python
def test_manifest_round_trip_hashes_only_direct_year_hdf5(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, {2016: [b"a", b"bb"], 2017: [b"ccc"]})
    summary = clusterctl.write_hdf5_manifest(root, csv_path, summary_path)
    assert summary["file_count"] == 3
    assert summary["total_bytes"] == 6
    assert summary["years"] == {"2016": 2, "2017": 1}
    assert clusterctl.verify_hdf5_manifest(root, csv_path, summary_path) == summary


@pytest.mark.parametrize("mutation", ["missing", "extra", "resize", "content"])
def test_verify_rejects_tree_mutation(tmp_path: Path, mutation: str) -> None:
    root = make_fixture(tmp_path, {2016: [b"a"]})
    clusterctl.write_hdf5_manifest(root, csv_path, summary_path)
    mutate_fixture(root, mutation)
    with pytest.raises(ValueError):
        clusterctl.verify_hdf5_manifest(root, csv_path, summary_path)
```

Add tests rejecting symlink files/directories when platform capability exists,
nested HDF5 files, non-year directories, duplicate CSV rows, unsafe relative
paths, non-lowercase SHA-256, wrong summary manifest hash, and non-atomic temp
leftovers.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_cluster_manifest.py -q
```

Expected: FAIL because manifest interfaces do not exist.

- [ ] **Step 3: Implement the manifest schema**

Use a frozen CSV header:

```csv
relative_path,size_bytes,sha256
```

Only direct `<year>/<filename>.hdf5` files are valid. Store POSIX relative paths,
integer bytes, and lowercase SHA-256. Sort by integer year then filename. Write
CSV and summary through sibling temporary files followed by `os.replace`.

The summary exact keys are:

```python
{
    "schema_version": 1,
    "dataset": "WSTS+ active fixed event-level HDF5",
    "file_count": 999,
    "total_bytes": 49816826985,
    "years": {"2016": 92, "2017": 110, "2018": 176, "2019": 74,
              "2020": 201, "2021": 156, "2022": 122, "2023": 68},
    "manifest_sha256": hashlib.sha256(csv_bytes).hexdigest(),
}
```

For fixture tests, the same function accepts arbitrary year subsets and derives
their counts; the production-generation command separately asserts the exact
contract above before publishing the tracked files.

- [ ] **Step 4: Generate the real manifest once**

Run against the active fixed tree:

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' scripts/cluster/clusterctl.py manifest create `
  'D:\WildFire Project\data\hdf5' `
  manifests/data/wstsplus-hdf5.csv `
  manifests/data/wstsplus-hdf5.summary.json `
  --require-production-contract
```

Expected: 999 files, 49,816,826,985 bytes, exact year counts above. This command
reads all HDF5 bytes to hash them but does not open or modify HDF5 content.

- [ ] **Step 5: Independently verify the generated manifest**

Use a short standard-library script that does not import `clusterctl` to recount
direct files, recompute total bytes, validate 64-character lowercase hashes,
and recompute the CSV file hash. Then run the production verifier:

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' scripts/cluster/clusterctl.py manifest verify `
  'D:\WildFire Project\data\hdf5' `
  manifests/data/wstsplus-hdf5.csv `
  manifests/data/wstsplus-hdf5.summary.json `
  --require-production-contract
```

- [ ] **Step 6: Add the weight-manifest pointer**

`manifests/weights/README.md` states that the canonical official 12-weight
manifest remains
`reproductions/wsts_res18_unet_t1/official_weights_manifest.json` at Hub
revision `acf70a37394849f4ec8d108a51d6f4325a554d0a`; do not duplicate its rows.

- [ ] **Step 7: Run GREEN and commit**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_cluster_manifest.py -q
git diff --check
git add scripts/cluster/clusterctl.py manifests/data manifests/weights/README.md tests/test_cluster_manifest.py
git commit -m "feat: add verified WSTS+ migration manifest"
```

---

### Task 4: Add thin Slurm wrappers and compact run evidence

**Files:**
- Modify: `scripts/cluster/clusterctl.py`
- Create: `scripts/cluster/bootstrap.sh`
- Create: `scripts/cluster/preflight.sh`
- Create: `scripts/cluster/submit.sh`
- Create: `scripts/cluster/job.sh`
- Create: `scripts/cluster/collect.sh`
- Create: `tests/test_cluster_scripts.py`

**Interfaces:**
- Adds CLI: `clusterctl.py preflight PROFILE --data-manifest CSV --data-summary JSON`
- Adds CLI: `clusterctl.py run start PROFILE RUN_DIR -- COMMAND...`
- Adds CLI: `clusterctl.py run finish RUN_DIR EXIT_CODE`
- Adds CLI: `clusterctl.py collect RUN_DIR OUTPUT_JSON`
- Shell wrappers pass arguments once to these commands and contain no scientific defaults.

- [ ] **Step 1: Write shell-boundary RED tests**

```python
def test_wrappers_are_thin_lf_posix_files() -> None:
    for name in ("bootstrap.sh", "preflight.sh", "submit.sh", "job.sh", "collect.sh"):
        raw = (SCRIPTS / name).read_bytes()
        assert raw.startswith(b"#!/usr/bin/env bash\n")
        assert b"\r\n" not in raw
        assert b"set -euo pipefail" in raw
        assert len(raw.splitlines()) <= 80


def test_job_has_no_retry_or_scientific_override() -> None:
    text = (SCRIPTS / "job.sh").read_text(encoding="utf-8")
    assert "retry" not in text.lower()
    for forbidden in ("max_steps", "batch_size", "learning_rate", "fold_id", "features_to_keep"):
        assert forbidden not in text


def test_run_finish_writes_one_terminal_marker_and_preserves_start(tmp_path: Path) -> None:
    clusterctl.write_run_start(run_dir, profile, ["python", "train.py"], environ)
    before = (run_dir / "started.json").read_bytes()
    clusterctl.write_run_finish(run_dir, 0)
    assert (run_dir / "completed.json").is_file()
    assert not (run_dir / "failure.json").exists()
    assert (run_dir / "started.json").read_bytes() == before
```

Add cases for nonzero exit, existing terminal marker, dirty formal Git checkout,
missing data manifest, unwritable output root, no CUDA device in formal GPU
mode, source profile mutation between start and finish, and collection that
packs declared small JSON/CSV files without traversing checkpoints or arbitrary
directories.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest tests/test_cluster_scripts.py -q
```

- [ ] **Step 3: Implement compact run records**

`started.json` contains only:

```python
{
    "schema_version": 1,
    "status": "started",
    "created_utc": utc_now(),
    "slurm_job_id": environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": environ.get("SLURM_ARRAY_TASK_ID"),
    "git_commit": exact_git_head,
    "git_dirty": False,
    "profile_path": str(profile_path.resolve()),
    "profile_sha256": sha256_file(profile_path),
    "command": command,
    "python": platform.python_version(),
    "environment": selected_nonsecret_versions,
}
```

`completed.json` or `failure.json` records the exit code, end UTC, elapsed
seconds, and the original `started.json` SHA-256. Never include all environment
variables. Never auto-resume or auto-retry.

- [ ] **Step 4: Implement wrappers**

Use this pattern for `submit.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$repo_root/scripts/cluster/clusterctl.py" submit "$@"
```

Other wrappers use the same resolved-root pattern and delegate one CLI action.
`job.sh` activates the profile's explicit `ENV_ACTIVATE`, chooses the named
scratch environment variable when set, creates one job/run directory, writes
start evidence, executes the command exactly once, captures the exit code, and
writes the matching terminal record.

`bootstrap.sh` accepts exactly `PROFILE ENV_DIR REQUIREMENTS_FILE`; it loads the
comma-separated modules, creates one `virtualenv --no-download`, activates it,
and runs `python -m pip install --no-index -r REQUIREMENTS_FILE`. It refuses an
existing nonempty environment directory and does not invent training package
versions.

- [ ] **Step 5: Implement preflight and collection**

Preflight validates profile, Git cleanliness, manifest contents, persistent
roots, environment activation file, and GPU availability when `--require-gpu`
is passed. Unit tests inject command runners; local tests never call
`nvidia-smi`, `sbatch`, or Slurm.

Collection accepts an explicit list of result JSON/CSV paths inside one run
directory and creates one `results.tar.gz` plus `collection.json`. Reject
symlinks, path escapes, checkpoints, files over a small declared limit, and an
existing collection. This is inode consolidation, not a general artifact
system.

- [ ] **Step 6: Run GREEN, static checks, and commit**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest `
  tests/test_cluster_profile.py tests/test_cluster_manifest.py tests/test_cluster_scripts.py -q
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m py_compile scripts/cluster/clusterctl.py
git diff --check
git add scripts/cluster tests/test_cluster_scripts.py
git commit -m "feat: add thin Slurm research runners"
```

---

### Task 5: Add the cluster migration guide and environment boundary

**Files:**
- Create: `docs/cluster-migration.md`
- Create: `environments/README.md`
- Modify: `tests/test_project_research_organization.py`
- Modify: `tests/test_cluster_scripts.py`

**Interfaces:**
- Consumes the exact profile and CLI names from Tasks 2–4.
- Produces the student-facing clone-to-smoke sequence used after pushing `main`.

- [ ] **Step 1: Extend tests before writing docs**

```python
def test_cluster_guide_orders_read_only_gates_before_submission() -> None:
    text = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    ordered = [
        "Clone the reviewed commit",
        "Create the site-local profile",
        "Qualify the environment",
        "Transfer and verify HDF5",
        "Fetch pinned upstream code and weights",
        "Run the one-batch smoke",
        "Run Fold-2 test-only equivalence",
        "Run the 500-step calibration",
    ]
    positions = [text.index(item) for item in ordered]
    assert positions == sorted(positions)


def test_docs_do_not_present_cluster_execution_as_completed() -> None:
    text = (ROOT / "docs" / "cluster-migration.md").read_text(encoding="utf-8")
    assert "No cluster scientific job has been run by this repository reorganization" in text
```

Add exact command-literal assertions for profile validation, manifest verify,
submit dry-run, smoke, equivalence, and calibration. Add a scan that rejects
`D:\\WildFire Project`, `C:\\Users\\Ji`, real tokens, and Rorqual-specific
accounts in every newly tracked cluster file.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest `
  tests/test_project_research_organization.py tests/test_cluster_scripts.py -q
```

- [ ] **Step 3: Write `docs/cluster-migration.md`**

Use these sections:

```markdown
# Slurm cluster migration
## Scope and evidence boundary
## Persistent and node-local layout
## Clone the reviewed commit
## Create the site-local profile
## Qualify the environment
## Transfer and verify HDF5
## Fetch pinned upstream code and weights
## Run the one-batch smoke
## Run Fold-2 test-only equivalence
## Run the 500-step calibration
## Choose time limits from measured runtime
## Monitor, stop, and collect
## Inode hygiene
## Failure handling
```

Commands use generic `${PROJECT_ROOT}`-style shell variables only. Explain that
the first target cluster commit will add a verified training requirements lock
after module/PyTorch/CUDA qualification; do not silently reuse the Windows lock.

- [ ] **Step 4: Write `environments/README.md`**

Document two environments:

- audit: the `wildfire-phase0` package and deterministic rules;
- training: PyTorch and official/new model execution.

Freeze exact versions only after the target cluster smoke. Record Python,
PyTorch, CUDA runtime, driver, compiler, Lightning, NumPy, and GPU. Passing the
Fold-2 checkpoint gate establishes numerical executable equivalence, not
byte-identical hardware equivalence.

- [ ] **Step 5: Run GREEN and commit**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest `
  tests/test_project_research_organization.py tests/test_cluster_scripts.py -q
git diff --check
git add docs/cluster-migration.md environments/README.md `
  tests/test_project_research_organization.py tests/test_cluster_scripts.py
git commit -m "docs: add Slurm migration quick start"
```

---

### Task 6: Whole-branch review, integration, and authorized push

**Files:**
- Review all files changed since `e20aeb4`.
- Modify only files required to close reviewer findings.

**Interfaces:**
- Consumes all prior task deliverables.
- Produces a clean `main` commit that is safe to clone on the cluster.

- [ ] **Step 1: Run focused repository-organization tests**

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest `
  tests/test_project_research_organization.py `
  tests/test_cluster_profile.py `
  tests/test_cluster_manifest.py `
  tests/test_cluster_scripts.py -q
```

Expected: PASS with no Slurm or scientific process.

- [ ] **Step 2: Run the complete existing suite once**

```powershell
$env:PATH = 'C:\Users\Ji\miniconda3\condabin;C:\Users\Ji\miniconda3\Scripts;C:\Users\Ji\miniconda3;' + $env:PATH
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' -m pytest -q
```

Expected: all tests pass; the PATH prefix is required by the existing Conda
bootstrap contract and does not change project files.

- [ ] **Step 3: Run completion scans**

```powershell
git diff --check e20aeb4..HEAD
git status --short
git ls-files | ForEach-Object {
  $item = Get-Item -LiteralPath $_
  if ($item.Length -gt 5MB) { "{0}`t{1}" -f $item.Length,$_.ToString() }
}
rg -n -S '(hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE KEY|C:\\Users\\Ji|D:\\WildFire Project)' `
  README.md docs/cluster-migration.md docs/research-roadmap.md configs/cluster environments manifests scripts/cluster
```

Expected: no secret/local-path hits, no unexpected tracked file over 5 MiB,
clean diff, and only the intentional branch commits.

- [ ] **Step 4: Request independent code and research-scope review**

Reviewer scope:

```text
Review e20aeb4..HEAD. Report Critical/Important only.
Check: research-prototype scope, no deployment creep, profile fail-closed behavior,
one-GPU/array rendering, no retry, manifest exactness, inode policy, ignored heavy
state, roadmap claim boundaries, and absence of scientific execution.
```

Fix every confirmed Critical/Important finding with a focused RED/GREEN test and
one fix commit. Re-run focused tests and the full suite only when the fix can
affect shared behavior.

- [ ] **Step 5: Integrate into `main`**

Implementation begins in an isolated worktree/branch. After review:

```powershell
git -C 'D:\WildFire Project\wildfire-research-plan' status --short
git -C 'D:\WildFire Project\wildfire-research-plan' merge --ff-only cluster-research-prototype
```

If fast-forward is impossible, stop and inspect rather than creating an
unreviewed merge commit automatically.

- [ ] **Step 6: Verify `main` and push once**

```powershell
git status --short
git log -1 --oneline
git push origin main
git status --short
```

Expected: push succeeds, local `main` is clean, and no scientific process was
started. Report the pushed commit and the first five server commands, but do not
claim that cluster equivalence or training has already run.

---

## Plan Self-Review

- **Spec coverage:** repository entry point, roadmap, portable profile, thin
  Slurm boundary, environment distinction, inode-safe data, manifest, tests,
  Git integration, and push are each assigned to a task.
- **No deployment creep:** no service, database, scheduler, DDP, container
  platform, or artifact registry appears in any task.
- **No placeholder implementation:** the example profile intentionally uses
  dummy values; formal parsing rejects them. Actual function names, CLI names,
  schemas, test commands, commits, and production data totals are specified.
- **Type consistency:** Tasks 2–5 use the same `parse_profile`,
  `render_submit_command`, manifest, and CLI interfaces.
- **Scientific safety:** all local commands are tests, hashing, documentation,
  or Git operations. Actual cluster jobs remain after the pushed migration
  commit.
