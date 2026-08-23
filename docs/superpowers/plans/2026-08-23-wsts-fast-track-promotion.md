# WSTS+ Fast-Track Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a typed WSTS+ follow-on matrix and a read-only promotion CLI that unlocks fresh C00/C02 seed-0 10,000-step commands only after both 3,000-step screening records pass.

**Architecture:** Keep model choices in the existing `ExperimentSpec`, add immutable run/corruption registries, and let the current entrypoint resolve declared follow-on run IDs without accepting arbitrary seeds or budgets. A promotion module validates prerequisite `completed.json` files and emits deterministic, non-overwriting manifests for the existing cluster layer; it never submits a job.

**Tech Stack:** Python 3.13 control environment, Python 3.10 scientific environment, standard-library `argparse`, `dataclasses`, `json`, `math`, and `pathlib`, pytest, existing PyTorch Lightning adapter.

**Spec:** `docs/superpowers/specs/2026-08-23-wsts-fast-track-promotion-design.md`

## Global Constraints

- Preserve `entrypoint --experiment {C00,C02}` and its default 3,000-step command rendering.
- Both seed-0 promotions train fresh for exactly 10,000 steps; no resume checkpoint is accepted.
- Train on 2016--2020, validate on 2021, set `do_test=false`, and withhold 2022--2023.
- Require both passing 3K results for either seed-0 promotion; never select by AP rank.
- Declare seed 1/2 and corruption scenarios but keep them non-launchable.
- Refuse existing output paths; never call `sbatch` or retry a run.

---

### Task 1: Add the Typed Matrices

**Files:**
- Create: `reproductions/wsts_fast_track/matrix.py`
- Modify: `tests/test_wsts_fast_track.py`

**Interfaces:**
- Produces: `RunSpec`, `CorruptionSpec`, `CLEAN_RUNS`, `CORRUPTIONS`, `run_spec(run_id: str) -> RunSpec`, and `matrix_payload() -> dict[str, object]`.
- Consumes: base experiment IDs `C00` and `C02`.

- [ ] **Step 1: Write failing exact-matrix tests**

```python
from reproductions.wsts_fast_track import matrix

def test_follow_on_matrix_has_exact_clean_runs() -> None:
    assert tuple(matrix.CLEAN_RUNS) == (
        "C00-S0-3K", "C02-S0-3K", "C00-S0-10K", "C02-S0-10K",
        "C00-S1-10K", "C00-S2-10K", "C02-S1-10K", "C02-S2-10K",
    )
    promoted = matrix.run_spec("C00-S0-10K")
    assert (promoted.seed, promoted.max_steps) == (0, 10_000)
    assert promoted.prerequisites == ("C00-S0-3K", "C02-S0-3K")
    assert promoted.launch_state == "promotable"

def test_follow_on_matrix_has_exact_corruptions() -> None:
    assert tuple(matrix.CORRUPTIONS) == tuple(f"M{i:02d}" for i in range(8))
    assert matrix.CORRUPTIONS["M01"].feature_indices == (22,)
    assert matrix.CORRUPTIONS["M03"].feature_indices == tuple(range(5, 12))
    assert matrix.CORRUPTIONS["M04"].feature_indices == tuple(range(17, 22))
    assert matrix.CORRUPTIONS["M06"].severity == "area=0.25"
    assert matrix.CORRUPTIONS["M07"].severity == "area=0.50"
```

- [ ] **Step 2: Run RED**

Run the audit-environment pytest on `tests/test_wsts_fast_track.py`. Require collection failure because `matrix.py` is absent.

- [ ] **Step 3: Implement immutable registries**

Use these exact frozen dataclass fields:

```python
class RunSpec:
    run_id: str
    experiment_id: str
    stage: Literal["screening", "promotion", "replication"]
    seed: int
    max_steps: int
    prerequisites: tuple[str, ...]
    launch_state: Literal["active", "promotable", "gated"]

class CorruptionSpec:
    scenario_id: str
    condition: str
    family: str
    feature_indices: tuple[int, ...]
    severity: str
    matrix_seed: int
    implementation_state: Literal["declared"]
```

Populate the eight rows from the spec. M06/M07 use raw dynamic indices `tuple(range(12)) + (15,) + tuple(range(17, 23))`. Strict lookup raises `ValueError("unknown fast-track run")`. `matrix_payload` uses `asdict`, schema version 1, and registry order.

- [ ] **Step 4: Run GREEN and commit**

Run the focused test file, then commit matrix and tests as `feat: declare WSTS follow-on experiment matrix`.

---

### Task 2: Render Declared Seeds and Budgets

**Files:**
- Modify: `reproductions/wsts_fast_track/contract.py`
- Modify: `reproductions/wsts_fast_track/entrypoint.py`
- Modify: `tests/test_wsts_fast_track.py`

**Interfaces:**
- Consumes: `matrix.run_spec`.
- Produces: `upstream_arguments(..., *, max_steps: int = 3000, seed: int | None = None) -> list[str]` and `resolve_run(experiment_id: str | None, run_id: str | None) -> tuple[ExperimentSpec, int, int | None]`.

- [ ] **Step 1: Write RED compatibility and promotion tests**

Capture a default C00/C02 argument list before the change and assert the default call remains identical. Add:

```python
def test_declared_promotion_renders_seed_and_10k_without_test() -> None:
    spec, max_steps, seed = entrypoint.resolve_run(None, "C02-S0-10K")
    arguments = contract.upstream_arguments(
        spec, Path("/upstream"), Path("/data"), Path("/run"),
        max_steps=max_steps, seed=seed,
    )
    assert "--seed_everything=0" in arguments
    assert "--trainer.max_steps=10000" in arguments
    assert "--do_test=false" in arguments
    assert "--model.class_path=models.SMPTempModel" in arguments
```

Parametrize the four seed 1/2 IDs and require resolved seed in `{1, 2}` and 10,000 steps. Require two selectors and zero selectors to raise `ValueError`.

- [ ] **Step 2: Run RED**

Run only new test names. Require `TypeError` for new keywords or `AttributeError` for missing `resolve_run`.

- [ ] **Step 3: Extend command rendering minimally**

Validate `max_steps` is a positive integer and `seed` is `None` or a nonnegative integer. Format the max-step argument. Insert `--seed_everything={seed}` only for non-`None`; preserve the exact default list and order.

- [ ] **Step 4: Resolve declared runs in the entrypoint**

Use an argparse mutually exclusive required group for `--experiment` and `--run-id`. Legacy experiment returns its spec, 3,000, `None`; run ID returns matrix experiment, budget, seed. Pass resolved values to `upstream_arguments` and keep runtime/test locks unchanged.

- [ ] **Step 5: Run GREEN, import help, and commit**

Run focused tests and scientific-environment `python -m reproductions.wsts_fast_track.entrypoint --help`. Require both selectors in help. Commit as `feat: resolve declared fast-track runs`.

---

### Task 3: Add the Promotion Gate and Manifest CLI

**Files:**
- Create: `reproductions/wsts_fast_track/promotion.py`
- Create: `tests/test_wsts_fast_track_promotion.py`

**Interfaces:**
- Produces: `load_completed(path: Path) -> dict[str, object]`, `validate_prerequisites(target: RunSpec, paths: Sequence[Path]) -> tuple[dict[str, object], ...]`, `promotion_manifest(...) -> dict[str, object]`, and `write_json_new(path: Path, payload: Mapping[str, object]) -> None`.
- Produces CLI: `matrix --output PATH` and `render --run-id ID --result PATH --result PATH --upstream-root PATH --data-root PATH --run-root PATH --stats-path PATH --output PATH`.

- [ ] **Step 1: Write a passing synthetic RED test**

Use exact completed records with `status=pass`, experiment C00/C02, max steps 3,000, seed 0, train years 2016--2020, validation `[2021]`, `test_enabled=false`, withheld `[2022,2023]`, positive CUDA bytes/checkpoint step, nonempty checkpoint/job ID, and finite `val_avg_precision`, `val_f1`, `val_loss`. Assert a C00 seed-0 10K manifest has schema 1, target seed 0/10K, ordered C00/C02 prerequisites, and a command containing `--run-id C00-S0-10K`.

- [ ] **Step 2: Run RED**

Run the new test file and require import failure because `promotion.py` is absent.

- [ ] **Step 3: Implement strict result validation**

Require the exact completed-record key set. Require exact status, experiment, steps, seed, splits, test flag, and withheld years. Require finite AP/F1/loss, AP/F1 in `[0,1]`, positive CUDA bytes, nonempty checkpoint/job ID, and checkpoint step in `1..3000`. Normalize C00 then C02. Reject missing, duplicate, or extra records.

- [ ] **Step 4: Implement deterministic manifests**

Allow only `launch_state == "promotable"`. Emit this command shape:

```python
[
    "python", "-m", "reproductions.wsts_fast_track.entrypoint",
    "--upstream-root", str(upstream_root.resolve()),
    "--run-id", target.run_id,
    "--data-root", str(data_root.resolve()),
    "--run-root", str(run_root.resolve()),
    "--stats-path", str(stats_path.resolve()),
]
```

Include `schema_version`, `run=asdict(target)`, compact prerequisite identities/metrics, and exact split/test boundary. Rendering does not require runtime paths to exist.

- [ ] **Step 5: Write all gate mutation tests**

Parametrize failed status, wrong experiment/steps/seed/splits, enabled test, wrong withheld years, NaN/Inf metrics, AP/F1 out of range, zero CUDA bytes, empty checkpoint/job ID, invalid checkpoint step, duplicate C00, missing C02, and extra third result. Require field-specific `ValueError`. Require seed 1/2 and unknown targets to fail rendering.

- [ ] **Step 6: Implement non-overwriting CLI output**

Create parents, open output with `"x"`, write sorted indent-2 JSON plus LF, and preserve an existing file. The CLI only writes matrix or promotion-manifest JSON and never invokes Slurm.

- [ ] **Step 7: Run GREEN and commit**

Run both fast-track test files. Commit promotion module/tests as `feat: gate WSTS 10K promotion manifests`.

---

### Task 4: Publish and Verify the Workflow

**Files:**
- Modify: `reproductions/wsts_fast_track/README.md`
- Modify: `docs/research-roadmap.md`
- Test: `tests/test_wsts_fast_track.py`
- Test: `tests/test_wsts_fast_track_promotion.py`
- Test: `tests/test_project_research_organization.py`

**Interfaces:**
- Consumes: Tasks 1--3.
- Produces: reviewed matrix/export/render instructions; no job submission.

- [ ] **Step 1: Document matrices and states**

Add all eight clean IDs and M00--M07 from the spec. Label seed-0 10K promotable only after both 3K records pass, seed 1/2 gated, and corruption scenarios declared/non-launchable.

- [ ] **Step 2: Document both read-only commands**

Show `python -m reproductions.wsts_fast_track.promotion matrix --output ...` and a complete `render` example with two result paths plus upstream/data/run/stats paths. State that an operator reviews the manifest and passes it explicitly to the existing cluster layer.

- [ ] **Step 3: Update the roadmap**

Link Stage 2 to the matrix, freeze fresh matched 10K seed-0 promotion for both controls, name seed 1/2 as gated replications, and link Stage 3 to M00--M07 while preserving label-independent test-only scope.

- [ ] **Step 4: Run final focused verification**

Run audit-environment pytest for both fast-track files plus `tests/test_project_research_organization.py`, then `git diff --check` and `git status --short --branch`. Require all tests green and only intended changes.

- [ ] **Step 5: Commit and preserve execution boundary**

Commit README, roadmap, and this plan as `docs: publish WSTS promotion workflow`. Do not submit any 10K job. Report code/tests/commits separately from the still-active 3K Nibi chain.
