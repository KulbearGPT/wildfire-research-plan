# Rule Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate deterministic T=1 no-fire and latest-day persistence baselines on the repaired frozen WSTS+ split with streaming, event-aware metrics and reproducible artifacts.

**Architecture:** Add count-based binary-score metrics so billions of pixels never need to be concatenated, then build an event evaluator that reads only consecutive active-fire frames from HDF5. Expose a read-only CLI that writes event metrics, split summaries, and a Markdown report atomically. Run it only after the repaired Phase 0 gate verifies nonzero labels in every year and split.

**Tech Stack:** Python 3.13, NumPy 2.x, pandas 2.x, h5py 3.x, scikit-learn only as an independent test oracle, pytest, standard-library `argparse`, `dataclasses`, `json`, and `pathlib`.

## Global Constraints

- Do not start until the active-fire repair plan finishes with 999 valid events, nonzero labels in all eight years and all three frozen splits, and Phase 0 status `continue_controlled`.
- Use the exact frozen split `2016--2020 train / 2021 validation / 2022--2023 test`; never create a random split.
- Evaluate T=1 rules only: target day `t+1`, latest persistence score from active-fire mask at day `t`, and no-fire score zero everywhere.
- Rule outputs are fixed binary scores. Do not tune thresholds, select variants, or fit any parameter on validation or test labels.
- Report event-macro AP over defined positive-target events, the defined and undefined event counts, pooled AP, pooled target prevalence, zero-target-day false-alarm rate, target days, and positive target pixels.
- Keep zero-positive events and days in artifacts; never silently discard them. AP for a zero-positive event is undefined, while its false alarms contribute to zero-target-day FAR.
- Stream counts per day/event. Do not concatenate a split's pixel arrays in memory and do not write pixel-level predictions.
- The evaluation CLI is read-only with respect to HDF5 input and writes only under a caller-supplied output root through atomic staged artifacts.
- Do not describe no-fire or persistence as learned models and do not compare raw AP across years without reporting prevalence.
- Commit implementation and verified real-data result separately; never commit HDF5 or ignored raw artifacts.

---

### Task 1: Streaming Binary-Score AP and Count Accumulator

**Files:**
- Modify: `src/wildfire_phase0/metrics.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Produces immutable `BinaryScoreCounts(tp: int, fp: int, fn: int, tn: int)` with `total`, `positives`, `predicted_positives`, `prevalence`, and additive `__add__`.
- Produces: `binary_score_counts(y_true: object, y_score: object) -> BinaryScoreCounts` for binary target and binary scores.
- Produces: `binary_average_precision(counts: BinaryScoreCounts) -> float`.
- Produces: `binary_false_alarm_rate(counts: BinaryScoreCounts) -> float` for zero-positive subsets.

- [ ] **Step 1: Write failing count and AP equivalence tests**

Add to `tests/test_metrics.py`:

```python
from sklearn.metrics import average_precision_score

from wildfire_phase0.metrics import (
    BinaryScoreCounts,
    binary_average_precision,
    binary_false_alarm_rate,
    binary_score_counts,
)


def test_binary_counts_add_without_retaining_pixels() -> None:
    first = binary_score_counts([1, 0], [1, 1])
    second = binary_score_counts([1, 0], [0, 0])
    assert first + second == BinaryScoreCounts(tp=1, fp=1, fn=1, tn=1)
    assert (first + second).total == 4
    assert (first + second).positives == 2


@pytest.mark.parametrize("seed", range(10))
def test_binary_average_precision_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=1000, dtype=np.uint8)
    scores = rng.integers(0, 2, size=1000, dtype=np.uint8)
    counts = binary_score_counts(target, scores)
    assert binary_average_precision(counts) == pytest.approx(
        average_precision_score(target, scores)
    )


def test_binary_average_precision_is_nan_without_positive_target() -> None:
    counts = binary_score_counts([0, 0], [1, 0])
    assert np.isnan(binary_average_precision(counts))
    assert binary_false_alarm_rate(counts) == 0.5
```

Add validation tests for non-binary score/target, mismatched or empty arrays,
negative dataclass counts, and FAR called with positive targets.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_metrics.py -v
```

Expected: import failure for the new count interfaces.

- [ ] **Step 3: Implement count metrics**

Use this AP definition for binary scores, matching scikit-learn's non-interpolated
average precision:

```python
@dataclass(frozen=True)
class BinaryScoreCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.tp, self.fp, self.fn, self.tn)):
            raise ValueError("binary score counts must be nonnegative")

    def __add__(self, other: "BinaryScoreCounts") -> "BinaryScoreCounts":
        if not isinstance(other, BinaryScoreCounts):
            return NotImplemented
        return BinaryScoreCounts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
        )

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def predicted_positives(self) -> int:
        return self.tp + self.fp

    @property
    def prevalence(self) -> float:
        return self.positives / self.total if self.total else float("nan")


def binary_average_precision(counts: BinaryScoreCounts) -> float:
    if counts.total == 0:
        raise ValueError("binary score counts must not be empty")
    if counts.positives == 0:
        return float("nan")
    recall_at_one = counts.tp / counts.positives
    precision_at_one = (
        counts.tp / counts.predicted_positives
        if counts.predicted_positives
        else 0.0
    )
    return (
        recall_at_one * precision_at_one
        + (1.0 - recall_at_one) * counts.prevalence
    )
```

`binary_score_counts` must reuse `_validated_arrays`, require score values in
`{0, 1}`, and compute confusion cells through NumPy boolean counts.

- [ ] **Step 4: Run focused and full tests**

```powershell
python -m pytest tests/test_metrics.py -v
python -m pytest -q
```

Expected: all tests pass and every randomized AP case matches scikit-learn.

- [ ] **Step 5: Commit the streaming metric core**

```powershell
git add src/wildfire_phase0/metrics.py tests/test_metrics.py
git commit -m "feat: add streaming binary baseline metrics"
```

---

### Task 2: Event-Level Rule Evaluation

**Files:**
- Create: `src/wildfire_phase0/rule_eval.py`
- Create: `tests/test_rule_eval.py`

**Interfaces:**
- Consumes `BinaryScoreCounts`, `binary_score_counts`, and `binary_average_precision` from Task 1.
- Produces `evaluate_rule_event(path: Path, data_root: Path, split: str) -> pandas.DataFrame` with exactly two records, `no_fire` and `persistence_latest`.
- Produces `evaluate_rule_dataset(data_root: Path, split_manifest: pandas.DataFrame) -> pandas.DataFrame` sorted by `(split, year, fire_name, baseline)`.
- Event record columns: `event_id`, `year`, `fire_name`, `path`, `split`, `baseline`, `target_days`, `zero_target_days`, `positive_target_pixels`, `total_pixels`, `tp`, `fp`, `fn`, `tn`, `event_ap`, `event_ap_defined`, `zero_target_pixels`, `zero_target_predicted_positive_pixels`.

- [ ] **Step 1: Write failing event-evaluation tests**

Create a three-day synthetic HDF5 where day masks are:

```text
day 0: [[1, 0], [0, 0]]
day 1: [[1, 1], [0, 0]]
day 2: [[0, 0], [0, 0]]
```

Add:

```python
def test_event_evaluation_uses_next_day_and_latest_persistence(tmp_path: Path) -> None:
    path = tmp_path / "2021" / "fire_a.hdf5"
    _write_active_event(path, masks)
    rows = evaluate_rule_event(path, tmp_path, "validation").set_index("baseline")
    persistence = rows.loc["persistence_latest"]
    assert persistence["target_days"] == 2
    assert persistence["zero_target_days"] == 1
    assert persistence["positive_target_pixels"] == 2
    assert (persistence["tp"], persistence["fp"], persistence["fn"]) == (1, 2, 1)
    assert persistence["zero_target_predicted_positive_pixels"] == 2
    no_fire = rows.loc["no_fire"]
    assert (no_fire["tp"], no_fire["fp"], no_fire["fn"]) == (0, 0, 2)


def test_zero_positive_event_keeps_row_and_marks_ap_undefined(tmp_path: Path) -> None:
    # All target masks are zero; an input mask may still contain fire.
    rows = evaluate_rule_event(path, tmp_path, "test")
    assert rows["event_ap_defined"].tolist() == [False, False]
    assert rows["event_ap"].isna().all()
```

Also test missing `data`, invalid channel values, n_days less than 2, manifest
path/event mismatch, and deterministic dataset ordering.

- [ ] **Step 2: Run tests to verify RED**

```powershell
python -m pytest tests/test_rule_eval.py -v
```

Expected: collection fails because `wildfire_phase0.rule_eval` does not exist.

- [ ] **Step 3: Implement streaming event evaluation**

For each event, open HDF5 read-only, validate required attrs and 23 channels,
then iterate target indices `1..n_days-1`. Keep only the preceding and current
active masks in memory:

```python
previous = np.asarray(data[0, 22]) > 0
for target_index in range(1, n_days):
    target = np.asarray(data[target_index, 22]) > 0
    latest = previous
    no_fire = np.zeros_like(target, dtype=np.uint8)
    # Accumulate per-baseline BinaryScoreCounts and zero-target counters.
    previous = target
```

Compute each event AP from its accumulated counts. Keep the event row even when
AP is NaN. Require split in `{train, validation, test}`. In dataset evaluation,
join through exact manifest `path`, `year`, `fire_name`, and `event_id`; reject
missing, duplicate, or extra manifest entries rather than guessing.

- [ ] **Step 4: Run focused and full tests**

```powershell
python -m pytest tests/test_rule_eval.py -v
python -m pytest -q
```

Expected: all tests pass with bounded per-event memory.

- [ ] **Step 5: Commit the event evaluator**

```powershell
git add src/wildfire_phase0/rule_eval.py tests/test_rule_eval.py
git commit -m "feat: evaluate wildfire rule baselines"
```

---

### Task 3: Split Summaries, Atomic Artifacts, and CLI

**Files:**
- Modify: `src/wildfire_phase0/rule_eval.py`
- Modify: `src/wildfire_phase0/cli.py`
- Modify: `tests/test_rule_eval.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/experiments/phase0.md`

**Interfaces:**
- Produces `summarize_rule_metrics(event_metrics: pandas.DataFrame) -> pandas.DataFrame`.
- Summary columns: `split`, `baseline`, `events`, `event_ap_defined`, `event_ap_undefined`, `event_macro_ap`, `pooled_ap`, `positive_prevalence`, `target_days`, `zero_target_days`, `zero_target_day_far`, `positive_target_pixels`, `total_pixels`.
- Produces `render_rule_report(summary: pandas.DataFrame) -> str`.
- Produces CLI: `python -m wildfire_phase0.cli evaluate-rules --data-root PATH --split-manifest PATH --output-root PATH`.
- Writes atomically: `rule_event_metrics.csv`, `rule_summary.csv`, `rule_report.md`.

- [ ] **Step 1: Write failing summary tests**

Build a small event frame containing one defined and one undefined persistence
event. Use a defined event with counts `tp=2, fp=1, fn=0, tn=5`,
`event_ap=2/3`, and no zero-target pixels; use an undefined event with counts
`tp=0, fp=1, fn=0, tn=3`, `event_ap=NaN`, four zero-target pixels, and one
zero-target predicted-positive pixel. Set `target_days=1` and
`zero_target_days=0` for the defined event, and `target_days=1` and
`zero_target_days=1` for the undefined event. Assert:

```python
summary = summarize_rule_metrics(frame).set_index(["split", "baseline"])
row = summary.loc[("test", "persistence_latest")]
assert row["events"] == 2
assert row["event_ap_defined"] == 1
assert row["event_ap_undefined"] == 1
assert row["event_macro_ap"] == pytest.approx(2 / 3)
assert row["pooled_ap"] == pytest.approx(
    binary_average_precision(BinaryScoreCounts(tp=2, fp=2, fn=0, tn=8))
)
assert row["zero_target_day_far"] == pytest.approx(1 / 4)
```

Add the analogous two no-fire rows with the same targets and zero predicted
positives. Test that every split/baseline group is sorted and that empty or
malformed frames are rejected with exact messages.

- [ ] **Step 2: Run summary tests to verify RED**

```powershell
python -m pytest tests/test_rule_eval.py -k "summary" -v
```

Expected: failure because summary interfaces do not exist.

- [ ] **Step 3: Implement summaries and deterministic Markdown**

Aggregate confusion counts by sum, pooled AP from summed counts, event-macro AP
from only `event_ap_defined=True`, and prevalence from total positives / total
pixels. Compute zero-target FAR from the two explicit pixel columns; return NaN
only when a group has no zero-target pixels. The report must state:

- fixed T=1 rules and next-day target semantics;
- frozen split definition;
- no threshold/model tuning;
- every summary column including undefined AP counts;
- warning that raw AP is prevalence-dependent.

- [ ] **Step 4: Add failing end-to-end CLI artifact test**

Create a synthetic three-split dataset and manifest. Call `cli.main` with
`evaluate-rules`; assert exit `0`, exact three artifact names, deterministic CSV
order, report text, and no `.tmp`/`.bak`. Reuse the existing publication failure
injection pattern to prove all three prior final artifacts roll back together.

- [ ] **Step 5: Implement CLI using existing artifact staging semantics**

Refactor only the minimal generic artifact helpers needed from `cli.py`; do not
change Phase 0 artifact behavior. The evaluate command must read the supplied
manifest CSV, run event evaluation, stage all outputs, publish them as one
generation, clean temps, and return `2` with no mixed final generation on data
validation failure.

Document the command in `docs/experiments/phase0.md` and state that generated
rule artifacts remain ignored local derivatives.

- [ ] **Step 6: Run full verification and commit**

```powershell
python -m pytest tests/test_rule_eval.py tests/test_cli.py -v
python -m pytest -q
git diff --check
git add src/wildfire_phase0/rule_eval.py src/wildfire_phase0/cli.py tests/test_rule_eval.py tests/test_cli.py docs/experiments/phase0.md
git commit -m "feat: add reproducible rule baseline evaluation"
```

---

### Task 4: Real Rule Baseline Run and Website Result

**Files:**
- Derived only: `artifacts/rule-baselines/*` (ignored local artifacts)
- Modify: `docs/experiments/phase0.md`
- Modify: `index.html`

**Interfaces:**
- Consumes repaired `D:\WildFire Project\data\hdf5` and `artifacts\phase0\split_manifest.csv`.
- Produces verified T=1 no-fire and persistence results for train, validation, and untouched test.

- [ ] **Step 1: Verify prerequisites fresh**

Require:

- clean repository worktree;
- active Phase 0 report covers 999 events and returns `continue_controlled`;
- split manifest counts `653/156/190`;
- every year and split has nonzero positive target pixels;
- retained pre-repair backup still exists.

Do not run baselines when any prerequisite is indirect or missing.

- [ ] **Step 2: Run the real evaluation once**

```powershell
python -m wildfire_phase0.cli evaluate-rules `
  --data-root 'D:\WildFire Project\data\hdf5' `
  --split-manifest 'artifacts\phase0\split_manifest.csv' `
  --output-root 'artifacts\rule-baselines'
```

Expected exit `0` and exactly three complete artifacts.

- [ ] **Step 3: Independently recompute result invariants**

Read the event and summary CSVs with a separate verification script and require:

- 999 events x 2 baselines = 1,998 event rows;
- exact event counts `653/156/190` for both baselines;
- no-fire `tp=0`, `fp=0`, and zero-target FAR `0` in every split;
- persistence confusion totals sum to total pixels;
- pooled positive prevalence matches Phase 0 target totals;
- undefined event count equals the event rows with zero positive targets;
- every reported AP is within `[0, 1]` or explicitly NaN where undefined;
- no duplicate `(event_id, baseline)` rows and no `.tmp`/`.bak` files.

If any invariant fails, fix code with a regression test and a new commit before
rerunning the real evaluation.

- [ ] **Step 4: Publish the baseline result with claim boundaries**

Update `docs/experiments/phase0.md` and the bilingual Phase 0/experiment section
in `index.html` with the fresh summary values. Include:

- T=1 no-fire and latest persistence definitions;
- split-wise event-macro AP with defined/undefined event counts;
- pooled AP, prevalence, and zero-target FAR;
- no threshold tuning and untouched-test statement;
- a warning that these are deterministic reference rules, not learned-model
  performance and not evidence for natural-missingness or operational claims.

- [ ] **Step 5: Run fresh repository and website verification**

```powershell
python -m pytest -q
git diff --check
node -e "const fs=require('fs'),vm=require('vm');const s=fs.readFileSync('index.html','utf8');const m=s.match(/<script>([\\s\\S]*?)<\\/script>/);if(!m)throw new Error('script missing');new vm.Script(m[1]);"
git status --short
```

Expected: green suite, clean static checks, and only intended documentation
files modified.

- [ ] **Step 6: Commit the verified result**

```powershell
git add docs/experiments/phase0.md index.html
git commit -m "docs: publish rule baseline results"
```

Do not push, merge, delete the retained backup, or start a learned model without
explicit user direction.
