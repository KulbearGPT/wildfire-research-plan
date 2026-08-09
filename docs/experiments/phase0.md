# Phase 0 data gate

Run the reproducible audit from the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
$wstsDataRoot = (Resolve-Path -LiteralPath $env:WSTSPLUS_DATA_ROOT).Path
python -m wildfire_phase0.cli audit --data-root "$wstsDataRoot" --output-root artifacts\phase0
```

The target is the `next-calendar-day active-fire proxy`. A `continue_controlled`
decision preserves the approved project route but removes natural-missingness and
operational claims.

The command writes four derived artifacts under the requested output root:

- `inventory.csv` is the deterministic event metadata and NaN inventory.
- `split_manifest.csv` is the frozen temporal event split.
- `contract_decision.json` is the machine-readable contract decision.
- `phase0_report.md` is the human-readable gate report and command record.

## Full eight-year audit result (2026-08-08)

The verified source archives were converted to event-level HDF5 before the
audit. The original WSTS archive contributed 607 valid events from 2018--2021.
The WSTS+ extension archive contained 397 event directories from 2016, 2017,
2022, and 2023; five 2022 directories contained no GeoTIFF files, leaving 392
valid added-year events. The final audit therefore covers 999 valid events.

| split | years | events |
| --- | --- | ---: |
| train | 2016--2020 | 653 |
| validation | 2021 | 156 |
| test | 2022--2023 | 190 |

The added-year conversion had treated source values that were already integer
detection hours as HHMM values and divided them by 100 a second time, zeroing
the active-fire target channel. Active-fire repair version `1` restored the
four affected years. An independent verifier opened all 392 repaired files,
and a separate all-year invariant scan validated all 999 active files before
the audit was rerun.

### Corrected target counts by year

| year | events | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| 2016 | 92 | 2,102 | 886 | 303,648 |
| 2017 | 110 | 2,490 | 1,481 | 177,972 |
| 2018 | 176 | 3,597 | 1,373 | 381,935 |
| 2019 | 74 | 1,351 | 642 | 45,021 |
| 2020 | 201 | 4,091 | 1,615 | 670,920 |
| 2021 | 156 | 3,961 | 1,332 | 768,324 |
| 2022 | 122 | 3,424 | 2,158 | 105,377 |
| 2023 | 68 | 2,442 | 1,297 | 167,952 |

### Corrected target counts by frozen split

| split | events | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| train | 653 | 13,631 | 5,997 | 1,579,496 |
| validation | 156 | 3,961 | 1,332 | 768,324 |
| test | 190 | 5,866 | 3,455 | 273,329 |

The target-integrity gate now requires exactly years 2016--2023, exactly the
`train`/`validation`/`test` split labels, and the canonical mapping
`2016--2020 / 2021 / 2022--2023`; it rejects any missing, remapped, or
zero-positive group. All eight years and all three splits pass that strengthened
gate in the fresh report.

The pixel-weighted dataset NaN fraction is `0.0166683592232`, and the maximum
single-event NaN fraction is `0.276663755051`. All 607 original WSTS events
retain finite GeoTIFF-derived longitude and latitude. Only one of the 392 valid
added-year events has finite ROI coordinates, so ROI provenance remains an
operational blocker for the extension years.

The field contract is missing `observation_time`, `availability_time`, `qa`,
`coverage`, and `target_validity`. The resulting decision is
`continue_controlled`, with `availability_time`, `event_roi_provenance`, and
`target_validity` recorded as blockers. This does not change the approved model
route or temporal split. It limits the main claims to prespecified controlled
missingness and excludes natural-missingness and operational-deployment claims
unless the missing provenance is recovered from upstream products.

The generated CSV, JSON, and Markdown artifacts remain local derived data under
`artifacts/phase0/`; they can be regenerated with the command above and are not
committed to the repository.

## Fixed rule baseline evaluation

Evaluate the prespecified no-fire and latest-mask persistence rules at fixed
`T=1` against the next-day target using the frozen audit split manifest:

```powershell
if ([string]::IsNullOrWhiteSpace($env:WSTSPLUS_DATA_ROOT)) {
  throw 'WSTSPLUS_DATA_ROOT must name the active HDF5 data directory'
}
$wstsDataRoot = (Resolve-Path -LiteralPath $env:WSTSPLUS_DATA_ROOT -ErrorAction Stop).Path
python -m wildfire_phase0.cli evaluate-rules `
  --data-root "$wstsDataRoot" `
  --split-manifest artifacts\phase0\split_manifest.csv `
  --output-root artifacts\rule-baselines
```

The command reads the HDF5 inputs without modifying them and publishes
`rule_event_metrics.csv`, `rule_summary.csv`, and `rule_report.md` together as
one artifact generation. It performs no threshold or model tuning. The report
retains counts of events whose AP is undefined and warns that raw AP depends on
positive prevalence. These rule-baseline artifacts are reproducible, ignored
local derivatives and are not committed to the repository.

### Measured fixed-rule result (2026-08-09)

The fresh audit passed all frozen prerequisites before evaluation: 999 events,
`continue_controlled`, split counts `653/156/190`, and nonzero positive target
pixels in every year and split. The evaluation was then run once. `no_fire`
assigns score zero to every target pixel. `persistence_latest` copies the most
recent active-fire mask forward as the fixed `T=1` next-day score.

| split | rule | events | event-macro AP | AP defined / undefined | pooled AP | positive prevalence | zero-target FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | no_fire | 653 | 0.001156603331 | 626 / 27 | 0.001367787980 | 0.001367787980 | 0 |
| train | persistence_latest | 653 | 0.074670022088 | 626 / 27 | 0.126383035985 | 0.001367787980 | 0.000127607813 |
| validation | no_fire | 156 | 0.001885736126 | 156 / 0 | 0.002529419492 | 0.002529419492 | 0 |
| validation | persistence_latest | 156 | 0.143628059298 | 156 / 0 | 0.283867881947 | 0.002529419492 | 0.000268773985 |
| test | no_fire | 190 | 0.000530785830 | 174 / 16 | 0.000458901563 | 0.000458901563 | 0 |
| test | persistence_latest | 190 | 0.070538373867 | 174 / 16 | 0.105814987028 | 0.000458901563 | 0.000063380673 |

Event-macro AP excludes events with no positive target pixels; the table reports
those defined and undefined counts explicitly. Pooled AP and prevalence aggregate
all pixels within each split. Zero-target FAR is the fraction of predicted-positive
pixels over pixels on zero-target days. All event rows, split totals, confusion
totals, Phase 0 target totals, undefined-AP counts, AP ranges, and publication
files were independently recomputed from the CSV artifacts after the run.

No threshold or model tuning was performed. The 2022--2023 test split remained
untouched by threshold or model selection and was evaluated only after the
protocol and prerequisites were frozen. These are deterministic reference rules,
not learned-model performance. They do not provide evidence for
natural-missingness performance or operational deployment, and raw AP must not
be compared across splits without accounting for prevalence.

## Active-fire repair operator workflow

The added-year source GeoTIFFs encode positive active-fire pixels as detection
hours, but the original conversion double-converted those hour values and
incorrectly zeroed the HDF5 targets. Repair version `1` records the detected
source encoding and stores finite integer hours in `[0, 23]`.
The repair command writes only to the caller-supplied staging root. It does not
activate staged files and does not delete, overwrite, rename, or otherwise
modify an active HDF5 year directory.
`active_fire_repair_manifest.csv` contains one deterministic row for every
matched success or failure and every unmatched nonempty source/HDF5 event. Its
`source_event_dir` column preserves source provenance; an empty source-HDF5 or
staged-HDF5 cell means that file is actually absent.
`active_fire_repair_decision.json` is published last as the commit marker for
one evidence generation. Its `manifest_sha256` binds it to the exact manifest
bytes, and its `generation` binds that digest to the canonical decision data.
Never activate from either file in isolation.

Stage the four affected years with operator-resolved paths:

```powershell
python -m wildfire_phase0.cli repair-active-fire `
  --source-tiff-root "$wstsPlusTiffRoot" `
  --hdf5-root "$wstsHdf5Root" `
  --staging-root "$repairStagingRoot" `
  --years 2016 2017 2022 2023
```

Independently verify the staged tree against raw TIFF samples and the complete
source-derived totals:

```powershell
python -m wildfire_phase0.verify_repair `
  --hdf5-root "$repairStagingRoot" `
  --source-tiff-root "$wstsPlusTiffRoot" `
  --expect 2016:92:2102:886:303648 `
  --expect 2017:110:2490:1481:177972 `
  --expect 2022:122:3424:2158:105377 `
  --expect 2023:68:2442:1297:167952
```

Activation is blocked until every hard precondition below is satisfied:

- `verify_repair_evidence` confirms one committed manifest/decision generation,
  and the verified decision has status `ready`.
- Its sorted `excluded_empty_source_directories` value contains exactly these
  five IDs and no others:

  - `2022/fire_CA4186812327820220730`
  - `2022/fire_ID4570411652620220904`
  - `2022/fire_OR4513211711020220825`
  - `2022/fire_WA4687912083320220803`
  - `2022/fire_WA4796412068520220909`

- The staging root contains exactly 392 staged HDF5 files, zero recorded
  errors, zero `.tmp` files, and no interrupted evidence transaction journal.
- Every staged `data` dataset uses LZF compression with shuffle enabled.
- The verifier confirms these exact counts:

  | year | files | target days | zero-target days | positive target pixels |
  | --- | ---: | ---: | ---: | ---: |
  | 2016 | 92 | 2,102 | 886 | 303,648 |
  | 2017 | 110 | 2,490 | 1,481 | 177,972 |
  | 2022 | 122 | 3,424 | 2,158 | 105,377 |
  | 2023 | 68 | 2,442 | 1,297 | 167,952 |

- The four active year paths, their four staged replacements, and the exact
  backup root `hdf5-active-fire-bug-backup` are individually resolved and
  checked to be on the same volume.

Verify the evidence generation, then compare its decision exclusions in
deterministic sorted order before any activation rename:

```powershell
$verifiedDecisionJson = @'
from dataclasses import asdict
import json
import sys
from pathlib import Path

from wildfire_phase0.repair import verify_repair_evidence

print(json.dumps(asdict(verify_repair_evidence(Path(sys.argv[1]))), sort_keys=True))
'@ | python - $repairStagingRoot
if ($LASTEXITCODE -ne 0) {
  throw 'repair manifest and decision do not form a committed evidence generation'
}
$decision = $verifiedDecisionJson | ConvertFrom-Json
$expectedExclusions = @(
  '2022/fire_CA4186812327820220730',
  '2022/fire_ID4570411652620220904',
  '2022/fire_OR4513211711020220825',
  '2022/fire_WA4687912083320220803',
  '2022/fire_WA4796412068520220909'
)
$actualExclusions = @($decision.excluded_empty_source_directories)
$exclusionDiff = Compare-Object -ReferenceObject $expectedExclusions -DifferenceObject $actualExclusions -SyncWindow 0
if ($actualExclusions.Count -ne $expectedExclusions.Count -or $null -ne $exclusionDiff) {
  throw 'empty-source exclusions differ from the frozen activation precondition'
}
```

Activation is a separate, explicit same-volume directory rename for each of
the four years: rename each active year into
`hdf5-active-fire-bug-backup`, then rename its staged replacement into the
active location. Retain the backup through the corrected audit and baseline
smoke run. Rollback reverses those exact renames: move the activated year back
to its staging location and restore the corresponding retained backup year to
the active location. Do not proceed if any resolved path, destination, count,
or volume check differs from the preflight record.

After activation, regenerate the Phase 0 artifacts and require a non-blocked
decision containing exactly years 2016--2023, the canonical
`2016--2020 train / 2021 validation / 2022--2023 test` mapping, and nonzero
positive target pixels in all eight years and all three splits:

```powershell
python -m wildfire_phase0.cli audit `
  --data-root "$wstsHdf5Root" `
  --output-root artifacts\phase0
```

Task 5 completed those explicit activation and corrected-audit steps on
2026-08-08. The independently verified replacements are now active, the
pre-repair year directories remain in a local recoverable backup, and the
generated audit artifacts remain local and ignored. The repair changes the
label evidence only: the approved research route and frozen temporal split are
unchanged. Model experiments must use the repaired active data.
