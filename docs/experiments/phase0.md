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

## Full eight-year audit result (2026-08-07)

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
