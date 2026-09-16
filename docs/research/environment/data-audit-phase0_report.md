# Phase 0 Data Gate

## Target definition

next-calendar-day active-fire proxy

## Frozen split

2016–2020 train / 2021 validation / 2022–2023 test

### Target counts by year

| year | events | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| 2016 | 92 | 2102 | 886 | 303648 |
| 2017 | 110 | 2490 | 1481 | 177972 |
| 2018 | 176 | 3597 | 1373 | 381935 |
| 2019 | 74 | 1351 | 642 | 45021 |
| 2020 | 201 | 4091 | 1615 | 670920 |
| 2021 | 156 | 3961 | 1332 | 768324 |
| 2022 | 122 | 3424 | 2158 | 105377 |
| 2023 | 68 | 2442 | 1297 | 167952 |

### Target counts by split

| split | events | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| test | 190 | 5866 | 3455 | 273329 |
| train | 653 | 13631 | 5997 | 1579496 |
| validation | 156 | 3961 | 1332 | 768324 |

## Inventory validation

Invalid-file errors: none

## NaN summary

Events scanned: 999
Pixel-weighted dataset NaN fraction: 0.0166683592232
Maximum event NaN fraction: 0.276663755051

## Contract audit

Missing contract fields: observation_time, availability_time, qa, coverage, target_validity

Operational blockers: availability_time, event_roi_provenance, target_validity

Notes: Natural missingness is not recoverable from this contract.

## Gate decision

Status: continue_controlled

Implication: continue_controlled preserves the approved project route but removes natural-missingness and operational claims.

## Reproducible commands used

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m wildfire_phase0.cli audit --data-root "/project/6085198/kulbear/wildfire/handoff-20260916/public environment/data/wstsplus-active-fixed" --output-root "/project/6085198/kulbear/wildfire/handoff-20260916/public environment/qualification/data-audit"
```
