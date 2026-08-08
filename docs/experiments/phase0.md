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
