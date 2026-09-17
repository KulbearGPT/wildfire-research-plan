# Verified evaluation populations

The earlier **2,312** figure for 2023 in the quantitative reliability ledger was a documentation error. The original completed B0/B2/B3/D1-ERM/D1-KL/D2-STD/D12-SARP summaries all report **2,102**, consistent with the cross-history population. No dataset change is needed to explain this discrepancy.

On 2026-09-16, all 21 original year/method summary JSON files were read without loading HDF5 or model weights. Every scenario in every method reports 3,181 samples for 2021, 2,856 for 2022, and 2,102 for 2023. All 84 M00/M01/M06/M07 AP entries reproduce the ledger's six-decimal values exactly. Only the population sentence was corrected; metric values and method conclusions were not changed.

## Original source directories

These paths identify historical evidence, not required runtime locations. The original root is `/project/6085198/kulbear/wildfire/runs`. Within each listed directory the inspected file is `results-YEAR/summary.json`.

| Method | 2021 source directory | 2022 source directory | 2023 source directory |
| --- | --- | --- | --- |
| B0 | `corrected-B0-S0-3K-21093263` | `heldout-B0-2022-21098505` | `heldout-B0-2023-21098506` |
| B2 | `corrected-B2-S0-3K-21093265` | `heldout-B2-2022-21098505` | `heldout-B2-2023-21098506` |
| B3 | `corrected-B3-S0-3K-21093266` | `heldout-B3-2022-21099907` | `heldout-B3-2023-21099907` |
| D1-ERM | `D1-erm-S0-3K-21102676` | `heldout-D1-ERM-2022-21105328` | `heldout-D1-ERM-2023-21105328` |
| D1-KL | `D1-kl-S0-3K-21102677` | `heldout-D1-KL-2022-21105328` | `heldout-D1-KL-2023-21105328` |
| D2-STD | `D2-standard-S0-3K-21094929` | `heldout-D2-STD-2022-21122938` | `heldout-D2-STD-2023-21122938` |
| D12-SARP | `D12-SARP-S0-3K-21122172` | `heldout-D12-SARP-2022-21122938` | `heldout-D12-SARP-2023-21122938` |

For all seven 2023 runs, each of the four scenarios has `sample_count: 2102` and `pixel_count: 34439168`, equal to 2,102 × 128 × 128. Their saved upstream commits are all `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`. The archived runner scripts select `hdf5/wstsplus-active-fixed`; the saved data summary records 999 HDF5 events, including 68 events in 2023. Event count and evaluated window count are different quantities.

## Statistics identity

Every inspected 2023 run's `inputs.sha256` records the same training-statistics SHA-256:

```text
4f6d308f00afcfcafcd14404dca9fff4f5beb89e1c2d0f615723f4d08b690f49
```

The original 824-byte `runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz` and the 824-byte transferred `handoff-20260916/public environment/data/train-2016-2020-stats.npz`, both beneath `/project/6085198/kulbear/wildfire`, were independently hashed during this check and both match that recorded value. These files were hashed as small metadata; no bulk data or checkpoint I/O occurred on login.

This verification resolves the false historical population mismatch. It does not by itself prove byte identity of all HDF5 files or claim a new model replay; fresh allocated evaluations must still compare their metrics with these recorded results. The archive directory inspected contains run artifacts rather than a separate historical HDF5 snapshot, but a separate snapshot is not needed to explain the erroneous 2,312 statement.
