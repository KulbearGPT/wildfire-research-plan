# Active-Fire Label Repair and Target Gate Design

**Date:** 2026-08-08  
**Status:** Approved design, awaiting written-spec review  
**Scope:** Repair the four WSTS+ added-year HDF5 label channels, prevent recurrence in Phase 0, rerun the data gate, and then evaluate deterministic rule baselines. The approved WSTS+ research route and frozen temporal split do not change.

## Problem and evidence

The current event HDF5 files combine two source encodings for channel 22
(`active fire`):

- Original WSTS GeoTIFFs from 2018--2021 store positive detection times as
  HHMM values, observed in the range `806--2200`. Converting these values to
  hours with floor division by 100 is correct.
- WSTS+ added-year GeoTIFFs from 2016, 2017, 2022, and 2023 already store
  positive detection times as integer hours, observed exhaustively in the
  range `6--22`.

The environment-local converter applied floor division by 100 to both forms.
As a result, every positive label in the four added years became zero. The
structural Phase 0 audit did not detect the problem because file shape,
attributes, dates, and NaN fractions remained valid.

An independent comparison of representative raw and converted events found
that channels 0--21, spatial shapes, and dates are unchanged and exact. Only
channel 22 requires repair. The 392 affected HDF5 files occupy 22.31 GiB, and
the data volume currently has more than 250 GiB free.

## Goals

1. Repair channel 22 for all valid added-year events without modifying the
   currently active derived copy in place.
2. Make source active-fire encoding explicit and reject ambiguous or mixed
   positive values.
3. Add a target-integrity gate so an all-zero benchmark year or split cannot
   pass Phase 0 again.
4. Preserve event identities, non-label channels, compression, dates, ROI
   attributes, and the frozen split.
5. Produce deterministic, machine-readable repair and audit evidence before
   running no-fire and persistence baselines.

## Non-goals

- Do not change the forecasting target, temporal split, backbone, gating idea,
  or ten-month research route.
- Do not recover missing acquisition time, availability time, QA, coverage,
  target validity, or ROI provenance in this change.
- Do not train a neural model in this change.
- Do not treat the five empty 2022 source directories as valid events.

The five empty 2022 source directories are recorded as explicit nonblocking
exclusions in the repair decision. A nonempty source event without a matching
HDF5 file, or a source HDF5 file without a matching nonempty event directory,
is a blocking repair error.

## Chosen approach

Use a staged, event-atomic channel repair rather than rebuilding every channel
or modifying files in place.

For each affected event:

1. Validate that the source TIFF dates and the HDF5 `img_dates` attribute are
   identical and in the same order.
2. Copy the existing HDF5 file to a sibling staging root.
3. Read only active-fire band 23 from each source TIFF.
4. Validate and normalize its encoding.
5. Replace only `data[day, 22, :, :]` in the staged HDF5 file.
6. Record explicit source and stored encodings as dataset attributes.
7. Flush, reopen, and validate the staged event before continuing.

The existing four year directories remain untouched until every staged event
passes validation. Final activation uses same-volume directory renames per
year. The old directories move to the deterministic backup root
`hdf5-active-fire-bug-backup` and remain available until the repaired dataset,
Phase 0 audit, and baseline smoke evaluation all pass.

This is faster than a complete 23-channel rebuild because channels 0--21 are
already verified and are copied rather than decoded from 10,850 GeoTIFF files.
It is safer than an in-place patch because activation is delayed until the
staging tree is complete and validated.

## Encoding rules

Encoding is inferred per event from all finite positive channel-22 values:

- No positive values: preserve zeros and mark the source encoding as
  `no_positive_values`.
- Every positive value is an integer in `[1, 23]`: source encoding is `hour`;
  preserve the values.
- Every positive value is greater than 23 and is a valid HHMM value with hour
  in `[0, 23]` and minute in `[0, 59]`: source encoding is `hhmm`; convert
  using floor division by 100.
- Positive values spanning both `[1, 23]` and values greater than 23 are
  ambiguous mixed encoding and are rejected.
- Any non-integer, negative, out-of-range, or mixed invalid representation:
  reject the event and leave the active dataset unchanged.

The stored HDF5 representation is always integer-valued hour data in `[0, 23]`
with source NaNs replaced by zero, matching the downstream official loader.
Dataset attributes added by the repair are:

- `active_fire_source_encoding`: `hour`, `hhmm`, or `no_positive_values`.
- `active_fire_stored_encoding`: always `hour`.
- `active_fire_repair_version`: a stable version string defined by the package.

## Repository interfaces

Add a focused preprocessing module under `src/wildfire_phase0` and expose it
through the existing CLI. The operator interface is:

```powershell
python -m wildfire_phase0.cli repair-active-fire `
  --source-tiff-root PATH `
  --hdf5-root PATH `
  --staging-root PATH `
  --years 2016 2017 2022 2023
```

The command never activates or deletes directories. It writes staged HDF5
files plus `active_fire_repair_manifest.csv` and
`active_fire_repair_decision.json`. Activation remains a separate, explicit
operator step after validation so a repair command cannot silently replace the
benchmark.

The implementation uses `tifffile` and `imagecodecs` for TIFF decoding and
adds compatible bounded dependencies to `pyproject.toml`. It does not add
`rasterio` as a package dependency because geospatial transforms are not
needed to restore a single existing band.

## Target-integrity gate

Extend the Phase 0 inventory evidence with label statistics computed while
event data are already being scanned:

- `positive_target_pixels`
- `target_days`
- `zero_target_days`
- `active_fire_min_positive`
- `active_fire_max_positive`

The report aggregates these fields by year and split. Phase 0 returns
`blocked` when any expected benchmark year or any frozen split has zero
positive target pixels, when stored positive active-fire values fall outside
`[1, 23]`, or when the target channel cannot be read. Individual events and
days may legitimately have no positives and are reported rather than rejected.

This rule detects the current failure while allowing the observed 27
zero-positive events in 2017 and 16 in 2022.

## Error handling and rollback

- Every staged file is written through a sibling temporary path and renamed
  only after event validation succeeds.
- A failed event is recorded with its year, event name, source path, HDF5 path,
  and exact error. The command exits nonzero and does not activate any data.
- Existing staging files are accepted only after their repair attributes and
  source fingerprint match; otherwise they are rejected rather than silently
  skipped. The fingerprint is SHA-256 over the ordered source-relative TIFF
  paths, file sizes, and nanosecond modification timestamps.
- Activation first verifies exact year paths, file counts, free space, and the
  absence of conflicting backup directories.
- If a post-activation audit fails, the four directory renames can be reversed
  without re-downloading or reconverting source data.

## Verification and acceptance criteria

Unit and integration tests use synthetic HDF5/TIFF fixtures to cover:

- hour values preserved;
- HHMM values converted once;
- zero-only events accepted and marked;
- invalid and mixed encodings rejected;
- date, shape, event, and year mismatches rejected;
- source HDF5 files left byte-for-byte untouched by staging;
- atomic staging cleanup on failure;
- Phase 0 blocking an all-zero year or split;
- legitimate zero-positive events/days reported without blocking a split that
  contains positive labels.

Before activation, the staged four-year dataset must contain 392 valid files,
zero temporary files, zero repair errors, LZF compression with shuffle enabled,
and these exact next-day target counts derived independently from source TIFFs:

| year | events | target days | zero-target days | positive target pixels |
| --- | ---: | ---: | ---: | ---: |
| 2016 | 92 | 2,102 | 886 | 303,648 |
| 2017 | 110 | 2,490 | 1,481 | 177,972 |
| 2022 | 122 | 3,424 | 2,158 | 105,377 |
| 2023 | 68 | 2,442 | 1,297 | 167,952 |

The repair decision must also contain exactly these five sorted nonblocking
exclusions and no others: `2022/fire_CA4186812327820220730`,
`2022/fire_ID4570411652620220904`,
`2022/fire_OR4513211711020220825`,
`2022/fire_WA4687912083320220803`, and
`2022/fire_WA4796412068520220909`. The operator workflow must compare the
actual and expected arrays with an order-sensitive deterministic command and
block activation on any missing, extra, duplicated, or reordered value.

After activation:

1. Validate all 999 HDF5 files and the four exact added-year label totals.
2. Verify representative first and last days against raw TIFFs, including
   exact equality for channels 0--21 and normalized equality for channel 22.
3. Rerun the full Phase 0 CLI and require a non-blocked data decision with
   nonzero labels in every year and split.
4. Update the website and experiment documentation to record the discovered
   encoding mismatch, corrected label statistics, and strengthened gate.
5. Run the full test suite and static website checks, then commit the repair
   implementation and the corrected real-data result in small, intentional
   commits.

## Next experiment

Only after the repaired data and strengthened gate pass, run deterministic
T=1 no-fire and latest-day persistence baselines on the frozen split. Report
event-macro AP only over events for which it is defined, the number and fraction
of undefined zero-positive events, pooled AP and prevalence for context, and
zero-target-day false-alarm rate separately. Do not tune a threshold or model
on the 2022--2023 test targets.
