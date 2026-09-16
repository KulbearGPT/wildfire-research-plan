# Reproduce the two positive diagnostic observations

These are historical **24-event, 2021, T1 diagnostics**, not adopted forecasting methods. Recovered source is `4b843add48862dcfac668b643ecd2a564cbe00df`. The original compact results are preserved in [diagnostics-results.json](diagnostics-results.json), copied from completed archived jobs without recomputation.

| Diagnostic | Exact recorded observation | Limitation |
|---|---|---|
| Natural VIIRS attention screen, job 20814289 | AP .3822925055272726 → .38257901744809514; delta +.00028651192082251864 | F1 .26853028377806015 → .2632027257240204; Brier worsens by .00002638172979156292; no forecasting-method gain |
| Target-QA censoring, job 20823565 | Frozen P00 AP .3822925055272726 → .39878926089761657 | Evaluation population changes; not improvement to the predictor |

The original population has 393,216 pixels and 1,867 positives. QA retains all positives and only reliably observed zero labels: 49,310 unknown zero pixels are excluded, leaving 343,906 pixels. This is 12.54% of **all** pixels, or 12.60% of zero-label pixels; some older prose rounds/describes the denominator loosely. After censoring, attention AP .3979113680650778 is below P00 by .0008778928325387514. Do not present the diagnostic as attention becoming useful after QA.

## Recovered code and changes

The entrypoints are:

- `reproductions.wsts_fast_track.viirs_reliability`: public-product acquisition and fixed reliability-map preparation.
- `reproductions.wsts_fast_track.evaluate_viirs_reliability`: natural-observation attention versus frozen P00.
- `reproductions.wsts_fast_track.evaluate_viirs_target_quality`: standard versus QA-censored metrics for the same predictions.
- `reproductions.wsts_fast_track.train_belief_state`: regenerate the attention head from an available valid historical P00 completion record.

The recovered closure includes the historical attention/filter/reconstruction heads, latent-state support, belief evaluator, and censored-cohort support used by the builder. These dependencies are retained for source integrity; this guide covers only the two T1 diagnostic entrypoints. The general `evaluate_belief_state` CLI is dependency code here: its broader T5 and standalone relocation paths are not supported or qualified by this recovery, and no general negative-belief replay is promised. No scientific objective, mask classes, cohort selection, layer math or evaluation arithmetic was changed. Imports use retained runtime/corruption code and `diagnostic_support.py` contains the exact historical sampling/index/checkpoint-record helpers. `diagnostic_artifacts.py` adds explicit checksum-verified path relocation. Original absolute checkpoint provenance remains untouched inside transferred files.

## Prerequisites and transfer status

Run data preparation, hashing large files, training and evaluation **inside Slurm**. Use the configured environment from [reproduce.md](reproduce.md), with the pinned upstream checkout and corrected 999-event HDF5 dataset/statistics. Diagnostics additionally need `requests`, `rasterio` (including its GDAL/PROJ runtime), `h5py`, NumPy and the system `curl` executable. The model steps use the shared PyTorch/upstream model dependencies. These extra imports have not yet been qualified in a fresh environment by this recovery task.

| Asset | Acquisition / exact requirement |
|---|---|
| Original georeferenced daily rasters | `WildfireSpreadTS.zip`, Zenodo record `8006177`, MD5 `dc1a04e63ccc70037b277d585b8fe761`; existing `docs/tutorials/res18/download.sh original` downloads and verifies it. Builder expects `2021/<event>/<date>.tif` (also accepts a `WSTSPlus/` prefix). Corrected HDF5 alone cannot supply the georeferencing. |
| VIIRS products | Builder queries NASA CMR collection `C2734202914-LPCLOUD` (VNP14IMG) and downloads matching VNP14IMG fire-mask and VNP03IMG geolocation products from LAADS. An Earthdata bearer-token file is required; obtain credentials yourself and keep the file mode 600. A transferred cache can avoid repeated downloads, but the current builder still requires a nonempty token file. |
| Input/target reliability | Either regenerate the `model` and `target` gates below or transfer each entire `output/` directory, including `manifest.json` and all 24 referenced NPZ files. Their names are relative and need no path rewrite. The provenance gate is not a substitute for either gate. |
| Historical frozen P00 | Transfer `completed.json` and its exact checkpoint; corrected B2/B3 is not a substitute. P00 is seed 0, 10,000 updates, legacy FireDrop and legacy pooled-year training exposure. |
| Historical attention head | Transfer `belief-state.pt` from `prototype-fire-belief-attention-t1-20655225`, or regenerate the attention head from that P00 as below. Historical SHA256: `e576bafff00776ef3482cb09092bd43972a367c34e30dc0eaa65cf762f09942a`. |
| Historical statistics | Original attention input log records SHA256 `4f6d308f00afcfcafcd14404dca9fff4f5beb89e1c2d0f615723f4d08b690f49` for `train-2016-2020-stats.npz`. Verify the transferred file before claiming numeric reproduction. |

Original run-directory names under the archived run root are `prototype-P00-FireDrop-C00-20398173`, `prototype-fire-belief-attention-t1-20655225`, `viirs-reliability-24-model-20807932`, and `viirs-reliability-24-target-20823563`. The original site root is recorded in `docs/experiments/artifact-archive-manifest.tsv`; it is provenance, not a required runtime location. Small metadata/existence checks on 2026-09-16 found the P00 record/checkpoint, attention checkpoint and all 48 referenced map files present at that archive. No large checkpoint hashing, transfer or fresh model execution was performed for this recovery.

The historical P00 record SHA256 is `9b74bd49d15bf7aaf8beec9b830b6db224a50209110a8f133b65293115b6d8ac`. Leave its bytes and checkpoint field unchanged. The map below redirects the old field without requiring the old path to exist. A SHA computed on transfer verifies that transferred content stays unchanged; the P00 checkpoint itself still needs comparison with a trusted source-side checksum generated during transfer. Its existence alone does not establish integrity.

## Prepare public VIIRS maps

Set `WSTS_ZIP`, `VIIRS_CACHE`, `EARTHDATA_TOKEN_FILE`, `INPUT_QA`, `TARGET_QA` to site-selected paths. Both output directories must be new. Run the following submission commands from the committed Git checkout on the login shell, after setting `WILDFIRE_SITE_ENV`, sourcing it and changing to `WILDFIRE_REPO`. The submitter creates the allocation; do not execute the Python commands directly on login. The two map jobs are independent. Wait for both to finish with Slurm state COMPLETED and exit 0:0 before submitting dependent evaluations:

```bash
bash scripts/research/submit.sh cpu python -m reproductions.wsts_fast_track.viirs_reliability \
  --archive "$WSTS_ZIP" --cache-root "$VIIRS_CACHE" \
  --token-file "$EARTHDATA_TOKEN_FILE" --gate model --output-root "$INPUT_QA"
bash scripts/research/submit.sh cpu python -m reproductions.wsts_fast_track.viirs_reliability \
  --archive "$WSTS_ZIP" --cache-root "$VIIRS_CACHE" \
  --token-file "$EARTHDATA_TOKEN_FILE" --gate target --output-root "$TARGET_QA"
```

The 24 identities are frozen in `FIXED_2021_MODEL_GATE` and `FIXED_2021_TARGET_GATE`; do not replace them by convenient available events. Valid fire-mask classes remain 5/8/9, acquisition ages are anchored at next-day 00:00 UTC, and unobserved age is 24 hours. Reliability-map generation is CPU/geospatial work and can use a CPU allocation; model evaluation requires the configured GPU allocation. Historical model/target mean reliable fractions were .8583887641562177/.8792395447601579. The 2016–2020 cohort extension failed because its added-year TIFFs lacked usable georeferencing; it is not part of these successful 2021 map preparations.

## Relocate the frozen artifacts and evaluate

Set `P00_RECORD`, `P00_CHECKPOINT`, `ATTENTION_CHECKPOINT` to transferred files, `ARTIFACT_MAP` to a new JSON path, `DIAG_OUTPUT` to an output directory, and the shared `WILDFIRE_UPSTREAM`, `WILDFIRE_DATA`, `WILDFIRE_STATS` configuration. Submit map creation to a CPU allocation; the long original record string below is an immutable provenance key, never a file read:

```bash
bash scripts/research/submit.sh cpu python -m reproductions.wsts_fast_track.diagnostic_artifacts \
  --p00-record "$P00_RECORD" --p00-checkpoint "$P00_CHECKPOINT" \
  --recorded-p00-record /project/6085198/kulbear/wildfire/runs/prototype-P00-FireDrop-C00-20398173/completed.json \
  --output "$ARTIFACT_MAP"
```

Wait for the map job to complete successfully, then submit the GPU evaluations (they may run independently):

```bash
common=(--p00-record "$P00_RECORD" --attention-checkpoint "$ATTENTION_CHECKPOINT"
        --artifact-map "$ARTIFACT_MAP" --upstream-root "$WILDFIRE_UPSTREAM"
        --data-root "$WILDFIRE_DATA" --stats-path "$WILDFIRE_STATS"
        --batch-size 8 --device cuda)
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.evaluate_viirs_reliability "${common[@]}" \
  --reliability-root "$INPUT_QA" --output "$DIAG_OUTPUT/natural.json"
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.evaluate_viirs_target_quality "${common[@]}" \
  --input-reliability-root "$INPUT_QA" --target-reliability-root "$TARGET_QA" \
  --output "$DIAG_OUTPUT/target-qa.json"
```

The map has schema version 1 and an `artifacts` object keyed by recorded source path. Each entry has `path`, `bytes`, `sha256`; `path` can be relative to the map file. With a map provided, missing entries, changed bytes, or checksum mismatch fail instead of falling back to the historical filesystem. Every input/target NPZ is checked against its sample manifest SHA256 before its first decode; cached decoded arrays reuse that verified content. Missing digests and checksum mismatches fail explicitly. Scientific record/checkpoint fields and finite state checks remain enforced. A map generated from the wrong files is not evidence they match the historical run; first verify the source checksums described above.

## Regenerate the attention head, and the public-only boundary

With the historical P00 record/checkpoint available, this preserves the archived attention recipe (seed 0, 3,000 AdamW updates, LR .001, effective batch 64, state-loss coefficient .1):

```bash
DIAGNOSTIC_SOURCE_COMMIT=$(git rev-parse HEAD)
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.train_belief_state \
  --method attention --history 1 --p00-record "$P00_RECORD" --artifact-map "$ARTIFACT_MAP" \
  --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
  --stats-path "$WILDFIRE_STATS" --output-path "$NEW_ATTENTION_CHECKPOINT" \
  --git-commit "$DIAGNOSTIC_SOURCE_COMMIT" --batch-size 64 --accumulation-steps 1 \
  --num-workers 8 --device cuda
```

Compute `DIAGNOSTIC_SOURCE_COMMIT` in the submission checkout as shown; it is passed literally into the archived job. Do not run `git rev-parse` inside the archive, which has no Git metadata. Wait for this training job to complete before evaluating its checkpoint.

The regenerated checkpoint records the current P00 paths. The map generator includes checksum-verified identity entries for those paths as well as historical aliases, so the same map supports the regenerated head: set `ATTENTION_CHECKPOINT="$NEW_ATTENTION_CHECKPOINT"` and rerun the two evaluation commands with new result filenames. Do not silently edit immutable source metadata.

If historical P00 is unavailable, use the recovered [legacy P00 bootstrap and completion recipe](legacy-p00.md). It preserves the archived 10,000-step training policy and known pooled-year indexing defect; it is not corrected B2. On the current prepared data this regenerates the old algorithm, while exact numerical replay additionally requires the historical data/statistics snapshot. Pair a newly regenerated P00 with a newly trained attention head, not a transferred historical head. Use fresh checksums and provenance, and do not claim that qualification or algorithm regeneration reproduced the historical scores.

## Verification status

Source AST parsing and relative-import closure checks passed. Five stdlib tests verify relocation, unchanged provenance, missing-map-entry rejection, independent byte counts, NPZ manifest integrity, path containment and corruption detection. The two original VIIRS mechanism test files are recovered for execution in Slurm, not run on login. Fresh environment imports and the actual 24-event evaluations remain pending controller qualification; historical results above are not new successful reproduction results.
