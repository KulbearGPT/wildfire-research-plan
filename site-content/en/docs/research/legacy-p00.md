# Legacy P00 initialization for historical diagnostics

This is the **historical invalid-foundation replay**, needed only by the two [VIIRS/target-QA diagnostic observations](diagnostics-reproduction.md). It does not replace corrected B2/B3 or establish a new forecasting result.

`reproductions.wsts_fast_track.legacy_p00` recovers the P00 entrypoint from `4b843ad:reproductions/wsts_fast_track/prototype_entrypoint.py`. It uses that archive's C00 argument contract, seed0, 10,000 updates, batch64, ImageNet ResNet18 initialization and training-only FireDrop probability .3. The retained `apply_training_fire_dropout` and `install_training_fire_dropout` functions are AST-identical to that archive. Shared runtime split/statistics code is equivalent for C00. Crucially it **does not install corrected dataset indexing**: it leaves the pinned upstream pooled-year index loop leak intact, preserving the invalid exposure that was later diagnosed. A contaminated process with an already replaced resolver is rejected. Run it as a separate process.

Public prerequisites are the pinned upstream/import patch, complete prepared data, ImageNet initialization, and train statistics from the normal setup. Exact reproduction of original numerical scores additionally needs the same historical data/statistics snapshot. Regenerating from today's corrected data produces a replay of the old algorithm and its known indexing defect, not proof of byte-identical original weights. Keep its fresh provenance and checksums.

After [setup and data preparation](reproduce.md), submit the full legacy initialization only if its weights are needed:

```bash
source "$WILDFIRE_SITE_ENV"
P00_RUN="$WILDFIRE_ROOT/runs/legacy-P00"
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.legacy_p00 \
  --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
  --stats-path "$WILDFIRE_STATS" --run-root "$P00_RUN"
```

Wait for this job to complete successfully; do not submit completion concurrently. The training wrapper captures the actual final fit step and best validation-AP checkpoint/score/hash, and writes `legacy-training-completion.json` only after fit and final validation return successfully. A best checkpoint may precede step10,000; its presence alone does not demonstrate full fitting.

```bash
bash scripts/research/submit.sh cpu python -m reproductions.wsts_fast_track.legacy_p00_completion \
  --run-root "$P00_RUN"
```

The completion module requires a 10,000-step receipt with the exact legacy policy, verifies the checkpoint digest, actual selected step and saved ModelCheckpoint best-score metadata, then creates `$P00_RUN/completed.json`. It keeps `scientific_claim: false` and `legacy_index_semantics: true`. Existing completed records are not overwritten. Its required P00 fields are compatible with the recovered attention/diagnostic loaders; it cannot be used as a corrected B3 completion record.

Set `P00_RECORD="$P00_RUN/completed.json"` and `P00_CHECKPOINT` to the checkpoint named in that record. Continue with `diagnostic_artifacts` and the attention-head training command in [diagnostic reproduction](diagnostics-reproduction.md), then regenerate the public maps and evaluate. For a fresh regenerated head, use the map's current-path identity entries; the original recorded P00 alias is needed only to replay an original transferred attention checkpoint. Do not combine a regenerated P00 with a transferred attention head and claim they are the same original pair.

For execution qualification only, the dedicated case locally changes the budget to two steps and batch2/workers0 with one validation batch; it does not run a full scientific job:

```bash
bash scripts/research/submit.sh gpu python scripts/research/qualify-legacy.py \
  --case legacy-P00 --output "$WILDFIRE_ROOT/qualification/legacy-P00"
```

It verifies strict checkpoint reload and two-example real-data evaluation, asserts that formal P00 completion rejects its two-step receipt, and retains only a clearly named `legacy-p00-qualification.json`. It never emits a successful 10,000-step completion. No full legacy training or efficacy rerun is performed as part of handoff qualification.
