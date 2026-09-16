# Regenerating corrected B3 and B5 baselines

Run inside a configured Slurm GPU allocation with the pinned upstream checkout, complete corrected dataset, train-only statistics and ImageNet ResNet18 encoder weights available. Set `U`, `D`, `S` and `R` to the upstream, dataset, stats NPZ and a new external run parent. The historical `initialization: from_scratch` label means no wildfire task checkpoint; the encoder starts from ImageNet weights.

```bash
python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id B3 --upstream-root "$U" --data-root "$D" \
  --stats-path "$S" --run-root "$R/B3"
python -m reproductions.wsts_fast_track.complete_baseline \
  --baseline-id B3 --run-root "$R/B3" \
  --checkpoint-output "$R/artifacts/B3.ckpt" \
  --record-output "$R/artifacts/B3-completed.json"

python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id B5 --upstream-root "$U" --data-root "$D" \
  --stats-path "$S" --run-root "$R/B5"
python -m reproductions.wsts_fast_track.complete_baseline \
  --baseline-id B5 --run-root "$R/B5" \
  --checkpoint-output "$R/artifacts/B5.ckpt" \
  --record-output "$R/artifacts/B5-completed.json"
```

B3 is C00/T1 and B5 is C02/T5. Both retain seed zero, 3,000 optimizer steps, AdamW learning rate .001, batch64 and their original FireDrop/BlockDrop policy. Training uses 2016–2020; checkpoint selection maximizes 2021 validation AP. The selected checkpoint can precede step 3,000.

The trainer writes `training-completion.json` only after upstream fitting and final validation return successfully. It records the actual completed fit step separately from the selected best checkpoint. The completion command verifies baseline configuration, full fit length, selection monitor/mode, source SHA256, selected checkpoint step and stored ModelCheckpoint best-score metadata. It then copies the checkpoint, verifies its digest and creates a compatible immutable completion record. Existing outputs are rejected. A lone `best.ckpt` or interrupted training run is insufficient; this helper cannot retroactively certify old runs without a receipt.

Use `--b3-record "$R/artifacts/B3-completed.json"` for the recovered T1 D-series continuations and `--b5-record "$R/artifacts/B5-completed.json"` for `train_temporal_reliability_prompting` with `--variant standard` or `--variant sarp`. Records point to absolute external checkpoint paths; after moving artifacts, update only the record's `checkpoint` field to the transferred file and verify its recorded SHA256. Keep the scientific provenance fields intact.
