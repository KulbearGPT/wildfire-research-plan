# Regenerating corrected B3 and B5 baselines

Use the generic submitter from the committed checkout after environment/data setup. The historical `initialization: from_scratch` label means no wildfire task checkpoint; the encoder starts from ImageNet weights.

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
for baseline in B3 B5; do
  bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.train_corrected_baseline \
    --baseline-id "$baseline" --upstream-root "$WILDFIRE_UPSTREAM" \
    --data-root "$WILDFIRE_DATA" --stats-path "$WILDFIRE_STATS" \
    --run-root "$WILDFIRE_ROOT/runs/$baseline"
done
```

Wait for both training jobs to finish successfully. Then export the selected checkpoints in CPU allocations:

```bash
for baseline in B3 B5; do
  bash scripts/research/submit.sh cpu python -m reproductions.wsts_fast_track.complete_baseline \
    --baseline-id "$baseline" --run-root "$WILDFIRE_ROOT/runs/$baseline" \
    --checkpoint-output "$WILDFIRE_ROOT/checkpoints/${baseline,,}.ckpt" \
    --record-output "$WILDFIRE_ROOT/checkpoints/$baseline-completed.json"
done
```

These checkpoint destinations match the example site configuration; if you customized them, use your configured destinations instead. B0/B1/B2 use the same two commands with their own baseline ID and new output paths. After export succeeds, evaluate the completed baseline, for example B3:

```bash
bash scripts/research/submit.sh gpu python -m reproductions.wsts_fast_track.evaluate_corrected_baseline \
  --record "$WILDFIRE_ROOT/checkpoints/B3-completed.json" \
  --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA" \
  --stats-path "$WILDFIRE_STATS" --year 2021 --device cuda \
  --output-root "$WILDFIRE_ROOT/runs/B3/eval-2021"
```

For a frozen held-out evaluation, change the year and output directory and add `--heldout-authorized`. Do not retrain or select checkpoints using 2022/2023.

B3 is C00/T1 and B5 is C02/T5. Both retain seed zero, 3,000 optimizer steps, AdamW learning rate .001, batch64 and their original FireDrop/BlockDrop policy. Training uses 2016–2020; checkpoint selection maximizes 2021 validation AP. The selected checkpoint can precede step 3,000.

The trainer writes `training-completion.json` only after upstream fitting and final validation return successfully. It records the actual completed fit step separately from the selected best checkpoint. The completion command verifies baseline configuration, full fit length, selection monitor/mode, source SHA256, selected checkpoint step and stored ModelCheckpoint best-score metadata. It then copies the checkpoint, verifies its digest and creates a compatible immutable completion record. Existing outputs are rejected. A lone `best.ckpt` or interrupted training run is insufficient; this helper cannot retroactively certify old runs without a receipt.

Use `--b3-record "$WILDFIRE_ROOT/checkpoints/B3-completed.json"` for the recovered T1 D-series continuations and `--b5-record "$WILDFIRE_ROOT/checkpoints/B5-completed.json"` for `train_temporal_reliability_prompting` with `--variant standard` or `--variant sarp`. Records point to absolute external checkpoint paths; after moving artifacts, update only the record's `checkpoint` field to the transferred file and verify its recorded SHA256. Keep the scientific provenance fields intact.
