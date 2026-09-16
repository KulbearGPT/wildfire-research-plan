# Swin and SegFormer reproduction

The cross-history runner retains Swin-Unet and SegFormer-B2 for both T1 and T5, including single-setting positive signals that did not pass the cross-history gate. Their historical campaign uses a 10,000-step seed-zero architecture bootstrap, then independently initialized 3,000-step matched continuations from that same bootstrap. Do not replace those initializations with B3/B5.

The public Swin-T ImageNet-1K asset is linked by the [official Swin model hub](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md). SegFormer uses NVIDIA's [pinned MiT-B2 revision](https://huggingface.co/nvidia/mit-b2/tree/3bb39e8739149c3777d0325349b2a6c32c6413db). The fetch script pins all three SHA256 values already enforced by `reproductions/cross_history/architectures.py`:

| Artifact | SHA256 |
|---|---|
| Swin-T `swin_tiny_patch4_window7_224.pth` | `9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3` |
| MiT-B2 `config.json` | `d9a879499e7d73e2b33af0638cee320b1070c8f0dadb620eac8907df3d18caa9` |
| MiT-B2 `pytorch_model.bin` | `4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f` |

After committing the handoff and configuring `WILDFIRE_SITE_ENV`, submit the public download on CPU:

```bash
bash scripts/research/submit.sh cpu bash scripts/research/fetch-architectures.sh
```

Downloads go beneath `$WILDFIRE_ROOT/cache/swin` and `$WILDFIRE_ROOT/cache/huggingface/nvidia-mit-b2-3bb39e87`. Files are downloaded to unique `.partial` paths, checked, then atomically published without replacing an existing destination. Valid existing assets are reused; bad existing files cause an error and remain untouched. No download occurs outside Slurm. Network failures can be retried by submitting a new job. The configured training environment must include the pinned Transformers dependency for SegFormer.

For each architecture/history, run the following bootstrap and wait for successful completion before submitting its continuations. Set `ARCH=swin_unet` or `ARCH=segformer_b2`; set `HISTORY=1` or `HISTORY=5`. These commands assume the configured `WILDFIRE_ROOT` is also exported in the submission shell. Each output directory must be new.

```bash
ARCH=segformer_b2
HISTORY=1
RUN="$WILDFIRE_ROOT/runs/reproduced-${ARCH}-t${HISTORY}-s0"

bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
  --architecture "$ARCH" --history "$HISTORY" --method control \
  --bootstrap --seed 0 --steps 10000 --batch-size 8 --workers 3 \
  --output "$RUN/bootstrap"
```

After that job completes, submit the matched control and independent cosine/block branches:

```bash
BASE="$RUN/bootstrap/checkpoint.pt"
for METHOD in control cosine_erm block_specialist; do
  bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
    --architecture "$ARCH" --history "$HISTORY" --method "$METHOD" \
    --initial-checkpoint "$BASE" --seed 0 --steps 3000 \
    --batch-size 8 --workers 3 --output "$RUN/$METHOD"
done
for FRACTION in .25 .5; do
  bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
    --architecture "$ARCH" --history "$HISTORY" --method block_specialist \
    --block-fraction "$FRACTION" --initial-checkpoint "$BASE" \
    --seed 0 --steps 3000 --batch-size 8 --workers 3 \
    --output "$RUN/block-$FRACTION"
done
```

The runner accumulates physical microbatches to its fixed effective batch64; batch8 is a conservative supported allocation choice, not a changed optimizer-step budget. Use the historical per-job batch settings when seeking exact floating-point reruns. Repeat with the other history/architecture to retain matched comparisons. Historical seed-zero positives and failures are described in [the positive catalog](positive-signals.md) and [negative archive](negative-results.md); these commands do not imply every block variant passed a gate. They evaluate 2021 by default and do not authorize new seed sweeps or test-set selection.

`--bootstrap` loads the public architecture initialization and emits a **cross-history checkpoint.pt** containing architecture/history, metadata and its trained state. `--initial-checkpoint` here requires that cross-history checkpoint format from the same architecture and history. It does not accept the raw downloaded ImageNet `.pth`/`.bin`, a B3/B5 Lightning `.ckpt`, or a B3/B5 completion-record JSON. Canonical ResNet B3/B5 regeneration instead uses [the baseline recipe](baselines.md). Public architecture assets remain required at their verified cache paths during reconstruction even when loading a task checkpoint.
