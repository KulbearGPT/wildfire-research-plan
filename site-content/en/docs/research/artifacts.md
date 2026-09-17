# Portable teacher artifact bundles

The committed manifests identify historical checkpoints and their SHA256 digests. To transfer existing teachers, create a self-contained bundle while the original files are accessible. Bundling verifies checkpoint size where recorded and SHA256 before copying, verifies every copied file, and writes relative paths in `manifest.json` plus file digests in `checksums.json`. It preserves the original manifests and refuses an existing output directory.

Run from the repository in a Slurm allocation. The helper uses only the Python standard library and refuses to operate without `SLURM_JOB_ID`. For example, after selecting your site's account and partition:

```bash
srun --account="$SLURM_ACCOUNT" --partition="$CPU_PARTITION" \
  --cpus-per-task=1 --mem=4G --time=00:30:00 \
  python -m reproductions.artifact_bundle \
  --format reliability \
  --manifest docs/experiments/three_directions_manifest.json \
  --output "$ARTIFACT_ROOT/reliability-teachers"

srun --account="$SLURM_ACCOUNT" --partition="$CPU_PARTITION" \
  --cpus-per-task=1 --mem=4G --time=00:30:00 \
  python -m reproductions.artifact_bundle \
  --format routed \
  --manifest reproductions/three_directions/sources.json \
  --output "$ARTIFACT_ROOT/routed-teachers"
```

Use `--histories 1 5 --seeds 0` to restrict either bundle to seed zero. Omit selection flags to include every manifest entry. Each selected history/seed includes all its teacher roles. Choose sufficient time and disk capacity for the selected weights; copying can require several GB.

Transfer the entire output directory. Point the relevant runner's source manifest setting at the transferred `manifest.json`. Checkpoint and summary paths resolve relative to that file. For the archived routed teacher whose summary is null, the helper also copies the checkpoint's sibling `M00.json`, preserving the evaluator's existing fallback. `metadata.initial_checkpoint` deliberately remains the original string: it is a provenance identity compared against checkpoint metadata, not a file that the runner opens.

This is a verified transfer route and requires access to the original checkpoint files. It does not download private artifacts, train missing teachers, or reconstruct unavailable historical datasets. If originals are unavailable, use the baseline and teacher regeneration recipes in [the reproduction guide](reproduce.md); regenerated checkpoints form a new manifest with their actual checksums and provenance. Do not relabel their hashes as historical artifacts. Historical 2023 populations also remain distinct from current corrected-data populations.
