# Regenerate teacher checkpoints and manifests

Teacher manifests describe newly trained source runs. Generate them from actual checkpoints and completed 2021 evaluation summaries; do not copy historical SHA-256 values onto regenerated artifacts. This procedure prepares sources shared by the reliability-fusion driver (`reproductions.cross_history.run_three_directions`) and the separate routed-teacher driver (`reproductions.three_directions.run`). It is a regeneration recipe, not a claim that these full training jobs were rerun during handoff qualification.

First finish [environment and B3/B5 setup](reproduce.md) and [corrected dataset preparation](data-preparation.md). Set `WILDFIRE_B3_CHECKPOINT` and `WILDFIRE_B5_CHECKPOINT` in your site configuration to the corresponding regenerated Lightning baseline checkpoints. Each history uses its same baseline checkpoint for all source methods and seeds. Keep those paths fixed throughout source training: `initial_checkpoint` is an embedded provenance identity, and the builder checks literal equality across all sources for each history.

## Train the five source runs

Use these output directory names directly beneath a common runs root:

| Directory suffix | Cross-history method | Block fraction | Reliability role | Routed role |
| --- | --- | --- | --- | --- |
| `tH-sS-control` | `control` | unspecified | `clean` | `erm` |
| `tH-sS-cosine_erm` | `cosine_erm` | unspecified | `fire` | `x22` |
| `tH-sS-block_specialist` | `block_specialist` | unspecified | `block_mixed` | `x14` |
| `tH-sS-block025` | `block_specialist` | `0.25` | `mild` | `mild` |
| `tH-sS-block050` | `block_specialist` | `0.5` | `severe` | `severe` |

Here `H` is history 1 or 5 and `S` is seed 0, 1, or 2. A run directory must contain `checkpoint.pt` and `summary.json`, not an additional `result` subdirectory. The commands below retain the source recipe's 3,000 optimizer steps and physical/effective batch size 64. Choose sufficient GPU memory in your site configuration; reducing physical batch changes this exact source recipe and the manifest builder rejects it.

```bash
export WILDFIRE_SITE_ENV="/path/to/site.env"
source "$WILDFIRE_SITE_ENV"
TEACHER_RUNS="$WILDFIRE_ROOT/runs/teacher-sources"
mkdir -p "$TEACHER_RUNS"

for history in 1 5; do
  for seed in 0 1 2; do
    for method in control cosine_erm block_specialist; do
      bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
        --history "$history" --seed "$seed" --method "$method" \
        --steps 3000 --batch-size 64 --year 2021 \
        --output "$TEACHER_RUNS/t${history}-s${seed}-${method}"
    done
    bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
      --history "$history" --seed "$seed" --method block_specialist \
      --block-fraction 0.25 --steps 3000 --batch-size 64 --year 2021 \
      --output "$TEACHER_RUNS/t${history}-s${seed}-block025"
    bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run \
      --history "$history" --seed "$seed" --method block_specialist \
      --block-fraction 0.5 --steps 3000 --batch-size 64 --year 2021 \
      --output "$TEACHER_RUNS/t${history}-s${seed}-block050"
  done
done
```

This submits 30 training/evaluation jobs. For an initial single-seed regeneration, change `for seed in 0 1 2` to `for seed in 0`, then request only seed 0 in manifest generation and downstream runs. Do not pass `--smoke`: smoke outputs are rejected. Do not substitute a cross-history continuation checkpoint for the baseline environment variables; those variables select upstream Lightning base-model state.

## Build both manifest formats

Wait until every requested source job has finished successfully, including its full 3,181-sample 2021 evaluation in each of M00/M01/M06/M07. The submitter does not wait or automatically add job dependencies. Then run the builder in CPU Slurm allocations using the training environment:

```bash
bash scripts/research/submit.sh cpu python -m reproductions.build_teacher_manifest \
  --runs-root "$TEACHER_RUNS" \
  --output "$WILDFIRE_ROOT/manifests/reliability-teachers.json" \
  --format reliability --histories 1 5 --seeds 0 1 2

bash scripts/research/submit.sh cpu python -m reproductions.build_teacher_manifest \
  --runs-root "$TEACHER_RUNS" \
  --output "$WILDFIRE_ROOT/manifests/routed-teachers.json" \
  --format routed --histories 1 5 --seeds 0 1 2
```

The builder loads checkpoints on CPU and validates history, seed, architecture, method, block fraction, 3,000 steps, batch sizes, source initialization identity, and matching summary metadata. It hashes checkpoint bytes afresh. Missing summaries, incomplete evaluation populations, smoke runs, mismatched sources, and existing output manifests fail explicitly. `--histories` defaults to `1 5`; `--seeds` defaults to `0`.

Artifact `checkpoint` and `summary` paths are relative to the manifest's directory. Transfer manifests and run folders together, preserving their relative layout. `metadata.initial_checkpoint` is deliberately left as the exact identity stored in the checkpoint; do not rewrite it when relocating files. The downstream loaders verify checkpoint hashes and metadata before use. Manifest generation alone does not qualify a scientific improvement or reproduce historical scores.

## Run the downstream drivers

After both manifest jobs succeed, an example reliability-fusion distillation run and matched control are:

```bash
for arm in control distill; do
  bash scripts/research/submit.sh gpu python -m reproductions.cross_history.run_three_directions \
    --history 1 --seed 0 --arm "$arm" --steps 3000 --batch-size 64 \
    --manifest "$WILDFIRE_ROOT/manifests/reliability-teachers.json" \
    --output "$WILDFIRE_ROOT/runs/reliability-t1-s0-$arm"
done
```

For reliability-fusion normalization diagnostics, use `--arm bn`; `--bn-forward-mode batch_stats` explicitly requests the corrected batch-statistics calibration variant. The legacy default `fixed` remains available for auditing the historical fixed-forward result. For the separate routed-teacher family:

```bash
for mode in student_control student_distill; do
  bash scripts/research/submit.sh gpu python -m reproductions.three_directions.run \
    --history 1 --seed 0 --mode "$mode" \
    --sources "$WILDFIRE_ROOT/manifests/routed-teachers.json" \
    --output "$WILDFIRE_ROOT/runs/routed-t1-s0-$mode"
done
```

Other routed modes are `bn_shared`, `bn_conditional`, `merge`, and `x14_shared`; select the exact recipe described in the evidence catalog. Repeat with history 5 and the required seeds only after their source manifests exist. Held-out evaluation uses each driver's separate evaluation/checkpoint arguments; it must not refit on held-out years.
