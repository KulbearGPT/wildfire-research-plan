# X22 + X17: active method recipe

Only this route and its necessary controls are active. The [archived recipe
collection](../archive/method-recipes.md) preserves all other methods and original
commands; [Chinese archive](../archive/method-recipes-zh.md). Their experimental
settings and outcomes remain in the [inventory](method-inventory.json).

Complete [setup and data](reproduce.md) and [B3/B5 preparation](baselines.md) first.
Use a committed checkout and a trusted `WILDFIRE_SITE_ENV`. These examples are
submission instructions, not evidence of new runs or authorization for a sweep.
Choose the history, seed and run set within the agreed compute budget.

## 1. Check the intended command

The narrow `mainline` entrypoint accepts only canonical T1/T5, `control`,
`cosine_erm`, and `block_specialist`. It fixes 3000 optimizer steps and physical
batch 64, allows the recorded seeds 0/1/2, and rejects archived methods and
architecture overrides. It delegates training unchanged to the shared runner.
B3/B5 initialization comes from the configured site paths. Do not pass the old
runner's `--steps`, `--batch-size`, `--architecture`, or `--bootstrap` switches.

This preview is safe on a login node: it prints text without importing a model,
creating an output directory, querying Slurm, or submitting anything:

```bash
python3 -m reproductions.cross_history.mainline \
  --history 1 --method cosine_erm --seed 0 \
  --output /path/to/new/run --print-command
```

Use `--help` for options. A real invocation requires a Slurm allocation before
loading the trainer. For a changed execution path, `--smoke` checks one training
step and save/reload; it retains batch 64 and does not reproduce scientific scores.
Use a separate smoke directory. Do not replace a formal run with its checkpoint.

## 2. Train one matched set

Set a new group name for every attempt. These commands show the five distinct
roles; submit only the roles needed for the approved comparison. Reuse a control
only when its initialization, protocol and seed match.

```bash
source "$WILDFIRE_SITE_ENV"
cd "$WILDFIRE_REPO"
history=1
seed=0
run_group="$WILDFIRE_ROOT/runs/x22-x17-t${history}-s${seed}-attempt1"

bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method control \
  --output "$run_group/control"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method cosine_erm \
  --output "$run_group/x22"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method block_specialist \
  --output "$run_group/mixed"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method block_specialist --block-fraction 0.25 \
  --output "$run_group/mild"
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method block_specialist --block-fraction 0.5 \
  --output "$run_group/severe"
```

The mixed expert is X14's attribution control. Fixed mild/severe experts are X17.
Fresh `control` supplies M00, X22 supplies M01, mild supplies M06, severe supplies
M07 in the final route. Frozen B3/B5 are separate references, not fresh controls.

Check each job's exit status and actual `checkpoint.pt`/`summary.json` before
submitting dependent work. Outputs must be new; do not restart into an old result
directory. Log the source commit and job ID printed by the submitter.

## 3. Evaluate the same checkpoints

Training validates on 2021. To reproduce historical 2022/2023 reporting, evaluate
fixed checkpoints; do not train on those years or select new methods with them.
For example, after the mild expert completes:

```bash
bash scripts/research/submit.sh gpu python -m reproductions.cross_history.mainline \
  --history "$history" --seed "$seed" --method block_specialist --block-fraction 0.25 \
  --evaluate-only "$run_group/mild/checkpoint.pt" --year 2022 \
  --output "$run_group/mild-2022"
```

Repeat only as authorized, with each role's original method and block fraction,
checkpoint, history and seed, and new year-specific output paths. Historical test
years are already exposed; this is reproduction, not untouched confirmation.
The runner rejects a checkpoint whose recorded history, method, seed or block
fraction differs from the evaluation request. Do not relabel a different expert
or seed by changing command-line arguments. This guard applies to new evaluations;
old summary files still need their provenance checked.

## 4. Compose and interpret

After matching 2021 summaries exist, these CPU Slurm jobs produce the system
report and the necessary attribution comparison:

```bash
bash scripts/research/submit.sh cpu python -m reproductions.cross_history.compose_complete_routes --mainline \
  --control "$run_group/control/summary.json" --fire "$run_group/x22/summary.json" \
  --mild "$run_group/mild/summary.json" --severe "$run_group/severe/summary.json" \
  --output "$run_group/x22-x17-2021.json"
bash scripts/research/submit.sh cpu python -m reproductions.cross_history.compose_severity_routes --mainline \
  --control "$run_group/control/summary.json" --mixed "$run_group/mixed/summary.json" \
  --mild "$run_group/mild/summary.json" --severe "$run_group/severe/summary.json" \
  --output "$run_group/x17-attribution-2021.json"
```

Use `--mainline` for the active route: it requires the correct method in each
role, explicit canonical architecture, finite AP, the recorded sample count for
each year, and equal pixel counts across matched inputs. Both composers reject
duplicate history/seed/year cells. Their three-seed and historical-test gates
require seeds 0/1/2 in every relevant cell; partial reports remain useful, but
cannot pass those gates. Mainline mode refuses to overwrite an existing report.
Without this flag, the broad composer remains available for archived routes.

These checks validate summary fields, not the full experimental provenance.
`training_provenance_verified` remains `false`: the historical summary format
does not contain training steps, physical batch, initialization or normalization
identity. Review checkpoint metadata and the source/configuration/data preparation
records before treating runs as matched. An evaluation-only `started.json`
records that invocation, not proof of the checkpoint's training settings.
Equal sample counts do not establish identical samples or labels.

This is summary composition, not deployment of a single fused model. Pass repeated
matching groups of arguments to aggregate additional histories/seeds/years,
keeping their order aligned. A `goal_evidence_pass` flag describes the historical
numerical gates only; it does not establish independent confirmation or imply
that the separate `magnitude_target_met` flag is true.

Report primary AP, clean AP, block AP and comparator identity. Preserve X17's
failed held-out attribution cells even when the complete system improves over
ERM. See [scope and evidence limits](../CODE_LIFECYCLE.md). Current selection does
not promote X17 to an independently confirmed mechanism or reactivate any archive.
