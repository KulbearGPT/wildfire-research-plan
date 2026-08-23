# WSTS+ Fast-Track Promotion and Follow-on Matrix Design

## Purpose

Extend the active WSTS+ fast track from two 3,000-step screening runs into a
small, explicit experiment program. The implementation must prepare clean
10,000-step promotion commands now, without launching them or exposing the
withheld 2022--2023 test years. It must also publish the bounded follow-on
matrix so future runs do not grow through ad hoc command edits.

## Frozen decisions

- C00 and C02 both receive a fresh seed-0 10,000-step run. Promotion does not
  resume the 3,000-step checkpoint.
- Promotion is allowed only after both C00 and C02 screening records pass the
  same machine-checkable gate.
- Seed-0 promotion still trains on 2016--2020, selects checkpoints on 2021,
  and keeps 2022--2023 inaccessible.
- Seed 1 and seed 2 are declared now but remain gated until both seed-0 10K
  runs pass.
- The controlled-missingness matrix is declared and validated now. Its data
  transforms and evaluations are a later implementation and cannot be
  launched by the promotion CLI.
- No 10K or follow-on GPU job is submitted merely by installing or testing
  this feature. Submission remains an explicit operator action through the
  existing cluster layer.

## Alternatives considered

1. **Declarative matrix plus one promotion CLI (selected).** Run identities,
   budgets, seeds, prerequisites, and corruption scenarios live in one typed
   registry. The CLI validates screening evidence and renders exact commands.
   This adds little machinery while preventing shell-script drift.
2. **One shell script per experiment.** This is initially shorter, but repeats
   split, budget, seed, and test-lock values and makes matched comparisons easy
   to break.
3. **A general sweep engine.** A reusable scheduler could support arbitrary
   search spaces, retries, and arrays, but those features are outside the
   current research question and would blur the distinction between frozen
   experiments and hyperparameter search.

## Architecture

### Typed matrix

`reproductions/wsts_fast_track/matrix.py` owns two immutable registries.

`RunSpec` contains:

- `run_id`;
- base experiment ID (`C00` or `C02`);
- stage (`screening`, `promotion`, or `replication`);
- seed;
- exact optimizer-step budget;
- prerequisite run IDs;
- whether the run is launchable by the promotion CLI.

The clean matrix is:

| Run ID | Model | Seed | Steps | Prerequisites | Launch state |
| --- | --- | ---: | ---: | --- | --- |
| `C00-S0-3K` | Res18-UNet | 0 | 3,000 | none | already active |
| `C02-S0-3K` | Res18-UTAE | 0 | 3,000 | none | already active |
| `C00-S0-10K` | Res18-UNet | 0 | 10,000 | both 3K runs pass | promotable |
| `C02-S0-10K` | Res18-UTAE | 0 | 10,000 | both 3K runs pass | promotable |
| `C00-S1-10K` | Res18-UNet | 1 | 10,000 | both seed-0 10K runs pass | declared, gated |
| `C00-S2-10K` | Res18-UNet | 2 | 10,000 | both seed-0 10K runs pass | declared, gated |
| `C02-S1-10K` | Res18-UTAE | 1 | 10,000 | both seed-0 10K runs pass | declared, gated |
| `C02-S2-10K` | Res18-UTAE | 2 | 10,000 | both seed-0 10K runs pass | declared, gated |

`CorruptionSpec` contains a scenario ID, family, affected raw feature indices,
severity, deterministic seed, and implementation state. The declared Stage 3
matrix is evaluated against every accepted 10K clean checkpoint:

| Scenario ID | Condition | Raw feature indices | Severity |
| --- | --- | --- | --- |
| `M00` | clean reference | none | 0 |
| `M01` | active-fire history absent | 22 | all available history |
| `M02` | active-fire history stale | 22 | one day |
| `M03` | observed weather absent | 5--11 | full modality |
| `M04` | forecast weather absent | 17--21 | full modality |
| `M05` | observed and forecast weather absent | 5--11, 17--21 | both modalities |
| `M06` | structured spatial blocks | all dynamic inputs | 25% area |
| `M07` | structured spatial blocks | all dynamic inputs | 50% area |

Spatial masks will later be generated from a stable hash of scenario ID,
event-relative path, target date, and matrix seed. They must be
label-independent. The matrix records this contract but does not implement the
transform in the current change.

### Command rendering

The existing `ExperimentSpec` remains the sole definition of model and feature
choices. `contract.upstream_arguments` gains a keyword-only `max_steps` input
defaulting to 3,000 and an optional keyword-only `seed` input defaulting to
`None`. The default call remains byte-for-byte identical for the active jobs;
declared follow-on runs pass both values explicitly and render the seed into
the Lightning command.

The existing entrypoint keeps its current `--experiment` interface unchanged
for already-submitted jobs and adds a mutually exclusive `--run-id` interface
for declared follow-on runs. A new `promotion.py` CLI consumes a declared
`run_id`, validates its prerequisites, and renders the exact invocation of the
same entrypoint with that run ID. The entrypoint resolves experiment, seed, and
budget only from the typed matrix. No arbitrary budget, seed, experiment ID,
test flag, or split override is accepted from the operator.

The CLI has two read-only operations:

- `matrix`: write or print the canonical clean and corruption matrices;
- `render`: validate prerequisite `completed.json` files and print a JSON
  promotion manifest containing the exact scientific command and run metadata.

The manifest is passed explicitly to the existing Nibi cluster submission
layer. The promotion CLI does not call `sbatch`, retry a run, or mutate a run
directory.

### Screening and prerequisite gate

Each prerequisite record must be a regular JSON file with one exact run
identity. The gate requires:

- `status == "pass"`;
- the expected experiment ID, seed, and exact step budget;
- `test_enabled == false` and withheld years `[2022, 2023]`;
- train years 2016--2020 and validation year 2021;
- finite validation AP, F1, and loss, with AP/F1 in `[0, 1]`;
- a positive CUDA allocation;
- a nonempty checkpoint path;
- no duplicate or unexpected prerequisite identities.

The gate does not compare C00 and C02 AP or select a winner. Both are promoted
to retain a matched bridge baseline and temporal-backbone baseline.

### Outputs and error handling

Matrix and promotion-manifest JSON use a versioned schema and deterministic key
ordering. Output paths are refused if they already exist; run artifacts remain
immutable. Missing, malformed, failed, non-finite, test-enabled, mismatched, or
extra prerequisite records fail before a command is emitted. Declared-but-
gated seed 1/2 runs and all Stage 3 scenarios cannot render launch commands.

## Tests

Focused tests will prove:

1. the eight clean run IDs and eight corruption scenarios are exact and
   duplicate-free;
2. C00/C02 seed-0 promotion commands differ only where their frozen model
   contracts differ and both contain seed 0, 10,000 steps, and `do_test=false`;
3. current 3K rendering remains byte-for-byte stable for already-submitted
   jobs;
4. a complete pair of synthetic passing screening records unlocks both seed-0
   10K manifests;
5. every gate mutation is rejected, including one missing result, wrong run
   ID, wrong steps/seed/split, enabled test, non-finite metric, failed status,
   duplicate input, or extra prerequisite;
6. seed 1/2 and corruption scenarios remain non-launchable;
7. CLI output is deterministic and refuses overwrite.

Only the focused fast-track and project-organization tests are required for
this implementation. Existing historical reproduction campaigns are not a
launch gate.

## Documentation and scope boundary

`reproductions/wsts_fast_track/README.md` will show the clean promotion matrix,
the declared controlled-missingness matrix, the explicit operator workflow,
and the boundary between prepared and launched work. `docs/research-roadmap.md`
will point to this registry as the authoritative run naming scheme.

This change does not run 10K jobs, evaluate 2022--2023, implement corruption
transforms, add reliability models, or make performance claims. Those actions
require their own completed prerequisites and explicit execution step.
