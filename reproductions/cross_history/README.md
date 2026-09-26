# Cross-history code: active X22 + X17 route

The active route is **X22 + X17**, with fresh ERM and mixed-severity X14 controls.
See [code lifecycle](../../docs/CODE_LIFECYCLE.md) for the scientific limits,
maintenance status and archived directions. Other switches remain available for
historical reproduction; they are not active research recommendations.

## Active training settings

Use `python -m reproductions.cross_history.mainline` for the active route. Its
`--help` and `--print-command` paths load no model dependencies. Training delegates
to the unchanged shared runner after checking for a Slurm allocation.

Use canonical T1 Res18-U-Net / T5 Res18-UTAE, corresponding B3/B5 initialization,
3000 continuation steps and explicit physical batch 64. Run through
`scripts/research/submit.sh`; `run_slurm.sh` is a legacy Nibi-specific runner.

| Mainline method | Additional setting | Role |
|---|---|---|
| `control` | no block fraction | Fresh constant-schedule ERM comparator |
| `cosine_erm` | no block fraction | X22; final route's M01 component |
| `block_specialist` | no block fraction | Mixed-severity X14 attribution control |
| `block_specialist` | `--block-fraction 0.25` | X17 M06 component |
| `block_specialist` | `--block-fraction 0.5` | X17 M07 component |

Use a unique output directory for each history, seed, method and block fraction.
The mainline entrypoint fixes 3000 steps and physical batch 64 and rejects
archived methods and architecture overrides. The shared `run.py` retains its
historical batch default of 16 for archive compatibility; use the narrow entrypoint
for new mainline commands. Full commands are in the [active recipe](../../docs/research/method-recipes.md).

Evaluation checks checkpoint history, method, seed and block fraction before
constructing the model, so command-line metadata cannot relabel a trained expert.
Historical missing block-fraction metadata still denotes the mixed setting; the
archived normalized-inpaint transform of a control model remains supported.

`compare.py` checks matched summaries. `compose_severity_routes.py` compares X17
with mixed X14 and ERM. `compose_complete_routes.py` assembles M00 from control,
M01 from X22 and M06/M07 from fixed experts. Compose matching history/seed/year
rows using `--mainline` on both composers. This checks role identity, explicit
canonical architecture, AP validity and population counts, and protects existing
output reports. Duplicate cells are rejected in both modes; historical-test gates
require all three seeds in every history/year cell. Training provenance still
needs checkpoint/configuration review; see the active recipe for the limits.

## Source and archive boundary

`run.py`, `models.py` and `architectures.py` contain shared dispatch and archived
variants. Keep their imports and checkpoint interfaces stable. All methods other
than the three names above, and noncanonical architectures, are archived.
`compose_complete_routes.py` also supports archived X8+X17 results; that capability
does not make X8 active. `compose_routes.py` and `compose_table1.py` retain earlier
composition/architecture-table work.

`run_three_directions.py`, `three_directions.py`, `assess_three_directions.py` and
the BN audits belong to the archived RF campaign. The separate
`../three_directions/` package is the archived TD campaign. Use their distinct
source ledgers in [the inventory](../../docs/research/method-inventory.json).

The archived BN audits require `--run-root RUNS`; optional `--t1-run`, `--t5-run`
and `--output` select explicit locations. Their defaults identify the historical
RF runs and do not launch or select new experiments. Importing either audit or
requesting `--help` loads no model; actual diagnosis requires Slurm before any
model imports or checkpoint/data reads. The activation audit uses a GPU. The
statistics audit retains its historical behavior of skipping missing checkpoints.

Both confirmation bundle scripts are retired and exit before touching Slurm.
Their old job IDs and commands remain visible only as historical implementation.
