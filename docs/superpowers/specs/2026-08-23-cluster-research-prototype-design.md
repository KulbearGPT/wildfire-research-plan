# Cluster-Ready Wildfire Research Prototype Design

**Date:** 2026-08-23
**Status:** Approved in chat; awaiting written-spec review
**Repository:** `KulbearGPT/wildfire-research-plan`

## 1. Purpose

Organize the current wildfire research repository so that it can be pushed to
GitHub, cloned on a Slurm-compatible GPU cluster, and used to continue the
research without losing the provenance of the completed Res18-U-Net baseline
reproduction.

This remains a research prototype. The implementation must optimize for fast,
controlled scientific iteration rather than production deployment,
serviceability, or platform engineering.

## 2. Current scientific state

The repository already contains the following completed foundations:

- an audited eight-year WSTS+ event-level dataset contract covering 999 valid
  events;
- the frozen temporal split `2016--2020 / 2021 / 2022--2023`;
- deterministic no-fire and latest-mask persistence baselines;
- an official-code Res18-U-Net, `T=1`, Fold-2 training reproduction;
- independently verified test-only evaluation of all twelve official released
  weights;
- a sealed twelve-fold measured AP aggregate of
  `0.452764004469 +/- 0.088217319908` (population standard deviation), with
  explicit separation from the paper-reported `0.460 +/- 0.084` provenance;
- English lecture pages documenting the related work and baseline
  reproduction.

The remaining scientific gaps are:

1. the official configuration/code discrepancy in positive weighting
   (`236` in YAML versus `608.465...` at Fold-2 runtime) has not been tested as
   a paired training ablation;
2. the frozen WSTS+ split has rule baselines but not a project-owned learned
   baseline;
3. the effect of controlled missingness has not yet been measured on the
   project-owned learned baseline;
4. the reliability-aware method and its matched baselines have not yet been
   implemented.

## 3. Design principles

### 3.1 Research-first, not deployment-first

The repository will not add a web service, database, workflow platform,
model-serving layer, Kubernetes deployment, generic experiment scheduler, or
custom artifact registry. Slurm already provides scheduling. Plain shell,
small Python validators, Git, JSON/CSV summaries, and the filesystem are
sufficient.

Every added component must answer one of four needs:

1. reproduce an experiment;
2. prevent an invalid experiment from starting;
3. preserve enough evidence to interpret a result;
4. reduce the time required to launch the next controlled experiment.

### 3.2 Preserve completed evidence

Existing baseline artifacts and claim boundaries remain frozen. Migration does
not retroactively reinterpret the original Windows/RTX 3090 environment as a
cluster run. The cluster begins a new evidence lineage linked to, but distinct
from, the completed local lineage.

### 3.3 Portable Slurm profile

No Rorqual account, partition, module, or absolute path is hard-coded into a
generic job script. A small cluster profile supplies site-specific values. The
same research command and job script must work on another similar Slurm cluster
after changing only that profile.

### 3.4 One scientific variable at a time

Backbone, split, history length, feature group, loss, seed, corruption, and
reliability mechanism are explicit configuration fields. A run must not change
multiple scientific variables under one ablation label.

### 3.5 Inode-aware storage

The cluster storage may have ample bytes but a constrained inode budget. The
design therefore uses event-level HDF5, archived raw small files, bounded
checkpoint retention, aggregated logs, and one compact result record per run.

## 4. Repository organization

The reorganization is additive. Existing website, baseline reproduction,
source, test, and evidence-document paths remain stable.

```text
wildfire-research-plan/
|-- README.md
|-- index.html
|-- baseline-reproduction/
|-- related-work/
|-- docs/
|   |-- experiments/
|   |-- research-roadmap.md
|   |-- cluster-migration.md
|   `-- superpowers/
|-- configs/
|   |-- cluster/
|   |   `-- profile.example.env
|   `-- wstsplus_field_contract.csv
|-- environments/
|   |-- audit/
|   `-- training/
|-- manifests/
|   |-- data/
|   `-- weights/
|-- scripts/
|   `-- cluster/
|       |-- bootstrap.sh
|       |-- preflight.sh
|       |-- submit.sh
|       |-- job.sh
|       `-- collect.sh
|-- reproductions/
|-- src/
`-- tests/
```

`README.md` becomes the human entry point. It links the live research plan,
the completed baseline evidence, the roadmap, and the cluster quick start.

`docs/research-roadmap.md` is the single current experiment-order document.
It distinguishes completed work, next required gates, mainline experiments,
matched baselines, and conditional extensions.

`docs/cluster-migration.md` explains the site-independent migration procedure,
the local-versus-cluster evidence boundary, and the commands a student runs on
a new Slurm cluster.

The cluster scripts are deliberately thin. They do not form a Python package or
an orchestration framework.

## 5. Git and storage boundary

### 5.1 Tracked in Git

- source code and tests;
- website and documentation;
- small experiment and cluster-profile templates;
- dependency inputs or locks once verified on the target cluster;
- data and weight manifests containing counts, sizes, revisions, and hashes;
- generic Slurm scripts;
- compact, reviewed experiment summaries and scientific claim boundaries.

### 5.2 Never tracked in Git

- WSTS/WSTS+ HDF5 or raw imagery;
- extracted GeoTIFF trees;
- downloaded official weights and new checkpoints;
- virtual environments, package caches, and compiler caches;
- real cluster profiles, accounts, tokens, credentials, and private paths;
- full run directories, batch-level predictions, and large logs;
- node-local scratch content.

The ignore rules must cover the real cluster profile, environments, data,
weights, caches, run outputs, checkpoints, and scratch paths without hiding the
example profile, manifests, source, tests, or reviewed summaries.

## 6. Cluster portability layer

### 6.1 Cluster profile

The untracked site profile supplies only operational values:

- CPU and GPU account;
- CPU and GPU partition;
- GPU resource request syntax and default GPU type;
- maximum array concurrency;
- module load commands or an environment activation path;
- persistent project, data, run, cache, and log roots;
- preferred node-local scratch variable;
- default smoke, calibration, and full-run time limits.

The tracked example profile contains explanatory dummy values and no real
account, token, or absolute user path.

### 6.2 Submission wrapper

`submit.sh` reads the profile and experiment arguments, validates required
fields, and invokes `sbatch` with site-specific flags supplied on the command
line. This avoids relying on shell-variable expansion inside `#SBATCH`
directives.

The wrapper supports a no-side-effect `--dry-run` mode that prints the exact
quoted `sbatch` command. The first implementation supports one GPU per task and
optional Slurm arrays. It does not implement multi-node or distributed
training.

For the expected allocation of two to four H100 GPUs, independent folds,
seeds, or corruption settings run as separate one-GPU array tasks. DDP is added
only if later profiling demonstrates that a single model cannot fit or a single
run is the actual throughput bottleneck.

### 6.3 Generic worker

`job.sh` performs only the common run boundary:

1. enable fail-fast shell behavior;
2. activate the configured environment;
3. select node-local scratch when available;
4. record Git, Slurm, Python, PyTorch, CUDA, and GPU metadata;
5. run the exact experiment command supplied by the submission wrapper;
6. copy only declared outputs to persistent storage;
7. write a compact completion or failure record.

The worker does not decide the scientific configuration and does not silently
resume or retry a failed scientific process.

### 6.4 Preflight

`preflight.sh` fails before scientific execution when any required boundary is
invalid:

- missing or dirty code identity for a formal run;
- missing cluster-profile field;
- unavailable GPU or environment;
- mismatched data count/bytes/hash manifest;
- mismatched official weight revision/hash;
- missing output/log root;
- insufficient writable scratch;
- an existing immutable run directory or launch marker.

The preflight remains fast and bounded. It does not rescan raw imagery or run a
full engineering test suite inside every Slurm job.

## 7. Environment strategy

The completed local reproduction retains its recorded Python 3.10.4,
PyTorch 2.0.0, CUDA 11.8, and RTX 3090 lineage.

The cluster uses a separately locked, cluster-supported environment. Alliance
style modules plus `virtualenv` are the default portable path; Conda or
Apptainer is optional rather than required. The exact cluster environment is
frozen only after module discovery and a successful installation smoke test.

Two logical environments remain separate:

- **audit environment:** Phase 0 data validation and deterministic rule
  evaluation;
- **training environment:** PyTorch models, GPU training, inference, and
  corruption experiments.

Before any new training, the training environment must test one pinned official
Fold-2 checkpoint on the original data. Metrics, tensor shapes, finite outputs,
and raw evidence are compared with the sealed local result under a declared
numerical tolerance. Passing this gate establishes executable equivalence; it
does not claim byte-identical execution across hardware and software stacks.

## 8. Data and inode strategy

The authoritative training representation on the cluster is the audited set of
approximately 999 event-level HDF5 files. The migration transfers these files
plus a manifest containing canonical relative path, byte size, and SHA-256.

Raw small-file products are retained as a few immutable archives. They are not
expanded into the persistent project tree. If a source audit later requires
them, a compute job extracts one bounded subset into node-local scratch and
deletes it at job completion.

Checkpoint retention defaults to:

- one best checkpoint by the frozen validation selection metric;
- one last resumable checkpoint;
- explicitly declared milestone checkpoints only when an experiment requires
  learning-curve analysis.

Logs are aggregated per Slurm job. Metrics are written to compact JSON/CSV. The
prototype does not emit one prediction or sidecar file per event unless a
specific error-analysis experiment has been approved; such outputs are packed
before persistence.

## 9. Research roadmap after migration

### Stage 0: Cluster migration gate

1. clone the reviewed Git commit;
2. create the site-local cluster profile;
3. create and record the server-native environments;
4. transfer and verify the HDF5 manifest;
5. fetch pinned upstream code and official weights from GitHub/Hugging Face;
6. run import and one-batch smoke checks;
7. run the Fold-2 checkpoint test-only equivalence gate;
8. run a 500-step timing and memory calibration before choosing a Slurm time
   tier.

### Stage 1: Close the training contract

Run a paired Fold-2 experiment in the same cluster environment. Every field is
identical except the positive-weight behavior:

- official runtime recomputation (`608.465...` in the previous Fold-2 run);
- fixed YAML value (`236`).

Materiality criteria and any rule for expanding to additional predetermined
folds are frozen before launch. This stage is configuration sensitivity, not a
hyperparameter search.

### Stage 2: Establish WSTS+ learned controls

Run the frozen WSTS+ split with:

1. Res18-U-Net, `T=1`, All features, as the bridge from the completed official
   reproduction;
2. UTAE(Res18), `T=5`, Multi features, as the stronger clean-observation
   backbone candidate.

Each starts with seed 0. Additional seeds are launched only after data,
numerical, runtime, and metric gates pass. Results are reported by test year as
well as in aggregate.

### Stage 3: Controlled-missingness diagnosis

Freeze a small corruption matrix before model development. It includes the
minimum set needed to test the research premise: missing or stale fire history,
single dynamic-modality loss, combined dynamic-modality loss, and structured
spatial block missingness at prespecified severities.

First apply the matrix test-only to the clean learned controls. If the baseline
does not show stable and scientifically meaningful degradation, the project
revisits its premise before building a complex method.

### Stage 4: Matched robust baselines and main method

Implement in increasing cost order:

1. zero-fill and last-observation-carried-forward controls;
2. direct concatenate-plus-mask;
3. uniform available-modality fusion;
4. a lightweight direct reliability gate;
5. MaskUNet reconstruction followed by the same frozen forecaster;
6. FireEx-style modality experts plus a capacity-matched generalist ensemble;
7. residual reliability gating augmented with SGMA-style semantic-prototype
   reliability, while retaining the same backbone and training budget.

The central comparison asks whether explicit reliability conditioning improves
robustness beyond capacity, masking, imputation, reconstruction, and uniform
fusion controls.

### Stage 5: Conditional extensions

These are not part of the initial implementation commitment:

- MaskCVAE, only if severe block missingness makes reconstruction competitive;
- STARS-style alignment, only if measured modality-representation drift remains
  after prototype reliability;
- an arbitrary-modal attention baseline, only if the direct gate requires a
  stronger generic-fusion control;
- a direct/reconstruction hybrid, only if the two paths win in distinct,
  reproducible missingness regimes;
- timestamp-centric methods such as AnytimeFormer/AGFlow, only if defensible
  acquisition or availability timestamps are recovered.

The linked planning conversation is treated as user-provided ideation. Before
the roadmap makes factual claims about FireEx, SGMA, MAGIC/Any2Seg, STARS,
MaskUNet/MaskCVAE, AnytimeFormer, or AGFlow, the implementation phase verifies
the relevant claim against the primary paper or official source. Unverified
ideas remain labelled hypotheses or candidate baselines.

## 10. Validation strategy

The repository reorganization is complete only when all of the following pass:

1. existing repository tests remain green;
2. focused tests validate required profile fields and rejection of secrets or
   machine-specific example values;
3. `submit.sh --dry-run` produces the expected one-GPU `sbatch` command and
   bounded array concurrency without invoking Slurm;
4. shell syntax checks pass for every tracked cluster script;
5. manifest validation rejects missing, extra, resized, or hash-mismatched
   files;
6. ignore-contract tests prove that real profiles, data, environments, weights,
   runs, checkpoints, caches, and scratch remain untracked;
7. secret, absolute-local-path, and accidental-large-file scans pass;
8. documentation links and existing GitHub Pages content remain valid;
9. `git diff --check` and final status checks pass.

Actual Slurm submission, server environment installation, data transfer, and
scientific execution are subsequent server-side operations. Local tests use
dry-run or fake command boundaries and must not start training.

## 11. Git integration and delivery

Implementation occurs in an isolated Git worktree. Changes are divided into
reviewable research-roadmap, cluster-boundary, and validation tasks. After
focused and full verification, the completed branch is merged into `main` and
`main` is pushed to `origin` as explicitly authorized by the user.

The push contains no data, weight, checkpoint, environment, cache, credential,
or local artifact. The resulting Git commit is the server migration starting
point.

## 12. Success criteria

The design succeeds when a student can clone one reviewed commit on a similar
Slurm cluster, fill one small local profile, verify the event-level data and
official weights, pass a one-GPU smoke/equivalence gate, and submit the next
declared experiment without editing scientific code or copying a site-specific
Slurm script.

It must remain visibly a fast research prototype: few scripts, explicit
configs, no hidden retries, no deployment stack, and no infrastructure work
that does not protect or accelerate a scientific decision.
