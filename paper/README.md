# Paper materials

The current author package is **2026-09-26, version 2**. It covers the selected
X22+X17 mainline, necessary controls, and useful archived evidence without
reactivating archived research directions.

- [Download the local ZIP](../deliverables/paper-materials-20260926-v2.zip)
- [Read the seven-page PDF overview](../deliverables/paper-materials-20260926-v2/report.pdf)
- [Open the offline figure gallery](../deliverables/paper-materials-20260926-v2/index.html)
- [Read the maintained materials guide](materials/README.md)
- [Review the proposed contributions and evidence limits](materials/CLAIMS_AND_LIMITATIONS.md)

Generated deliverables are local author artifacts and are ignored by Git. The
download links work in the build workspace; a repository clone alone does not
contain the ZIP. The source scripts, prose, and this build record are committed.

## Contents and evidence

The package includes six figures as vector PDF, editable SVG, and 300 dpi PNG;
full-precision CSVs; two LaTeX tables; English experiment prose and captions;
455 copied source records with checksums; 1,564 final scenario records; and
the complete 100-entry method catalogue with lifecycle and evidence status.
The final row count includes 144 explicitly derived system-composition rows.
Archived single-run results retain their original comparators and scope.

The principal effect is +0.014642 primary AP over matched fresh ERM, averaged
equally over two settings, three years, and three seeds. The separate frozen
B3/B5-reference effect is +0.034527 AP. The closer X22+mixed-X14 control gains
+0.014287 AP over ERM; X22+X17 adds only +0.000355 AP overall and -0.000226 AP
on historical 2022/2023 data, with four retained models versus three.

The audit also found that the historical complete-route artifact uses X22 on
M00, while the current documented route uses ERM. Both are preserved explicitly;
their missingness primary scores agree, but clean scores and model counts differ.

## Build provenance

- Evidence snapshot: `5b457c3ea6f2f6063ee260edae7ff8404709b505`.
- Final builder and manuscript source: `82c6b4d5918eb633903b58602eb4d334bd0a7dc8`.
- Final evidence collection: Slurm CPU job `22716756`.
- Initial build and independent audit: CPU jobs `22716927` and `22717041`.
- Final build and independent audit: CPU job `22717106`, node `c549`.
- Resources per build/audit allocation: two CPUs, 4 GiB, 15-minute limit; no GPU.
- ZIP SHA-256: `b01221ca1c7cd5c55158e34f8da9481520ef5aee037b60415cff92ff5bd5cb93`.

The build checks source hashes, all original-summary AP references, unique
cells, complete mainline coverage, historical/current route identity, and
reported aggregate effects. Five focused synthetic tests check metric and
composition semantics. An independent extraction audit verifies every file
listed in the package checksum manifest and regenerates all six figures from
the extracted CSVs. `BUILD_REPORT.json`, `checksums.json`, source manifests,
and `REPRODUCE.md` travel inside the ZIP.

In this workspace, the additional audit output is
`tmp/paper-materials-20260926/verification-v2/validation.json`. Slurm source,
command, and log records are under the site's jobs directory:
`/project/6085198/kulbear/wildfire/handoff-20260916/public environment/jobs/20260926T134652Z-82c6b4d5918e-OuqvKe/`.

No training, model inference, or dataset tensor processing was performed.
This is an author materials package, not an anonymized conference submission.
It does not add qualitative predictions, event-level confidence intervals,
or independent generalization evidence absent from the retained experiments.
