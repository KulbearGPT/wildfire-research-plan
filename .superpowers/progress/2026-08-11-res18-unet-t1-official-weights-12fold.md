# Res18-U-Net T=1 official-weights 12-fold publication

Status: publication documentation and completion audit complete.

The committed released-weight campaign is
`official-weight-12fold-20260812T052555Z`, generation
`generations/3c076108f46e4b519e65f8603b9d97a5`. Publication is explicitly a
test-only evaluation of released weights, not twelve new training runs. The
result supports released-weight executable reproducibility but does not prove
paper-table provenance identity; official focal-alpha behavior remains a future
training ablation outside this baseline claim.

Completion evidence: fresh focused publication tests passed (`4 passed`), and
the unique fresh full suite passed (`683 passed, 5 skipped in 46.45s`). The
read-only Fold-2 `verify_run` exactly matched its sealed JSON, including the
607-file / 24,242,259,023-byte source inventory; the original upstream was
clean and the derived checkout had only the initializer patch. Read-only
`load_committed_publication`, all 12 raw seals, CSV, summary, and all 12 weight
hashes passed; the publication hash and all three output hashes were unchanged.
Secret/local-path hits were 0, `git diff --check` was clean, and Python process
count was 0. The campaign verifier CLI was not rerun, and no scientific code or
artifact mutation occurred.

Task 5 TDD evidence is recorded in the task report.
