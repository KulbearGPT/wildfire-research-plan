# Finish an interrupted environment setup

Ordinary `submit.sh cpu setup` intentionally refuses existing nonempty environment or upstream directories. If setup completed the training dependency installation and upstream clone but stopped before completion, inspect its Slurm log, fix the relevant tracked source, and commit that fix. Then use the same site configuration and root:

```bash
export WILDFIRE_SITE_ENV="/path/to/site.env"
bash scripts/research/submit.sh cpu setup-finish
```

Wait for the original setup job to terminate first; never run two setup jobs against one root. The new job runs from the newly committed archive and records its own source, allocation, upstream diff, and dependency freezes. It does not change the already submitted or completed job's archived source or logs.

`setup-finish` requires an existing executable training environment and cloned upstream repository. It reruns the pinned pip requirements without forcing reinstall, checks dependency consistency, validates the exact upstream commit, applies any missing approved compatibility change, reuses the encoder download cache, and creates or finishes the audit environment. It does not delete environments, reset the upstream checkout, or discard partial outputs. A nonempty audit directory without an executable Python is rejected for explicit inspection.

Upstream may be pristine or already contain the retained model-import patch and/or removal of the obsolete `T_co` import. Both affected files are compared in full against their pinned originals and exact expected transformed bytes. Any other tracked change, untracked file, unexpected commit, or unexpected contents in those files is rejected. The model-import patch includes the pinned upstream file's missing terminal-newline marker; it does not silently accept source drift.

This action verifies requirements rather than proving that earlier installations came from public wheels. If a previous job used a site vendor wheelhouse, `setup-finish` can reuse those installed packages. To qualify an entirely public-PyPI installation, choose a distinct empty root/upstream and run ordinary `setup` with the corrected launcher. Completing setup also does not verify a training or checkpoint-resume path; those require separate allocated qualification jobs.
