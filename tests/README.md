# Test scope after mainline selection

Tests for both active and archived implementations remain here. Their presence
does not reactivate a research direction. Use [CODE_LIFECYCLE.md](../docs/CODE_LIFECYCLE.md)
and the source file's lifecycle comment to choose affected checks.

The X22+X17 route uses cross-history training/data/checkpoint contracts,
severity/complete-route composition, corrected B3/B5 foundations and shared
evaluation/data support. Tests for D12, belief states, RF/TD, optional architectures
and other X-series methods preserve archive reproducibility. Run them when their
behavior or shared dependencies change, not automatically for every cleanup.

Model imports, tensor tests and numerical evaluation require Slurm. Pure source
inspection, JSON metadata checks and shell syntax checks do not require a training
job. A successful archive smoke check establishes execution, not scientific benefit.

`test_mainline_cli.py` checks only command construction and mocked dispatch using
the standard library. It imports no trainer, numerical package or dataset and
does not call the scheduler. Run this control-plane check with:

```bash
python3 -m unittest discover -s tests -p test_mainline_cli.py -v
```
