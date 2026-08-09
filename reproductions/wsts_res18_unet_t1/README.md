# WSTS Res18-U-Net T=1 fold-2 timing calibration

This directory controls a provenance-checked timing calibration of the authors'
released Res18-U-Net, `T=1`, All-features configuration on official WSTS fold 2.
The 500-step timing run is **not a scientific reproduction result**: it does not
invoke the test loader and cannot establish the paper target `0.460 +/- 0.084`.

The official baseline command, run only from the pinned upstream checkout and
its isolated environment, is:

```powershell
python src/train.py
```

Before any run, validate the data inventory and exact non-scientific calibration
overrides with `scripts/control.py`. Keep the upstream checkout unmodified and
store all local clones, runs, logs, and learned artifacts under ignored paths.
