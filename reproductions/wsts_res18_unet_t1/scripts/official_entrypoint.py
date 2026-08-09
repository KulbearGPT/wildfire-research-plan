"""Execute the pinned upstream train.py unchanged and emit peak CUDA memory."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2 or arguments[0] != "--upstream-root":
        raise ValueError("the first arguments must be --upstream-root PATH")
    upstream_root = Path(arguments[1]).resolve()
    upstream_src = (upstream_root / "src").resolve()
    train_path = upstream_src / "train.py"
    if not train_path.is_file():
        raise ValueError(f"pinned upstream train.py is missing: {train_path}")

    sys.path.insert(0, str(upstream_src))
    sys.argv = [str(train_path), *arguments[2:]]
    runpy.run_path(str(train_path), run_name="__main__")

    import torch

    print(f"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
