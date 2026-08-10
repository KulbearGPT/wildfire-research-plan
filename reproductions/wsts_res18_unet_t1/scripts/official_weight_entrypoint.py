"""Load one official raw state dict strictly and invoke only Trainer.test."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    strict_load_only = bool(arguments and arguments[0] == "--strict-load-only")
    if strict_load_only:
        arguments = arguments[1:]
    if len(arguments) < 4 or arguments[0] != "--upstream-root" or arguments[2] != "--weights-path":
        raise ValueError(
            "the first arguments must be --upstream-root PATH --weights-path PATH"
        )
    upstream_root = Path(arguments[1]).resolve()
    weights_path = Path(arguments[3]).resolve()
    upstream_src = (upstream_root / "src").resolve()
    train_path = upstream_src / "train.py"
    if not train_path.is_file() or not weights_path.is_file():
        raise ValueError("pinned train.py or released weight is missing")

    sys.path.insert(0, str(upstream_src))
    sys.argv = [str(train_path), *arguments[4:]]
    namespace = runpy.run_path(str(train_path), run_name="wsts_official_weight_eval")

    import torch

    cli = namespace["MyLightningCLI"](
        namespace["BaseModel"],
        namespace["FireSpreadDataModule"],
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
    )
    raw_state = torch.load(weights_path, map_location="cpu")
    if not isinstance(raw_state, dict) or "state_dict" in raw_state:
        raise ValueError("released Fold-2 weight must be a raw state_dict")
    cli.model.load_state_dict(raw_state, strict=True)
    print(
        f"WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors={len(raw_state)}",
        flush=True,
    )
    if strict_load_only:
        return 0
    cli.trainer.test(cli.model, cli.datamodule)
    print(
        f"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
