"""Independent released-weight result validation primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path


def validate_weight_result(payload: Mapping[str, object]) -> dict[str, float]:
    if payload.get("exit_code") != 0:
        raise ValueError("released-weight child exit code is not zero")
    if payload.get("strict_load") is not True:
        raise ValueError("released weight was not loaded with strict=True")
    for action in ("train", "validation", "predict"):
        if payload.get(f"{action}_invoked") is not False:
            raise ValueError(f"released-weight evaluation invoked {action}")
    metrics = payload.get("test_metrics")
    if not isinstance(metrics, Mapping) or "test_AP" not in metrics:
        raise ValueError("released-weight test_AP is missing")
    test_ap = float(metrics["test_AP"])
    filename_ap = float(payload.get("filename_ap", math.nan))
    if not math.isfinite(test_ap) or not 0.0 <= test_ap <= 1.0:
        raise ValueError("released-weight test_AP is invalid")
    if not math.isfinite(filename_ap) or not 0.0 <= filename_ap <= 1.0:
        raise ValueError("released-weight filename AP is invalid")
    return {
        "test_AP": test_ap,
        "filename_ap": filename_ap,
        "filename_ap_absolute_difference": abs(test_ap - filename_ap),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run(run_directory: Path, weight_path: Path) -> dict[str, object]:
    run = run_directory.resolve()
    result = json.loads((run / "weight-result.json").read_text(encoding="utf-8"))
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (
        run / "stderr.log"
    ).read_text(encoding="utf-8", errors="replace")
    if int((run / "exit-code.txt").read_text(encoding="utf-8").strip()) != 0:
        raise ValueError("released-weight exit code is not zero")
    if len(re.findall(r"WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=\d+", output)) != 1:
        raise ValueError("released-weight strict-load evidence is missing")
    if re.search(r"\bEpoch\s+\d+:|Validation DataLoader|Predicting DataLoader", output):
        raise ValueError("released-weight output contains a forbidden action")
    if _sha256(weight_path) != result.get("weight_sha256"):
        raise ValueError("released-weight SHA-256 differs from the result")
    summary = validate_weight_result(result)
    return {
        "status": "pass",
        "weight_sha256": result["weight_sha256"],
        **summary,
        "independent_implementation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--weight-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = verify_run(arguments.run_directory, arguments.weight_path)
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
