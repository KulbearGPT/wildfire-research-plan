"""Independently verify a preserved full Fold-2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_recorded_command(
    payloads: Mapping[str, Mapping[str, object]],
    expected_command: Sequence[str],
    expected_hash: str,
) -> None:
    required = {
        "launch": ("command", "command_sha256"),
        "started": ("command", "command_sha256"),
        "effective": ("command", "command_sha256"),
        "result": ("command_sha256",),
    }
    for label, fields in required.items():
        payload = payloads.get(label)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} marker is missing")
        for field in fields:
            if field not in payload:
                raise ValueError(f"{label}.{field} is required")
        if "command" in fields and payload["command"] != list(expected_command):
            raise ValueError(f"{label}.command differs from the exact full command")
        if payload["command_sha256"] != expected_hash:
            raise ValueError(f"{label}.command_sha256 mismatch")


def verify_result_summary(
    raw: Mapping[str, object], recorded: Mapping[str, object]
) -> None:
    if dict(raw) != dict(recorded):
        raise ValueError("full result summary differs from independently reconstructed values")
    checkpoint = Path(str(raw.get("best_checkpoint", "")))
    if not checkpoint.is_file():
        raise ValueError("recorded best checkpoint is missing")
    if raw.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("best checkpoint SHA-256 mismatch")
    if raw.get("optimizer_steps") != 10_000:
        raise ValueError("full result optimizer step count mismatch")


def _parse_metrics(output: str) -> dict[str, float]:
    values = {
        name: float(raw)
        for name, raw in re.findall(
            r"(?m)[│|]\s*(test_(?:AP|f1|iou|loss|precision|recall))\s*[│|]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            output,
        )
    }
    if "test_AP" not in values:
        raise ValueError("independent verifier could not reconstruct test_AP")
    return values


def verify_run(run_directory: Path) -> dict[str, object]:
    run = run_directory.resolve()
    effective = json.loads((run / "effective-command.json").read_text(encoding="utf-8"))
    command = effective["command"]
    command_hash = _command_sha256(command)
    if command_hash != effective["command_sha256"]:
        raise ValueError("effective command SHA-256 mismatch")
    global_lock = json.loads(
        (run.parent / "fold2-full-launch.lock.json").read_text(encoding="utf-8")
    )
    run_lock = json.loads((run / "launch.lock.json").read_text(encoding="utf-8"))
    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    recorded = json.loads((run / "full-result.json").read_text(encoding="utf-8"))
    verify_recorded_command(
        {"launch": global_lock, "started": started, "effective": effective, "result": recorded},
        command,
        command_hash,
    )
    output = (run / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (
        run / "stderr.log"
    ).read_text(encoding="utf-8", errors="replace")
    if int((run / "exit-code.txt").read_text(encoding="utf-8").strip()) != 0:
        raise ValueError("full child exit code is not zero")
    if re.search(r"max_steps=10000`?\s+reached", output) is None:
        raise ValueError("independent max_steps=10000 evidence is missing")
    if re.search(r"Testing DataLoader 0:\s*100%", output) is None:
        raise ValueError("independent test completion evidence is missing")
    if "Predicting DataLoader" in output:
        raise ValueError("predict action exists in full run output")
    checkpoints = sorted(run.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError("independent checkpoint inventory is not exactly one")
    metrics = _parse_metrics(output)
    raw = dict(recorded)
    raw["test_metrics"] = metrics
    raw["best_checkpoint"] = str(checkpoints[0].resolve())
    raw["checkpoint_sha256"] = _sha256(checkpoints[0])
    raw["optimizer_steps"] = 10_000
    verify_result_summary(raw, recorded)
    return {
        "status": "pass",
        "command_sha256": command_hash,
        "optimizer_steps": 10_000,
        "test_metrics": metrics,
        "checkpoint_sha256": _sha256(checkpoints[0]),
        "independent_implementation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--original-upstream", required=True, type=Path)
    parser.add_argument("--derived-upstream", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = verify_run(arguments.run_directory)
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

