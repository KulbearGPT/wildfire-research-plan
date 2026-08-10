import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_full_fold import verify_recorded_command, verify_result_summary  # noqa: E402


def test_independent_full_verifier_requires_exact_command_and_hash() -> None:
    command = ["python.exe", "entrypoint.py", "--trainer.max_steps=10000", "--do_test=true"]
    payloads = {
        "launch": {"command": command, "command_sha256": "abc"},
        "started": {"command": command, "command_sha256": "abc"},
        "effective": {"command": command, "command_sha256": "abc"},
        "result": {"command_sha256": "abc"},
    }

    verify_recorded_command(payloads, command, "abc")
    del payloads["started"]["command_sha256"]
    with pytest.raises(ValueError, match="started.command_sha256"):
        verify_recorded_command(payloads, command, "abc")


def test_independent_full_verifier_rejects_metric_or_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"weights")
    raw = {
        "optimizer_steps": 10_000,
        "test_metrics": {"test_AP": 0.5712, "test_f1": 0.4},
        "best_checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
    }
    recorded = json.loads(json.dumps(raw))

    verify_result_summary(raw, recorded)
    recorded["test_metrics"]["test_AP"] = 0.6
    with pytest.raises(ValueError, match="full result summary"):
        verify_result_summary(raw, recorded)
