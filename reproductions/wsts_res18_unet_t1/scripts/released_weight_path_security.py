"""Fail-closed filesystem identity checks for released-weight recovery evidence."""

from __future__ import annotations

import os
import stat
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


def _is_reparse_or_link(path: Path) -> bool:
    metadata = path.lstat()
    is_junction = getattr(path, "is_junction", lambda: False)
    return bool(
        path.is_symlink()
        or is_junction()
        or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    )


def _require_exact_ancestry(
    path: Path, *, checked_root: Path, description: str, leaf_exists: bool
) -> Path:
    lexical = path.absolute()
    root_lexical = checked_root.absolute()
    try:
        root_resolved = checked_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{description} root is missing or inaccessible") from error
    if root_lexical != root_resolved:
        raise ValueError(f"{description} root is aliased or reparse-backed")
    current = lexical if leaf_exists else lexical.parent
    while True:
        if not os.path.lexists(current):
            raise ValueError(f"{description} ancestor is missing")
        try:
            if _is_reparse_or_link(current):
                raise ValueError(f"{description} is aliased or reparse-backed")
        except OSError as error:
            raise ValueError(f"{description} ancestry is inaccessible") from error
        if current == root_lexical:
            break
        if current.parent == current:
            raise ValueError(f"{description} escapes its root")
        current = current.parent
    # A caller-selected containment root is not itself a trust boundary: every
    # lexical ancestor up to the volume root must also be a real directory.
    current = root_lexical.parent
    while current != root_lexical:
        if not os.path.lexists(current):
            raise ValueError(f"{description} ancestor is missing")
        try:
            if _is_reparse_or_link(current):
                raise ValueError(f"{description} is aliased or reparse-backed")
        except OSError as error:
            raise ValueError(f"{description} ancestry is inaccessible") from error
        if current.parent == current:
            break
        current = current.parent
    if leaf_exists:
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{description} is missing or inaccessible") from error
        if resolved != lexical:
            raise ValueError(f"{description} is aliased or reparse-backed")
    return lexical


def require_sealed_regular_file(
    path: Path, *, checked_root: Path, description: str
) -> Path:
    lexical = _require_exact_ancestry(
        path, checked_root=checked_root, description=description, leaf_exists=True
    )
    metadata = lexical.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} is not a sealed regular file")
    if metadata.st_nlink != 1:
        raise ValueError(f"{description} is a hardlink or multiply linked")
    return lexical


def require_sealed_directory(
    path: Path, *, checked_root: Path, description: str
) -> Path:
    lexical = _require_exact_ancestry(
        path, checked_root=checked_root, description=description, leaf_exists=True
    )
    if not stat.S_ISDIR(lexical.lstat().st_mode):
        raise ValueError(f"{description} is not a directory")
    return lexical


def require_absent_recovery_path(
    path: Path, *, checked_root: Path, description: str
) -> Path:
    lexical = _require_exact_ancestry(
        path, checked_root=checked_root, description=description, leaf_exists=False
    )
    if os.path.lexists(lexical):
        raise ValueError(f"{description} exists, is dangling, or is reparse-backed")
    return lexical


def require_pairwise_distinct_files(paths: Iterable[Path], *, description: str) -> None:
    items = list(paths)
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            try:
                if left.samefile(right):
                    raise ValueError(f"{description} files are aliased")
            except OSError as error:
                raise ValueError(f"{description} identity is inaccessible") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reviewed_authorization_manifest(
    path: Path,
    *,
    expected_path: Path,
    repository_root: Path,
    campaign_directory: Path,
    run_directory: Path,
    reviewed_paths: Sequence[Path],
) -> dict[str, object]:
    """Validate an externally created approval against one clean reviewed commit."""
    repository = require_sealed_directory(
        repository_root,
        checked_root=repository_root,
        description="reviewed authorization repository",
    )
    campaign = require_sealed_directory(
        campaign_directory,
        checked_root=campaign_directory,
        description="reviewed authorization campaign",
    )
    run = require_sealed_directory(
        run_directory,
        checked_root=run_directory,
        description="reviewed authorization run",
    )
    expected = expected_path.absolute()
    approval = require_sealed_regular_file(
        path,
        checked_root=campaign,
        description="reviewed authorization manifest",
    )
    if approval != expected or approval.parent != campaign:
        raise ValueError("reviewed authorization manifest path mismatch")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.splitlines():
        raise ValueError("reviewed authorization requires an entirely clean worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("reviewed authorization commit is invalid")
    sealed_files: list[dict[str, object]] = []
    lexical_paths: list[Path] = []
    for item in reviewed_paths:
        reviewed = require_sealed_regular_file(
            item,
            checked_root=repository,
            description="reviewed authorization code file",
        )
        try:
            relative = reviewed.relative_to(repository)
        except ValueError as error:
            raise ValueError("reviewed authorization code path escapes repository") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise ValueError("reviewed authorization code file is not tracked")
        lexical_paths.append(reviewed)
        sealed_files.append(
            {
                "path": str(reviewed),
                "bytes": reviewed.stat().st_size,
                "sha256": _sha256(reviewed),
            }
        )
    require_pairwise_distinct_files(
        [approval, *lexical_paths], description="reviewed authorization"
    )
    expected_payload = {
        "schema_version": 1,
        "status": "APPROVED",
        "approval_scope": "exact Fold 0 offline recovery code after independent review",
        "campaign_id": campaign.name,
        "campaign_directory": str(campaign),
        "fold_id": 0,
        "run_directory": str(run),
        "reviewed_commit": commit,
        "reviewed_files": sealed_files,
    }
    try:
        payload = json.loads(approval.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("reviewed authorization manifest is invalid JSON") from error
    if type(payload) is not dict or payload != expected_payload:
        raise ValueError("reviewed authorization manifest schema or identity mismatch")
    return payload
