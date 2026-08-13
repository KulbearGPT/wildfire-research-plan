"""Fail-closed filesystem identity checks for released-weight recovery evidence."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterable


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
