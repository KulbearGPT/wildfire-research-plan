"""Canonical filesystem-root and containment checks for mutating workflows."""

from collections.abc import Mapping
from pathlib import Path


def canonical_root(path: Path, label: str, *, must_exist: bool = False) -> Path:
    """Resolve one directory root without mutating it."""
    resolved = Path(path).expanduser().resolve(strict=False)
    if must_exist and not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def require_pairwise_disjoint_roots(roots: Mapping[str, Path]) -> None:
    """Reject equal, nested, ancestor, or resolved-aliased roots."""
    items = tuple(roots.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "filesystem roots must be pairwise disjoint: "
                    f"{left_name}={left} overlaps {right_name}={right}"
                )


def require_contained_path(root: Path, path: Path, designation: str) -> Path:
    """Resolve a path and require it to remain strictly below its root."""
    resolved = Path(path).resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"path resolves outside designated {designation}: {path}")
    return resolved
