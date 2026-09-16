"""Controlled-area masks derived from cached VIIRS observation footprints."""
from functools import lru_cache
from pathlib import Path
from reproductions.paths import load_paths

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


DEFAULT_BANK = load_paths().root / (
    "archive/pre-t1-cleanup-2026-09-04/"
    "runs/viirs-reliability-24-20804128/output"
)


def _digest_bytes(key_digest: str) -> bytes:
    if len(key_digest) != 64:
        raise ValueError("key_digest must contain 64 hexadecimal characters")
    try:
        return bytes.fromhex(key_digest)
    except ValueError as exc:
        raise ValueError("key_digest must contain 64 hexadecimal characters") from exc


def footprint_mask(
    reliability: np.ndarray,
    height: int,
    width: int,
    fraction: float,
    *,
    key_digest: str,
) -> np.ndarray:
    """Resize and area-normalize one observed reliability footprint."""
    key = _digest_bytes(key_digest)
    values = np.asarray(reliability)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("reliability must be a non-trivial 2D array")
    if not np.isfinite(values).all():
        raise ValueError("reliability must be finite")
    count = round(height * width * fraction)
    if height < 1 or width < 1 or count < 1 or count >= height * width:
        raise ValueError("target shape and fraction must define a partial mask")

    missing = values < 0.5
    rotation = key[0] % 4
    missing = np.rot90(missing, rotation)
    if key[1] & 1:
        missing = np.flip(missing, axis=0)
    if key[1] & 2:
        missing = np.flip(missing, axis=1)
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(missing.copy())[None, None].float(),
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy().astype(bool)

    # Signed distance preserves the observed footprint topology while ranking
    # every pixel, allowing a fair exact-area comparison with M06/M07 blocks.
    score = distance_transform_edt(~resized) - distance_transform_edt(resized)
    rng = np.random.default_rng(int.from_bytes(key[2:10], "big"))
    score = score + rng.random(score.shape) * 1e-6
    selected = np.argpartition(score.ravel(), count - 1)[:count]
    result = np.zeros(height * width, dtype=bool)
    result[selected] = True
    return result.reshape(height, width)


@lru_cache(maxsize=4)
def load_footprint_bank(root: str = str(DEFAULT_BANK)) -> tuple[np.ndarray, ...]:
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no VIIRS reliability footprints under {root}")
    bank = []
    for path in paths:
        with np.load(path) as payload:
            bank.append(np.asarray(payload["reliability"], dtype=np.uint8))
    return tuple(bank)


def sampled_footprint_mask(
    height: int, width: int, fraction: float, *, key_digest: str
) -> np.ndarray:
    key = _digest_bytes(key_digest)
    bank = load_footprint_bank()
    source = bank[int.from_bytes(key[10:18], "big") % len(bank)]
    return footprint_mask(source, height, width, fraction, key_digest=key_digest)
