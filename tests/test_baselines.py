import numpy as np
import pytest

from wildfire_phase0.baselines import no_fire, persistence


def test_persistence_modes() -> None:
    history = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.uint8)

    assert no_fire((2, 2)).sum() == 0
    assert persistence(history, "latest").tolist() == [[0, 1], [0, 0]]
    assert persistence(history, "all").tolist() == [[1, 1], [0, 0]]


def test_no_fire_returns_fresh_uint8_array() -> None:
    first = no_fire((1, 2))
    second = no_fire((1, 2))
    first[0, 0] = 1

    assert first.dtype == np.uint8
    assert second.tolist() == [[0, 0]]


@pytest.mark.parametrize("shape", [(2,), (2, 0), (0, 2), (1, 2, 3)])
def test_no_fire_rejects_non_positive_or_non_2d_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        no_fire(shape)


@pytest.mark.parametrize(
    "history",
    [np.zeros((2, 2), dtype=np.uint8), np.zeros((0, 2, 2), dtype=np.uint8)],
)
def test_persistence_rejects_non_3d_or_empty_time_history(history: np.ndarray) -> None:
    with pytest.raises(ValueError):
        persistence(history, "latest")


def test_persistence_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        persistence(np.zeros((1, 2, 2), dtype=np.uint8), "earliest")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["latest", "all"])
def test_persistence_rejects_non_binary_history(mode: str) -> None:
    history = np.array([[[0.0, 1.0], [2.0, np.nan]]])

    with pytest.raises(ValueError, match="binary"):
        persistence(history, mode)  # type: ignore[arg-type]
