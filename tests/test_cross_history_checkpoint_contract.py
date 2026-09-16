"""Formal artifact guards; scheduled with the controller's Slurm tests."""
import pytest
from reproductions.cross_history.checkpoint_contract import require_formal_checkpoint


def test_historical_formal_checkpoint_remains_compatible():
    require_formal_checkpoint({'steps': 3000})


def test_complete_new_formal_checkpoint_is_accepted():
    require_formal_checkpoint({'smoke': False, 'steps': 3000, 'completed_steps': 3000})


@pytest.mark.parametrize('payload', [
    {'smoke': True, 'steps': 3000, 'completed_steps': 1},
    {'smoke': True, 'steps': 1, 'completed_steps': 1},
    {'smoke': False, 'steps': 3000, 'completed_steps': 1},
    {'smoke': False, 'steps': 3000, 'completed_steps': 0},
    {'smoke': False, 'steps': 1, 'completed_steps': True},
    {'smoke': False, 'steps': 0, 'completed_steps': 0},
])
def test_smoke_and_incomplete_artifacts_cannot_be_formal_sources(payload):
    with pytest.raises(ValueError):
        require_formal_checkpoint(payload)
