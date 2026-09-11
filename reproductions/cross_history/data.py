"""Share targets and corruption semantics across existing T1 and T5 models."""
from pathlib import Path
import hashlib
import importlib
import numpy as np
import torch

from reproductions.wsts_fast_track.runtime import (
    TRAIN_YEARS, _install_runtime_contract, load_training_stats,
)
from reproductions.wsts_fast_track.corrected_baselines import install_corrected_baseline
from reproductions.wsts_fast_track.evaluation import ControlledMissingnessDataset
from reproductions.wsts_fast_track.processed_reliability import (
    PROCESSED_DYNAMIC_NON_FIRE,
)
from reproductions.wsts_fast_track.missingness import structured_block_mask
from .reliability_masks import sampled_footprint_mask

ROOT = Path('/project/6085198/kulbear/wildfire')
UPSTREAM = ROOT / 'cache/WildfireSpreadTS-res18-runtime'
DATA = ROOT / 'hdf5/wstsplus-active-fixed'
STATS = ROOT / 'runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz'
MULTI_FEATURES = (0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,
                  22,23,24,25,26,27,28,29,30,31,32,38,39)


def setup():
    _install_runtime_contract(UPSTREAM, 'C00', load_training_stats(STATS))
    install_corrected_baseline(UPSTREAM, 'B0')


def base_dataset(history, *, year=None):
    cls = importlib.import_module('dataloader.FireSpreadDataset').FireSpreadDataset
    return cls(data_dir=str(DATA), included_fire_years=list(TRAIN_YEARS) if year is None else [year],
               n_leading_observations=history,
               n_leading_observations_test_adjustment=None if year is None else 6,
               crop_side_length=128, load_from_hdf5=True, is_train=year is None,
               remove_duplicate_features=history == 1,
               features_to_keep=None if history == 1 else list(MULTI_FEATURES),
               return_doy=False, stats_years=list(TRAIN_YEARS), is_pad=False)


class PairedDataset:
    def __init__(self, base, history, *, fire_probability=.3, block_probability=.3,
                 block_fraction=None, block_candidates=1, reliability_footprints=False):
        self.base = base
        self.columns = tuple(range(40)) if history == 1 else MULTI_FEATURES
        self.dynamic = tuple(i for i, c in enumerate(self.columns) if c in PROCESSED_DYNAMIC_NON_FIRE)
        self.fire_value = self.columns.index(38)
        self.fire_binary = self.columns.index(39)
        self.missing_value = float(-base.means[0,22,0,0] / base.stds[0,22,0,0])
        self.fire_probability = fire_probability
        self.block_probability = block_probability
        if block_fraction not in (None, 0.25, 0.5):
            raise ValueError('block_fraction must be None, 0.25, or 0.5')
        self.block_fraction = block_fraction
        if block_candidates not in (1, 2):
            raise ValueError('block_candidates must be one or two')
        self.block_candidates = block_candidates
        self.reliability_footprints = reliability_footprints

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        clean, target = self.base[index]
        fire_drop = bool(np.random.random() < self.fire_probability)
        block_drop = bool(np.random.random() < self.block_probability)
        sampled_fraction = 0.25 if np.random.random() < 0.5 else 0.5
        fraction = self.block_fraction if self.block_fraction is not None else sampled_fraction
        digest = np.random.bytes(32).hex()

        packed_candidates = []
        for candidate in range(self.block_candidates):
            x = clean.clone()
            invalid = torch.zeros(clean.shape[-2:], dtype=torch.bool)
            if block_drop:
                # Derive extra hard-mining choices without consuming another
                # RNG draw: candidate zero remains exactly paired to the
                # ordinary one-block control for every sample and worker.
                candidate_digest = digest if candidate == 0 else hashlib.sha256(
                    f'{digest}-{candidate}'.encode('ascii')).hexdigest()
                mask_fn = sampled_footprint_mask if self.reliability_footprints else structured_block_mask
                invalid = torch.from_numpy(mask_fn(
                    *clean.shape[-2:], fraction, key_digest=candidate_digest))
                x[:,self.dynamic] = x[:,self.dynamic].masked_fill(invalid[None,None], 0.)
            fire_invalid = invalid | fire_drop
            x[:,self.fire_value] = x[:,self.fire_value].masked_fill(
                fire_invalid[None], self.missing_value)
            x[:,self.fire_binary] = x[:,self.fire_binary].masked_fill(fire_invalid[None], 0.)
            masks = torch.stack((invalid, fire_invalid)).to(x.dtype)[None].expand(
                x.shape[0],-1,-1,-1)
            packed_candidates.append(torch.cat((x,masks),dim=1))
        packed = packed_candidates[0] if self.block_candidates == 1 else torch.stack(
            packed_candidates, dim=0)
        return packed, target, clean


class EvaluationDataset(ControlledMissingnessDataset):
    def __getitem__(self, index):
        packed, target = super().__getitem__(index)
        fire_invalid = torch.ones_like(packed[:,-1:]) if self.scenario_id == 'M01' else packed[:,-1:]
        return torch.cat((packed,fire_invalid),dim=1), target


def evaluation_dataset(history, year, scenario):
    return EvaluationDataset(base_dataset(history, year=year), scenario,
        evaluation_year=year, heldout_authorized=year != 2021,
        routing_mask_channel=True, allowed_histories=(1,5))
