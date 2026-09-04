"""Run the fixed D7-CRA continuation from corrected B3."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import experiment_spec, validate_inventory
from .corrected_baselines import install_corrected_baseline
from .counterfactual_impact_consistency import (
    IMPACT_WEIGHTING,
    counterfactual_impact_kl_from_logits,
)
from .counterfactual_reliability_adapter import (
    ADAPTER_PARAMETER_COUNT,
    RELIABILITY_MAPS,
    CounterfactualReliabilityAdapter,
    pack_reliability_input,
)
from .entrypoint import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .evaluate_missingness import load_checkpoint_model
from .predictive_consistency import CleanCorruptPairDataset
from .prototype import BLOCK_DROPOUT_PROBABILITY, FIRE_DROPOUT_PROBABILITY
from .train_predictive_consistency import validate_b3_record


TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-3
CIWC_WEIGHT = 0.1


def cra_training_objective(
    model: Any,
    clean_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return D5 supervision and impact consistency on adapted logits."""

    clean_loss = model.compute_loss(clean_logits, target)
    adapted_loss = model.compute_loss(adapted_logits, target)
    supervised = 0.5 * (clean_loss + adapted_loss)
    ciwc = counterfactual_impact_kl_from_logits(clean_logits, adapted_logits)
    objective = supervised + CIWC_WEIGHT * ciwc
    return objective, {
        "supervised": float(supervised.detach().cpu()),
        "ciwc": float(ciwc.detach().cpu()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-record", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    record_path = args.b3_record.resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("B3 completion record must be an object")
    checkpoint = Path(validate_b3_record(record)).resolve(strict=True)
    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    install_corrected_baseline(upstream, "B0")

    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    spec = experiment_spec("C00")
    base_dataset = dataset_class(
        data_dir=str(data_root),
        included_fire_years=list(TRAIN_YEARS),
        n_leading_observations=spec.n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=spec.remove_duplicate_features,
        features_to_keep=None,
        return_doy=False,
        stats_years=list(TRAIN_YEARS),
        is_pad=False,
    )
    means = np.asarray(base_dataset.means)
    stds = np.asarray(base_dataset.stds)
    active_fire_missing_value = float(
        (0.0 - means[0, 22, 0, 0]) / stds[0, 22, 0, 0]
    )
    dataset = CleanCorruptPairDataset(
        base_dataset,
        fire_probability=FIRE_DROPOUT_PROBABILITY,
        block_probability=BLOCK_DROPOUT_PROBABILITY,
        active_fire_missing_value=active_fire_missing_value,
        return_reliability=True,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but unavailable")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    base_model = load_checkpoint_model(
        checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    model = CounterfactualReliabilityAdapter(base_model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    iterator = iter(loader)
    final_loss = float("nan")
    final_parts: dict[str, float] = {}
    for step in range(1, TRAINING_STEPS + 1):
        try:
            clean_x, corrupt_x, target, fire_missing, block_missing = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            clean_x, corrupt_x, target, fire_missing, block_missing = next(iterator)
        clean_x = clean_x.to(device, non_blocking=True)
        corrupt_x = corrupt_x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        fire_missing = fire_missing.to(device, non_blocking=True)
        block_missing = block_missing.to(device, non_blocking=True)
        packed_corrupt = pack_reliability_input(
            corrupt_x, fire_missing, block_missing
        )
        optimizer.zero_grad(set_to_none=True)
        clean_logits = model.clean_forward(clean_x).squeeze(1)
        adapted_logits = model(packed_corrupt).squeeze(1)
        loss, final_parts = cra_training_objective(
            model, clean_logits, adapted_logits, target
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"D7_STEP={step} LOSS={final_loss:.8f} "
                f"SUPERVISED={final_parts['supervised']:.8f} "
                f"CIWC={final_parts['ciwc']:.8f}",
                flush=True,
            )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D7-CRA",
        "matched_pair": "D7",
        "base_control": "D1-ERM",
        "closest_ablations": ["D5-CIWC", "D4-TOKEN", "P04-P06"],
        "base_b3_record": str(record_path),
        "base_b3_checkpoint": str(checkpoint),
        "experiment": "C00",
        "train_years": list(TRAIN_YEARS),
        "steps": TRAINING_STEPS,
        "seed": 0,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "lambda_ciwc": CIWC_WEIGHT,
        "impact_weighting": IMPACT_WEIGHTING,
        "adapter_parameter_count": ADAPTER_PARAMETER_COUNT,
        "reliability_maps": list(RELIABILITY_MAPS),
        "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
        "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
        "final_loss": final_loss,
        "final_loss_components": final_parts,
        "hyper_parameters": dict(model.base_model.hparams),
        "base_state_dict": model.base_model.state_dict(),
        "adapter_state_dict": model.adapter.state_dict(),
        "global_step": TRAINING_STEPS,
    }
    torch.save(payload, output)
    print(f"D7_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
