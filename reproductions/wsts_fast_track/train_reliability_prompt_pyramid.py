"""Run the fixed D10 reliability prompt pyramid continuation."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .contract import experiment_spec, validate_inventory
from .corrected_baselines import install_corrected_baseline
from .runtime import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .evaluate_missingness import load_checkpoint_model
from .corruption_training import BLOCK_DROPOUT_PROBABILITY, FIRE_DROPOUT_PROBABILITY
from .reliability_prompt_pyramid import (
    INPUT_PROMPT,
    PROMPT_POOLING,
    SEVERITY_THRESHOLD,
    CompleteReliabilityPromptPyramid,
    ReliabilityPromptPyramid,
    SeverityAdaptiveReliabilityPrompting,
)
from .train_predictive_consistency import validate_b3_record
from .train_reliability_normalized import ProcessedReliabilityDataset


TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-record", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variant",
        choices=(
            "prompt-pyramid",
            "complete-prompt-pyramid",
            "severity-adaptive-prompts",
        ),
        default="prompt-pyramid",
    )
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
    dataset = ProcessedReliabilityDataset(
        base_dataset, active_fire_missing_value
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
    model_class = {
        "prompt-pyramid": ReliabilityPromptPyramid,
        "complete-prompt-pyramid": CompleteReliabilityPromptPyramid,
        "severity-adaptive-prompts": SeverityAdaptiveReliabilityPrompting,
    }[args.variant]
    model = model_class(base_model).to(device)
    if tuple(model.prompt_channels) != (64, 64, 128, 256, 512):
        raise ValueError("D10 requires the fixed ResNet-18 encoder channels")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    iterator = iter(loader)
    final_loss = float("nan")
    for step in range(1, TRAINING_STEPS + 1):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target = next(iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = model.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"D10_STEP={step} LOSS={final_loss:.8f}", flush=True
            )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    complete = args.variant == "complete-prompt-pyramid"
    adaptive = args.variant == "severity-adaptive-prompts"
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": (
            "D12-SARP" if adaptive else "D11-CRPP" if complete else "D10-RPP"
        ),
        "matched_pair": "D12" if adaptive else "D11" if complete else "D10",
        "base_control": "D2-STD",
        "closest_ablation": "D4-TOKEN",
        "closest_ablations": (
            ["D4-TOKEN", "D10-RPP", "D11-CRPP"]
            if adaptive
            else ["D4-TOKEN", "D10-RPP"]
            if complete
            else None
        ),
        "base_b3_record": str(record_path),
        "base_b3_checkpoint": str(checkpoint),
        "experiment": "C00",
        "train_years": list(TRAIN_YEARS),
        "steps": TRAINING_STEPS,
        "seed": 0,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "variant": args.variant,
        "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
        "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
        "processed_space_matched_corruption": True,
        "prompt_channels": list(model.prompt_channels),
        "prompt_parameter_count": model.prompt_parameter_count,
        "prompt_pooling": PROMPT_POOLING,
        "input_prompt": INPUT_PROMPT if complete or adaptive else None,
        "severity_threshold": SEVERITY_THRESHOLD if adaptive else None,
        "mild_prompt": "hierarchical" if adaptive else None,
        "severe_prompt": "input" if adaptive else None,
        "final_loss": final_loss,
        "hyper_parameters": dict(model.base_model.hparams),
        "state_dict": model.state_dict(),
        "global_step": TRAINING_STEPS,
    }
    torch.save(payload, output)
    print(f"D10_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
