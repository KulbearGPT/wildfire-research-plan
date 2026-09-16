"""Train matched standard/SARP continuations for the C02 T=5 model."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .contract import experiment_spec, validate_inventory
from .corrected_baselines import install_corrected_baseline
from .runtime import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .evaluate_corrected_baseline import validate_evaluation_record
from .evaluate_missingness import load_checkpoint_model
from .corruption_training import BLOCK_DROPOUT_PROBABILITY, FIRE_DROPOUT_PROBABILITY
from .reliability_prompt_pyramid import PROMPT_POOLING, SEVERITY_THRESHOLD
from .temporal_reliability_prompting import (
    TEMPORAL_FEATURE_COUNT,
    TEMPORAL_STEPS,
    ProcessedTemporalReliabilityDataset,
    TemporalSeverityAdaptiveReliabilityPrompting,
)


TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-3


def validate_b5_record(record: Mapping[str, object]) -> str:
    """Return the checkpoint from the exact corrected T=5 robust base."""

    try:
        baseline = validate_evaluation_record(record)
    except ValueError as error:
        raise ValueError("T5 continuation requires a corrected B5 record") from error
    if baseline.baseline_id != "B5":
        raise ValueError("T5 continuation requires a corrected B5 record")
    return str(record["checkpoint"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b5-record", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--variant", choices=("standard", "sarp"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    record_path = args.b5_record.resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("B5 completion record must be an object")
    checkpoint = Path(validate_b5_record(record)).resolve(strict=True)
    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C02", stats)
    install_corrected_baseline(upstream, "B1")

    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    spec = experiment_spec("C02")
    base_dataset = dataset_class(
        data_dir=str(data_root),
        included_fire_years=list(TRAIN_YEARS),
        n_leading_observations=spec.n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=spec.remove_duplicate_features,
        features_to_keep=list(spec.features_to_keep or ()),
        return_doy=False,
        stats_years=list(TRAIN_YEARS),
        is_pad=False,
    )
    means = np.asarray(base_dataset.means)
    stds = np.asarray(base_dataset.stds)
    active_fire_missing_value = float(
        (0.0 - means[0, 22, 0, 0]) / stds[0, 22, 0, 0]
    )
    dataset = ProcessedTemporalReliabilityDataset(
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
        experiment_id="C02",
        upstream_root=upstream,
        device=device,
    )
    model: torch.nn.Module
    if args.variant == "sarp":
        model = TemporalSeverityAdaptiveReliabilityPrompting(base_model).to(device)
        if tuple(model.prompt_channels) != (64, 64, 128, 256, 512):
            raise ValueError("T5 SARP requires ResNet-18 encoder channels")
    else:
        model = base_model
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
        model_input = x if args.variant == "sarp" else x[:, :, :-1]
        logits = model(model_input).squeeze(1)
        loss = model.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"D13_STEP={step} VARIANT={args.variant} LOSS={final_loss:.8f}",
                flush=True,
            )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    sarp = args.variant == "sarp"
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D13-SARP-T5" if sarp else "D13-STD-T5",
        "matched_pair": "D13-T5",
        "base_control": "B5",
        "base_b5_record": str(record_path),
        "base_b5_checkpoint": str(checkpoint),
        "experiment": "C02",
        "train_years": list(TRAIN_YEARS),
        "steps": TRAINING_STEPS,
        "seed": 0,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "variant": args.variant,
        "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
        "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
        "processed_space_matched_corruption": True,
        "temporal_steps": TEMPORAL_STEPS,
        "feature_count": TEMPORAL_FEATURE_COUNT,
        "prompt_channels": list(model.prompt_channels) if sarp else None,
        "prompt_parameter_count": model.prompt_parameter_count if sarp else 0,
        "prompt_pooling": PROMPT_POOLING if sarp else None,
        "severity_threshold": SEVERITY_THRESHOLD if sarp else None,
        "mild_prompt": "hierarchical-before-temporal-fusion" if sarp else None,
        "severe_prompt": "per-step-input" if sarp else None,
        "final_loss": final_loss,
        "hyper_parameters": dict(base_model.hparams),
        "state_dict": model.state_dict(),
        "global_step": TRAINING_STEPS,
    }
    torch.save(payload, output)
    print(f"D13_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
