"""Train the TCN to reconstruct one-second spindle velocity waveforms."""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.tcn import build_autoencoder
from utils.reconstruction import evaluate, fit_normalization
from utils.spindle_dataset import REPO_ROOT, SpindleDataset, split_runs


def train_epoch(
    model, loader, normalizer, optimizer, scaler, device, use_amp, gradient_clip_norm,
    progress_desc="Train", augmentation=None, augmentation_generator=None, sample_rate_hz=2048,
):
    model.train()
    center = normalizer["center"].to(device)
    scale = normalizer["scale"].to(device)
    loss_sum = 0.0
    sample_count = 0
    skipped_steps = 0
    with tqdm(
        loader, desc=progress_desc, unit="batch", dynamic_ncols=True,
        mininterval=0.5, ascii=True, file=sys.stdout,
    ) as progress:
        for batch_index, velocity in enumerate(progress, start=1):
            velocity = velocity.to(device, non_blocking=True)
            input_velocity = target_velocity = velocity
            if augmentation is not None:
                from utils.augmentation import augment_pair

                input_velocity, target_velocity = augment_pair(
                    velocity, augmentation, sample_rate_hz, augmentation_generator,
                )
            normalized = (input_velocity - center) / scale
            target = (target_velocity - center) / scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(normalized)
            # Keep the small reconstruction residual and its square in FP32.
            loss = (prediction.float() - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # unscale_ has already recorded non-finite gradients for GradScaler.
            # Let it skip/back off on AMP overflow; FP32 still fails immediately.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm,
                error_if_nonfinite=not scaler.is_enabled(),
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                skipped_steps += 1
                tqdm.write(
                    f"{progress_desc}, batch {batch_index}: AMP gradient overflow; "
                    f"update skipped, scale {scale_before:g} -> {scale_after:g}",
                    file=sys.stdout,
                )
            batch_loss = loss.item()
            loss_sum += batch_loss * len(velocity)
            sample_count += len(velocity)
            progress.set_postfix({
                "loss": f"{batch_loss:.6g}", "avg": f"{loss_sum / sample_count:.6g}",
                "lr": f"{optimizer.param_groups[0]['lr']:.3g}", "skips": skipped_steps,
            }, refresh=False)
    if skipped_steps == len(loader):
        raise FloatingPointError("AMP skipped every update in this epoch; try training.amp: false")
    return loss_sum / sample_count, skipped_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tcn.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    print(f"Config: {args.config.resolve()}", flush=True)
    training = config["training"]
    if training["epochs"] < 1:
        raise ValueError("epochs must be at least 1")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(training["device"])
    use_amp = device.type == "cuda" and training["amp"]

    print("Preparing run splits and datasets...", flush=True)
    files = split_runs(config)
    datasets = {
        split: SpindleDataset(
            files[split],
            sample_rate_hz=config["data"]["sample_rate_hz"],
            input_length=config["data"]["input_length"],
            max_windows_per_run=config["data"]["max_windows_per_run"],
        )
        for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=training["batch_size"],
            shuffle=split == "train",
            num_workers=training["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=training["num_workers"] > 0,
        )
        for split, dataset in datasets.items()
    }
    print(f"Runs: { {split: len(paths) for split, paths in files.items()} }", flush=True)
    print(f"Windows: { {split: len(dataset) for split, dataset in datasets.items()} }", flush=True)
    initial_checkpoint = None
    if training.get("initial_checkpoint"):
        initial_checkpoint = torch.load(
            REPO_ROOT / training["initial_checkpoint"], map_location="cpu", weights_only=True,
        )
        if initial_checkpoint["config"]["model"] != config["model"]:
            raise ValueError("Initial checkpoint architecture differs from the configured model")
        if set(initial_checkpoint["run_ids"]["train"]) != set(datasets["train"].run_ids):
            raise ValueError("Fine-tuning must retain the initial checkpoint's training runs")
        normalizer = initial_checkpoint["normalizer"]
        print(f"Initializing from {training['initial_checkpoint']}; retaining fixed training normalization", flush=True)
    else:
        print("Fitting training-set channel normalization...", flush=True)
        with tqdm(
            loaders["train"], desc="Normalization", unit="batch", dynamic_ncols=True,
            mininterval=0.5, ascii=True, file=sys.stdout,
        ) as progress:
            normalizer = fit_normalization(progress)
    model = build_autoencoder(**config["model"]).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=training["lr_factor"],
        # PyTorch reduces when bad epochs > patience: 4 implements exactly 5.
        patience=training["lr_patience"] - 1, threshold=0.0,
        threshold_mode="abs", min_lr=training["minimum_learning_rate"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    # A new directory on every invocation preserves existing weights and logs.
    output_dir = REPO_ROOT / config["output_dir"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    run_files = {
        split: [str(path.relative_to(REPO_ROOT / config["data"]["root"])) for path in paths]
        for split, paths in files.items()
    }
    with (output_dir / "splits.json").open("w", encoding="utf-8") as stream:
        json.dump(run_files, stream, ensure_ascii=False, indent=2)
    print(f"Model: {config['model'].get('implementation', 'custom')}; device: {device}; "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Precision: {'AMP FP16 (MSE in FP32)' if use_amp else 'FP32'}; "
          f"batch size: {training['batch_size']}; max epochs: {training['epochs']}")
    print(f"Output: {output_dir}", flush=True)

    augmentation = config.get("augmentation")
    augmentation_generator = None
    if augmentation is not None:
        augmentation_generator = torch.Generator(device=device).manual_seed(augmentation["seed"])
        print(f"Denoising augmentation: {augmentation}", flush=True)
    robust_validation = config.get("robustness_validation")
    selection_metric = "validation_mse_normalized"
    if robust_validation is not None:
        from utils.robustness import evaluate_robustness

        validation_scenarios = [
            {"name": f"{kind}_{snr}dB", "noise_type": kind, "snr_db": snr}
            for kind in robust_validation["noise_types"] for snr in robust_validation["snr_dbs"]
        ]
        selection_metric = "weighted_clean_and_corrupted_validation_mse"

    visualization = config.get("visualization", {})
    cases = []
    if visualization.get("enabled", False):
        from utils.visualization import save_reconstruction_cases, select_cases

        if visualization["interval_epochs"] < 1:
            raise ValueError("visualization.interval_epochs must be positive")
        cases = select_cases(datasets["validation"], visualization["sample_count"])
        print(f"Visualization: {len(cases)} fixed validation cases, epoch 1 and every "
              f"{visualization['interval_epochs']} epochs, plus best checkpoint", flush=True)

    best_loss = float("inf")
    best_epoch = 0
    best_clean_loss = float("inf")
    epochs_without_improvement = 0
    stop_reason = "maximum_epochs"
    with SummaryWriter(str(output_dir / "tensorboard")) as writer:
        for epoch in range(1, training["epochs"] + 1):
            epoch_start = time.perf_counter()
            train_loss, skipped_steps = train_epoch(
                model, loaders["train"], normalizer, optimizer, scaler, device,
                use_amp, training["gradient_clip_norm"],
                progress_desc=f"Epoch {epoch}/{training['epochs']} train",
                augmentation=augmentation, augmentation_generator=augmentation_generator,
                sample_rate_hz=config["data"]["sample_rate_hz"],
            )
            metrics = evaluate(
                model, loaders["validation"], normalizer, device,
                use_amp, config["data"]["sample_rate_hz"],
                progress_desc=f"Epoch {epoch}/{training['epochs']} validation",
            )
            if not math.isfinite(metrics["mse_normalized"]):
                raise FloatingPointError("Validation loss is non-finite")
            selection_loss = metrics["mse_normalized"]
            robust_metrics = []
            if robust_validation is not None:
                robust_metrics = evaluate_robustness(
                    {"model": model}, loaders["validation"], {"model": normalizer}, device,
                    use_amp, config["data"]["sample_rate_hz"], validation_scenarios,
                    robust_validation["seeds"], progress_desc=f"Epoch {epoch} interference validation",
                )
                clean_weight = robust_validation["clean_weight"]
                selection_loss = (clean_weight * selection_loss + (1 - clean_weight)
                                  * sum(row["mse_normalized"] for row in robust_metrics) / len(robust_metrics))
                writer.add_scalar("loss/validation_robust_selection_mse", selection_loss, epoch)
            writer.add_scalar("loss/train_mse_normalized", train_loss, epoch)
            writer.add_scalar("loss/validation_mse_normalized", metrics["mse_normalized"], epoch)
            writer.add_scalar(
                "metric/validation_rmse_mm_s",
                math.sqrt(metrics["time_mse_mm2_s2"]["mean"]), epoch,
            )
            writer.add_scalar("metric/validation_nrmse", metrics["time_nrmse"]["mean"], epoch)
            writer.add_scalar("metric/validation_correlation", metrics["time_correlation"]["mean"], epoch)
            writer.add_scalar("training/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("training/amp_scale", scaler.get_scale(), epoch)
            writer.add_scalar("training/amp_skipped_steps", skipped_steps, epoch)
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_clean_loss = metrics["mse_normalized"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "normalizer": normalizer,
                    "config": config,
                    "epoch": epoch,
                    "run_ids": {split: dataset.run_ids for split, dataset in datasets.items()},
                    "validation_metrics": metrics,
                    "selection_metric": selection_metric,
                    "validation_score": selection_loss,
                    "robust_validation_metrics": robust_metrics,
                }, output_dir / "best.pt")
                with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as stream:
                    json.dump(metrics, stream, ensure_ascii=False, indent=2)
                if robust_metrics:
                    with (output_dir / "robust_validation_metrics.json").open("w", encoding="utf-8") as stream:
                        json.dump(robust_metrics, stream, ensure_ascii=False, indent=2)
            else:
                epochs_without_improvement += 1
            if cases and (epoch == 1 or epoch % visualization["interval_epochs"] == 0):
                print(f"Saving validation figures for epoch {epoch}...", flush=True)
                report = save_reconstruction_cases(
                    model, datasets["validation"], cases, normalizer, device, use_amp,
                    config["data"]["sample_rate_hz"], output_dir, epoch,
                    config["model"].get("implementation", "custom"), writer=writer,
                    zoom_time_s=visualization["zoom_time_s"],
                )
                print(f"Visualization: {report}", flush=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(selection_loss)
            next_learning_rate = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - epoch_start
            writer.add_scalar("training/epoch_seconds", elapsed, epoch)
            writer.flush()
            print(
                f"Epoch {epoch}/{training['epochs']} | train_mse={train_loss:.6g} "
                f"val_mse={metrics['mse_normalized']:.6g} "
                f"val_NRMSE={metrics['time_nrmse']['mean']:.3%} | "
                f"lr={learning_rate:.3g} next_lr={next_learning_rate:.3g}\n"
                f"  val_score={selection_loss:.6g} best_val_score={best_loss:.6g} (epoch {best_epoch}) | "
                f"no_improvement={epochs_without_improvement}/{training['early_stopping_patience']} | "
                f"AMP_skips={skipped_steps} scale={scaler.get_scale():g} | {elapsed:.1f}s",
                flush=True,
            )
            if epochs_without_improvement >= training["early_stopping_patience"]:
                stop_reason = "early_stopping"
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
        if cases:
            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            report = save_reconstruction_cases(
                model, datasets["validation"], cases, checkpoint["normalizer"], device, use_amp,
                config["data"]["sample_rate_hz"], output_dir, checkpoint["epoch"],
                config["model"].get("implementation", "custom"), writer=writer, best=True,
                zoom_time_s=visualization["zoom_time_s"],
            )
            writer.flush()
            print(f"Best-checkpoint visualization: {report}", flush=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_validation_mse_normalized": best_clean_loss,
            "best_validation_score": best_loss, "stop_reason": stop_reason,
            "selection_metric": selection_metric,
        }, stream, indent=2)
    print(f"Saved checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
