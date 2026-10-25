"""Evaluate held-out reconstruction with ``python -m test.test``."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from model.tcn import TCNAutoencoder, build_autoencoder
from model.tcn_standard import StandardTCNAutoencoder
from utils.reconstruction import evaluate
from utils.spindle_dataset import REPO_ROOT, SpindleDataset, split_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tcn.yaml")
    parser.add_argument("--output", type=Path, help="Save metrics JSON in addition to printing")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    # The original TCN checkpoint used one ConvAutoencoder wrapper.
    historical = "variant.encoder.0.weight" in state
    if historical:
        if checkpoint["config"]["model"]["architecture"] != "tcn":
            raise ValueError("This entry point loads TCN checkpoints only")
        if checkpoint["config"]["duration_s"] * 2048 != config["data"]["input_length"]:
            raise ValueError("Configured input length differs from the checkpoint")
        model = TCNAutoencoder()
        state = {key.removeprefix("variant."): value for key, value in state.items()}
    else:
        saved_data = checkpoint["config"]["data"]
        for key in (
            "sample_rate_hz", "input_length", "target_speed_rpm",
            "speed_tolerance_rpm", "stable_wait_s",
        ):
            if config["data"][key] != saved_data[key]:
                raise ValueError(f"Configured {key} differs from the checkpoint")
        saved_model = checkpoint["config"]["model"]
        if saved_model.get("implementation") == "standard" and "num_channels" in saved_model["standard"]:
            # Earlier standard checkpoints stored their own full architecture.
            model = StandardTCNAutoencoder(**saved_model["standard"])
        else:
            model = build_autoencoder(**saved_model)
    model.load_state_dict(state)
    if checkpoint["normalizer"]["mode"] != "channel_zscore":
        raise ValueError("TCN testing requires training-set channel Z-score normalization")

    files = split_runs(config)["test"]
    dataset = SpindleDataset(
        files,
        sample_rate_hz=config["data"]["sample_rate_hz"],
        input_length=config["data"]["input_length"],
        max_windows_per_run=config["data"]["max_windows_per_run"],
    )
    if historical:
        # Old checkpoints did not save split metadata. Limit them to their
        # original held-out runs rather than accepting a changed YAML split.
        historical_test_runs = {
            "run_20260722_221740", "run_20260723_204016", "run_20260724_225918",
            "run_20260725_210750", "run_20260726_215234",
        }
        if not set(dataset.run_ids).issubset(historical_test_runs):
            raise ValueError("Historical TCN weights require their original held-out test runs")
    else:
        used_runs = set(checkpoint["run_ids"]["train"] + checkpoint["run_ids"]["validation"])
        if used_runs.intersection(dataset.run_ids):
            raise ValueError("Test runs overlap checkpoint training or validation runs")
    device = torch.device(config["training"]["device"])
    model.to(device)
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda",
    )
    metrics = evaluate(
        model, loader, checkpoint["normalizer"], device,
        device.type == "cuda" and config["training"]["amp"], config["data"]["sample_rate_hz"],
    )
    metrics["run_ids"] = dataset.run_ids
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
