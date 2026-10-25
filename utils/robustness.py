"""Paired interference evaluation against untouched physical reference signals."""

import sys

import torch
from torch.utils.data import SequentialSampler, TensorDataset
from tqdm import tqdm

from utils.augmentation import add_noise
from utils.reconstruction import CHANNELS


def _totals():
    return {
        "sample_count": 0, "point_count": 0, "normalized_sse": 0.0,
        "reference_energy": torch.zeros(3, dtype=torch.float64),
        "input_sse": torch.zeros(3, dtype=torch.float64),
        "output_sse": torch.zeros(3, dtype=torch.float64),
    }


def _accumulate(total, energy, input_sse, output_sse, normalized_sse, length):
    total["sample_count"] += len(energy)
    total["point_count"] += len(energy) * length
    total["reference_energy"] += energy.sum(dim=0)
    total["input_sse"] += input_sse.sum(dim=0)
    total["output_sse"] += output_sse.sum(dim=0)
    total["normalized_sse"] += normalized_sse.sum().item()


def _metrics(total):
    energy = total["reference_energy"]
    points = total["point_count"]
    values = {
        "rmse_mm_s": (total["output_sse"] / points).sqrt(),
        "reference_rms_mm_s": (energy / points).sqrt(),
        "input_snr_db": 10 * (energy / total["input_sse"]).log10(),
        "output_snr_db": 10 * (energy / total["output_sse"]).log10(),
    }
    return {
        "sample_count": total["sample_count"],
        "mse_normalized": total["normalized_sse"] / (points * len(CHANNELS)),
        **{name: dict(zip(CHANNELS, values.tolist())) for name, values in values.items()},
    }


@torch.no_grad()
def evaluate_robustness(
    models, loader, normalizers, device, use_amp, sample_rate_hz,
    scenarios, seeds, progress_desc=None,
):
    """Read references once and use identical corrupted tensors for all models.

    Noise is scaled per window/channel in mm/s before each model's fixed
    checkpoint normalization. SSE and signal energy are aggregated before
    deriving global and run-level RMSE/SNR; windows are not bootstrap units.
    """
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Robustness evaluation requires shuffle=False for run attribution")
    dataset = loader.dataset
    if hasattr(dataset, "samples") and hasattr(dataset, "run_ids"):
        sample_runs = [dataset.run_ids[file_index] for file_index, _ in dataset.samples]
    elif isinstance(dataset, TensorDataset):
        sample_runs = ["all"] * len(dataset)
    else:
        raise TypeError("Dataset must expose samples/run_ids, or be TensorDataset for checks")
    normalization = {}
    for name, model in models.items():
        model.eval()
        normalization[name] = (
            normalizers[name]["center"].to(device), normalizers[name]["scale"].to(device),
        )
    conditions, generators, totals = [], {}, {}
    for index, scenario in enumerate(scenarios):
        for seed in ([None] if scenario["noise_type"] == "clean" else seeds):
            key = (index, seed)
            conditions.append((key, scenario))
            if seed is not None:
                generators[key] = torch.Generator(device=device).manual_seed(int(seed) + index * 100003)
            for model_name in models:
                totals[(model_name, key)] = {"global": _totals(), "runs": {}}
    offset = 0
    with tqdm(
        loader, desc=progress_desc, disable=progress_desc is None, unit="batch",
        dynamic_ncols=True, mininterval=0.5, ascii=True, file=sys.stdout,
    ) as progress:
        for reference in progress:
            if isinstance(reference, (tuple, list)):
                reference = reference[0]
            reference = reference.to(device, non_blocking=True)
            batch, _, length = reference.shape
            batch_runs = sample_runs[offset:offset + batch]
            offset += batch
            run_indices = {
                run_id: [i for i, value in enumerate(batch_runs) if value == run_id]
                for run_id in dict.fromkeys(batch_runs)
            }
            energy = reference.square().sum(dim=-1).double().cpu()
            for key, scenario in conditions:
                corrupted = reference if scenario["noise_type"] == "clean" else add_noise(
                    reference, scenario["noise_type"], scenario["snr_db"],
                    sample_rate_hz, generators[key],
                )
                input_sse = (corrupted - reference).square().sum(dim=-1).double().cpu()
                for model_name, model in models.items():
                    center, scale = normalization[model_name]
                    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                        prediction = model((corrupted - center) / scale)
                    prediction = prediction.float()
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError(f"Non-finite prediction: {model_name}, {scenario['name']}")
                    reconstructed = prediction * scale + center
                    output_sse = (reconstructed - reference).square().sum(dim=-1).double().cpu()
                    normalized_sse = (prediction - (reference - center) / scale).square().sum(dim=-1).double().cpu()
                    total = totals[(model_name, key)]
                    _accumulate(total["global"], energy, input_sse, output_sse, normalized_sse, length)
                    for run_id, indices in run_indices.items():
                        run_total = total["runs"].setdefault(run_id, _totals())
                        _accumulate(run_total, energy[indices], input_sse[indices],
                                    output_sse[indices], normalized_sse[indices], length)
    rows = []
    for key, scenario in conditions:
        for model_name in models:
            total = totals[(model_name, key)]
            rows.append({
                "model": model_name, "condition": scenario["name"],
                "noise_type": scenario["noise_type"], "snr_db": scenario["snr_db"], "seed": key[1],
                **_metrics(total["global"]),
                "run_metrics": {run_id: _metrics(value) for run_id, value in total["runs"].items()},
            })
    return rows
