"""Training-set normalization and streaming waveform reconstruction metrics."""

import sys

import torch
from tqdm import tqdm


CHANNELS = ("ACC1", "ACC2", "ACC3")


@torch.no_grad()
def fit_normalization(loader):
    total = torch.zeros(3, dtype=torch.float64)
    total_square = torch.zeros(3, dtype=torch.float64)
    count = 0
    for velocity in loader:
        velocity = velocity.double()
        total += velocity.sum(dim=(0, 2))
        total_square += velocity.square().sum(dim=(0, 2))
        count += velocity.shape[0] * velocity.shape[2]
    mean = total / count
    std = (total_square / count - mean.square()).clamp_min(0).sqrt()
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Training data must have finite, nonzero channel variance")
    return {
        "mode": "channel_zscore",
        "center": mean.float().reshape(1, 3, 1),
        "scale": std.float().reshape(1, 3, 1),
    }


def correlation(original, reconstructed):
    original = original - original.mean(dim=-1, keepdim=True)
    reconstructed = reconstructed - reconstructed.mean(dim=-1, keepdim=True)
    denominator = original.square().sum(dim=-1).sqrt() * reconstructed.square().sum(dim=-1).sqrt()
    return (original * reconstructed).sum(dim=-1) / denominator.clamp_min(1e-12)


def amplitude_spectrum(signal, sample_rate_hz):
    """Existing Hann/rFFT amplitude definition, shared by metrics and figures."""
    window = torch.hann_window(signal.shape[-1], device=signal.device)
    amplitude_scale = 2.0 / window.sum()
    spectrum = torch.fft.rfft(signal * window).abs() * amplitude_scale
    frequency = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sample_rate_hz, device=signal.device)
    band = (frequency >= 10) & (frequency <= 900)
    return frequency[band], spectrum[..., band]


def reconstruction_metrics(original, reconstructed, sample_rate_hz):
    time_mse = (reconstructed - original).square().mean(dim=-1)
    time_rmse = time_mse.sqrt()
    time_nrmse = time_rmse / original.square().mean(dim=-1).sqrt().clamp_min(1e-12)

    _, original_spectrum = amplitude_spectrum(original, sample_rate_hz)
    _, reconstructed_spectrum = amplitude_spectrum(reconstructed, sample_rate_hz)
    spectrum_mse = (reconstructed_spectrum - original_spectrum).square().mean(dim=-1)
    spectrum_rmse = spectrum_mse.sqrt()
    spectrum_nrmse = spectrum_rmse / original_spectrum.square().mean(dim=-1).sqrt().clamp_min(1e-12)

    return {
        "time_mse_mm2_s2": time_mse,
        "time_rmse_mm_s": time_rmse,
        "time_nrmse": time_nrmse,
        "time_correlation": correlation(original, reconstructed),
        "spectrum_mse_mm2_s2": spectrum_mse,
        "spectrum_rmse_mm_s": spectrum_rmse,
        "spectrum_nrmse": spectrum_nrmse,
        "spectrum_correlation": correlation(original_spectrum, reconstructed_spectrum),
    }


@torch.no_grad()
def evaluate(model, loader, normalizer, device, use_amp, sample_rate_hz, progress_desc=None):
    model.eval()
    center = normalizer["center"].to(device)
    scale = normalizer["scale"].to(device)
    totals = {}
    normalized_error = 0.0
    sample_count = 0
    point_count = 0
    with tqdm(
        loader, desc=progress_desc, disable=progress_desc is None,
        unit="batch", dynamic_ncols=True, mininterval=0.5, ascii=True, file=sys.stdout,
    ) as progress:
        for velocity in progress:
            velocity = velocity.to(device, non_blocking=True)
            normalized = (velocity - center) / scale
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(normalized)
            prediction = prediction.float()
            batch_error = (prediction - normalized).square().sum().item()
            normalized_error += batch_error
            reconstructed = prediction * scale + center
            for name, values in reconstruction_metrics(velocity, reconstructed, sample_rate_hz).items():
                if name not in totals:
                    totals[name] = torch.zeros(3, dtype=torch.float64)
                totals[name] += values.double().sum(dim=0).cpu()
            sample_count += len(velocity)
            point_count += velocity.numel()
            progress.set_postfix(
                loss=f"{batch_error / velocity.numel():.6g}",
                mean_loss=f"{normalized_error / point_count:.6g}", refresh=False,
            )

    result = {"sample_count": sample_count, "mse_normalized": normalized_error / point_count}
    for name, total in totals.items():
        means = total / sample_count
        result[name] = {"mean": means.mean().item(), "channels": dict(zip(CHANNELS, means.tolist()))}
    return result
