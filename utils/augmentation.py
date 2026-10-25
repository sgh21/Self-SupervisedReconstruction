"""Paired denoising augmentation in physical units, before fixed normalization."""

import math

import torch


def _unit_rms(noise):
    noise = noise - noise.mean(dim=-1, keepdim=True)
    return noise / noise.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)


def _raw_noise(reference, kind, sample_rate_hz, generator):
    shape = reference.shape
    kwargs = {"device": reference.device, "dtype": reference.dtype, "generator": generator}
    if kind in ("white", "pink"):
        noise = torch.randn(shape, **kwargs)
        if kind == "pink":
            # Multiplying amplitude by 1/sqrt(f) gives 1/f power; remove DC.
            weight = torch.arange(shape[-1] // 2 + 1, device=reference.device).clamp_min(1).rsqrt()
            weight[0] = 0
            noise = torch.fft.irfft(torch.fft.rfft(noise) * weight, n=shape[-1])
        return noise
    if kind == "narrowband":
        frequency = 40 + torch.rand((*shape[:2], 1), **kwargs) * (min(850, sample_rate_hz / 2) - 40)
        phase = torch.rand((*shape[:2], 1), **kwargs) * (2 * math.pi)
        time = torch.arange(shape[-1], device=reference.device, dtype=reference.dtype) / sample_rate_hz
        return torch.sin(2 * math.pi * frequency * time + phase)
    if kind == "impulse":
        # TEST ONLY: synthetic interference, never a claim that real impacts are noise.
        count = max(1, shape[-1] // 200)
        positions = torch.randint(shape[-1], (*shape[:2], count), device=reference.device, generator=generator)
        amplitudes = torch.randn((*shape[:2], count), **kwargs)
        return torch.zeros_like(reference).scatter_add_(-1, positions, amplitudes)
    raise ValueError(f"Unknown interference kind: {kind}")


def add_noise(reference, kind, snr_db, sample_rate_hz=2048, generator=None):
    """Add RMS-scaled interference to [B,3,T] physical signals.

    SNR may be scalar, [B], [B,C], or broadcastable [B,C,1]. Components
    in a '+' mixture have equal RMS before the final total-SNR scaling.
    A supplied torch.Generator must use the same device as reference.
    """
    noise = sum(
        _unit_rms(_raw_noise(reference, component, sample_rate_hz, generator))
        for component in kind.split("+")
    )
    snr = torch.as_tensor(snr_db, device=reference.device, dtype=reference.dtype)
    if snr.ndim == 1:
        snr = snr[:, None, None]
    elif snr.ndim == 2:
        snr = snr[..., None]
    signal_rms = reference.square().mean(dim=-1, keepdim=True).sqrt()
    target_noise_rms = signal_rms * torch.pow(10.0, -snr / 20.0)
    return reference + _unit_rms(noise) * target_noise_rms


def augment_pair(reference, config, sample_rate_hz=2048, generator=None):
    """Return (corrupted_input, clean_target), retaining complete references.

    clean_probability keeps samples untouched. Optional circular shifts use
    one shift per augmented sample, shared by the input, target, and channels.
    Noise is never added to targets; time reversal is not applied.
    """
    batch, channels, length = reference.shape
    device = reference.device
    augmented = torch.rand(batch, device=device, generator=generator) >= config.get("clean_probability", 0.25)
    target = reference
    max_shift = int(config.get("max_shift_samples", 0))
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=device, generator=generator)
        shifts = shifts * augmented
        indices = (torch.arange(length, device=device)[None, None, :] - shifts[:, None, None]) % length
        target = reference.gather(-1, indices.expand(batch, channels, length))
    corrupted = target.clone()
    noise_types = config.get("noise_types", ["white"])
    selected = torch.randint(len(noise_types), (batch,), device=device, generator=generator)
    low, high = config.get("snr_db_range", [10, 30])
    snr = low + torch.rand(batch, device=device, generator=generator) * (high - low)
    for index, kind in enumerate(noise_types):
        mask = augmented & (selected == index)
        if mask.any():
            corrupted[mask] = add_noise(target[mask], kind, snr[mask], sample_rate_hz, generator)
    return corrupted, target
