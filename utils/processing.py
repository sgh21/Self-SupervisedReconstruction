"""Shared, amplitude-preserving velocity processing from native model states."""

from math import gcd
import numpy as np
from scipy.signal import fftconvolve, firwin, resample_poly

FS_OUT_HZ = 8192.0


def process_velocity(values_mm_s, source_fs_hz):
    """Apply the existing two-stage FIR chain without importing HDF5 readers."""
    source_fs = int(round(source_fs_hz))
    if abs(source_fs - source_fs_hz) > 1e-6:
        raise ValueError("native source sampling rate must be integer-valued")
    divisor = gcd(source_fs, int(FS_OUT_HZ))
    resampled = resample_poly(
        np.asarray(values_mm_s, dtype=float),
        int(FS_OUT_HZ) // divisor,
        source_fs // divisor,
        axis=-1,
        padtype="line",
    )
    common_band = firwin(
        4097, [10.0, 2048.0], pass_zero=False, fs=FS_OUT_HZ, window=("kaiser", 8.6)
    )
    inband = fftconvolve(resampled, common_band.reshape(1, -1), mode="same", axes=-1)
    lowpass = firwin(4097, 900.0, fs=FS_OUT_HZ, window=("kaiser", 8.6))
    return fftconvolve(inband, lowpass.reshape(1, -1), mode="same", axes=-1)
