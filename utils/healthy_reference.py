"""Read legacy 2048 Hz healthy velocity references for physical model calibration."""
from pathlib import Path

import h5py
import numpy as np


def read_healthy_runs(config):
    """Select complete runs before taking evenly spaced, valid 1 s windows."""
    result, provenance, seen = {}, [], set()
    for split in ("train", "validation"):
        signals = []
        for relative in config[f"{split}_runs"]:
            path = Path(config["root"]) / relative
            with h5py.File(path, "r") as h5:
                run_id = str(h5.attrs["run_id"])
                if run_id in seen:
                    raise ValueError(f"Run occurs in multiple selections: {run_id}")
                seen.add(run_id)
                data = h5["windows_1s/velocity"]
                channels = list(data.attrs["channel_order"])
                if (data.attrs["unit"] != "mm/s" or data.attrs["sample_rate_hz"] != 2048
                        or not np.array_equal(h5.attrs["velocity_frequency_band_hz"], [10, 900])
                        or h5.attrs["target_speed_rpm"] != 7000 or h5.attrs["stable_wait_s"] != 600):
                    raise ValueError(f"Unexpected healthy preprocessing: {path}")
                valid = np.flatnonzero(h5["windows_1s/sample_valid"][:])
                selected = valid[np.linspace(0, len(valid)-1,
                                             min(len(valid), config["windows_per_run"]), dtype=int)]
                values = data[selected][:, [channels.index("ACC2"), channels.index("ACC1")]]
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite healthy input: {path}")
                starts = h5["windows_1s/start_time_s"][selected]
                if np.any(starts < h5.attrs["stable_cutoff_s"]):
                    raise ValueError(f"Nonsteady windows: {path}")
                signals.append(values)
                provenance.append(dict(path=str(path), run_id=run_id, split=split,
                                       indices=selected.tolist(), start_time_s=starts.tolist(),
                                       channel_order=["ACC2", "ACC1"], unit="mm/s"))
        result[split] = np.concatenate(signals)
    return result, provenance
