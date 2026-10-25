"""Read run-level HDF5 velocity windows and keep runs in separate splits."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("ACC1", "ACC2", "ACC3")


def split_runs(config: dict) -> dict[str, list[Path]]:
    data = config["data"]
    root = (REPO_ROOT / data["root"]).resolve()
    all_files = sorted(root.rglob("*.h5"))
    if not all_files:
        raise FileNotFoundError(f"No HDF5 runs found in {root}")

    splits = {}
    for name in ("validation", "test", "train"):
        if name == "train" and "train_runs" not in data:
            continue
        files = []
        for pattern in data[f"{name}_runs"]:
            matches = sorted(root.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No run matches {name} pattern: {pattern}")
            files.extend(matches)
        splits[name] = sorted(set(files))
    held_out = set(splits["validation"] + splits["test"])
    if "train" not in splits:
        splits["train"] = [path for path in all_files if path not in held_out]

    seen_runs = {}
    for name, files in splits.items():
        if not files and name != "test":
            raise ValueError(f"The {name} split is empty")
        for path in files:
            with h5py.File(path, "r") as h5:
                run_id = str(h5.attrs["run_id"])
                if float(h5.attrs["target_speed_rpm"]) != data["target_speed_rpm"]:
                    raise ValueError(f"Wrong target speed in {path}")
                if float(h5.attrs["stable_wait_s"]) != data["stable_wait_s"]:
                    raise ValueError(f"Wrong steady-state waiting time in {path}")
            if run_id in seen_runs:
                raise ValueError(f"Run {run_id} appears more than once: {seen_runs[run_id]}, {name}")
            seen_runs[run_id] = name
    return splits


class SpindleDataset(Dataset):
    """Return unnormalized float32 tensors shaped [3, 2048], in mm/s."""

    def __init__(
        self,
        files: list[Path],
        sample_rate_hz: int = 2048,
        input_length: int = 2048,
        max_windows_per_run: int | None = None,
    ) -> None:
        if sample_rate_hz != 2048 or input_length != sample_rate_hz:
            raise ValueError("The TCN task uses 1 s windows at 2048 Hz")
        if max_windows_per_run is not None and max_windows_per_run < 1:
            raise ValueError("max_windows_per_run must be positive or null")
        self.files = [Path(path) for path in files]
        self.samples = []
        self.run_ids = []
        for file_index, path in enumerate(self.files):
            with h5py.File(path, "r") as h5:
                self.run_ids.append(str(h5.attrs["run_id"]))
                velocity = h5["windows_1s/velocity"]
                if velocity.shape[1:] != (3, input_length):
                    raise ValueError(f"Expected [N, 3, {input_length}] velocity in {path}")
                if tuple(velocity.attrs["channel_order"]) != CHANNELS:
                    raise ValueError(f"Expected ACC1/ACC2/ACC3 order in {path}")
                if velocity.attrs["sample_rate_hz"] != sample_rate_hz:
                    raise ValueError(f"Wrong velocity sample rate in {path}")
                if velocity.attrs["unit"] != "mm/s":
                    raise ValueError(f"Wrong velocity unit in {path}")
                if not np.array_equal(h5.attrs["velocity_frequency_band_hz"], [10, 900]):
                    raise ValueError(f"Expected 10–900 Hz velocity in {path}")
                count = len(velocity)
                if max_windows_per_run is not None:
                    count = min(count, max_windows_per_run)
                self.samples.extend((file_index, index) for index in range(count))
        if not self.samples:
            raise ValueError("The dataset contains no velocity windows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_index, sample_index = self.samples[index]
        with h5py.File(self.files[file_index], "r") as h5:
            velocity = np.asarray(h5["windows_1s/velocity"][sample_index], dtype=np.float32)
        if not np.isfinite(velocity).all():
            raise ValueError(f"Non-finite velocity in {self.files[file_index]} window {sample_index}")
        return torch.from_numpy(velocity)
