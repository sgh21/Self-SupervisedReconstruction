"""Convert complete steady raw acceleration segments to 1 s velocity HDF5 windows."""

import argparse
import csv
import io
import lzma
from pathlib import Path

import h5py
import numpy as np
import yaml

from utils.spindle_dataset import CHANNELS, REPO_ROOT


SOURCE_RATE_HZ = 25_600
SOURCE_POINTS = 256_000
OUTPUT_RATE_HZ = 2048
OUTPUT_POINTS = 20_480


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def acceleration_to_velocity(acceleration_g: np.ndarray) -> np.ndarray:
    """Keep the existing 10 s FFT integration and exact 10–900 Hz synthesis."""
    acceleration = np.asarray(acceleration_g, dtype=np.float64) * 9806.65
    if not np.isfinite(acceleration).all():
        raise ValueError("Raw acceleration contains non-finite values")
    acceleration -= acceleration.mean(axis=1, keepdims=True)
    frequency = np.fft.rfftfreq(SOURCE_POINTS, d=1 / SOURCE_RATE_HZ)
    band = (frequency >= 10) & (frequency <= 900)
    spectrum = np.fft.rfft(acceleration, axis=1)
    # Both 10 s frequency grids have 0.1 Hz spacing; retained coefficients
    # have unit weight in the original 10 Hz high-pass / 1000 Hz low-pass.
    output_spectrum = np.zeros((3, OUTPUT_POINTS // 2 + 1), dtype=np.complex128)
    indices = np.flatnonzero(band)
    output_spectrum[:, indices] = (
        spectrum[:, band] / (2j * np.pi * frequency[band])
        * (OUTPUT_POINTS / SOURCE_POINTS)
    )
    velocity = np.fft.irfft(output_spectrum, n=OUTPUT_POINTS, axis=1)
    velocity -= velocity.mean(axis=1, keepdims=True)
    return velocity.astype(np.float32)


def read_acceleration(path: Path, channel_for_sensor: dict[str, str]) -> tuple[np.ndarray, float]:
    with lzma.open(path, "rb") as compressed:
        with np.load(io.BytesIO(compressed.read()), allow_pickle=False) as payload:
            acceleration = payload["data"]
            time_s = payload["time_s"]
            channels = payload["channels"].astype(str).tolist()
            if acceleration.shape != (3, SOURCE_POINTS) or time_s.shape != (SOURCE_POINTS,):
                raise ValueError(f"Expected a complete 3-channel 10 s segment in {path}")
            if len(channels) != 3 or len(set(channels)) != 3:
                raise ValueError(f"Invalid acceleration channel metadata in {path}")
            if float(payload["sample_rate_hz"].item()) != SOURCE_RATE_HZ:
                raise ValueError(f"Wrong acceleration sample rate in {path}")
            if str(payload["signal_type"].item()) != "acceleration" or str(payload["unit"].item()) != "g":
                raise ValueError(f"Expected acceleration in g in {path}")
            if not np.isfinite(time_s).all() or not np.allclose(
                np.diff(time_s), 1 / SOURCE_RATE_HZ, rtol=0, atol=1e-10
            ):
                raise ValueError(f"Irregular acceleration time axis in {path}")
            ordered = []
            for sensor in CHANNELS:
                channel = channel_for_sensor[sensor]
                if channel not in channels:
                    raise ValueError(f"Missing {sensor} channel {channel} in {path}")
                ordered.append(channels.index(channel))
            return acceleration[ordered], float(time_s[0])


def prepare_run(run_dir: Path, output_root: Path, data: dict) -> int:
    experiment = read_csv(run_dir / "experiment_record.csv")[0]
    target = experiment["target_speed_rpm"]
    if not target or float(target) != data["target_speed_rpm"]:
        return 0
    telemetry = read_csv(run_dir / "spindle_telemetry.csv")
    time_s = np.array([float(row["time_s"]) for row in telemetry])
    valid = np.array([
        abs(float(row["target_rpm"]) - data["target_speed_rpm"]) <= 0.01
        and row["speed_ok"].lower() in ("true", "1")
        and abs(float(row["actual_speed_rpm"]) - data["target_speed_rpm"])
        <= data["speed_tolerance_rpm"]
        for row in telemetry
    ])
    if valid.sum() < 100:
        return 0
    platform_start = float(time_s[valid].min())
    platform_end = float(time_s[valid].max())
    cutoff = platform_start + data["stable_wait_s"]
    rows = []
    for row in read_csv(run_dir / "segment_records.csv"):
        if row["signal_type"] != "acceleration" or row["partial"].lower() in ("true", "1"):
            continue
        start, end = float(row["time_start_s"]), float(row["time_end_s"])
        if int(row["sample_count"]) != SOURCE_POINTS or float(row["sample_rate_hz"]) != SOURCE_RATE_HZ:
            continue
        if start < cutoff or end > platform_end:
            continue
        inside = (time_s >= start) & (time_s < end)
        if inside.any() and valid[inside].mean() >= data["minimum_valid_telemetry_fraction"]:
            rows.append(row)
    rows.sort(key=lambda row: float(row["time_start_s"]))
    if not rows:
        return 0

    sensor_rows = [
        row for row in read_csv(run_dir / "sensor_info.csv")
        if row["signal_type"] == "acceleration"
    ]
    mapping = {row["sensor_id"]: row["channel"] for row in sensor_rows}
    if set(mapping) != set(CHANNELS) or len(sensor_rows) != 3 or len(set(mapping.values())) != 3:
        raise ValueError(f"Expected one channel for each of ACC1/ACC2/ACC3 in {run_dir}")
    count = len(rows) * 10
    output = output_root / run_dir.parent.name / f"{run_dir.name}.h5"
    if output.exists():
        raise FileExistsError(f"Prepared run already exists: {output}; choose a new data.root")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as h5:
        h5.attrs.update(
            run_id=rows[0]["run_id"], run_name=run_dir.name, run_date=run_dir.parent.name,
            target_speed_rpm=data["target_speed_rpm"], platform_start_s=platform_start,
            stable_wait_s=data["stable_wait_s"], stable_cutoff_s=cutoff,
            platform_end_s=platform_end, velocity_sample_rate_hz=OUTPUT_RATE_HZ,
            velocity_unit="mm/s", velocity_frequency_band_hz=[10.0, 900.0],
            source_raw_run=str(run_dir.resolve()),
        )
        h5.attrs["velocity_channels"] = np.asarray(CHANNELS, dtype=h5py.string_dtype())
        group = h5.create_group("windows_1s")
        velocity_data = group.create_dataset(
            "velocity", shape=(count, 3, OUTPUT_RATE_HZ), dtype="float32",
            chunks=(1, 3, OUTPUT_RATE_HZ), compression="gzip", compression_opts=4,
        )
        velocity_data.attrs["channel_order"] = np.asarray(CHANNELS, dtype=h5py.string_dtype())
        velocity_data.attrs["sample_rate_hz"] = OUTPUT_RATE_HZ
        velocity_data.attrs["unit"] = "mm/s"
        starts = group.create_dataset("start_time_s", shape=(count,), dtype="float64")
        ends = group.create_dataset("end_time_s", shape=(count,), dtype="float64")
        sources = group.create_dataset("source_segment_index", shape=(count,), dtype="int32")
        for parent_index, row in enumerate(rows):
            source_path = run_dir / "acceleration_25600Hz_256000samples" / row["data_file"]
            acceleration, start = read_acceleration(source_path, mapping)
            if not np.isclose(start, float(row["time_start_s"]), rtol=0, atol=1e-8):
                raise ValueError(f"Payload and segment time disagree in {source_path}")
            velocity = acceleration_to_velocity(acceleration)
            velocity = velocity.reshape(3, 10, OUTPUT_RATE_HZ).transpose(1, 0, 2)
            begin, end = parent_index * 10, (parent_index + 1) * 10
            velocity_data[begin:end] = velocity
            starts[begin:end] = start + np.arange(10)
            ends[begin:end] = start + np.arange(10) + 1
            sources[begin:end] = int(row["segment_index"])
    print(f"Prepared {run_dir.name}: {count} windows")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tcn.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)["data"]
    if data["sample_rate_hz"] != OUTPUT_RATE_HZ or data["input_length"] != OUTPUT_RATE_HZ:
        raise ValueError("Preparation uses 1 s, 2048 Hz, 10–900 Hz velocity")
    raw_root = (REPO_ROOT / data["raw_root"]).resolve()
    output_root = (REPO_ROOT / data["root"]).resolve()
    if output_root.is_relative_to(raw_root):
        raise ValueError("data.root must be outside the read-only raw data directory")
    runs = sorted(raw_root.glob("*/*/experiment_record.csv"))
    if not runs:
        raise FileNotFoundError(f"No raw runs found in {raw_root}")
    count = sum(prepare_run(path.parent, output_root, data) for path in runs)
    if not count:
        raise ValueError("No complete steady segments matched the configured conditions")
    print(f"Prepared {count} windows in {output_root}")


if __name__ == "__main__":
    main()
