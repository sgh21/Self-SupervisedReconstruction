"""Read-only raw acquisition and guarded 10–2048 Hz velocity preparation."""
import csv
import io
import json
import lzma
from collections import Counter
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from utils.pitting_analysis import filter_vibration, resample_to


CHANNELS = ("ACC2", "ACC1", "ACC3")
SOURCE_FS = 25600
PROCESSING = dict(fs_hz=8192, band_hz=[10, 2048], fir_taps=4097, kaiser_beta=8.6)
HISTORICAL_TEST = ("20260722_221740", "20260723_204016", "20260724_225918",
                   "20260725_210750", "20260726_215234")
DEFAULT_SPLITS = {"train": ["20260722_212303", "20260722_215018"],
                  "validation": ["20260723_093018", "20260723_095732"],
                  "test": list(HISTORICAL_TEST)}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_split(run_name, splits=DEFAULT_SPLITS):
    matches = [split for split, patterns in splits.items()
               if any(Path(pattern).stem in run_name or pattern in run_name for pattern in patterns)]
    if len(matches) > 1:
        raise ValueError(f"Run assigned to multiple splits: {run_name}")
    result = matches[0] if matches else "unassigned"
    if result in ("train", "validation") and any(key in run_name for key in HISTORICAL_TEST):
        raise ValueError(f"Historical reconstruction test run cannot enter model-development selections: {run_name}")
    return result


def read_run_index(run_dir):
    run_dir = Path(run_dir)
    sensors = [row for row in read_csv(run_dir / "sensor_info.csv")
               if row["signal_type"] == "acceleration"]
    mapping = {row["sensor_id"]: row["channel"] for row in sensors}
    if set(mapping) != set(CHANNELS) or len(set(mapping.values())) != 3:
        raise ValueError(f"Invalid acceleration mapping: {run_dir}")
    for sensor in sensors:
        if float(sensor["sample_rate_hz"]) != SOURCE_FS or sensor["unit"] != "g":
            raise ValueError(f"Unexpected acceleration sensor metadata: {run_dir}")
    segments = [row for row in read_csv(run_dir / "segment_records.csv")
                if row["signal_type"] == "acceleration"]
    segments.sort(key=lambda row: int(row["sample_start_index"]))
    return mapping, segments


def read_acceleration(run_dir, row, mapping):
    path = Path(run_dir) / "acceleration_25600Hz_256000samples" / row["data_file"]
    with lzma.open(path, "rb") as handle, np.load(io.BytesIO(handle.read()), allow_pickle=False) as raw:
        values = raw["data"]
        time = raw["time_s"]
        channels = raw["channels"].astype(str).tolist()
        if (values.shape != (3, int(row["sample_count"])) or time.shape != (values.shape[1],)
                or float(raw["sample_rate_hz"].item()) != SOURCE_FS
                or str(raw["unit"].item()) != "g"
                or str(raw["signal_type"].item()) != "acceleration"
                or int(raw["sample_start_index"].item()) != int(row["sample_start_index"])):
            raise ValueError(f"Unexpected raw payload: {path}")
        if (not np.isfinite(values).all() or not np.isfinite(time).all()
                or not np.allclose(np.diff(time), 1 / SOURCE_FS, rtol=0, atol=1e-10)
                or not np.isclose(time[0], float(row["time_start_s"]), rtol=0, atol=1e-8)):
            raise ValueError(f"Invalid values or time axis: {path}")
        order = [channels.index(mapping[sensor]) for sensor in CHANNELS]
        return np.asarray(values[order], dtype=np.float64), float(time[0])


def acceleration_to_velocity(acceleration_g, processing=PROCESSING, return_acceleration=False):
    """Resample, apply the common centered FIR once, then integrate the complex FFT.

    Caller supplies real adjacent guards and crops after this function. No window
    normalization is performed. DC velocity is zero; all values remain physical.
    """
    resampled = resample_to(np.asarray(acceleration_g) * 9.80665, SOURCE_FS, processing["fs_hz"])
    half = processing["fir_taps"] // 2
    # FFT integration is periodic: apply the same FIR with periodic boundary
    # conditions too. Zero-padding the FIR first creates a boundary acceleration
    # impulse whose integral contaminates even the center of long records.
    padding = [(0, 0)] * resampled.ndim
    padding[-1] = (half, half)
    acceleration = filter_vibration(np.pad(resampled, padding, mode="wrap"),
                                    processing["fs_hz"], processing)[..., half:-half]
    frequency = np.fft.rfftfreq(acceleration.shape[-1], d=1 / processing["fs_hz"])
    spectrum = np.fft.rfft(acceleration, axis=-1)
    spectrum[..., 0] = 0
    spectrum[..., 1:] *= 1000 / (2j * np.pi * frequency[1:])
    velocity = np.fft.irfft(spectrum, n=acceleration.shape[-1], axis=-1)
    return (velocity, acceleration) if return_acceleration else velocity


def telemetry_state(run_dir, target=7000, tolerance=10, stable_wait=600):
    rows = read_csv(Path(run_dir) / "spindle_telemetry.csv")
    time = np.array([float(row["time_s"]) for row in rows])
    speed = np.array([float(row["actual_speed_rpm"]) for row in rows])
    valid = np.array([row["speed_ok"].lower() in ("true", "1")
                      and abs(float(row["target_rpm"]) - target) < 0.01 for row in rows])
    valid &= np.isfinite(speed) & (np.abs(speed - target) <= tolerance)
    start = float(time[valid].min()) if valid.any() else float("nan")
    end = float(time[valid].max()) if valid.any() else float("nan")
    return dict(time=time, speed=speed, valid=valid, platform_start_s=start,
                platform_end_s=end, stable_cutoff_s=start + stable_wait, stable_wait_s=stable_wait)


def steady_interval(state, start, end):
    time = state["time"]
    left = max(0, np.searchsorted(time, start, side="right") - 1)
    right = min(len(time), np.searchsorted(time, end, side="left") + 1)
    selected = time[left:right]
    return bool(len(selected) > 1 and selected[0] <= start and selected[-1] >= end
                and start >= state["stable_cutoff_s"] and end <= state["platform_end_s"]
                and state["valid"][left:right].all() and np.max(np.diff(selected)) <= 1)


def guarded_indices(segments, center, guard_s=10):
    """Require actual contiguous full acquisition on both sides, never pad a run."""
    row = segments[center]
    required = round(guard_s * SOURCE_FS)
    first, last = center, center
    previous = following = 0
    while previous < required and first > 0:
        left, right = segments[first - 1], segments[first]
        if int(left["sample_start_index"]) + int(left["sample_count"]) != int(right["sample_start_index"]):
            break
        first -= 1
        previous += int(left["sample_count"])
    while following < required and last + 1 < len(segments):
        left, right = segments[last], segments[last + 1]
        if int(left["sample_start_index"]) + int(left["sample_count"]) != int(right["sample_start_index"]):
            break
        last += 1
        following += int(right["sample_count"])
    if previous < required or following < required:
        return None
    return first, last, previous


def prepare_run(run_dir, output_root, split, max_segments=None, all_conditions=False, guard_s=10):
    started = perf_counter()
    run_dir, output_root = Path(run_dir), Path(output_root)
    mapping, segments = read_run_index(run_dir)
    state = telemetry_state(run_dir)
    candidates, skipped = [], []
    for index, row in enumerate(segments):
        start = float(row["time_start_s"])
        complete = (int(row["sample_count"]) == SOURCE_FS * 10
                    and row["partial"].lower() not in ("true", "1"))
        steady = steady_interval(state, start, start + 10)
        guards = guarded_indices(segments, index, guard_s) if complete else None
        if complete and guards and (all_conditions or steady):
            candidates.append((index, steady, guards))
        else:
            skipped.append(dict(segment_index=int(row["segment_index"]),
                                reason="incomplete" if not complete else
                                "insufficient_guard" if not guards else "not_steady_7000rpm"))
    available = len(candidates)
    if max_segments and len(candidates) > max_segments:
        selected = np.linspace(0, len(candidates) - 1, max_segments, dtype=int)
        candidates = [candidates[index] for index in selected]
    if not candidates:
        return dict(run=run_dir.name, split=split, segments=0, windows=0, skipped=skipped)
    output = output_root / run_dir.parent.name / (run_dir.name + ".h5")
    output.parent.mkdir(parents=True, exist_ok=True)
    processing = dict(PROCESSING, raw_fs_hz=SOURCE_FS, input_unit="g", output_unit="mm/s",
                      channel_order=list(CHANNELS), guard_s=guard_s,
                      integration="complex_fft_after_periodic_centered_fir_zero_dc", stable_wait_s=600,
                      speed_tolerance_rpm=10, minimum_valid_telemetry_fraction=1.0,
                      max_telemetry_gap_s=1.0, all_conditions=all_conditions,
                      filter_cutoff_definition="firwin half amplitude, approximately -6 dB")
    @lru_cache(maxsize=8)
    def load(index):
        return read_acceleration(run_dir, segments[index], mapping)
    count = len(candidates) * 10
    provenance = []
    experiment = read_csv(run_dir / "experiment_record.csv")[0]
    with h5py.File(output, "x") as h5:
        h5.attrs.update(run_id=segments[0]["run_id"], run_name=run_dir.name,
                        run_date=run_dir.parent.name, split=split, source_raw_run=str(run_dir.resolve()),
                        velocity_unit="mm/s", velocity_sample_rate_hz=8192,
                        velocity_frequency_band_hz=[10, 2048], stable_wait_s=600,
                        target_speed_rpm=7000 if not all_conditions else float("nan"),
                        platform_start_s=state["platform_start_s"], platform_end_s=state["platform_end_s"],
                        stable_cutoff_s=state["stable_cutoff_s"],
                        preprocessing_json=json.dumps(processing),
                        raw_experiment_json=json.dumps(experiment),
                        raw_target_speed_rpm=float(experiment["target_speed_rpm"] or "nan"),
                        condition_name=run_dir.name)
        group = h5.create_group("windows_1s")
        velocity = group.create_dataset("velocity", shape=(count, 3, 8192), dtype="float32",
                                        chunks=(1, 3, 8192), compression="lzf")
        velocity.attrs.update(sample_rate_hz=8192, unit="mm/s",
                              channel_order=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        starts = group.create_dataset("start_time_s", shape=(count,), dtype="float64")
        ends = group.create_dataset("end_time_s", shape=(count,), dtype="float64")
        sources = group.create_dataset("source_segment_index", shape=(count,), dtype="int32")
        valid = group.create_dataset("sample_valid", shape=(count,), dtype="bool")
        healthy = group.create_dataset("healthy_eligible", shape=(count,), dtype="bool")
        group.create_dataset("sample_time_offset_s", data=np.arange(8192) / 8192)
        parent = h5.create_group("segments_10s")
        parent.attrs["velocity_reconstruction"] = "windows_1s/velocity[10*i:10*(i+1)].transpose(1,0,2).reshape(3,81920)"
        for output_index, (index, steady, (first, last, previous)) in enumerate(candidates):
            row = segments[index]
            for item in range(first, last):
                if not np.isclose(load(item)[1] + load(item)[0].shape[-1] / SOURCE_FS,
                                  load(item + 1)[1], rtol=0, atol=1e-8):
                    raise ValueError(f"Payload time gap between segments in {run_dir}")
            raw = np.concatenate([load(item)[0] for item in range(first, last + 1)], axis=-1)
            center_start = previous - round(guard_s * SOURCE_FS)
            raw = raw[:, center_start:center_start + round((10 + 2 * guard_s) * SOURCE_FS)]
            converted = acceleration_to_velocity(raw)
            crop = round(guard_s * 8192)
            center = converted[:, crop:crop + 81920]
            begin, end = output_index * 10, (output_index + 1) * 10
            start = load(index)[1]
            velocity[begin:end] = center.reshape(3, 10, 8192).transpose(1, 0, 2)
            starts[begin:end] = start + np.arange(10)
            ends[begin:end] = start + np.arange(10) + 1
            sources[begin:end] = int(row["segment_index"])
            valid[begin:end] = True
            healthy[begin:end] = steady
            provenance.append(dict(source_segment_index=int(row["segment_index"]), start_s=start,
                                   sample_start_index=int(row["sample_start_index"]),
                                   raw_file=row["data_file"], guard_source_files=[segments[item]["data_file"]
                                   for item in range(first, last + 1)], channel_mapping=mapping,
                                   healthy_eligible=steady, first_window=begin, window_count=10))
        h5.create_dataset("provenance_json", data=json.dumps(provenance), dtype=h5py.string_dtype())
        h5.create_dataset("skipped_segments_json", data=json.dumps(skipped), dtype=h5py.string_dtype())
        h5.attrs["conversion_complete"] = True
    load.cache_clear()
    return dict(run=run_dir.name, split=split, path=str(output.resolve()), segments=len(candidates),
                available_segments=available, windows=count, healthy_windows=10 * sum(item[1] for item in candidates),
                elapsed_s=perf_counter() - started, bytes=output.stat().st_size)


def audit_collection(output_root):
    """Audit saved metadata and provenance without computing test waveform metrics."""
    output_root = Path(output_root)
    inventory = []
    for path in sorted(output_root.glob("*/*.h5")):
        with h5py.File(path, "r") as h5:
            velocity = h5["windows_1s/velocity"]
            channels = tuple(velocity.attrs["channel_order"])
            if (not h5.attrs.get("conversion_complete", False) or velocity.shape[1:] != (3, 8192)
                    or channels != CHANNELS or velocity.attrs["unit"] != "mm/s"
                    or velocity.attrs["sample_rate_hz"] != 8192
                    or not np.array_equal(h5.attrs["velocity_frequency_band_hz"], [10, 2048])):
                raise ValueError(f"Incomplete or incompatible conversion: {path}")
            source_run = Path(h5.attrs["source_raw_run"])
            _, raw_rows = read_run_index(source_run)
            provenance = json.loads(h5["provenance_json"][()])
            referenced = {name for row in provenance for name in row["guard_source_files"]}
            skipped = Counter(row["reason"] for row in json.loads(h5["skipped_segments_json"][()]))
            unselected = len(raw_rows) - sum(skipped.values()) - len(provenance)
            if unselected:
                skipped["not_selected_by_subset_sampling"] = unselected
            inventory.append(dict(path=str(path.resolve()), run_id=h5.attrs["run_id"], split=h5.attrs["split"],
                                  windows=len(velocity), converted_10s_segments=len(provenance),
                                  healthy_windows=int(h5["windows_1s/healthy_eligible"][:].sum()),
                                  raw_indexed_files=len(raw_rows), raw_payload_files_checked=len(referenced),
                                  skipped_segments=dict(skipped), bytes=path.stat().st_size))
    totals = {key: sum(row[key] for row in inventory) for key in
              ("windows", "converted_10s_segments", "healthy_windows", "raw_indexed_files",
               "raw_payload_files_checked", "bytes")}
    reasons = Counter()
    for row in inventory:
        reasons.update(row["skipped_segments"])
    result = dict(runs=len(inventory), **totals, skipped_segments=dict(reasons),
                  split_run_counts=dict(Counter(row["split"] for row in inventory)),
                  test_access="Metadata and deterministic conversion only; no test signal metrics", records=inventory)
    (output_root / "collection_manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (output_root / "collection_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
