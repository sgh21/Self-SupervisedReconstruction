"""Small JSON/CSV helpers and CRC32 input identities."""

import csv
import json
import zlib
from pathlib import Path
import numpy as np


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(serial(v), ensure_ascii=False)
                    if isinstance(v, (dict, list, tuple, np.ndarray))
                    else serial(v)
                    for k, v in row.items()
                }
            )


def crc_file(path):
    value = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return f"{value & 0xFFFFFFFF:08x}"


def config_crc(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
