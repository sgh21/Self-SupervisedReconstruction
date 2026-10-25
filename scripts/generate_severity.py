"""Save independent phase groups, matched severity sweeps and healthy comparisons."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample

from scripts.run_prototype import calibrate_forcing
from spindle.dynamics_model import (area_and_link, default_parameters, displacement_metrics,
                                    modes_and_stability, simulate)
from utils.data_io import bandlimit, save_json
from utils.healthy_reference import read_healthy_runs


def processed_velocity(result, cfg):
    """Remove the duplicate endpoint, band-limit, then Fourier-resample the full run."""
    sim = cfg["simulation"]
    n = round(sim["duration_s"] * sim["output_fs_hz"])
    signal = bandlimit(result["H_velocity_m_s"][:n].T, sim["output_fs_hz"])
    return resample(signal, round(sim["duration_s"] * cfg["data"]["sample_rate_hz"]), axis=-1) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/severity_sweep.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output
    (out / "signals").mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", cfg)
    healthy, provenance = read_healthy_runs(cfg["data"])
    np.savez_compressed(out / "healthy_reference.npz", **healthy)
    save_json(out / "healthy_provenance.json", provenance)
    p = default_parameters()
    frequencies, coefficients, calibration = calibrate_forcing(healthy["train"] * 1e-3, 2048, p)
    save_json(out / "calibration.json", calibration)
    area, _ = area_and_link(p, 0)
    sim = cfg["simulation"]
    cases, rows = [], []
    start = time.perf_counter()
    for group in sim["groups"]:
        rng = np.random.default_rng(group["seed"])
        # Shared phase offsets across X/Y preserve their relative phase per line.
        drive = coefficients * np.exp(1j*rng.uniform(-np.pi, np.pi, (len(frequencies), 1)))
        gp = dict(p, cage_phase=float(rng.uniform(0, 2*np.pi)))
        np.savez_compressed(out / "signals" / f"{group['id']}_forcing.npz",
                            frequencies_hz=frequencies, coefficients_N=drive)
        base = None
        for label, fraction in enumerate(sim["loss_fractions"]):
            name = f"{group['id']}_loss_{round(100*fraction):02d}"
            print(f"Integrating {name}", flush=True)
            result = simulate(gp, area*fraction, frequencies, drive,
                              duration=sim["duration_s"], warmup=sim["warmup_s"],
                              fs=sim["output_fs_hz"], max_step=sim["max_step_s"])
            velocity = processed_velocity(result, cfg)
            result["H_velocity_band_mm_s_2048"] = velocity
            result["parameters_json"] = json.dumps(gp)
            result["lost_area_fraction"] = fraction
            result["group_id"] = group["id"]
            result["split"] = group["split"]
            np.savez_compressed(out / "signals" / f"{name}.npz", **result)
            metrics = displacement_metrics(result["T_center_m"][:-1])
            if base is None:
                base = velocity
            _, kl = area_and_link(gp, area*fraction)
            row = dict(case=name, group=group["id"], split=group["split"], label=label,
                       loss_fraction=fraction, lost_area_mm2=area*fraction*1e6,
                       link_stiffness_N_m=kl,
                       H_radial_rms_mm_s=float(np.sqrt(np.mean(np.sum(velocity**2, axis=0)))),
                       H_x_rms_mm_s=float(np.sqrt(np.mean(velocity[0]**2))),
                       H_y_rms_mm_s=float(np.sqrt(np.mean(velocity[1]**2))),
                       H_relative_change_from_matched_healthy=float(np.linalg.norm(velocity-base)/np.linalg.norm(base)),
                       T_dynamic_radial_rms_um=metrics["dynamic_radial_rms_um"],
                       T_x_pp_um=metrics["dynamic_x_pp_um"], T_y_pp_um=metrics["dynamic_y_pp_um"],
                       equilibrium_residual_N=result["equilibrium_residual_N"], nfev=result["nfev"])
            rows.append(row)
            cases.append(dict(**row, parameters=gp, displacement=metrics,
                              modes=modes_and_stability(gp, area*fraction)))
            print(f"  H={row['H_radial_rms_mm_s']:.6f} mm/s; T={row['T_dynamic_radial_rms_um']:.6f} um", flush=True)
    with (out / "severity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save_json(out / "generation.json", dict(parameters=p, nominal_area_mm2=area*1e6, cases=cases,
              elapsed_seconds=time.perf_counter()-start,
              split_note="Phase groups share the training-calibrated force spectrum; validation is not independent machine validation."))
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
