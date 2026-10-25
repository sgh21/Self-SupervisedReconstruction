"""Quantitative plots and checks for a saved severity sweep; never touches raw data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from spindle.dynamics_model import area_and_link, harmonic_response, internal_force, simulate
from utils.data_io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--check-integration", action="store_true")
    args = parser.parse_args()
    out = args.experiment
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((out / "generation.json").read_text(encoding="utf-8"))
    cases = generation["cases"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#475569", "#0891b2", "#d97706", "#be123c"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fractions = cfg["simulation"]["loss_fractions"]
    with np.load(out / "healthy_reference.npz", allow_pickle=False) as data:
        real = {key:data[key] for key in data.files}
    signals = {}
    for case in cases:
        with np.load(out / "signals" / (case["case"] + ".npz"), allow_pickle=False) as data:
            signals[case["case"]] = {key:data[key] for key in
                                     ("H_velocity_band_mm_s_2048", "T_center_m", "H_velocity_m_s", "t")}
    representative = [c for c in cases if c["group"] == "phase_0"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, key, title, unit in [
        (axes[0,0], "link_stiffness_N_m", "Effective shaft-holder stiffness", "MN/m"),
        (axes[0,1], "H_radial_rms_mm_s", "Housing H: 10-900 Hz velocity", "mm/s"),
        (axes[1,0], "T_dynamic_radial_rms_um", "Holder T: dynamic radial RMS", "um")]:
        scale = 1e-6 if key == "link_stiffness_N_m" else 1
        for group in cfg["simulation"]["groups"]:
            rows = [c for c in cases if c["group"] == group["id"]]
            ax.plot([100*c["loss_fraction"] for c in rows], [scale*c[key] for c in rows],
                    "o-", label=group["id"], alpha=0.8)
        ax.set(xlabel="Equivalent contact area loss (%)", ylabel=unit, title=title)
        ax.grid(alpha=0.2)
    real_radial = np.sqrt(np.mean(np.sum(real["validation"]**2, axis=1)))
    axes[0,1].axhline(real_radial, color="black", linestyle=":", label="Real healthy comparison")
    axes[0,1].legend(fontsize=8)
    modes = np.array([c["modes"]["mode_frequencies_hz"] for c in representative])
    for i in range(modes.shape[1]):
        axes[1,1].plot(np.array(fractions)*100, modes[:,i], "o-", label=f"mode {i+1}")
    axes[1,1].set(xlabel="Equivalent contact area loss (%)", ylabel="Hz", title="Loaded tangent modes (not full nonlinear stability)")
    axes[1,1].grid(alpha=0.2)
    fig.suptitle("Changing physical severity changes the response; amplitude need not be monotonic", fontsize=12)
    fig.savefig(figures / "severity_response.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(len(fractions), 2, figsize=(12, 10), constrained_layout=True)
    f, measured_psd = welch(real["validation"][:, 0], fs=2048, nperseg=1024, axis=-1)
    measured_psd = measured_psd.mean(axis=0)
    band = (f>=10)&(f<=900)
    psd_min, psd_max = measured_psd[band].min(), measured_psd[band].max()
    comparisons = []
    for i, case in enumerate(representative):
        signal = signals[case["case"]]["H_velocity_band_mm_s_2048"]
        count = round(.12*2048)
        axes[i,0].plot(np.arange(count)/2048, real["validation"][0,0,:count], color="#a3a3a3", lw=0.8,
                       label="Real healthy (different time origin)")
        axes[i,0].plot(np.arange(count)/2048, signal[0,:count], color=colors[i], lw=1,
                       label=f"Simulation, loss {100*case['loss_fraction']:.0f}%")
        axes[i,0].set(xlabel="Relative time (s)", ylabel="H X velocity (mm/s)")
        axes[i,0].legend(fontsize=8, loc="upper right")
        _, psd = welch(signal[0], fs=2048, nperseg=1024)
        axes[i,1].semilogy(f[band], measured_psd[band], color="#a3a3a3", label="Real healthy mean")
        axes[i,1].semilogy(f[band], psd[band], color=colors[i], label="Simulation")
        psd_min, psd_max = min(psd_min, psd[band].min()), max(psd_max, psd[band].max())
        axes[i,1].set(xlim=(10,900), xlabel="Frequency (Hz)", ylabel="PSD ((mm/s)^2/Hz)")
        comparisons.append(dict(case=case["case"], loss_fraction=case["loss_fraction"],
            x_log_psd_rmse_db=float(np.sqrt(np.mean((10*np.log10(psd[band]/measured_psd[band]))**2)))))
    for ax in axes[:,1]:
        ax.set_ylim(psd_min*.5, psd_max*2)
    fig.suptitle("Matched excitation, phase_0 | no measured/simulated phase alignment", fontsize=12)
    fig.savefig(figures / "waveforms_spectra.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fractions), figsize=(12,3.3), constrained_layout=True)
    maximum = 0
    for i, case in enumerate(representative):
        center = signals[case["case"]]["T_center_m"][:-1]
        center = (center-center.mean(axis=0))*1e6
        maximum = max(maximum, abs(center).max())
        # Show the first 0.1 s to keep the trajectory readable.
        axes[i].plot(center[:819,0], center[:819,1], color=colors[i], lw=0.7)
        axes[i].set(xlabel="Centered X (um)", ylabel="Centered Y (um)", title=f"Area loss {100*case['loss_fraction']:.0f}%")
        axes[i].set_aspect("equal")
    for ax in axes:
        ax.set(xlim=(-maximum,maximum), ylim=(-maximum,maximum))
        ax.grid(alpha=0.2)
    fig.suptitle("Simulated holder-node trajectory; not measured ISO error motion", fontsize=12)
    fig.savefig(figures / "holder_trajectories.png", dpi=170)
    plt.close(fig)

    summary = dict(real_healthy_rms_xy_mm_s={k:np.sqrt(np.mean(v**2, axis=(0,2))).tolist() for k,v in real.items()},
                   real_healthy_validation_radial_rms_mm_s=float(real_radial), comparisons=comparisons)
    # Pooled RMS can hide opposite errors in individual acquisitions.
    provenance = json.loads((out / "healthy_provenance.json").read_text(encoding="utf-8"))
    healthy_model = signals[representative[0]["case"]]["H_velocity_band_mm_s_2048"]
    model_rms = np.sqrt(np.mean(healthy_model**2, axis=-1))
    _, model_psd = welch(healthy_model, fs=2048, nperseg=1024, axis=-1)
    offsets = {"train": 0, "validation": 0}
    summary["per_run_healthy_comparison"] = []
    for record in provenance:
        split, count = record["split"], len(record["indices"])
        start = offsets[split]
        values = real[split][start:start+count].astype(np.float64)
        offsets[split] += count
        rms = np.sqrt(np.mean(values**2, axis=(0,2)))
        _, psd = welch(values, fs=2048, nperseg=1024, axis=-1)
        spectral_error = np.sqrt(np.mean((10*np.log10(model_psd[:,band])
                                         -10*np.log10(psd.mean(0)[:,band]))**2, axis=1))
        summary["per_run_healthy_comparison"].append(dict(
            run_id=record["run_id"], split=split, windows_1s=count,
            measured_rms_xy_mm_s=rms.tolist(), model_rms_xy_mm_s=model_rms.tolist(),
            relative_rms_error_percent=(100*(model_rms/rms-1)).tolist(),
            log_psd_rmse_xy_db=spectral_error.tolist()))
    summary["by_severity"] = [dict(loss_fraction=fraction,
        **{key:dict(mean=float(np.mean([c[key] for c in cases if c["loss_fraction"]==fraction])),
                    min=float(min(c[key] for c in cases if c["loss_fraction"]==fraction)),
                    max=float(max(c[key] for c in cases if c["loss_fraction"]==fraction)))
           for key in ["H_radial_rms_mm_s", "T_dynamic_radial_rms_um", "H_relative_change_from_matched_healthy"]})
        for fraction in fractions]
    summary["finite_all_saved_signals"] = all(np.isfinite(s[k]).all().item() for s in signals.values() for k in s)
    summary["stable_loaded_tangents"] = all(c["modes"]["maximum_pole_real_s_inv"] < 0 for c in cases)
    summary["within_case_window_repetition"] = []
    for case in cases:
        windows = signals[case["case"]]["H_velocity_band_mm_s_2048"][0].reshape(-1, 1024)
        summary["within_case_window_repetition"].append(dict(
            case=case["case"], relative_l2_to_first=[float(np.linalg.norm(w-windows[0])/np.linalg.norm(windows[0]))
                                                    for w in windows[1:]]))
    if args.check_integration:
        case = representative[-1]
        p = case["parameters"]
        with np.load(out / "signals" / "phase_0_forcing.npz", allow_pickle=False) as drive:
            frequencies, coefficients = drive["frequencies_hz"], drive["coefficients_N"]
        area, _ = area_and_link(p, 0)
        print("Checking first 0.5 s of the highest loss at half integration step", flush=True)
        fine = simulate(p, area*case["loss_fraction"], frequencies, coefficients, duration=.5,
                        warmup=cfg["simulation"]["warmup_s"], fs=8192,
                        max_step=cfg["simulation"]["max_step_s"]/2)
        original = signals[case["case"]]
        n = len(fine["t"])
        def relative(a,b):
            return float(np.linalg.norm(a-b)/np.linalg.norm(b))
        coarse_t = original["T_center_m"][:n]
        fine_t = fine["T_center_m"]
        check = dict(H_velocity_relative_l2=relative(original["H_velocity_m_s"][:n], fine["H_velocity_m_s"]),
                     T_centered_relative_l2=relative(coarse_t-coarse_t.mean(0), fine_t-fine_t.mean(0)),
                     compared_seconds=.5, baseline_max_step_s=cfg["simulation"]["max_step_s"],
                     fine_max_step_s=cfg["simulation"]["max_step_s"]/2)
        q=np.arange(1,7)*1e-6
        r=internal_force(0,q,np.zeros(6),p,area*.5).reshape(3,2)
        check["reaction_balance_N"] = float(np.linalg.norm(r.sum(0)-p["kh"]*q[4:]))
        check["near_zero_loss_linear_relative_l2"] = relative(
            harmonic_response(p,area*1e-8,frequencies,coefficients),
            harmonic_response(p,0,frequencies,coefficients))
        assert check["H_velocity_relative_l2"] < 1e-3 and check["T_centered_relative_l2"] < 1e-3
        summary["numerical_check"] = check
        np.savez_compressed(out / "integration_check.npz", **fine)
    elif (out / "analysis.json").exists():
        previous = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        if "numerical_check" in previous:
            summary["numerical_check"] = previous["numerical_check"]
    save_json(out / "analysis.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
