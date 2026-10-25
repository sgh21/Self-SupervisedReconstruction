"""Reusable V9 comparison figures and frozen-tangent diagnostics.

Reads only completed case folders. Missing A/B cases are left out of the
corresponding curves; this module does not infer or synthesize missing runs.
The optional tangent command computes a local frozen-equilibrium diagnostic,
not a nonlinear time-domain stability result.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/clearance_v9_tai_bearing"
GAP_RE = re.compile(r"gap_([0-9]+(?:\.[0-9]+)?)um", re.IGNORECASE)
SELECTION_GAP_RE = re.compile(r"(?:gap_)?([0-9]+(?:\.[0-9]+)?)(?:um|µm)?$", re.IGNORECASE)
RING_NAMES = ("F1", "F2", "R1", "R2")


@dataclass
class Case:
    folder: Path
    group: str
    variant: str
    gap_um: float
    config: dict
    metrics: dict

    @property
    def p(self):
        return self.config.get("parameters", {})

    @property
    def drive(self):
        return self.config.get("excitation", {})

    @property
    def U(self):
        return self.drive.get("U_kg_m")

    @property
    def label(self):
        if self.group == "B":
            return f"B (U={_fmt(self.U, 3)} kg·m)"
        return f"{self.variant} (U={_fmt(self.U, 3)} kg·m)"


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_group(path: Path, base: Path):
    rel = path.parent.relative_to(base)
    parts = rel.parts
    group = "B" if base.name.lower() == "data_b" else "A"
    variant = "B" if group == "B" else "unknown"
    if group == "A" and parts:
        variant = parts[0]
    try:
        config = _read_json(path.parent / "actual_config.json")
    except (OSError, json.JSONDecodeError):
        return None
    config_variant = config.get("mechanism_variant")
    if group == "A" and config_variant:
        variant = str(config_variant)
    metrics = _read_json(path)
    match = GAP_RE.search(path.parent.as_posix())
    p = config.get("parameters", {})
    gap = (float(metrics["gap_um"]) if "gap_um" in metrics else
           float(match.group(1)) if match else
           p.get("front_group_clearance_um", p.get("fit_clearance_um")))
    if gap is None:
        return None
    return Case(path.parent, group, variant, float(gap), config, metrics)


def discover_cases(data_a: Path, data_b: Path):
    cases = []
    for group, base in (("A", data_a), ("B", data_b)):
        if not base.exists():
            continue
        for metric_path in sorted(base.rglob("metrics.json")):
            if not (metric_path.parent / "native.npz").exists():
                continue
            case = _case_group(metric_path, base)
            if case is not None:
                case.group = group
                cases.append(case)
    return sorted(cases, key=lambda c: (c.group, c.variant, c.gap_um, str(c.folder)))


def _selection_gap(gap_key, metrics, config, folder):
    key_match = SELECTION_GAP_RE.search(str(gap_key).strip())
    key_gap = float(key_match.group(1)) if key_match else None
    metric_gap = metrics.get("gap_um")
    if metric_gap is not None:
        gap = float(metric_gap)
        if key_gap is not None and abs(gap-key_gap) > 1e-6:
            raise ValueError(f"Selection gap {gap_key!r} disagrees with metrics in {folder}: {gap}")
        return gap
    if key_gap is not None:
        return key_gap
    match = GAP_RE.search(folder.as_posix())
    if match:
        return float(match.group(1))
    p = config.get("parameters", {})
    gap = p.get("front_group_clearance_um", p.get("fit_clearance_um"))
    if gap is None:
        raise ValueError(f"Cannot determine clearance for selected case: {folder}")
    return float(gap)


def discover_selected_cases(root: Path, selection_path: Path):
    """Load only folders named in case_selection.json; paths are root-relative."""
    root = Path(root).resolve()
    selection_path = Path(selection_path)
    if not selection_path.is_absolute():
        selection_path = root / selection_path
    selection = _read_json(selection_path)
    if not isinstance(selection, dict):
        raise ValueError(f"Selection file must contain an A/B object: {selection_path}")
    cases = []

    def add_case(group, variant, gap_key, relative_path):
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"Selection path for {group}/{variant}/{gap_key} must be a relative string")
        rel = Path(relative_path)
        if rel.is_absolute():
            raise ValueError(f"Selection paths must be relative to experiment root: {relative_path}")
        folder = (root / rel).resolve()
        if not (folder / "native.npz").is_file():
            raise FileNotFoundError(f"Selected case has no native.npz: {folder}")
        metrics_path = folder / "metrics.json"
        config_path = folder / "actual_config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            missing = [name for name, path in (("metrics.json", metrics_path),
                                                ("actual_config.json", config_path))
                       if not path.is_file()]
            raise FileNotFoundError(f"Selected case is incomplete ({', '.join(missing)}): {folder}")
        metrics = _read_json(metrics_path)
        config = _read_json(config_path)
        actual_variant = (str(config.get("mechanism_variant") or variant)
                          if group == "A" else "B")
        gap = _selection_gap(gap_key, metrics, config, folder)
        cases.append(Case(folder, group, actual_variant, gap, config, metrics))

    a_selection = selection.get("A", {})
    b_selection = selection.get("B", {})
    if not isinstance(a_selection, dict) or not isinstance(b_selection, dict):
        raise ValueError("Selection schema requires A and B objects")
    for variant, gaps in a_selection.items():
        if not isinstance(gaps, dict):
            raise ValueError(f"A/{variant} must map clearance keys to relative paths")
        for gap_key, relative_path in gaps.items():
            add_case("A", str(variant), gap_key, relative_path)
    for gap_key, relative_path in b_selection.items():
        add_case("B", "B", gap_key, relative_path)
    return sorted(cases, key=lambda c: (c.group, c.variant, c.gap_um, str(c.folder)))


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, output: Path, name: str):
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return [f"{name}.png", f"{name}.svg"]


def _annotate_scope(ax, title):
    ax.set_title(title)
    ax.text(.99, .02, "Completed runs only", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=7, color="0.4")


_VARIANT_MARKERS = {"B0": "o", "B0d": "s", "N1": "^", "N2": "D", "N3": "P"}


def _marker(group, variant):
    return "X" if group == "B" else _VARIANT_MARKERS.get(variant, "o")


def _gap_linestyle(n):
    # A two-point series is not evidence for the unrun clearances between them.
    return "-" if n >= 3 else "None"


def _read_csv_rows(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def _bool_field(value):
    return str(value).strip().lower() == "true" if value not in (None, "") else None


def _plot_e_gap(cases_a, cases_b, output):
    files = []
    variants = sorted({c.variant for c in cases_a})
    a_rows = [c for c in cases_a if _finite(c.metrics.get("E_dir_um"))]
    a_u = [float(c.U) for c in cases_a if _finite(c.U)]
    same_u = (None if not a_u else
              bool(np.ptp(a_u) <= max(1e-15, 1e-8*max(abs(x) for x in a_u))))
    if a_rows:
        reference_case = a_rows[0]
        z_m = reference_case.p.get("z_m", reference_case.p.get("z_T", .165))
        probe = reference_case.p.get("probe_reference", "ground")
        fig, ax = plt.subplots(figsize=(7.3, 4.5))
        for variant in variants:
            rows = sorted((c for c in a_rows if c.variant == variant), key=lambda c: c.gap_um)
            if rows:
                ax.plot([c.gap_um for c in rows], [c.metrics["E_dir_um"] for c in rows],
                        marker=_marker("A", variant), linestyle=_gap_linestyle(len(rows)),
                        linewidth=1.4,
                        label=f"{variant}, U={_fmt(rows[0].U, 3)} kg·m")
        ax.set(xlabel="Front radial clearance c (µm)", ylabel="E_dir (µm)")
        fig.suptitle(f"E_dir = max(mean per-turn X PP, mean per-turn Y PP); retains 1×; "
                     f"rotor section z={float(z_m):.3f} m relative to {probe}")
        title = ("A group — fixed old U, mechanism comparison" if same_u is True else
                 "A group — U differs between completed cases; see legend" if same_u is False else
                 "A group — U not recorded in actual configs")
        _annotate_scope(ax, title)
        ax.legend()
        files += _save(fig, output, "01_A_same_U_E_gap")

    b_rows = sorted((c for c in cases_b if _finite(c.metrics.get("E_dir_um"))),
                    key=lambda c: c.gap_um)
    if b_rows:
        fig, ax = plt.subplots(figsize=(7.3, 4.5))
        b_u = [float(c.U) for c in b_rows if _finite(c.U)]
        b_same_u = (None if not b_u else
                    bool(np.ptp(b_u) <= max(1e-15, 1e-8*max(abs(x) for x in b_u))))
        a_n3_paths = {c.folder.resolve() for c in cases_a if c.variant == "N3"}
        reused = [c for c in b_rows if c.folder.resolve() in a_n3_paths]
        new_runs = [c for c in b_rows if c.folder.resolve() not in a_n3_paths]
        ax.plot([c.gap_um for c in b_rows], [c.metrics["E_dir_um"] for c in b_rows],
                linestyle=_gap_linestyle(len(b_rows)), linewidth=1.6, color="#b24a3b",
                label=(f"B group, U={_fmt(b_rows[0].U, 3)} kg·m" if b_same_u is True else
                       "B group, U varies (see actual_config per point)" if b_same_u is False else
                       "B group, U not recorded"))
        if reused:
            ax.scatter([c.gap_um for c in reused], [c.metrics["E_dir_um"] for c in reused],
                       marker="s", facecolors="none", edgecolors="#b24a3b", s=54,
                       zorder=3, label="Reused A:N3 trajectories")
        if new_runs:
            ax.scatter([c.gap_um for c in new_runs],
                       [c.metrics["E_dir_um"] for c in new_runs], marker="o",
                       color="#b24a3b", s=36, zorder=3, label="New B integrations")
        ax.legend()
        ax.set(xlabel="Front radial clearance c (µm)", ylabel="E_dir (µm)")
        _annotate_scope(ax, "B group — recalibrated U, plotted separately from A's fixed-U sweep")
        reused_gaps = ", ".join(f"{c.gap_um:g}" for c in reused) or "none"
        new_gaps = ", ".join(f"{c.gap_um:g}" for c in new_runs) or "none"
        fig.suptitle(f"B E_dir curve; reused A:N3 at c={reused_gaps} µm; "
                     f"new B integrations at c={new_gaps} µm")
        files += _save(fig, output, "02_B_recalibrated_E_gap")

    # Small mechanism differences can be hard to see on the absolute E scale.
    # Compare completed c=2/5/10 points against B0 at the same clearance.
    baseline = {round(c.gap_um, 6): float(c.metrics["E_dir_um"])
                for c in a_rows if c.variant == "B0" and c.gap_um in (2.0, 5.0, 10.0)}
    delta_rows = []
    for case in a_rows:
        gap = round(float(case.gap_um), 6)
        if gap not in (2.0, 5.0, 10.0) or gap not in baseline:
            continue
        base = baseline[gap]
        value = float(case.metrics["E_dir_um"])
        delta_rows.append(dict(
            variant=case.variant, gap_um=gap, E_dir_um=value,
            B0_E_dir_um=base, delta_vs_B0_um=value-base,
            percent_vs_B0=(100*(value-base)/base if base else None),
            independent_error_budget=None,
        ))
    if delta_rows:
        acceptance_path = (Path(output).parent / "validation" / "acceptance_summary"
                           / "main_summary.csv")
        acceptance = {}
        for accepted in _read_csv_rows(acceptance_path):
            try:
                key = (accepted["group"], accepted["variant"],
                       round(float(accepted["gap_um"]), 6))
                acceptance[key] = accepted
            except (KeyError, TypeError, ValueError):
                continue
        for row in delta_rows:
            accepted = acceptance.get(("A", row["variant"], round(row["gap_um"], 6)), {})
            complete = _bool_field(accepted.get("epsilon_budget_complete"))
            epsilon = accepted.get("epsilon_E_um")
            row["case_epsilon_num_um"] = float(epsilon) if _finite(epsilon) else None
            row["case_epsilon_budget_complete"] = complete
            row["independent_error_budget"] = (
                "complete" if complete is True else
                "not available for this case" if accepted else
                "not listed in acceptance summary")
            # This acceptance column is specifically the B0-vs-N3 same-U
            # comparison, so do not project it onto B0d/N1/N2.
            row["B0_N3_S_E_vs_numeric_budget"] = (
                float(accepted["same_U_old_new_S_E"])
                if row["variant"] in ("B0", "N3") and
                _finite(accepted.get("same_U_old_new_S_E")) else None)
            row["B0_N3_numeric_comparison_status"] = (
                accepted.get("same_U_old_new_status")
                if row["variant"] in ("B0", "N3") else None)
            row["error_budget_source"] = (
                "validation/acceptance_summary/main_summary.csv"
                if accepted else "not listed in acceptance summary")

        _write_csv(output / "A_E_difference_vs_B0.csv", delta_rows)
        present_variants = sorted({row["variant"] for row in delta_rows})
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        xloc = {name: index for index, name in enumerate(present_variants)}
        for gap, marker, offset, color in (
                (2.0, "o", -.16, "#276b9a"),
                (5.0, "D", 0.0, "#49845a"),
                (10.0, "s", .16, "#c25b3c")):
            subset = [row for row in delta_rows if row["gap_um"] == gap]
            if subset:
                ax.scatter([xloc[row["variant"]]+offset for row in subset],
                           [row["delta_vs_B0_um"] for row in subset],
                           marker=marker, color=color, s=44, zorder=3,
                           label=f"c={gap:g} µm")
        ax.axhline(0, color="0.35", linewidth=.9)
        ax.set_xticks(range(len(present_variants)), present_variants)
        ax.set(xlabel="A-group mechanism", ylabel="ΔE_dir vs B0 at same c (µm)")
        comparison_note = []
        for gap in (2.0, 5.0, 10.0):
            b0 = acceptance.get(("A", "B0", gap), {})
            n3 = acceptance.get(("A", "N3", gap), {})
            significance = n3.get("same_U_old_new_S_E")
            if (_bool_field(b0.get("epsilon_budget_complete")) is True and
                    _bool_field(n3.get("epsilon_budget_complete")) is True and
                    _finite(significance)):
                comparison_note.append(f"c={gap:g}: S_E={float(significance):.1f}")
        if comparison_note:
            scope = ("B0–N3 numerical-budget comparison: " + ", ".join(comparison_note) +
                     "; this is not measured detectability. Other variants are descriptive.")
        else:
            scope = ("E-budget status follows validation/acceptance_summary/main_summary.csv; "
                     "unlisted comparisons are descriptive.")
        fig.suptitle("A-group E_dir differences relative to B0 at the same clearance")
        _annotate_scope(ax, scope)
        ax.legend(title="Completed point")
        files += _save(fig, output, "01b_A_E_delta_vs_B0")
    return files


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_native(case, *names):
    path = case.folder / "native.npz"
    with np.load(path, allow_pickle=False) as data:
        found = {name: data[name] for name in names if name in data.files}
        metadata = None
        if "metadata_json" in data.files:
            try:
                metadata = json.loads(str(data["metadata_json"]))
            except (ValueError, TypeError):
                metadata = {}
    return found, metadata or {}


def _chosen(cases, gap, variants, include_b=True):
    matches = [c for c in cases if abs(c.gap_um-gap) < 1e-6 and
               ((c.group == "A" and (not variants or c.variant in variants)) or
                (include_b and c.group == "B"))]
    # A repeated folder from a resumed run must not be drawn twice.
    unique = {}
    for case in matches:
        unique[(case.group, case.variant)] = case
    return list(unique.values())


def _plot_orbits(cases, output, variants):
    files = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plotted = False
    for row, gap in enumerate((2.0, 10.0)):
        selected = _chosen(cases, gap, variants)
        for case in selected:
            arrays, _ = _read_native(case, "t_raw", "r_m_raw_m")
            if not {"t_raw", "r_m_raw_m"}.issubset(arrays):
                continue
            t, r = arrays["t_raw"], arrays["r_m_raw_m"] * 1e6
            f1x = abs(float(case.p.get("rpm", 0.0)))/60
            if f1x > 0:
                use = t >= t[-1] - 10/f1x
                t, r = t[use], r[use]
            stride = max(1, len(t)//5000)
            label = f"{case.group}:{case.variant}" if case.group == "A" else case.label
            for col, centered in enumerate((False, True)):
                ax = axes[row, col]
                values = r - np.mean(r, axis=0) if centered else r
                ax.plot(values[::stride, 0], values[::stride, 1],
                        linewidth=.65, alpha=.8, label=label)
                plotted = True
        axes[row, 0].set_title(f"c={gap:g} µm — mean retained")
        axes[row, 1].set_title(f"c={gap:g} µm — mean removed")
        for col in range(2):
            axes[row, col].set_xlabel("X (µm)")
            axes[row, col].set_ylabel("Y (µm)")
            axes[row, col].set_aspect("equal", adjustable="datalim")
            if ax.lines:
                axes[row, col].legend(loc="best")
    if not plotted:
        plt.close(fig)
        return []
    fig.suptitle("Measured rotor-section center relative to the configured probe reference; "
                 "final 10 turns, static mean retained on left")
    files += _save(fig, output, "03_orbits_mean_and_centered")
    return files


def _turns(case):
    path = case.folder / "turns.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _plot_turns_tilt(cases, output, variants):
    files = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for gap in (2.0, 10.0):
        for case in _chosen(cases, gap, variants):
            rows = _turns(case)
            if not rows:
                continue
            turn = np.array([float(r.get("turn_id", i)) for i, r in enumerate(rows)])
            label = f"{case.group}:{case.variant} c={gap:g}"
            for col, key in enumerate(("Ex_um", "Ey_um")):
                axes[0].plot(turn, [float(r[key]) for r in rows],
                             linewidth=1.0, label=f"{label} {key[1]}")
    axes[0].set(xlabel="Turn index", ylabel="Per-turn peak-to-peak (µm)",
                title="Per-turn X/Y peak-to-peak")
    if axes[0].lines:
        axes[0].legend(ncol=2)

    # Tilts are shown for one representative c=2 case over its final ten turns.
    tilt_candidates = _chosen(cases, 2.0, variants)
    tilt_case = next((c for c in tilt_candidates if c.group == "A" and c.variant == "N3"), None)
    tilt_case = tilt_case or next((c for c in tilt_candidates if c.group == "B"), None)
    tilt_case = tilt_case or (tilt_candidates[0] if tilt_candidates else None)
    if tilt_case is not None:
        arrays, _ = _read_native(tilt_case, "t_raw", "q_raw_m")
        if {"t_raw", "q_raw_m"}.issubset(arrays):
            t, q = arrays["t_raw"], arrays["q_raw_m"]
            f1x = abs(float(tilt_case.p.get("rpm", 0.0)))/60
            use = (t >= t[-1]-10/f1x) if f1x > 0 else np.ones(len(t), dtype=bool)
            t, q = t[use], q[use]
            stride = max(1, len(t)//8000)
            rel_t = t[::stride]-t[0]
            label = f"{tilt_case.group}:{tilt_case.variant} c=2"
            axes[1].plot(rel_t, q[::stride, 8]*1e3, linewidth=.85,
                         label=f"{label} θx")
            axes[1].plot(rel_t, q[::stride, 9]*1e3, linewidth=.85, linestyle="--",
                         label=f"{label} θy")
    axes[1].set(xlabel="Time in final ten turns (s)", ylabel="Tilt (mrad)",
                title="Representative rotor tilt; final ten turns")
    if axes[1].lines:
        axes[1].legend(ncol=2)
    if not axes[0].lines and not axes[1].lines:
        plt.close(fig)
        return []
    fig.suptitle("Per-turn E over the saved retained window; rotor tilt shown for the final 10 turns")
    files += _save(fig, output, "04_per_turn_error_and_tilt")
    return files


def _plot_h_gap(cases, output):
    keys = (
        ("H_v_rms_{axis}_mm_s", "Stored H velocity RMS", "Velocity RMS (mm/s)"),
        ("H_v_1x_{axis}_mm_s", "H velocity 1× amplitude", "1× amplitude (mm/s)"),
        ("H_v_residual_{axis}_mm_s", "H velocity residual after 1× removal",
         "Residual RMS (mm/s)"),
    )
    files = []
    a_n3_paths = {c.folder.resolve() for c in cases if c.group == "A" and c.variant == "N3"}
    for group in ("A", "B"):
        group_cases = [c for c in cases if c.group == group]
        if not group_cases:
            continue
        labels = sorted({c.variant for c in group_cases})
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, (key_template, title, ylabel) in zip(axes, keys):
            for color, variant in zip(colors, labels):
                rows = sorted([c for c in group_cases if c.variant == variant],
                              key=lambda c: c.gap_um)
                for axis, marker in (("x", "o"), ("y", "s")):
                    usable = [(c.gap_um, c.metrics.get(key_template.format(axis=axis)))
                              for c in rows]
                    usable = [(x, y) for x, y in usable if _finite(y)]
                    if usable:
                        ax.plot([x for x, _ in usable], [y for _, y in usable],
                                marker=(_marker(group, variant) if group == "A" else marker),
                                markerfacecolor=("none" if axis == "y" else color),
                                markeredgecolor=color, color=color, linewidth=1.1,
                                linestyle=(_gap_linestyle(len(usable)) if axis == "x" else
                                           "--" if len(usable) >= 3 else "None"),
                                label=f"{group}:{variant} {axis.upper()}")
            ax.set(xlabel="Clearance c (µm)", ylabel=ylabel, title=title)
            if ax.lines:
                ax.legend(fontsize=7, ncol=2)
        if not any(ax.lines for ax in axes):
            plt.close(fig)
            continue
        bands = {tuple(c.config.get("processing", {}).get("passband_hz", ()))
                 for c in group_cases
                 if c.config.get("processing", {}).get("passband_hz")}
        rates = {c.config.get("simulation", {}).get("fs_out_hz") for c in group_cases
                 if c.config.get("simulation", {}).get("fs_out_hz")}
        band_note = (", ".join(f"{lo:g}–{hi:g} Hz" for lo, hi in sorted(bands))
                     if bands else "band per actual config")
        rate_note = (", ".join(f"{float(rate):g} Hz" for rate in sorted(rates))
                     if rates else "sample rate per actual config")
        if group == "A":
            group_note = "A fixed-U mechanism sweep; separate from B"
        else:
            reused_gaps = sorted(c.gap_um for c in group_cases
                                 if c.folder.resolve() in a_n3_paths)
            new_gaps = sorted(c.gap_um for c in group_cases
                              if c.folder.resolve() not in a_n3_paths)
            fmt_gaps = lambda values: ", ".join(f"{gap:g}" for gap in values) or "none"
            group_note = (f"B five-point curve; reused A:N3 at c={fmt_gaps(reused_gaps)} µm; "
                          f"new integrations at c={fmt_gaps(new_gaps)} µm")
        fig.suptitle(f"{group_note}; H primary band {band_note}; fs_out={rate_note}")
        files += _save(fig, output, f"05_{group}_H_RMS_1x_residual_vs_gap")
    return files


def _spectrum(t, y):
    if len(y) < 8:
        return np.array([]), np.array([])
    fs = 1 / np.median(np.diff(t))
    y = np.asarray(y, dtype=float) - np.mean(y)
    win = np.hanning(len(y))
    amplitude_peak = 2 * np.abs(np.fft.rfft(y * win)) / np.sum(win)
    if len(amplitude_peak) > 1:
        amplitude_peak[0] *= .5
        if len(y) % 2 == 0:
            amplitude_peak[-1] *= .5
    return np.fft.rfftfreq(len(y), d=1/fs), amplitude_peak/np.sqrt(2)


def _plot_h_time_spectrum(cases, output, variants):
    files = []
    for gap in (2.0, 10.0):
        selected = _chosen(cases, gap, variants)
        if not selected:
            continue
        fig, axes = plt.subplots(len(selected), 2,
                                 figsize=(10, max(3.4, 2.8*len(selected))),
                                 squeeze=False)
        for row, case in enumerate(selected):
            arrays, _ = _read_native(case, "t_out", "H_velocity_mm_s")
            if not {"t_out", "H_velocity_mm_s"}.issubset(arrays):
                continue
            t, h = arrays["t_out"], arrays["H_velocity_mm_s"]
            if h.ndim != 2 or h.shape[1] < 2 or len(t) != len(h):
                continue
            length = min(len(t), max(64, int(round(.12/np.median(np.diff(t))))))
            t_plot = t[:length]-t[0]
            for i, (axis, color) in enumerate((("X", "#276b9a"), ("Y", "#c25b3c"))):
                axes[row, 0].plot(t_plot, h[:length, i], color=color,
                                  linewidth=.85, label=f"H{axis}")
                freq, amp = _spectrum(t, h[:, i])
                use = freq <= 2048
                axes[row, 1].plot(freq[use], amp[use], color=color,
                                  linewidth=.8, label=f"H{axis}")
            rpm = abs(float(case.p.get("rpm", 0.0)))
            one_x = rpm/60
            axes[row, 1].axvline(one_x, color="0.25", linestyle=":", linewidth=.9,
                                 label=f"1×={one_x:.2f} Hz")
            passband = case.config.get("processing", {}).get("passband_hz", (10, 900))
            low, high = map(float, passband)
            sample_rate = float(case.config.get("simulation", {}).get(
                "fs_out_hz", 1/np.median(np.diff(t))))
            spectrum_limit = min(2048., sample_rate/2)
            axes[row, 1].axvspan(low, high, color="#49845a", alpha=.08,
                                 label=f"stored primary band {low:g}–{high:g} Hz")
            axes[row, 1].axvline(high, color="#49845a", linestyle="--", linewidth=.8,
                                 label=f"primary cutoff {high:g} Hz")
            axes[row, 0].set(xlabel="Time from export start (s)", ylabel="Velocity (mm/s)",
                             title=case.label)
            axes[row, 1].set(xlabel="Frequency (Hz)", ylabel="One-sided RMS amplitude (mm/s)",
                             xlim=(0, spectrum_limit),
                             title=f"Stored H spectrum; fs={sample_rate:g} Hz")
            axes[row, 0].legend(ncol=2)
            axes[row, 1].legend(ncol=3)
        if not any(ax.lines for ax in axes.ravel()):
            plt.close(fig)
            continue
        fig.suptitle(f"H time trace and spectrum at c={gap:g} µm; stored trace is primary 10–900 Hz; "
                     "above band shows FIR roll-off, not the separate 10–2048 Hz auxiliary metric")
        files += _save(fig, output, f"06_H_time_spectrum_c{gap:g}um")
    return files


def _tai_fields():
    # Import lazily: the plotting path can still handle a B0-only archive.
    try:
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from spindle import tai_bearing
        return tai_bearing.BALL_NAMES, tai_bearing
    except ImportError:
        return None, None


def _load_ball_array(case, bearing):
    path = case.folder / "balls_first10turns.npz"
    fields, tai = _tai_fields()
    if not path.exists() or fields is None:
        return None
    key = f"{bearing}_tai_32_fields"
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files or "t_raw" not in data.files:
            return None
        values = data[key]
        time = data["t_raw"]
    if values.ndim != 3 or values.shape[-1] != len(fields) or len(time) != len(values):
        return None
    return time, values, fields, tai


def _plot_ball_diagnostics(cases, output, variants, bearing):
    preferred = [c for c in cases if c.group == "A" and c.variant == "N3"]
    preferred += [c for c in cases if c.group == "B"]
    preferred += [c for c in cases if c.group == "A" and c.variant not in ("B0", "B0d")]
    preferred += [c for c in cases if c.group == "A" and c.variant in variants]
    selected = next((c for c in preferred if abs(c.gap_um-2.0) < 1e-6 and
                     _load_ball_array(c, bearing) is not None), None)
    if selected is None:
        return []
    loaded = _load_ball_array(selected, bearing)
    if loaded is None:
        return []
    t, balls, fields, tai = loaded
    field_i = {name: i for i, name in enumerate(fields)}
    d_m = selected.p.get("tai_bearing", {}).get("families", {}).get("front", {}).get("D_m")
    if not d_m or "alpha_i_rad" not in field_i:
        return []
    mid = len(t)//2
    ball_number = np.arange(1, balls.shape[1]+1)
    stride = max(1, len(t)//2500)
    ts = (t-t[0])[::stride]
    display = balls[::stride]
    q_i = display[..., field_i["Qi_N"]]
    qi_positive = q_i > 0
    safe_ratio = np.divide(np.abs(display[..., field_i["Mg_N_m"]]),
                           float(d_m)*q_i,
                           out=np.full_like(q_i, np.nan), where=qi_positive)
    active = np.rint(display[..., field_i["active_code"]]).astype(int)
    dual_contact_fraction = np.mean(active == 2, axis=1)
    any_contact_fraction = np.mean(active > 0, axis=1)
    fig, axes = plt.subplots(5, 2, figsize=(10, 13), sharex="col")
    fields_and_pairs = [
        (("alpha_i_rad", "alpha_o_rad"), "Contact angle (deg)", 180/np.pi, "left"),
        (("Qi_N", "Qo_N"), "Normal load (N)", 1.0, "left"),
    ]
    for row, (names, ylabel, scale, _) in enumerate(fields_and_pairs):
        ax = axes[row, 0]
        for name, color in zip(names, ("#276b9a", "#c25b3c")):
            ax.plot(ball_number, balls[mid, :, field_i[name]]*scale,
                    marker="o", linewidth=1.0, color=color,
                    label=name.replace("_rad", "").replace("_N", ""))
        ax.set(xlabel="Ball index (phase order)", ylabel=ylabel,
               title=f"Mid-window snapshot — {bearing}")
        ax.legend()
        ax = axes[row, 1]
        for name, color in zip(names, ("#276b9a", "#c25b3c")):
            values = display[..., field_i[name]]*scale
            ax.plot(ts, np.mean(values, axis=1), color=color, linewidth=.9,
                    label=f"{name.replace('_rad', '').replace('_N', '')} mean")
            ax.fill_between(ts, np.min(values, axis=1), np.max(values, axis=1),
                            color=color, alpha=.12)
        ax.set(xlabel="Time in first 10 retained turns (s)", ylabel=ylabel,
               title="Across-ball mean and min–max")
        ax.legend()

    # Keep Fc magnitude visible beside its separate |Mg|/(D Qi) ratio.
    fc_name, mg_name = "Fc_N", "Mg_N_m"
    if fc_name not in field_i or mg_name not in field_i:
        plt.close(fig)
        return []
    axes[2, 0].plot(ball_number, balls[mid, :, field_i[fc_name]], "o-",
                    color="#49845a", linewidth=1, label="Fc")
    axes[2, 0].set(xlabel="Ball index (phase order)", ylabel="Centrifugal force (N)",
                   title="Mid-window ball force")
    axes[2, 0].legend()
    fc = display[..., field_i[fc_name]]
    axes[2, 1].plot(ts, np.max(fc, axis=1), color="#49845a", label="max Fc")
    axes[2, 1].set(xlabel="Time in first 10 retained turns (s)",
                   ylabel="Centrifugal force (N)", title="Maximum across balls")
    axes[2, 1].legend()

    mid_display = min(int(np.argmin(np.abs(ts-(t[mid]-t[0])))), len(ts)-1)
    axes[3, 0].plot(ball_number, safe_ratio[mid_display], "o-",
                    color="#8a5aa3", linewidth=1)
    axes[3, 0].set(xlabel="Ball index (phase order)", ylabel="|Mg| / (D Qi)",
                   title="Only balls with Qi > 0 are shown")
    valid_ratio = np.isfinite(safe_ratio)
    count_ratio = np.sum(valid_ratio, axis=1)
    mean_ratio = np.divide(np.nansum(safe_ratio, axis=1), count_ratio,
                           out=np.full(len(ts), np.nan), where=count_ratio > 0)
    max_ratio = np.array([np.max(row[np.isfinite(row)]) if np.isfinite(row).any()
                          else np.nan for row in safe_ratio])
    axes[3, 1].plot(ts, mean_ratio, color="#8a5aa3", label="mean")
    axes[3, 1].plot(ts, max_ratio, color="#8a5aa3", linestyle="--", label="max")
    axes[3, 1].set(xlabel="Time in first 10 retained turns (s)", ylabel="|Mg| / (D Qi)",
                   title="Qi=0 samples excluded")
    axes[3, 1].legend()

    fc = display[..., field_i[fc_name]]
    fc_over_qi = np.divide(fc, q_i, out=np.full_like(q_i, np.nan),
                           where=qi_positive)
    axes[4, 0].plot(ball_number, fc_over_qi[mid_display], "o-",
                    color="#d08b2e", linewidth=1)
    axes[4, 0].set(xlabel="Ball index (phase order)", ylabel="Fc / Qi",
                   title="Mid-window Fc/Qi snapshot; Qi=0 excluded")
    axes[4, 1].plot(ts, dual_contact_fraction, color="#334a83", linewidth=1,
                    label="dual contact (ACTIVE=2)")
    axes[4, 1].plot(ts, any_contact_fraction, color="#49845a", linewidth=1,
                    linestyle="--", label="any contact (ACTIVE>0)")
    axes[4, 1].set(xlabel="Time in first 10 retained turns (s)", ylabel="Fraction",
                   ylim=(-.02, 1.02), title="Contact fraction across balls")
    axes[4, 1].legend()
    raw_rate = selected.config.get("simulation", {}).get("fs_raw_hz")
    fig.suptitle(f"Tai ball diagnostics — {selected.group}:{selected.variant}, "
                 f"c={selected.gap_um:g} µm; first 10 retained turns, "
                 f"source fs={_fmt(raw_rate, 5)} Hz, display stride={stride}; conditional local states\n"
                 "Source: balls_first10turns.npz; fields follow spindle.tai_bearing.BALL_NAMES")
    return _save(fig, output, "07_tai_ball_diagnostics")


def _wrench_case(case):
    path = case.folder / "bearing_wrenches.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            needed = {"t_raw"}
            for name in RING_NAMES:
                needed.add(f"{name}_inner_restoring_6")
                needed.add(f"{name}_outer_restoring_6")
                needed.add(f"{name}_outer_axial_constraint_N")
            if not needed.issubset(keys):
                return None
            result = {key: data[key] for key in needed}
            for name in RING_NAMES:
                key = f"{name}_outer_tilt_constraint_Nm"
                if key in keys:
                    result[key] = data[key]
        result["_diagnostic_schema"] = "v9_independent_inner_outer_wrenches"
        return result

    # V8 saved only the inner restoring terms and fixed-outer reaction terms.
    # This explicit equal/opposite reconstruction is a plotting convention;
    # it does not recover independent outer-ring Tai states.
    native_path = case.folder / "native.npz"
    if not native_path.exists():
        return None
    with np.load(native_path, allow_pickle=False) as data:
        keys = set(data.files)
        needed = {"t_raw"}
        for name in RING_NAMES:
            needed.update((f"{name}_radial_restoring_N",
                           f"{name}_local_restoring_Nm",
                           f"{name}_axial_restoring_N"))
        if not needed.issubset(keys):
            return None
        result = {"t_raw": data["t_raw"]}
        for name in RING_NAMES:
            radial = data[f"{name}_radial_restoring_N"]
            local = data[f"{name}_local_restoring_Nm"]
            axial = data[f"{name}_axial_restoring_N"]
            inner = np.zeros((len(result["t_raw"]), 6), dtype=float)
            inner[:, :2] = radial
            inner[:, 2] = axial
            inner[:, 3:5] = local
            result[f"{name}_inner_restoring_6"] = inner
            result[f"{name}_outer_restoring_6"] = -inner
            result[f"{name}_outer_axial_constraint_N"] = -axial
            result[f"{name}_outer_tilt_constraint_Nm"] = -local
    result["_diagnostic_schema"] = "legacy_v8_equal_opposite_reconstruction"
    return result


def _has_wrench_data(case):
    paths = (case.folder / "bearing_wrenches.npz", case.folder / "native.npz")
    for path in paths:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
            required = {"t_raw"}
            for name in RING_NAMES:
                if path.name == "bearing_wrenches.npz":
                    required.update((f"{name}_inner_restoring_6",
                                     f"{name}_outer_restoring_6",
                                     f"{name}_outer_axial_constraint_N"))
                else:
                    required.update((f"{name}_radial_restoring_N",
                                     f"{name}_local_restoring_Nm",
                                     f"{name}_axial_restoring_N"))
            if required.issubset(keys):
                return True
        except (OSError, ValueError):
            continue
    return False


def _wrench_summary_rows(case, data):
    rows = []
    for name in RING_NAMES:
        row = {"group": case.group, "variant": case.variant, "gap_um": case.gap_um,
               "bearing": name, "U_kg_m": case.U,
               "wrench_schema": data.get("_diagnostic_schema", "unknown")}
        for side in ("inner", "outer"):
            arr = data[f"{name}_{side}_restoring_6"]
            for col, comp in enumerate(("Fx", "Fy", "Fz", "Mx", "My", "Mz")):
                unit = "N" if col < 3 else "Nm"
                if (comp == "Mz" and str(data.get("_diagnostic_schema", "")).startswith("legacy_v8")):
                    row[f"{side}_{comp}_rms_{unit}"] = None
                    row[f"{side}_{comp}_peak_{unit}"] = None
                    continue
                row[f"{side}_{comp}_rms_{unit}"] = float(np.sqrt(np.mean(arr[:, col]**2)))
                row[f"{side}_{comp}_peak_{unit}"] = float(np.max(np.abs(arr[:, col])))
        axial = data[f"{name}_outer_axial_constraint_N"]
        row["outer_axial_constraint_rms_N"] = float(np.sqrt(np.mean(axial**2)))
        row["outer_axial_constraint_peak_N"] = float(np.max(np.abs(axial)))
        tilt_key = f"{name}_outer_tilt_constraint_Nm"
        if tilt_key in data:
            row["outer_tilt_constraint_peak_Nm"] = float(np.max(np.linalg.norm(data[tilt_key], axis=1)))
        rows.append(row)
    return rows


def _plot_wrench_summaries(rows, output):
    files = []
    if not rows:
        return files
    bearings = [name for name in RING_NAMES if any(r["bearing"] == name for r in rows)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len({(r["group"], r["variant"])
                                                         for r in rows}))))
    series_keys = sorted({(r["group"], r["variant"]) for r in rows})
    color_for = dict(zip(series_keys, colors))
    for kind, title, ylabel, component_keys in (
        ("load", "Bearing radial restoring load RMS", "Radial load RMS (N)",
         ("Fx", "Fy", "N")),
        ("moment", "Bearing local restoring moment RMS", "Tilt moment RMS (N·m)",
         ("Mx", "My", "Nm")),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        plotted = False
        xcomp, ycomp, unit = component_keys
        for ax, bearing in zip(axes.ravel(), bearings):
            bearing_rows = [r for r in rows if r["bearing"] == bearing]
            for key in series_keys:
                for side, linestyle in (("inner", "-"), ("outer", "--")):
                    usable = []
                    for row in bearing_rows:
                        if (row["group"], row["variant"]) != key:
                            continue
                        x = row.get(f"{side}_{xcomp}_rms_{unit}")
                        y = row.get(f"{side}_{ycomp}_rms_{unit}")
                        if _finite(x) and _finite(y):
                            usable.append((float(row["gap_um"]), np.hypot(float(x), float(y))))
                    usable.sort()
                    if usable:
                        ax.plot([x for x, _ in usable], [y for _, y in usable],
                                marker=_marker(key[0], key[1]),
                                markerfacecolor=(color_for[key] if side == "inner" else "none"),
                                markeredgecolor=color_for[key],
                                linestyle=(linestyle if len(usable) >= 3 else "None"),
                                linewidth=1,
                                color=color_for[key],
                                label=(f"{key[0]}:{key[1]} {side}" +
                                       (" (V8 reconstructed)" if key[1] == "B0" else "")))
                        plotted = True
            ax.set(title=bearing, xlabel="Clearance c (µm)", ylabel=ylabel)
            if ax.lines:
                ax.legend(fontsize=6, ncol=2)
        for ax in axes.ravel()[len(bearings):]:
            ax.axis("off")
        if plotted:
            fig.suptitle(f"{title} by bearing and mechanism; derived from saved wrench traces")
            files += _save(fig, output, f"08_{kind}_wrench_rms_by_case")
        else:
            plt.close(fig)
    return files


def _plot_wrenches(cases, output, variants, summary_only=False):
    available = [case for case in cases if _has_wrench_data(case)]
    if not available:
        return []
    if summary_only:
        rows = []
        for case in available:
            per_case = _wrench_case(case)
            if per_case is not None:
                rows.extend(_wrench_summary_rows(case, per_case))
            del per_case
        _write_csv(output / "bearing_wrench_summary.csv", rows)
        return _plot_wrench_summaries(rows, output)
    # Show all bearing-local moments and explicit fixed outer support reactions
    # for one selected representative case, plus per-case RMS bars when matched.
    preferred = [c for c in available if c.group == "A" and c.variant == "N3"
                 and abs(c.gap_um-2) < 1e-6]
    preferred += [c for c in available if c.group == "B" and abs(c.gap_um-2) < 1e-6]
    preferred += [c for c in available if abs(c.gap_um-2) < 1e-6
                  and (c.group == "B" or c.variant in variants)]
    case = preferred[0] if preferred else available[0]
    data = _wrench_case(case)
    t = data["t_raw"]
    one_x = abs(float(case.p.get("rpm", 0.0)))/60
    use = (t >= t[-1]-2/one_x) if one_x > 0 else np.ones(len(t), dtype=bool)
    t_show = t[use]
    stride = max(1, len(t_show)//1800)
    t_plot = t_show[::stride]-t_show[0]
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex="col")
    for row, name in enumerate(RING_NAMES):
        inner = data[f"{name}_inner_restoring_6"]
        outer = data[f"{name}_outer_restoring_6"]
        inner_force = np.linalg.norm(inner[use, :2], axis=1)
        outer_force = np.linalg.norm(outer[use, :2], axis=1)
        inner_moment = np.linalg.norm(inner[use, 3:5], axis=1)
        outer_moment = np.linalg.norm(outer[use, 3:5], axis=1)
        axes[row, 0].plot(t_plot, inner_force[::stride], linewidth=.8, label="inner")
        axes[row, 0].plot(t_plot, outer_force[::stride], linewidth=.8, linestyle="--",
                          label="outer")
        axes[row, 1].plot(t_plot, inner_moment[::stride], linewidth=.8, label="inner")
        axes[row, 1].plot(t_plot, outer_moment[::stride], linewidth=.8, linestyle="--",
                          label="outer")
        axes[row, 2].plot(t_plot,
            data[f"{name}_outer_axial_constraint_N"][use][::stride], linewidth=.8,
            label="axial reaction")
        tilt_key = f"{name}_outer_tilt_constraint_Nm"
        if tilt_key in data:
            tilt = np.linalg.norm(data[tilt_key][use], axis=1)
            axes[row, 3].plot(t_plot, tilt[::stride], linewidth=.8,
                              label="outer tilt reaction")
        axes[row, 0].set(ylabel=f"{name} force (N)")
        axes[row, 1].set(ylabel=f"{name} moment (N·m)")
        axes[row, 2].set(ylabel=f"{name} axial (N)")
        axes[row, 3].set(ylabel=f"{name} tilt (N·m)")
        for ax in axes[row]:
            if ax.lines:
                ax.legend(fontsize=7)
    for ax, title in zip(axes[0], ("Radial force magnitude", "Local moment magnitude",
                                   "Fixed outer axial reaction", "Fixed outer tilt reaction")):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Time in final 2 turns (s)")
    schema = data.get("_diagnostic_schema", "unknown")
    schema_note = ("legacy V8: outer terms reconstructed equal/opposite from saved inner terms"
                   if schema.startswith("legacy_v8") else "V9 independently saved inner/outer terms")
    raw_rate = case.config.get("simulation", {}).get("fs_raw_hz")
    fig.suptitle(f"Bearing wrench evidence — {case.group}:{case.variant}, "
                 f"c={case.gap_um:g} µm; final 2 turns of 480; source fs={_fmt(raw_rate, 5)} Hz, "
                 f"display stride={stride}; {schema_note}; full-window RMS/peaks are in CSV")
    files = _save(fig, output, "08_bearing_wrenches_and_constraints")
    del data, t, inner, outer

    rows = []
    for case in available:
        per_case = _wrench_case(case)
        if per_case is not None:
            rows.extend(_wrench_summary_rows(case, per_case))
        del per_case
    _write_csv(output / "bearing_wrench_summary.csv", rows)
    files += _plot_wrench_summaries(rows, output)
    return files


def _performance_rows(cases):
    rows = []
    for case in cases:
        m, p = case.metrics, case.p
        solver = []
        settings_path = case.folder / "actual_solver_settings.json"
        if settings_path.exists():
            try:
                solver = _read_json(settings_path)
            except (OSError, json.JSONDecodeError):
                solver = []
        accepted = sum(item.get("accepted_steps", {}).get("count", 0) for item in solver)
        residuals = [v for k, v in m.items() if k.endswith("max_force_residual_N") and _finite(v)]
        geometry = [v for k, v in m.items() if k.endswith("max_geometry_residual_normalized") and _finite(v)]
        failures = [v for k, v in m.items() if k.endswith("local_failure_count") and _finite(v)]
        start, end = m.get("measurement_start_s"), m.get("measurement_end_s")
        simulated_seconds = (float(end)-float(start)
            if _finite(start) and _finite(end) and float(end) > float(start) else None)
        wall = m.get("wall_seconds")
        nfev = m.get("nfev")
        iteration_total = 0.0
        iteration_count = 0
        stats_path = case.folder / "per_ball_statistics.csv"
        if stats_path.exists():
            with stats_path.open(encoding="utf-8-sig", newline="") as stream:
                for item in csv.DictReader(stream):
                    if item.get("field") == "iterations":
                        try:
                            iteration_total += float(item["mean"])*int(item["finite_samples"])
                            iteration_count += int(item["finite_samples"])
                        except (KeyError, TypeError, ValueError):
                            continue
        rows.append(dict(
            group=case.group, variant=case.variant, gap_um=case.gap_um,
            U_kg_m=case.U, excitation_peak_N=case.drive.get("peak_force_N"),
            E_dir_um=m.get("E_dir_um"), complete_turns=m.get("complete_turns"),
            steady_pass=m.get("steady_pass"), wall_seconds=wall,
            simulated_window_seconds=simulated_seconds, nfev=nfev,
            wall_seconds_per_RHS_eval=(float(wall)/float(nfev)
                if _finite(wall) and _finite(nfev) and float(nfev) > 0 else None),
            RHS_evals_per_simulated_second=(float(nfev)/simulated_seconds
                if _finite(nfev) and simulated_seconds and simulated_seconds > 0 else None),
            RHS_accepted_steps=accepted or None,
            accepted_steps_per_simulated_second=(accepted/simulated_seconds
                if accepted and simulated_seconds and simulated_seconds > 0 else None),
            summed_local_iterations_estimate=(int(round(iteration_total))
                if iteration_count else None),
            local_iteration_samples=iteration_count or None,
            local_iterations_max=max([v for k, v in m.items()
                if k.endswith("max_iterations") and _finite(v)], default=None),
            worst_force_residual_N=max(residuals, default=None),
            worst_geometry_residual_normalized=max(geometry, default=None),
            recorded_local_failures=max(failures, default=None),
            active_set_switches_full_window=None,
            bearing_dof=p.get("model_dof"), rhs_backend=p.get("rhs_backend"),
            raw_fs_hz=case.config.get("simulation", {}).get("fs_raw_hz"),
            output_fs_hz=case.config.get("simulation", {}).get("fs_out_hz"),
        ))
    baseline = {(row["group"], float(row["gap_um"])): row for row in rows
                if row["variant"] == "B0"}
    for row in rows:
        reference = baseline.get(("A", float(row["gap_um"])))
        row["wall_ratio_vs_B0_same_gap"] = (
            float(row["wall_seconds"])/float(reference["wall_seconds"])
            if reference and _finite(row["wall_seconds"])
            and _finite(reference["wall_seconds"]) and float(reference["wall_seconds"]) > 0
            else None)
        row["nfev_ratio_vs_B0_same_gap"] = (
            float(row["nfev"])/float(reference["nfev"])
            if reference and _finite(row["nfev"]) and _finite(reference["nfev"])
            and float(reference["nfev"]) > 0 else None)
    return rows


def _plot_performance_costs(rows, output):
    if not rows:
        return []
    metrics = (
        ("wall_seconds", "Measured wall time per run", "Wall time (s)",
         lambda row: row.get("wall_seconds")),
        ("rhs_rate", "Solver evaluations per simulated second",
         "RHS evaluations / simulated s", lambda row: row.get("RHS_evals_per_simulated_second")),
        ("local_iterations", "Mean local iterations per Tai sample",
         "Iterations / finite Tai sample",
         lambda row: (float(row["summed_local_iterations_estimate"])
                      / float(row["local_iteration_samples"])
                      if _finite(row.get("summed_local_iterations_estimate"))
                      and _finite(row.get("local_iteration_samples"))
                      and float(row["local_iteration_samples"]) > 0 else None)),
    )
    series_keys = sorted({(row["group"], row["variant"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(series_keys))))
    color_for = dict(zip(series_keys, colors))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    plotted_any = False
    for ax, (_, title, ylabel, value_fn) in zip(axes, metrics):
        for key in series_keys:
            usable = []
            for row in rows:
                if (row["group"], row["variant"]) == key:
                    value = value_fn(row)
                    if _finite(value):
                        usable.append((float(row["gap_um"]), float(value)))
            usable.sort()
            if usable:
                ax.plot([x for x, _ in usable], [y for _, y in usable],
                        marker=_marker(key[0], key[1]),
                        linestyle=_gap_linestyle(len(usable)),
                        linewidth=1.1, color=color_for[key],
                        label=f"{key[0]}:{key[1]}")
                plotted_any = True
        ax.set(xlabel="Clearance c (µm)", ylabel=ylabel, title=title)
        if ax.lines:
            ax.legend(fontsize=7, ncol=2)
    if not plotted_any:
        plt.close(fig)
        return []
    fig.suptitle("Recorded computational cost from completed runs")
    return _save(fig, output, "11_recorded_performance_costs")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _primary_point_rows(cases, output):
    wrench_rows = _read_csv_rows(output / "bearing_wrench_summary.csv")
    wrench_by_key = {}
    for item in wrench_rows:
        try:
            key = (item["group"], item["variant"], round(float(item["gap_um"]), 6),
                   item["bearing"])
            wrench_by_key[key] = item
        except (KeyError, TypeError, ValueError):
            continue

    rows = []
    for case in cases:
        metrics = case.metrics
        row = dict(
            group=case.group, variant=case.variant, gap_um=case.gap_um,
            source_path=str(case.folder), U_kg_m=case.U,
            complete_turns=metrics.get("complete_turns"),
            summary_window="full saved retained window",
            E_x_um=metrics.get("E_x_um"), E_y_um=metrics.get("E_y_um"),
            E_dir_um=metrics.get("E_dir_um"),
            center_mean_x_um=metrics.get("mean_x_um"),
            center_mean_y_um=metrics.get("mean_y_um"),
        )
        native_names = ["q_raw_m"]
        if not (_finite(row["center_mean_x_um"]) and _finite(row["center_mean_y_um"])):
            native_names.append("r_m_raw_m")
        arrays, _ = _read_native(case, *native_names)
        if "q_raw_m" in arrays:
            q = np.asarray(arrays["q_raw_m"])
            if q.ndim == 2 and q.shape[1] > 9:
                for index, axis in ((8, "x"), (9, "y")):
                    theta_mrad = q[:, index] * 1e3
                    row[f"tilt_theta_{axis}_mean_mrad"] = float(np.mean(theta_mrad))
                    row[f"tilt_theta_{axis}_rms_mrad"] = float(
                        np.sqrt(np.mean(theta_mrad**2)))
                del theta_mrad
        if "r_m_raw_m" in arrays:
            r_um = np.asarray(arrays["r_m_raw_m"]) * 1e6
            if not _finite(row["center_mean_x_um"]):
                row["center_mean_x_um"] = float(np.mean(r_um[:, 0]))
            if not _finite(row["center_mean_y_um"]):
                row["center_mean_y_um"] = float(np.mean(r_um[:, 1]))
            del r_um
        if "q_raw_m" in arrays:
            del q
        del arrays

        for metric in (
                "H_v_rms_x_mm_s", "H_v_rms_y_mm_s",
                "H_v_1x_x_mm_s", "H_v_1x_y_mm_s",
                "H_v_residual_x_mm_s", "H_v_residual_y_mm_s"):
            row[metric] = metrics.get(metric)

        for bearing in RING_NAMES:
            item = wrench_by_key.get((case.group, case.variant,
                                      round(case.gap_um, 6), bearing), {})
            row[f"{bearing}_wrench_schema"] = item.get("wrench_schema")
            for side in ("inner", "outer"):
                fx = item.get(f"{side}_Fx_rms_N")
                fy = item.get(f"{side}_Fy_rms_N")
                mx = item.get(f"{side}_Mx_rms_Nm")
                my = item.get(f"{side}_My_rms_Nm")
                row[f"{bearing}_{side}_radial_restoring_rms_N"] = (
                    float(np.hypot(float(fx), float(fy)))
                    if _finite(fx) and _finite(fy) else None)
                row[f"{bearing}_{side}_local_moment_rms_Nm"] = (
                    float(np.hypot(float(mx), float(my)))
                    if _finite(mx) and _finite(my) else None)
            row[f"{bearing}_outer_axial_constraint_rms_N"] = item.get(
                "outer_axial_constraint_rms_N")
            row[f"{bearing}_outer_axial_constraint_peak_N"] = item.get(
                "outer_axial_constraint_peak_N")
            row[f"{bearing}_outer_tilt_constraint_peak_Nm"] = item.get(
                "outer_tilt_constraint_peak_Nm")
        rows.append(row)
    return rows


def _tangent_case(case):
    """Compute M,C,K, poles and fixed-point receptance for one saved case."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from spindle import dynamics_model as model
    p = dict(case.p)
    # Strip runtime caches if a future config serializer happens to include any.
    for key in list(p):
        if key.startswith("_"):
            p.pop(key, None)
    p["_derived_cache_enabled"] = True
    m, c, k = model.linear_matrices(p, 0)
    n = len(m)
    state = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.linalg.solve(m, k), -np.linalg.solve(m, c)]])
    poles = np.linalg.eigvals(state)
    positive = poles[np.imag(poles) > 1e-7]
    modal_rows = []
    for pole in sorted(positive, key=lambda z: np.imag(z)):
        hz = abs(pole)/(2*np.pi)
        modal_rows.append(dict(group=case.group, variant=case.variant,
            gap_um=case.gap_um, U_kg_m=case.U,
            pole_real_s_inv=float(np.real(pole)), pole_imag_s_inv=float(np.imag(pole)),
            damped_frequency_Hz=float(np.imag(pole)/(2*np.pi)),
            natural_frequency_Hz=float(hz), damping_ratio=float(-np.real(pole)/abs(pole))))

    z_m = float(p.get("z_m", p.get("z_T", .165)))
    z_u = float(p.get("z_u", 0.0))
    h_output = np.zeros((2, n)); h_output[:, 4:6] = np.eye(2)
    measured = np.zeros((2, n))
    measured[:, model.ROTOR] = model.section_matrix(z_m)
    measured[:, 4:6] -= np.eye(2)
    force = np.column_stack([model.generalized_point_force([1., 0.], z_u, n),
                             model.generalized_point_force([0., 1.], z_u, n)])
    frequencies = np.unique(np.r_[np.geomspace(1., 2500., 450),
                                   abs(float(p.get("rpm", 0)))/60])
    frf_rows = []
    frf_measure, frf_h = [], []
    for freq in frequencies:
        omega = 2*np.pi*freq
        dynamic = k-omega**2*m+1j*omega*c
        response = np.linalg.solve(dynamic, force)
        tm = measured@response
        th = (1j*omega)*(h_output@response)
        value_m = float(np.linalg.svd(tm, compute_uv=False)[0])
        value_h = float(np.linalg.svd(th, compute_uv=False)[0]*1000)
        frf_measure.append(value_m*1e6)
        frf_h.append(value_h)
        frf_rows.append(dict(group=case.group, variant=case.variant, gap_um=case.gap_um,
            U_kg_m=case.U, frequency_Hz=float(freq),
            measured_receptance_um_per_N=value_m*1e6,
            H_velocity_transfer_mm_s_per_N=value_h))
    skew_c = float(np.linalg.norm(c-c.T)/max(np.linalg.norm(c), 1e-30))
    skew_k = float(np.linalg.norm(k-k.T)/max(np.linalg.norm(k), 1e-30))
    summary = dict(group=case.group, variant=case.variant, gap_um=case.gap_um,
        U_kg_m=case.U, dof=n, M_relative_asymmetry=float(np.linalg.norm(m-m.T)/max(np.linalg.norm(m), 1e-30)),
        C_relative_asymmetry=skew_c, K_relative_asymmetry=skew_k,
        gyro_included_once=True, tangent_source="dynamics_model.linear_matrices; frozen equilibrium",
        scope="local linear diagnostic only; no nonlinear time-domain stability claim")
    return modal_rows, frf_rows, summary, frequencies, np.asarray(frf_measure), np.asarray(frf_h)


def run_tangent(cases, output, variants):
    selected = [c for c in cases if abs(c.gap_um-2.) < 1e-6 or abs(c.gap_um-10.) < 1e-6]
    selected = [c for c in selected if c.group == "B" or c.variant in variants]
    modal, frf, summary, curves = [], [], [], []
    for case in selected:
        mrows, frows, srow, freq, receptance, htransfer = _tangent_case(case)
        modal.extend(mrows); frf.extend(frows); summary.append(srow)
        curves.append((case, freq, receptance, htransfer))
    _write_csv(output / "tangent_poles.csv", modal)
    _write_csv(output / "tangent_frf.csv", frf)
    _write_csv(output / "tangent_summary.csv", summary)
    if not curves:
        return []
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    case_keys = sorted({(row["group"], row["variant"], row["gap_um"]) for row in modal})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(case_keys))))
    for color, key in zip(colors, case_keys):
        rows = [row for row in modal if (row["group"], row["variant"], row["gap_um"]) == key]
        ax.scatter([row["pole_real_s_inv"] for row in rows],
                   [row["damped_frequency_Hz"] for row in rows], s=22, alpha=.85,
                   marker="o" if key[2] < 5 else "s", color=color,
                   label=f"{key[0]}:{key[1]} c={key[2]:g} µm")
    ax.axvline(0, color="0.25", linestyle="--", linewidth=.9)
    ax.set(xlabel="Real part of pole (s⁻¹)", ylabel="Imaginary part / 2π (Hz)",
           title="Positive-imaginary poles from each frozen local tangent")
    if case_keys:
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Frozen tangent poles — local work points only")
    files = _save(fig, output, "09_frozen_tangent_poles")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for case, freq, receptance, htransfer in curves:
        label = f"{case.group}:{case.variant} c={case.gap_um:g} µm"
        axes[0].loglog(freq, receptance, linewidth=1., label=label)
        axes[1].loglog(freq, htransfer, linewidth=1., label=label)
    rpm = abs(float(selected[0].p.get("rpm", 0))) if selected else 0
    if rpm:
        for ax in axes:
            ax.axvline(rpm/60, color="0.3", linestyle=":", label="1×")
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Largest singular value (µm/N)",
                title="Frozen measured-section receptance")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Largest singular value (mm/s/N)",
                title="Frozen H velocity transfer")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Frozen tangent frequency response — local work points only; "
                 "1–2500 Hz logarithmic grid plus 1×")
    files += _save(fig, output, "10_frozen_tangent_frequency_response")
    return files


def generate(root=DEFAULT_EXPERIMENT, data_a=None, data_b=None, output=None,
             variants=("B0", "B0d", "N1", "N2", "N3"), bearing="F1", tangent=False,
             selection=None, gap_only=False):
    root = Path(root)
    data_a = Path(data_a) if data_a else root / "data_A"
    data_b = Path(data_b) if data_b else root / "data_B"
    output = Path(output) if output else root / "figures"
    figures_dir = output
    old_manifest = {}
    try:
        old_manifest = _read_json(output / "figure_manifest.json")
    except (OSError, json.JSONDecodeError):
        old_manifest = {}
    cases = (discover_selected_cases(root, selection) if selection is not None
             else discover_cases(data_a, data_b))
    cases_a = [c for c in cases if c.group == "A"]
    cases_b = [c for c in cases if c.group == "B"]
    _style()
    files = []
    files += _plot_e_gap(cases_a, cases_b, figures_dir)
    files += _plot_h_gap(cases, figures_dir)
    if gap_only:
        files += _plot_wrenches(cases, figures_dir, variants, summary_only=True)
    else:
        files += _plot_orbits(cases, figures_dir, variants)
        files += _plot_turns_tilt(cases, figures_dir, variants)
        files += _plot_h_time_spectrum(cases, figures_dir, variants)
        files += _plot_ball_diagnostics(cases, figures_dir, variants, bearing)
        files += _plot_wrenches(cases, figures_dir, variants)
    performance = _performance_rows(cases)
    _write_csv(output / "performance.csv", performance)
    files += _plot_performance_costs(performance, figures_dir)
    primary_summary = _primary_point_rows(cases, output)
    _write_csv(output / "primary_point_summary.csv", primary_summary)
    b_summary = [row for row in primary_summary if row["group"] == "B"]
    if b_summary:
        _write_csv(output / "B_primary_five_point_summary.csv", b_summary)
    if tangent:
        files += run_tangent(cases, output, variants)
    if gap_only:
        # Preserve figures and prior tangent outputs from the official full run.
        files = list(dict.fromkeys(files + [name for name in old_manifest.get("figures", [])
                                             if (output / name).is_file()]))
    elif not tangent:
        # Keep the saved A-only tangent charts visible without recomputing them
        # for the completed B sweep.
        tangent_names = {"09_frozen_tangent_poles.png", "09_frozen_tangent_poles.svg",
                         "10_frozen_tangent_frequency_response.png",
                         "10_frozen_tangent_frequency_response.svg"}
        retained = [name for name in old_manifest.get("figures", [])
                    if name in tangent_names and (output / name).is_file()]
        files = list(dict.fromkeys(files + retained))
    a_u_values = [float(c.U) for c in cases_a if _finite(c.U)]
    b_u_values = [float(c.U) for c in cases_b if _finite(c.U)]
    a_same_u = (None if not a_u_values else bool(np.ptp(a_u_values) <= max(
        1e-15, 1e-8*max(abs(value) for value in a_u_values))))
    b_same_u = (None if not b_u_values else bool(np.ptp(b_u_values) <= max(
        1e-15, 1e-8*max(abs(value) for value in b_u_values))))
    manifest = dict(
        cases_found=len(cases), A_cases=len(cases_a), B_cases=len(cases_b),
        A_variants=sorted({c.variant for c in cases_a}),
        A_gaps_um=sorted({c.gap_um for c in cases_a}),
        A_same_U=a_same_u,
        B_gaps_um=sorted({c.gap_um for c in cases_b}),
        B_same_U=b_same_u,
        figures=files, tangent_requested=bool(tangent),
        gap_only=bool(gap_only),
        tangent_recomputed_this_run=bool(tangent),
        tangent_outputs_retained=(
            [name for name in ("09_frozen_tangent_poles.png",
                               "09_frozen_tangent_poles.svg",
                               "10_frozen_tangent_frequency_response.png",
                               "10_frozen_tangent_frequency_response.svg")
             if (output / name).is_file()] if not tangent else []),
        tables=[name for name in ("primary_point_summary.csv",
                                  "B_primary_five_point_summary.csv", "performance.csv",
                                  "bearing_wrench_summary.csv",
                                  "A_E_difference_vs_B0.csv", "tangent_summary.csv",
                                  "tangent_poles.csv", "tangent_frf.csv")
                if (output / name).is_file()],
        selection_file=(str(selection) if selection is not None else None),
        tangent_scope="local frozen-equilibrium linearization; not global nonlinear stability",
        wrench_source_schemas=[dict(group=c.group, variant=c.variant, gap_um=c.gap_um,
            schema=("v9_independent_inner_outer_wrenches"
                    if (c.folder/"bearing_wrenches.npz").exists() else
                    "legacy_v8_equal_opposite_reconstruction"))
            for c in cases if _has_wrench_data(c)],
        missing_sections=[name for name, needed in (
            ("Tai per-ball states", not any((c.folder/"balls_first10turns.npz").exists() for c in cases)),
            ("ring wrenches", not any((c.folder/"bearing_wrenches.npz").exists() for c in cases)),
            ("B recalibration curve", not cases_b)) if needed],
        note="Curves include only completed cases with native.npz and metrics.json.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_EXPERIMENT,
                        help="Experiment folder containing data_A and data_B")
    parser.add_argument("--data-a", type=Path, help="Override A-group folder")
    parser.add_argument("--data-b", type=Path, help="Override B-group folder")
    parser.add_argument("--selection", type=Path,
                        help="JSON case selection; selected source folders are relative to --root")
    parser.add_argument("--output", type=Path, help="Output folder; defaults to ROOT/figures")
    parser.add_argument("--variants", default="B0,B0d,N1,N2,N3",
                        help="A-group variants to include in time/orbit comparisons")
    parser.add_argument("--bearing", default="F1", choices=RING_NAMES,
                        help="Representative bearing for per-ball diagnostics")
    parser.add_argument("--tangent", action="store_true",
                        help="Compute frozen local tangent poles/FRF from saved actual configs; does not integrate time histories")
    parser.add_argument("--gap-only", action="store_true",
                        help="Regenerate only clearance comparison figures and tables; retain other existing figures")
    args = parser.parse_args()
    if args.gap_only and args.tangent:
        parser.error("--gap-only cannot be combined with --tangent")
    manifest = generate(args.root, args.data_a, args.data_b, args.output,
                        tuple(x.strip() for x in args.variants.split(",") if x.strip()),
                        args.bearing, args.tangent, selection=args.selection,
                        gap_only=args.gap_only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
