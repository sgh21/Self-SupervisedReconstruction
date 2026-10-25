#!/usr/bin/env python3
"""Reusable local and short assembled-system validation for V9 Tai bearings.

The module checks the independent bearing algebra, the 12-DOF force assembly,
and a short Python/Numba trajectory. It does not run the long experiment
matrix. Results include failed states so a local or global root failure cannot
disappear from an aggregate pass count.
"""
from __future__ import annotations

import argparse
import csv
import copy
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "GPTOrder" / "spindle_v9_tai_bearing_plan" / "templates" / "v9_contract.json"
DEFAULT_SYSTEM_CONFIG = ROOT / "config" / "clearance_v9_config.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "clearance_v9_tai_bearing" / "validation"


def _hertz_oracle(angle_rad: float, family: dict[str, Any], materials: dict[str, Any], *, inner: bool) -> dict[str, float]:
    """F04 Hertz coefficient from the plan, independent of tai_bearing.py."""
    D = float(family["D_m"])
    Dm = float(family["Dm_m"])
    f = float(family["fi" if inner else "fo"])
    if D <= 0 or Dm <= D or f <= 0.5:
        raise ValueError(f"invalid {'inner' if inner else 'outer'} race geometry")
    ball_radius = D / 2
    race_center_radius = ball_radius * (1 + (-1 if inner else 1) * D / Dm * math.cos(angle_rad))
    groove_radius = D * f / (2 * f - 1)
    rx, ry = sorted((race_center_radius, groove_radius))
    if rx <= 0 or ry <= 0:
        raise ValueError("Hertz principal radii must be positive")
    Eball = float(materials["ball_E_Pa"])
    nuball = float(materials["ball_nu"])
    Ering = float(materials["ring_E_Pa"])
    nuring = float(materials["ring_nu"])
    Eprime = 2.0 / ((1.0 - nuball**2) / Eball + (1.0 - nuring**2) / Ering)
    R = 1.0 / (1.0 / rx + 1.0 / ry)
    ratio = ry / rx
    kappa = 1.0339 * ratio**0.6360
    ellipse_E = 1.0003 + 0.5968 / ratio
    ellipse_F = 1.5277 + 0.6023 * math.log(ratio)
    K = math.pi * kappa * Eprime / (3.0 * ellipse_F) * math.sqrt(2.0 * ellipse_E * R / ellipse_F)
    return {
        "K_N_m32": K,
        "Eprime_Pa": Eprime,
        "R_m": R,
        "rx_m": rx,
        "ry_m": ry,
        "kappa": kappa,
        "ellipse_E": ellipse_E,
        "ellipse_F": ellipse_F,
    }


def _zero_speed_oracle(A_m: float, B_m: float, family: dict[str, Any], materials: dict[str, Any]) -> dict[str, float]:
    """Independent F03 zero-inertia, equal-load solution for a single ball."""
    D = float(family["D_m"])
    f_i = float(family["fi"])
    f_o = float(family["fo"])
    a_i = (f_i - 0.5) * D
    a_o = (f_o - 0.5) * D
    B0 = a_i + a_o
    length = math.hypot(A_m, B_m)
    alpha = math.atan2(A_m, B_m)
    delta_sum = max(length - B0, 0.0)
    Ki = _hertz_oracle(alpha, family, materials, inner=True)["K_N_m32"]
    Ko = _hertz_oracle(alpha, family, materials, inner=False)["K_N_m32"]
    Keq = (Ki ** (-2.0 / 3.0) + Ko ** (-2.0 / 3.0)) ** (-1.5)
    Q = Keq * delta_sum**1.5
    delta_i = (Q / Ki) ** (2.0 / 3.0) if Q > 0 else 0.0
    delta_o = (Q / Ko) ** (2.0 / 3.0) if Q > 0 else 0.0
    L_o = (f_o - 0.5) * D + delta_o
    U = L_o * math.sin(alpha)
    V = L_o * math.cos(alpha)
    return {
        "U_m": U,
        "V_m": V,
        "delta_i_m": delta_i,
        "delta_o_m": delta_o,
        "alpha_i_rad": alpha,
        "alpha_o_rad": alpha,
        "Qi_N": Q,
        "Qo_N": Q,
        "Ki_N_m32": Ki,
        "Ko_N_m32": Ko,
        "delta_sum_m": delta_sum,
        "B0_m": B0,
    }


def _preload_oracle(family: dict[str, Any], materials: dict[str, Any]) -> dict[str, float]:
    """Solve the F03 fixed-position preload from the independent F04 oracle."""
    D = float(family["D_m"])
    Dm = float(family["Dm_m"])
    Z = int(family["balls"])
    alpha0 = math.radians(float(family["nominal_contact_angle_deg"]))
    P_ref = float(family["preload_reference_at_rest_N"])
    f_i = float(family["fi"])
    f_o = float(family["fo"])
    B0 = (f_i + f_o - 1.0) * D
    A0 = B0 * math.sin(alpha0)
    B = B0 * math.cos(alpha0)

    def preload_residual(a_pre_m: float) -> float:
        ball = _zero_speed_oracle(A0 + a_pre_m, B, family, materials)
        return Z * ball["Qi_N"] * math.sin(ball["alpha_i_rad"]) - P_ref

    # This bracket is intentionally broad relative to the few-micrometre
    # reference offset; a missing root is recorded as a failed case by caller.
    a_pre = brentq(preload_residual, 0.0, 1e-3, xtol=1e-16, rtol=1e-14)
    ball = _zero_speed_oracle(A0 + a_pre, B, family, materials)
    return {
        "a_pre_m": a_pre,
        "axial_load_N": Z * ball["Qi_N"] * math.sin(ball["alpha_i_rad"]),
        "target_load_N": P_ref,
        "per_ball_Q_N": ball["Qi_N"],
        "alpha_rad": ball["alpha_i_rad"],
        "delta_i_m": ball["delta_i_m"],
        "delta_o_m": ball["delta_o_m"],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _save_results(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = result.get("configuration")
    if contract is not None:
        (output_dir / "local_bearing_contract_snapshot.json").write_text(
            json.dumps(_jsonable(contract), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    system = result.get("system_integration")
    if system is not None:
        system_config = system.get("configuration")
        if system_config is not None:
            (output_dir / "local_spindle_config_snapshot.json").write_text(
                json.dumps(_jsonable(system_config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        (output_dir / "system_integration_validation.json").write_text(
            json.dumps(_jsonable(system), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if system.get("short_trajectory"):
            trajectory = system["short_trajectory"]
            python_q = np.asarray(trajectory["python_q"])
            python_v = np.asarray(trajectory["python_v"])
            python_a = np.asarray(trajectory["python_acceleration"])
            numba_q = np.asarray(trajectory["numba_q"])
            numba_v = np.asarray(trajectory["numba_v"])
            numba_a = np.asarray(trajectory["numba_acceleration"])
            with (output_dir / "system_short_trajectory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                header = ["t_s"]
                for backend in ("python", "numba"):
                    header.extend(f"{backend}_{kind}_{i}" for kind in ("q", "v", "acc")
                                  for i in range(12))
                writer.writerow(header)
                for i, t in enumerate(trajectory["time_s"]):
                    writer.writerow([t, *python_q[i], *python_v[i], *python_a[i],
                                     *numba_q[i], *numba_v[i], *numba_a[i]])
    (output_dir / "local_bearing_validation.json").write_text(
        json.dumps(_jsonable(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = []
    for check in result["checks"]:
        rows.append({
            "check_id": check.get("check_id"),
            "case_id": check.get("case_id"),
            "status": check.get("status"),
            "message": check.get("message", ""),
            "metrics_json": json.dumps(_jsonable(check.get("metrics", {})), ensure_ascii=False),
        })
    with (output_dir / "local_bearing_validation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check_id", "case_id", "status", "message", "metrics_json"])
        writer.writeheader()
        writer.writerows(rows)
    system_checks = [row for row in rows if row["check_id"].startswith(("T07_", "T09_", "T10_", "T11_SYSTEM_", "T12_"))]
    with (output_dir / "system_integration_validation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check_id", "case_id", "status", "message", "metrics_json"])
        writer.writeheader()
        writer.writerows(system_checks)


def _append(checks: list[dict[str, Any]], check_id: str, case_id: str, status: str,
            metrics: dict[str, Any] | None = None, message: str = "") -> None:
    checks.append({"check_id": check_id, "case_id": case_id, "status": status,
                   "metrics": metrics or {}, "message": message})


def _error_record(error: BaseException) -> dict[str, Any]:
    record = {"type": type(error).__name__, "message": str(error)}
    state = getattr(error, "state", None)
    if state is not None:
        record["state"] = state
    return record


def _force_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    """Compare 12 generalized restoring components with force/moment units split."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    force_idx = np.r_[0:8, 10:12]
    force_diff = float(np.max(np.abs(a[force_idx] - b[force_idx])))
    moment_diff = float(np.max(np.abs(a[8:10] - b[8:10])))
    force_scale = max(float(np.max(np.abs(a[force_idx]))), float(np.max(np.abs(b[force_idx]))), 1e-12)
    moment_scale = max(float(np.max(np.abs(a[8:10]))), float(np.max(np.abs(b[8:10]))), 1e-15)
    force_rel, moment_rel = force_diff / force_scale, moment_diff / moment_scale
    return {
        "force_max_abs_difference_N": force_diff,
        "force_relative_difference": force_rel,
        "moment_max_abs_difference_Nm": moment_diff,
        "moment_relative_difference": moment_rel,
        "force_pass": force_diff <= max(1e-5, 1e-7 * force_scale),
        "moment_pass": moment_diff <= max(1e-7, 1e-7 * moment_scale),
    }


def _full_wrench_reference_assembly(t: float, q: np.ndarray, v: np.ndarray, p: dict[str, Any],
                                    tai_system: Any, dynamics: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble full per-ring outputs through the public wrench API."""
    args = tai_system.prepare(p)
    force = args[0] @ q + args[1] @ v
    wrenches = tai_system.bearing_wrenches(t, q, v, p)
    for j, name in enumerate(tai_system.NAMES):
        inner = np.asarray(wrenches[name]["inner_restoring_6"])
        outer = np.asarray(wrenches[name]["outer_restoring_6"])
        z, endpoint = args[4][j, :2]
        force[dynamics.ROTOR] += dynamics.section_matrix(z).T @ inner[:2]
        force[8:10] += inner[3:5]
        start = int(endpoint)
        force[start:start + 2] += outer[:2]
    fit1, _ = dynamics.fit_contact_force(q[6:8] - q[4:6], v[6:8] - v[4:6], p)
    fit2, _ = dynamics.fit_contact_force(
        q[10:12] - q[4:6], v[10:12] - v[4:6],
        dict(N=p["N"], Kf=p["Kf2"], Df=p["Df2"], fit_clearance_um=p["fit2_clearance_um"]))
    force[6:8] += fit1
    force[4:6] -= fit1
    force[10:12] += fit2
    force[4:6] -= fit2
    return force, wrenches


def _run_system_validation(config_path: Path) -> dict[str, Any]:
    """Check V9 assembly/signs and run a two-turn Python/Numba replay."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    p = copy.deepcopy(cfg["parameters"])
    sim_cfg, drive_cfg = cfg["simulation"], cfg["excitation"]
    from spindle import dynamics_model as dynamics
    from spindle import tai_bearing as tai
    from spindle import tai_system
    from spindle.fast_rhs import prepare_fast_rhs

    checks: list[dict[str, Any]] = []
    system: dict[str, Any] = {
        "schema": "spindle_v9_system_integration_validation_v1",
        "status": "running",
        "config_path": str(config_path.resolve()),
        "configuration": cfg,
        "scope": "12-DOF static equilibrium, force assembly and two-turn backend replay; no long experiment matrix",
        "checks": checks,
        "static_equilibrium_attempts": [],
        "limitations": [
            "This is numerical integration validation under the supplied conditional configuration, not machine validation.",
            "The two-turn trajectory checks the short backend path only; it is not a steady-state acceptance window.",
        ],
    }

    q_equilibrium: np.ndarray | None = None
    equilibrium_ok = False
    try:
        q_equilibrium, reported_residual = dynamics.equilibrium(p, 0.0)
        q_equilibrium = np.asarray(q_equilibrium, dtype=float)
        masses, gravity = dynamics.mass_and_gravity(p)
        residual_vector = dynamics.internal_force(0.0, q_equilibrium, np.zeros(12), p, 0.0) - gravity
        residual_equivalent = residual_vector.copy()
        residual_equivalent[8:10] /= float(p["L_ref"])
        independent_residual = float(np.max(np.abs(residual_equivalent)))
        equilibrium_ok = independent_residual <= 1e-6 and math.isfinite(float(reported_residual))
        attempt = {"gap_um": float(p["front_group_clearance_um"]), "status": "passed" if equilibrium_ok else "failed",
                   "reported_residual": float(reported_residual),
                   "independently_recomputed_max_residual_equivalent_N": independent_residual,
                   "q_m_rad": q_equilibrium.tolist()}
        system["static_equilibrium_attempts"].append(attempt)
        _append(checks, "T12_STATIC_EQUILIBRIUM", f"gap={p['front_group_clearance_um']:g}um",
                "passed" if equilibrium_ok else "failed", attempt)
    except Exception as exc:
        failure = _error_record(exc)
        if getattr(exc, "state", None) is not None:
            failure["state"] = exc.state
        attempt = {"gap_um": float(p["front_group_clearance_um"]), "status": "failed", "error": failure}
        system["static_equilibrium_attempts"].append(attempt)
        _append(checks, "T12_STATIC_EQUILIBRIUM", f"gap={p['front_group_clearance_um']:g}um",
                "failed", attempt, "production static equilibrium failed; failure state retained")

    # If the target-gap static solve fails, use a separate c=0 static point only
    # for assembly diagnostics. It never changes the reported target-gap result.
    diagnostic_state = "target_equilibrium"
    if q_equilibrium is None:
        try:
            p_zero = dynamics.front_group_parameters(p, 0.0)
            q_equilibrium, zero_residual = dynamics.equilibrium(p_zero, 0.0)
            q_equilibrium = np.asarray(q_equilibrium, dtype=float)
            diagnostic_state = "zero_gap_equilibrium_for_force_diagnostics_only"
            _append(checks, "T12_ZERO_GAP_DIAGNOSTIC_EQUILIBRIUM", "gap=0um",
                    "passed", {"reported_residual": float(zero_residual), "q_m_rad": q_equilibrium.tolist()},
                    "used only to exercise bearing assembly after the target-gap root failed")
        except Exception as exc:
            failure = _error_record(exc)
            q_equilibrium = np.zeros(12)
            diagnostic_state = "zero_state_after_both_equilibria_failed"
            _append(checks, "T12_ZERO_GAP_DIAGNOSTIC_EQUILIBRIUM", "gap=0um", "failed", failure)

    q_test = q_equilibrium.copy()
    q_test[[0, 1, 4, 5, 6, 7, 10, 11]] += np.array(
        [0.35, -0.22, 0.10, -0.08, 0.18, 0.07, -0.12, -0.16]) * 1e-6
    q_test[8:10] += np.array([1.7, -1.1]) * 1e-6
    v_test = np.array([2.0e-4, -1.1e-4, 1.2e-4, 0.7e-4, -0.5e-4, 0.8e-4,
                       1.0e-4, -0.9e-4, 0.03, -0.02, -0.6e-4, 0.4e-4])
    t_test = 3.71e-4

    try:
        args = tai_system.prepare(p)
        wrenches = tai_system.bearing_wrenches(t_test, q_test, np.zeros(12), p)

        # T07: independent virtual-work identity for all four assembled rings.
        rng = np.random.default_rng(9127)
        delta_q = rng.normal(size=12)
        delta_q[np.r_[0:8, 10:12]] *= 0.2e-6
        delta_q[8:10] *= 2e-5
        work_rows = []
        for j, name in enumerate(tai_system.NAMES):
            inner = np.asarray(wrenches[name]["inner_restoring_6"])
            outer = np.asarray(wrenches[name]["outer_restoring_6"])
            z, endpoint = args[4][j, :2]
            A = dynamics.section_matrix(float(z))
            generalized = np.zeros(12)
            generalized[dynamics.ROTOR] += A.T @ inner[:2]
            generalized[8:10] += inner[3:5]
            start = int(endpoint)
            generalized[start:start + 2] += outer[:2]
            rotor_delta = delta_q[dynamics.ROTOR]
            inner_point_delta = A @ rotor_delta
            inner_work = float(inner[:2] @ inner_point_delta + inner[3:5] @ delta_q[8:10])
            outer_work = float(outer[:2] @ delta_q[start:start + 2])
            from_generalized = float(generalized @ delta_q)
            relative = abs(from_generalized - inner_work - outer_work) / max(
                abs(from_generalized), abs(inner_work + outer_work), 1e-15)
            work_rows.append({"bearing": name, "inner_radial_work_J": float(inner[:2] @ inner_point_delta),
                              "inner_local_moment_work_J": float(inner[3:5] @ delta_q[8:10]),
                              "outer_radial_work_J": outer_work,
                              "assembled_work_J": from_generalized,
                              "relative_virtual_work_error": relative,
                              "inner_moment_norm_Nm": float(np.linalg.norm(inner[3:5]))})
        work_max = max(row["relative_virtual_work_error"] for row in work_rows)
        moment_nonzero = all(row["inner_moment_norm_Nm"] > 1e-12 for row in work_rows)
        _append(checks, "T07_RING_ASSEMBLY_VIRTUAL_WORK", diagnostic_state,
                "passed" if work_max <= 1e-11 and moment_nonzero else "failed", {
                    "max_relative_virtual_work_error": work_max,
                    "tolerance": 1e-11,
                    "all_inner_local_moments_nonzero": moment_nonzero,
                    "per_bearing": work_rows,
                })

        # T09: F08 force differences close on each ball's centrifugal load;
        # inner and outer restoring forces are intentionally not negatives.
        names = {name: i for i, name in enumerate(tai.BALL_NAMES)}
        ring_rows = []
        max_balance_error = 0.0
        max_pair_force_error = 0.0
        max_axial_error = 0.0
        for j, name in enumerate(tai_system.NAMES):
            inner = np.asarray(wrenches[name]["inner_restoring_6"])
            outer = np.asarray(wrenches[name]["outer_restoring_6"])
            balls = np.asarray(wrenches[name]["diagnostics"])
            row = args[4][j]
            params = args[5][j]
            z, endpoint, orientation, rho, cb, phase0, phase_factor = row
            omega = float(p["rpm"]) * 2 * math.pi / 60
            phases = phase0 + omega * float(params[tai.CAGE_RATIO]) * phase_factor * t_test
            phases = phases + 2 * math.pi * np.arange(len(balls)) / len(balls)
            fc = balls[:, names["Fc_N"]]
            mg = balls[:, names["Mg_N_m"]]
            qi = balls[:, names["Qi_N"]]
            qo = balls[:, names["Qo_N"]]
            ai = balls[:, names["alpha_i_rad"]]
            ao = balls[:, names["alpha_o_rad"]]
            g = mg / float(params[tai.D])
            pi = qi * np.cos(ai) - g * np.sin(ai)
            ax_i = qi * np.sin(ai) + g * np.cos(ai)
            po = qo * np.cos(ao) - g * np.sin(ao)
            ax_o = qo * np.sin(ao) + g * np.cos(ao)
            per_ball_force = (po - pi) - fc
            per_ball_axial = ax_o - ax_i
            expected_force_sum = -np.array([np.sum(fc * np.cos(phases)),
                                            np.sum(fc * np.sin(phases))])
            actual_force_sum = inner[:2] + outer[:2]
            force_sum_error = float(np.max(np.abs(actual_force_sum - expected_force_sum)))
            balance_error = float(max(np.max(np.abs(per_ball_force)), np.max(np.abs(per_ball_axial))))
            axial_error = abs(float(inner[2] + outer[2]))
            max_balance_error = max(max_balance_error, balance_error)
            max_pair_force_error = max(max_pair_force_error, force_sum_error)
            max_axial_error = max(max_axial_error, axial_error)
            ring_rows.append({
                "bearing": name,
                "sum_Fc_N": float(np.sum(fc)),
                "centrifugal_resultant_N": float(np.linalg.norm(expected_force_sum)),
                "inner_plus_outer_radial_N": actual_force_sum.tolist(),
                "expected_minus_Fc_phase_sum_N": expected_force_sum.tolist(),
                "max_per_ball_Po_minus_Pi_minus_Fc_N": float(np.max(np.abs(per_ball_force))),
                "max_per_ball_Ao_minus_Ai_N": float(np.max(np.abs(per_ball_axial))),
                "outer_not_negative_inner_radial_sum_norm_N": float(np.linalg.norm(actual_force_sum)),
            })
        front_load_gap = min(r["centrifugal_resultant_N"] for r in ring_rows if r["bearing"].startswith("F"))
        f08_ok = (max_balance_error <= 1e-6 and max_pair_force_error <= 1e-5 and
                  max_axial_error <= 1e-5 and front_load_gap > 1e-6)
        _append(checks, "T09_INNER_OUTER_FC_FORCE_BALANCE", diagnostic_state,
                "passed" if f08_ok else "failed", {
                    "max_per_ball_force_or_axial_residual_N": max_balance_error,
                    "max_ring_resultant_difference_N": max_pair_force_error,
                    "max_inner_plus_outer_axial_residual_N": max_axial_error,
                    "front_minimum_Fc_resultant_N": front_load_gap,
                    "explicitly_not_asserting_outer_equals_minus_inner": True,
                    "per_bearing": ring_rows,
                })

        # Constraint reactions balance the actual outer-race contact load
        # (-outer restoring wrench), so the support reaction is +outer.
        components = tai_system.force_components(t_test, q_test, np.zeros(12), p)
        axial_reactions = components.get("outer_axial_constraint_N_by_bearing")
        moment_reactions = components.get("outer_rotation_constraint_moment_Nm_by_bearing")
        support_force = np.asarray(components["healthy_support_forces_N"])
        rear_expected = np.stack([-wrenches[n]["outer_restoring_6"][:2] for n in ("R1", "R2")])
        support_error = float(np.max(np.abs(support_force - rear_expected)))
        if axial_reactions is None or moment_reactions is None:
            reaction_ok = False
            reaction_metrics = {"missing_outer_axial_or_rotation_reaction_export": True,
                                "force_components_keys": sorted(components)}
        else:
            axial_error_by_name = {n: abs(float(axial_reactions[n]) -
                float(wrenches[n]["outer_axial_constraint_N"])) for n in tai_system.NAMES}
            moment_error_by_name = {n: float(np.max(np.abs(np.asarray(moment_reactions[n]) -
                np.asarray(wrenches[n]["outer_tilt_constraint_Nm"])))) for n in tai_system.NAMES}
            reaction_ok = (max(axial_error_by_name.values()) <= 1e-6 and
                           max(moment_error_by_name.values()) <= 1e-7 and support_error <= 1e-6)
            reaction_metrics = {"max_outer_axial_reaction_difference_N": max(axial_error_by_name.values()),
                                "max_outer_rotation_reaction_difference_Nm": max(moment_error_by_name.values()),
                                "rear_outer_radial_support_force_difference_N": support_error,
                                "outer_reaction_sign_convention": "support balances actual outer contact (-outer restoring), so reaction is +outer restoring",
                                "axial_reaction_N_by_bearing": {n: float(axial_reactions[n]) for n in tai_system.NAMES},
                                "rotation_reaction_Nm_by_bearing": {n: np.asarray(moment_reactions[n]).tolist()
                                                                    for n in tai_system.NAMES}}
        _append(checks, "T09_OUTER_RING_CONSTRAINT_REACTIONS", diagnostic_state,
                "passed" if reaction_ok else "failed", reaction_metrics)

        # T10: F16 damping is positive semidefinite and is an action/reaction
        # pair between the rings; it is not folded into Hertz ball loads.
        damping_rows = []
        for j, name in enumerate(tai_system.NAMES):
            row = args[4][j]
            params, old = args[5][j], args[6][j]
            z, endpoint, orientation, rho, cb, phase0, phase_factor = row
            end = int(endpoint)
            vx = v_test[0] + z * v_test[9] - v_test[end]
            vy = v_test[1] - z * v_test[8] - v_test[end + 1]
            tx, ty = v_test[8], v_test[9]
            lever = orientation * rho * math.tan(float(params[tai.ALPHA0]))
            xi_dot = np.array([vx, vy, tx, ty])
            dx = q_test[0] + z * q_test[9] - q_test[end]
            dy = q_test[1] - z * q_test[8] - q_test[end + 1]
            combined_inner, combined_outer, ball_rows, reported_damp = tai_system.ring_kernel(
                t_test, q_test, v_test, row, params, old, *args[7:])
            active = bool(np.any(ball_rows[:, tai.ACTIVE] == 2))
            damp_expected = np.asarray(tai_system.damping_wrench(vx, vy, tx, ty, cb, lever, active))
            damp4 = damp_expected[[0, 1, 3, 4]]
            power = float(xi_dot @ damp4)
            power_formula = float(cb * ((vx - lever * ty) ** 2 + (vy + lever * tx) ** 2)) if active else 0.0
            elastic_inner, elastic_outer, _ = tai.bearing_wrenches_kernel(
                t_test, dx, dy, q_test[8], q_test[9], params, orientation, args[7], phase0,
                args[8], args[9], args[10], phase_factor)
            action_error = float(max(np.max(np.abs((combined_inner - elastic_inner) - reported_damp)),
                                     np.max(np.abs((combined_outer - elastic_outer) + reported_damp)),
                                     np.max(np.abs((combined_inner - elastic_inner) +
                                                   (combined_outer - elastic_outer)))))
            damping_rows.append({"bearing": name, "cb_N_s_m": float(cb), "lever_m": float(lever),
                                 "active_dual_contact": active,
                                 "relative_velocity_4": xi_dot.tolist(), "damping_wrench_4": damp4.tolist(),
                                 "dissipated_power_W": power, "formula_power_W": power_formula,
                                 "damping_action_reaction_max_error": action_error})
        max_negative_power = max(-row["dissipated_power_W"] for row in damping_rows)
        max_action_error = max(row["damping_action_reaction_max_error"] for row in damping_rows)
        damping_ok = max_negative_power <= 1e-12 and max_action_error <= 1e-9 and all(
            abs(row["dissipated_power_W"] - row["formula_power_W"]) <= 1e-12 for row in damping_rows)
        _append(checks, "T10_F16_DAMPING_DISSIPATION", diagnostic_state,
                "passed" if damping_ok else "failed", {
                    "minimum_dissipated_power_W": min(row["dissipated_power_W"] for row in damping_rows),
                    "maximum_action_reaction_error": max_action_error,
                    "per_bearing": damping_rows,
                    "closure": "common structural ring damping, gated by at least one dual-contact ball",
                })

        # T10: independent mass/gravity ledger and exactly one rotor gyro block.
        masses, gravity = dynamics.mass_and_gravity(p)
        expected_masses = np.r_[np.repeat([p["ms"], p["mt"], p["mh"], p["mO"]], 2),
                                [p["Jt"], p["Jt"], p["mO2"], p["mO2"]]]
        gravity_axis = np.asarray(p["gravity_radial_m_s2"], dtype=float)
        expected_gravity = np.zeros(12)
        expected_gravity[:8] = expected_masses[:8] * np.tile(gravity_axis, 4)
        expected_gravity[10:12] = expected_masses[10:12] * gravity_axis
        expected_gravity += dynamics.generalized_point_force(
            [0.0, float(p.get("radial_load_y", 0.0))], float(p.get("radial_load_z_m", 0.0)), 12)
        gyro = dynamics.gyro_matrix(p)
        difference = np.zeros((2, 12))
        difference[:, dynamics.ROTOR] = dynamics.section_matrix(p["z_T"])
        difference[:, 2:4] -= np.eye(2)
        expected_linear_damping = p["cl"] * difference.T @ difference
        expected_linear_damping[4:6, 4:6] += p["ch"] * np.eye(2)
        gyro_once_error = float(np.max(np.abs(args[1] - expected_linear_damping - gyro)))
        gyro_skew_error = float(np.max(np.abs(gyro + gyro.T)))
        v_rotor = v_test[dynamics.ROTOR]
        gyro_power = float(v_rotor @ (gyro[np.ix_(dynamics.ROTOR, dynamics.ROTOR)] @ v_rotor))
        mass_error = float(np.max(np.abs(np.asarray(masses) - expected_masses)))
        gravity_error = float(np.max(np.abs(np.asarray(gravity) - expected_gravity)))
        node_mass_total = float(p["ms"] + p["mt"] + p["mh"] + p["mO"] + p["mO2"])
        expected_node_mass_total = float(cfg["parameter_provenance"]["mass_ledger"]["spindle_total_kg"] + p["mt"])
        gyro_ok = (mass_error <= 1e-14 and gravity_error <= 1e-12 and gyro_once_error <= 1e-12 and
                   gyro_skew_error <= 1e-14 and abs(gyro_power) <= 1e-12 and
                   abs(node_mass_total - expected_node_mass_total) <= 1e-12)
        _append(checks, "T10_MASS_GRAVITY_GYRO_ONCE", diagnostic_state,
                "passed" if gyro_ok else "failed", {
                    "mass_diagonal_kg_or_kg_m2": np.asarray(masses).tolist(),
                    "expected_mass_diagonal_kg_or_kg_m2": expected_masses.tolist(),
                    "max_mass_diagonal_difference": mass_error,
                    "translational_node_mass_total_kg": node_mass_total,
                    "expected_translational_node_mass_total_kg": expected_node_mass_total,
                    "max_gravity_difference_N": gravity_error,
                    "gyro_matrix_skew_error": gyro_skew_error,
                    "gyro_power_vGv": gyro_power,
                    "gyro_added_to_linear_damping_once_max_error": gyro_once_error,
                    "gyro_nonzero_entries": np.argwhere(np.abs(gyro) > 0).tolist(),
                })

        # T11: full per-ball reporting assembly vs the Numba fast RHS kernel,
        # then independently compare the 24-component instantaneous RHS.
        full_reference, _ = _full_wrench_reference_assembly(t_test, q_test, v_test, p,
                                                             tai_system, dynamics)
        python_api = tai_system.internal_force(t_test, q_test, v_test, p)
        numba_internal = tai_system.internal_kernel(t_test, q_test, v_test, *args)
        assembly_errors = {
            "full_reporting_vs_python_api": _force_comparison(full_reference, python_api),
            "full_reporting_vs_numba_fast": _force_comparison(full_reference, numba_internal),
            "python_api_vs_numba_fast": _force_comparison(python_api, numba_internal),
        }
        assembly_ok = all(item["force_pass"] and item["moment_pass"]
                          for item in assembly_errors.values())
        _append(checks, "T11_SYSTEM_FORCE_12_COMPONENTS", diagnostic_state,
                "passed" if assembly_ok else "failed", assembly_errors)

        unbalance = (float(drive_cfg["U_kg_m"]), float(drive_cfg["phase_rad"]))
        omega = float(p["rpm"]) * 2 * math.pi / 60
        angle = omega * t_test + unbalance[1]
        input_force = dynamics.generalized_point_force(
            unbalance[0] * omega**2 * np.array([math.cos(angle), math.sin(angle)]), p["z_u"], 12)
        expected_rhs = np.r_[v_test, (gravity + input_force - full_reference) / masses]
        compiled_rhs = prepare_fast_rhs(dict(p, rhs_backend="numba"), unbalance)
        actual_rhs = compiled_rhs(t_test, np.r_[q_test, v_test])
        rhs_diff = np.abs(expected_rhs - actual_rhs)
        rhs_pass = (np.max(rhs_diff[:12]) <= 1e-12 and
                    np.max(rhs_diff[np.r_[12:20, 22:24]]) <= 1e-6 and
                    np.max(rhs_diff[20:22]) <= 1e-5)
        _append(checks, "T11_SYSTEM_RHS_24_COMPONENTS", diagnostic_state,
                "passed" if rhs_pass else "failed", {
                    "max_position_rate_difference_m_s": float(np.max(rhs_diff[:12])),
                    "max_translation_acceleration_difference_m_s2": float(np.max(rhs_diff[np.r_[12:20, 22:24]])),
                    "max_rotor_angular_acceleration_difference_rad_s2": float(np.max(rhs_diff[20:22])),
                    "reference_uses_full_per_ball_wrenches": True,
                })
    except Exception as exc:
        failure = _error_record(exc)
        _append(checks, "T07_T09_T10_T11_SYSTEM_ASSEMBLY", diagnostic_state,
                "failed", failure, "assembly checks aborted at first local/system error")

    # Configuration guard: V9 must be explicit and invalid selectors/layouts
    # must fail rather than silently selecting another physical branch.
    try:
        layout_names = [item["name"] for item in p["bearing_layout"]]
        family_vectors = p["tai_bearing"]["families"]
        cfg_ok = (p.get("bearing_contact_model") == "quasistatic_tai_v9" and
                  p.get("damping_policy") == "common_structural_ring" and
                  int(p.get("model_dof", 0)) == 12 and p.get("bearing_layout_name") == "DB_DB" and
                  set(layout_names) == {"F1", "F2", "R1", "R2"} and
                  p["tai_bearing"]["ball_inertia_mode"] == "full" and
                  p["tai_bearing"]["control_branch"] in ("outer", "inner") and
                  all(len(family_vectors[name]["initialized_parameter_vector"]) == 13
                      for name in ("front", "rear")) and
                  p["tai_bearing"].get("failed_local_solve") == "stop_and_save_state_no_fallback")
        bad_selector_rejected = False
        try:
            nominal = np.asarray(family_vectors["front"]["initialized_parameter_vector"], dtype=float)
            tai.solve_ball(*_geometry_for({"fi": nominal[tai.FI], "fo": nominal[tai.FO],
                                           "D_m": nominal[tai.D]}, nominal),
                           0.0, nominal, mode="N2")
        except (KeyError, ValueError):
            bad_selector_rejected = True
        bad_layout_rejected = False
        try:
            malformed = copy.deepcopy(p)
            malformed["bearing_layout_name"] = "single_legacy"
            tai_system.prepare(malformed)
        except ValueError:
            bad_layout_rejected = True
        guard_ok = cfg_ok and bad_selector_rejected and bad_layout_rejected
        _append(checks, "T12_EXPLICIT_CONFIG_AND_NO_FALLBACK", "N3-config",
                "passed" if guard_ok else "failed", {
                    "explicit_v9_configuration_valid": cfg_ok,
                    "bad_local_mode_rejected": bad_selector_rejected,
                    "bad_layout_rejected": bad_layout_rejected,
                    "layout_names": layout_names,
                    "inertia_mode": p["tai_bearing"].get("ball_inertia_mode"),
                    "control_branch": p["tai_bearing"].get("control_branch"),
                })
    except Exception as exc:
        _append(checks, "T12_EXPLICIT_CONFIG_AND_NO_FALLBACK", "N3-config", "failed", _error_record(exc))

    # Two-turn short trajectory: same equilibrium, force, and solver settings;
    # only the backend selector changes. Save all sampled q/v/a on success.
    trajectory_data = None
    if equilibrium_ok and q_equilibrium is not None:
        duration = 2.0 / abs(float(p["rpm"]) / 60.0)
        backend_results: dict[str, dict[str, Any]] = {}
        backend_failures = {}
        for backend in ("python", "numba"):
            local = copy.deepcopy(p)
            local["rhs_backend"] = backend
            try:
                backend_results[backend] = dynamics.simulate(
                    local, 0.0, np.empty(0), np.empty((0, 2)), duration=duration,
                    warmup=0.0, fs=8192, max_step=float(sim_cfg["max_step_s"]),
                    rtol=float(sim_cfg["rtol"]), unbalance=unbalance,
                    q_initial=q_equilibrium, method=str(sim_cfg.get("method", "DOP853")))
            except Exception as exc:
                backend_failures[backend] = _error_record(exc)
        if backend_failures:
            _append(checks, "T11_SYSTEM_SHORT_TRAJECTORY", "two-turn-backend-replay", "failed", {
                "duration_s": duration, "backend_failures": backend_failures,
                "configuration": cfg,
            }, "short trajectory failed; full failure state/configuration retained")
        else:
            py, nb = backend_results["python"], backend_results["numba"]
            same_time = np.array_equal(py["t"], nb["t"])
            qdiff = np.max(np.abs(py["q"] - nb["q"]), axis=0)
            vdiff = np.max(np.abs(py["v"] - nb["v"]), axis=0)
            adiff = np.max(np.abs(py["acc"] - nb["acc"]), axis=0)
            position_um = float(max(np.max(qdiff[np.r_[0:8, 10:12]]) * 1e6,
                                    np.max(qdiff[8:10]) * float(p["L_ref"]) * 1e6))
            velocity_mm_s = float(max(np.max(vdiff[np.r_[0:8, 10:12]]) * 1e3,
                                      np.max(vdiff[8:10]) * float(p["L_ref"]) * 1e3))
            acceleration_translation = float(np.max(adiff[np.r_[0:8, 10:12]]))
            acceleration_equivalent = float(np.max(adiff[8:10]) * float(p["L_ref"]))
            acceleration_max = max(acceleration_translation, acceleration_equivalent)
            trajectory_ok = (same_time and py["q"].shape == nb["q"].shape and
                             position_um <= float(cfg["numerical_thresholds"]["position_um"]) and
                             velocity_mm_s <= float(cfg["numerical_thresholds"]["velocity_mm_s"]) and
                             acceleration_max <= float(cfg["numerical_thresholds"]["acceleration_m_s2"]))
            _append(checks, "T11_SYSTEM_SHORT_TRAJECTORY", "two-turn-backend-replay",
                    "passed" if trajectory_ok else "failed", {
                        "duration_s": duration,
                        "revolutions": duration * abs(float(p["rpm"])) / 60.0,
                        "samples": len(py["t"]), "same_time_grid": same_time,
                        "position_difference_um_including_rotor_tilt_lever": position_um,
                        "velocity_difference_mm_s_including_rotor_tilt_lever": velocity_mm_s,
                        "translation_acceleration_difference_max_m_s2": acceleration_translation,
                        "rotor_angular_acceleration_lever_equivalent_m_s2": acceleration_equivalent,
                        "acceleration_difference_max_equivalent_m_s2": acceleration_max,
                        "python_equilibrium_residual_N": float(py["equilibrium_residual_N"]),
                        "numba_equilibrium_residual_N": float(nb["equilibrium_residual_N"]),
                        "python_nfev": int(py["nfev"]), "numba_nfev": int(nb["nfev"]),
                        "acceptance": {
                            "position_um": float(cfg["numerical_thresholds"]["position_um"]),
                            "velocity_mm_s": float(cfg["numerical_thresholds"]["velocity_mm_s"]),
                            "acceleration_m_s2": float(cfg["numerical_thresholds"]["acceleration_m_s2"]),
                        },
                    })
            trajectory_data = {"time_s": py["t"].tolist(),
                               "python_q": py["q"].tolist(), "python_v": py["v"].tolist(),
                               "python_acceleration": py["acc"].tolist(),
                               "numba_q": nb["q"].tolist(), "numba_v": nb["v"].tolist(),
                               "numba_acceleration": nb["acc"].tolist()}
    else:
        reason = "target-gap static equilibrium did not pass; trajectory intentionally not started"
        _append(checks, "T11_SYSTEM_SHORT_TRAJECTORY", "two-turn-backend-replay", "failed",
                {"duration_s": 2.0 / abs(float(p["rpm"]) / 60.0),
                 "static_equilibrium_passed": False,
                 "static_equilibrium_attempts": system["static_equilibrium_attempts"],
                 "configuration": cfg}, reason)

    system["short_trajectory"] = trajectory_data
    statuses = [item["status"] for item in checks]
    system["status"] = "failed" if "failed" in statuses else "numerical_pass"
    system["summary"] = {"checks_total": len(checks), "passed": statuses.count("passed"),
                         "conditional": statuses.count("conditional"), "failed": statuses.count("failed"),
                         "static_equilibrium_passed": equilibrium_ok,
                         "trajectory_revolutions": 2 if trajectory_data is not None else 0,
                         "machine_validated": False}
    return system


def _family_at_curvature(family: dict[str, Any], f_i: float, f_o: float) -> dict[str, Any]:
    changed = copy.deepcopy(family)
    changed["fi"], changed["fo"] = f_i, f_o
    return changed


def _geometry_for(family: dict[str, Any], params: np.ndarray) -> tuple[float, float]:
    B0 = (float(family["fi"]) + float(family["fo"]) - 1.0) * float(family["D_m"])
    alpha0 = float(params[tai_bearing.ALPHA0])
    return B0 * math.sin(alpha0) + float(params[tai_bearing.A_PRE]), B0 * math.cos(alpha0)


def _force_tolerance(params: np.ndarray, fc: float = 0.0) -> float:
    f_star = max(float(params[tai_bearing.P_REF]) /
                 (float(params[tai_bearing.Z]) * math.sin(float(params[tai_bearing.ALPHA0]))),
                 abs(fc), 1.0)
    return max(1e-6, 1e-7 * f_star)


def _check_ball_root(checks: list[dict[str, Any]], case_id: str, result: dict[str, Any],
                     params: np.ndarray, *, oracle: dict[str, float] | None = None) -> bool:
    """Check returned values against original-unit geometry and force limits."""
    D = float(params[tai_bearing.D])
    B0 = (float(params[tai_bearing.FI]) + float(params[tai_bearing.FO]) - 1.0) * D
    ai0, ao0 = (float(params[tai_bearing.FI]) - 0.5) * D, (float(params[tai_bearing.FO]) - 0.5) * D
    delta_i, delta_o = float(result["delta_i_m"]), float(result["delta_o_m"])
    Li, Lo = ai0 + delta_i, ao0 + delta_o
    resid = np.asarray(result["residuals"], dtype=float)
    g_norm = float(np.max(np.abs(resid[:2])) / (B0 * B0))
    force_residual = float(np.max(np.abs(resid[2:])))
    force_tol = _force_tolerance(params, float(result["Fc_N"]))
    radii_ok = Li > 0.0 and Lo > 0.0
    dual = result["active_set"] == "dual"
    if dual:
        geom_ok = g_norm <= 1e-9
        force_ok = force_residual <= force_tol
        A_m, B_m = result["state_AB_m"]
        si = (A_m - float(result["U_m"])) / Li
        ci = (B_m - float(result["V_m"])) / Li
        so = float(result["U_m"]) / Lo
        co = float(result["V_m"]) / Lo
        angle_residual = max(abs(si - math.sin(float(result["alpha_i_rad"]))),
                             abs(ci - math.cos(float(result["alpha_i_rad"]))),
                             abs(so - math.sin(float(result["alpha_o_rad"]))),
                             abs(co - math.cos(float(result["alpha_o_rad"]))))
        angle_ok = angle_residual <= 1e-8
        active_ok = delta_i >= -1e-15 and delta_o >= -1e-15 and result["Qi_N"] >= 0 and result["Qo_N"] >= 0
    elif result["active_set"] == "outer_only":
        # Inner contact is inactive in F10; only the outer equality and the
        # inactive inner-clearance inequality are physically required.
        A_m, B_m = result["state_AB_m"]
        inner_gap_sq = A_m * A_m + (B_m - ao0 - delta_o) ** 2
        inner_limit_sq = ai0 * ai0
        f10 = math.sqrt(max(inner_gap_sq, 0.0)) - ai0
        geom_ok = inner_gap_sq <= inner_limit_sq + 1e-14 * B0 * B0
        force_ok = (abs(float(result["Qi_N"])) <= force_tol and
                    abs(float(result["Qo_N"]) - float(result["Fc_N"])) <= force_tol and
                    abs(float(result["Mg_N_m"])) <= 1e-14)
        angle_residual = max(abs(float(result["U_m"])), abs(float(result["alpha_o_rad"])))
        angle_ok = angle_residual <= 1e-12
        active_ok = result["Qi_N"] == 0.0 and result["Qo_N"] >= 0.0 and result["Fc_N"] >= 0.0
        g_norm = abs(f10) / B0
    else:
        geom_ok = abs(float(result["Qi_N"])) <= force_tol and abs(float(result["Qo_N"])) <= force_tol
        force_ok = geom_ok
        angle_residual = 0.0
        angle_ok = True
        active_ok = result["Qi_N"] == 0.0 and result["Qo_N"] == 0.0

    numeric_ok = bool(geom_ok and force_ok and radii_ok and angle_ok and active_ok)
    metrics: dict[str, Any] = {
        "active_set": result["active_set"],
        "normalized_geometry_residual": g_norm,
        "geometry_tolerance": 1e-9,
        "force_residual_N": force_residual,
        "force_tolerance_N": force_tol,
        "Li_m": Li,
        "Lo_m": Lo,
        "angle_reconstruction_error": angle_residual,
        "radii_positive": radii_ok,
        "active_set_valid": active_ok,
    }
    if result["active_set"] == "outer_only":
        metrics.update({"inactive_inner_gap_residual_m": f10,
                        "Qi_N": result["Qi_N"], "Qo_N": result["Qo_N"],
                        "Fc_N": result["Fc_N"], "Mg_N_m": result["Mg_N_m"]})
    if oracle:
        comparisons = {}
        for field in ("U_m", "V_m", "delta_i_m", "delta_o_m", "alpha_i_rad", "alpha_o_rad", "Qi_N", "Qo_N"):
            actual = float(result[field])
            expected = float(oracle[field])
            scale = max(abs(expected), 1e-30)
            comparisons[field] = {"actual": actual, "oracle": expected,
                                  "relative_difference": abs(actual - expected) / scale}
        oracle_q = max(comparisons["Qi_N"]["relative_difference"],
                       comparisons["Qo_N"]["relative_difference"])
        oracle_d = max(comparisons["delta_i_m"]["relative_difference"],
                       comparisons["delta_o_m"]["relative_difference"])
        metrics["oracle_comparisons"] = comparisons
        metrics["oracle_max_relative_load_difference"] = oracle_q
        metrics["oracle_max_relative_compression_difference"] = oracle_d
        numeric_ok = numeric_ok and oracle_q <= 1e-6 and oracle_d <= 1e-6
    _append(checks, "T03_T05_LOCAL_ROOT", case_id, "passed" if numeric_ok else "failed", metrics)
    return numeric_ok


def _recompute_control_margins(result: dict[str, Any], params: np.ndarray) -> tuple[float, float]:
    from spindle import tai_bearing

    ei = tai_bearing.hertz_contact_properties(result["alpha_i_rad"], params, True,
                                               max(result["Qi_N"], 0.0))["ellipse_E"]
    eo = tai_bearing.hertz_contact_properties(result["alpha_o_rad"], params, False,
                                               max(result["Qo_N"], 0.0))["ellipse_E"]
    ai = tai_bearing.hertz_contact_properties(result["alpha_i_rad"], params, True,
                                               max(result["Qi_N"], 0.0))["semi_major_m"]
    ao = tai_bearing.hertz_contact_properties(result["alpha_o_rad"], params, False,
                                               max(result["Qo_N"], 0.0))["semi_major_m"]
    lhs_i = result["Qi_N"] * ai * ei
    lhs_o = result["Qo_N"] * ao * eo
    cosdiff = math.cos(result["alpha_i_rad"] - result["alpha_o_rad"])
    scale = max(lhs_i, lhs_o, 1e-30)
    return (lhs_o * cosdiff - lhs_i) / scale, (lhs_i * cosdiff - lhs_o) / scale


def _python_wrench_from_ball_rows(t: float, xy: tuple[float, float], tilt: tuple[float, float],
                                  params: np.ndarray, orientation: int, omega_global: float,
                                  phase0: float, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent Python implementation of the F11/F12 local wrench sums."""
    D, Dm = float(params[tai_bearing.D]), float(params[tai_bearing.DM])
    fi, fo = float(params[tai_bearing.FI]), float(params[tai_bearing.FO])
    alpha0 = float(params[tai_bearing.ALPHA0])
    Z = int(params[tai_bearing.Z])
    names = {name: index for index, name in enumerate(tai_bearing.BALL_NAMES)}
    ric = Dm / 2 + (fi - 0.5) * D * math.cos(alpha0)
    roc = Dm / 2 - (fo - 0.5) * D * math.cos(alpha0)
    inner = np.zeros(6)
    outer = np.zeros(6)
    for k, row in enumerate(rows):
        phase_global = phase0 + omega_global * float(params[tai_bearing.CAGE_RATIO]) * t + 2 * math.pi * k / Z
        phase = orientation * phase_global
        sn, cs = math.sin(phase), math.cos(phase)
        value = lambda name: float(row[names[name]])
        g = value("Mg_N_m") / D
        qi, qo = value("Qi_N"), value("Qo_N")
        ai, ao = value("alpha_i_rad"), value("alpha_o_rad")
        pi = qi * math.cos(ai) - g * math.sin(ai)
        ax_i = qi * math.sin(ai) + g * math.cos(ai)
        po = qo * math.cos(ao) - g * math.sin(ao)
        ax_o = qo * math.sin(ao) + g * math.cos(ao)
        mi = ric * ax_i - fi * value("Mg_N_m")
        mo = roc * ax_o + fo * value("Mg_N_m")
        inner += np.array([pi * cs, orientation * pi * sn, orientation * ax_i,
                           mi * sn, -orientation * mi * cs, 0.0])
        outer += np.array([-po * cs, -orientation * po * sn, -orientation * ax_o,
                           -mo * sn, orientation * mo * cs, 0.0])
    return inner, outer


def _backend_compare(checks: list[dict[str, Any]], tai_bearing: Any, family: dict[str, Any],
                     materials: dict[str, Any], params: np.ndarray, case_id: str) -> None:
    D = float(params[tai_bearing.D])
    B0 = (float(params[tai_bearing.FI]) + float(params[tai_bearing.FO]) - 1) * D
    alpha0, a_pre = float(params[tai_bearing.ALPHA0]), float(params[tai_bearing.A_PRE])
    # Keep a non-axisymmetric state so both radial directions and both local moments matter.
    xy = (0.35e-6, -0.22e-6)
    tilt = (1.7e-6, -1.1e-6)
    t = 7.31e-4
    omega_global = -7000 * 2 * math.pi / 60
    phase0 = 0.0
    outputs = {}
    for orientation in (-1, 1):
        # The local state is identical after applying the plan's proper rotation.
        local_xy = (xy[0], orientation * xy[1])
        local_tilt = (tilt[0], orientation * tilt[1])
        omega = orientation * omega_global
        try:
            out = tai_bearing.bearing_wrenches(t, local_xy, local_tilt, params, orientation,
                                               omega, phase0, mode="full", control="outer")
            inner_ref, outer_ref = _python_wrench_from_ball_rows(
                t, local_xy, local_tilt, params, orientation, omega, phase0, np.asarray(out["balls"])
            )
        except Exception as exc:
            _append(checks, "T01_T11_PYTHON_NUMBA_WRENCH", f"{case_id}:orientation={orientation:+d}",
                    "failed", message=f"{type(exc).__name__}: {exc}")
            return
        inner, outer = np.asarray(out["inner_global"]), np.asarray(out["outer_global"])
        inner_force_abs = float(np.max(np.abs(inner[:3] - inner_ref[:3])))
        outer_force_abs = float(np.max(np.abs(outer[:3] - outer_ref[:3])))
        inner_moment_abs = float(np.max(np.abs(inner[3:] - inner_ref[3:])))
        outer_moment_abs = float(np.max(np.abs(outer[3:] - outer_ref[3:])))
        inner_force_rel = inner_force_abs / max(float(np.max(np.abs(inner[:3]))), float(np.max(np.abs(inner_ref[:3]))), 1e-12)
        outer_force_rel = outer_force_abs / max(float(np.max(np.abs(outer[:3]))), float(np.max(np.abs(outer_ref[:3]))), 1e-12)
        inner_moment_rel = inner_moment_abs / max(float(np.max(np.abs(inner[3:]))), float(np.max(np.abs(inner_ref[3:]))), 1e-15)
        outer_moment_rel = outer_moment_abs / max(float(np.max(np.abs(outer[3:]))), float(np.max(np.abs(outer_ref[3:]))), 1e-15)
        passed = max(inner_force_rel, outer_force_rel, inner_moment_rel, outer_moment_rel) <= 1e-7
        _append(checks, "T01_T11_PYTHON_NUMBA_WRENCH", f"{case_id}:orientation={orientation:+d}",
                "passed" if passed else "failed", {
                    "inner_force_relative_difference": inner_force_rel,
                    "outer_force_relative_difference": outer_force_rel,
                    "inner_moment_relative_difference": inner_moment_rel,
                    "outer_moment_relative_difference": outer_moment_rel,
                    "inner_force_absolute_difference_N": inner_force_abs,
                    "outer_force_absolute_difference_N": outer_force_abs,
                    "inner_moment_absolute_difference_Nm": inner_moment_abs,
                    "outer_moment_absolute_difference_Nm": outer_moment_abs,
                    "phase_speed_relative_difference_max": out["max_phase_speed_relative_difference"],
                    "preload_offset_m": a_pre,
                    "nominal_angle_rad": alpha0,
                })
        try:
            fast_inner, fast_outer, dual_count, max_residual, max_iterations = tai_bearing.bearing_wrenches_fast_kernel(
                t, *local_xy, *local_tilt, params, orientation, omega, phase0,
                tai_bearing.MODE_CODES["full"], tai_bearing.CONTROL_CODES["outer"])
            fast_wrench_diff = max(float(np.max(np.abs(np.asarray(fast_inner) - inner))),
                                   float(np.max(np.abs(np.asarray(fast_outer) - outer))))
            fast_wrench_scale = max(float(np.max(np.abs(inner))), float(np.max(np.abs(outer))), 1e-12)
            fast_ok = (fast_wrench_diff / fast_wrench_scale <= 1e-7 and
                       int(dual_count) == int(np.sum(np.asarray(out["balls"])[:, tai_bearing.ACTIVE] == 2)) and
                       abs(float(max_residual) - float(out["max_force_residual_N"])) <= 1e-12 and
                       int(max_iterations) == int(np.max(np.asarray(out["balls"])[:, tai_bearing.ITERATIONS])))
            _append(checks, "T11_NUMBA_FAST_FULL_EQUIVALENCE", f"{case_id}:orientation={orientation:+d}",
                    "passed" if fast_ok else "failed", {
                        "wrench_relative_difference": fast_wrench_diff / fast_wrench_scale,
                        "dual_count_fast": int(dual_count),
                        "dual_count_full": int(np.sum(np.asarray(out["balls"])[:, tai_bearing.ACTIVE] == 2)),
                        "max_force_residual_fast_N": float(max_residual),
                        "max_force_residual_full_N": float(out["max_force_residual_N"]),
                        "max_iterations_fast": int(max_iterations),
                        "max_iterations_full": int(np.max(np.asarray(out["balls"])[:, tai_bearing.ITERATIONS])),
                    })
        except Exception as exc:
            _append(checks, "T11_NUMBA_FAST_FULL_EQUIVALENCE", f"{case_id}:orientation={orientation:+d}",
                    "failed", message=f"{type(exc).__name__}: {exc}")
        outputs[orientation] = (inner, outer)
    # Same local state with s=±1 must map back through T_s^T consistently.
    mapping_errors = []
    for plus, minus in zip(outputs[1], outputs[-1]):
        mapping_errors.extend((abs(plus[0] - minus[0]), abs(plus[1] + minus[1]),
                               abs(plus[2] + minus[2]), abs(plus[3] - minus[3]),
                               abs(plus[4] + minus[4]), abs(plus[5] - minus[5])))
    scale = max(max(float(np.max(np.abs(x))) for pair in outputs.values() for x in pair), 1e-12)
    map_rel = max(mapping_errors) / scale
    _append(checks, "T01_ORIENTATION_WRENCH_MIRROR", case_id,
            "passed" if map_rel <= 1e-7 else "failed", {"max_mapping_relative_difference": map_rel,
                                                         "max_mapping_absolute_difference": max(mapping_errors),
                                                         "tolerance": 1e-7})


def _run_validation(contract_path: Path, output_dir: Path,
                    system_config_path: Path = DEFAULT_SYSTEM_CONFIG) -> dict[str, Any]:
    global tai_bearing
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": "spindle_v9_bearing_and_system_validation_v2",
        "status": "running",
        "contract_path": str(contract_path.resolve()),
        "configuration": contract,
        "runtime": {"python_executable": sys.executable, "python_version": sys.version.split()[0],
                    "entrypoint": "python -m utils.clearance_v9_validation",
                    "required_pythonpath": str(ROOT / "src")},
        "scope": "independent bearing algebra/local sensitivity plus 12-DOF force assembly and short integration; excludes long experiment matrix and machine validation",
        "thresholds": {
            "normalized_geometry_residual_max": 1e-9,
            "force_residual_N": "max(1e-6, 1e-7*F_star)",
            "preload_residual_N": "max(1e-6, 1e-7*P_ref)",
            "zero_inertia_load_and_compression_relative_difference": 1e-6,
            "python_numba_force_and_moment_relative_difference": 1e-7,
            "control_self_consistency": "positive Table 2 margin for the branch being reported",
        },
        "checks": checks,
        "families": {},
        "limitations": [
            "fi/fo=0.52/0.54, preload and materials are reference assumptions, not GMN-specific measured geometry or in-machine preload.",
            "The prescribed cage-phase and structural-damping closures remain conditional physics.",
            "Passing this report does not validate the assembled spindle or machine measurements.",
        ],
    }

    try:
        tai = importlib.import_module("spindle.tai_bearing")
    except Exception as exc:
        result["status"] = "blocked"
        result["blocker"] = f"cannot import spindle.tai_bearing: {type(exc).__name__}: {exc}"
        return result
    tai_bearing = tai

    # T01: explicit proper DB coordinate rotations and reversibility.
    max_rotation_error, max_det_error = 0.0, 0.0
    for s in (-1, 1):
        T = np.array([[0.0, 0.0, float(s)], [1.0, 0.0, 0.0], [0.0, float(s), 0.0]])
        max_rotation_error = max(max_rotation_error, float(np.max(np.abs(T @ T.T - np.eye(3)))))
        max_rotation_error = max(max_rotation_error, float(np.max(np.abs(T.T @ (T @ np.array([1.2, -3.4, 5.6])) - np.array([1.2, -3.4, 5.6])))))
        max_det_error = max(max_det_error, abs(float(np.linalg.det(T)) - 1.0))
    coord_ok = max_rotation_error <= 1e-14 and max_det_error <= 1e-14
    _append(checks, "T01_COORDINATE_ROTATION", "both-orientations", "passed" if coord_ok else "failed", {
        "max_roundtrip_or_orthogonality_error": max_rotation_error, "max_determinant_error": max_det_error})

    materials = contract["nominal_materials"]
    speeds = (0.0, 3000.0, 7000.0, 10000.0)
    for family_name, family in contract["families"].items():
        family_out: dict[str, Any] = {"nominal": {}, "curvature_sensitivity": []}
        try:
            params = tai.make_bearing_parameters(family, materials)
            oracle_preload = _preload_oracle(family, materials)
            family_out["nominal"]["parameters"] = dict(zip(tai.PARAMETER_NAMES, map(float, params)))
            family_out["nominal"]["independent_preload_oracle"] = oracle_preload
            da = float(params[tai.A_PRE]) - float(oracle_preload["a_pre_m"])
            expected_preload_tol = max(1e-6, 1e-7 * float(family["preload_reference_at_rest_N"]))
            pforce_error = abs(float(oracle_preload["axial_load_N"]) - float(family["preload_reference_at_rest_N"]))
            apre_ok = abs(da) <= 1e-9 and pforce_error <= expected_preload_tol
            _append(checks, "T04_FIXED_POSITION_PRELOAD", family_name,
                    "passed" if apre_ok else "failed", {
                        "a_pre_production_m": float(params[tai.A_PRE]),
                        "a_pre_independent_m": oracle_preload["a_pre_m"],
                        "a_pre_difference_m": da,
                        "preload_force_N": oracle_preload["axial_load_N"],
                        "target_force_N": family["preload_reference_at_rest_N"],
                        "preload_residual_N": pforce_error,
                        "preload_tolerance_N": expected_preload_tol,
                    })

            A, B = _geometry_for(family, params)
            expected = _zero_speed_oracle(A, B, family, materials)
            zero = tai.solve_ball(A, B, 0.0, params, mode="none", control="outer")
            zero["state_AB_m"] = (A, B)
            _check_ball_root(checks, f"{family_name}:zero-speed-preload", zero, params, oracle=expected)
            preload_actual = float(params[tai.Z]) * float(zero["Qi_N"]) * math.sin(float(zero["alpha_i_rad"]))
            preload_error = abs(preload_actual - float(family["preload_reference_at_rest_N"]))
            _append(checks, "T03_T04_ZERO_SPEED_PRELOAD", family_name,
                    "passed" if preload_error <= expected_preload_tol else "failed", {
                        "axial_preload_N": preload_actual,
                        "target_N": family["preload_reference_at_rest_N"],
                        "residual_N": preload_error,
                        "tolerance_N": expected_preload_tol,
                        "Qi_N": zero["Qi_N"], "Qo_N": zero["Qo_N"],
                        "alpha_i_rad": zero["alpha_i_rad"], "alpha_o_rad": zero["alpha_o_rad"],
                    })

            # F04 coefficient, ellipse semi-axes and pressure are checked against
            # a separately coded material/curvature calculation.
            hertz_cases = []
            for inner in (True, False):
                angle = math.radians(17.0)
                load = 150.0
                actual = tai.hertz_contact_properties(angle, params, inner, load)
                ref = _hertz_oracle(angle, family, materials, inner=inner)
                exp_a = (6 * ref["kappa"]**2 * ref["ellipse_E"] * load * ref["R_m"] /
                         (math.pi * ref["Eprime_Pa"])) ** (1 / 3)
                exp_b = (6 * ref["ellipse_E"] * load * ref["R_m"] /
                         (math.pi * ref["kappa"] * ref["Eprime_Pa"])) ** (1 / 3)
                exp_p = 3 * load / (2 * math.pi * exp_a * exp_b)
                errs = {
                    "K_relative": abs(actual["K_N_m32"] - ref["K_N_m32"]) / ref["K_N_m32"],
                    "semi_major_relative": abs(actual["semi_major_m"] - exp_a) / exp_a,
                    "semi_minor_relative": abs(actual["semi_minor_m"] - exp_b) / exp_b,
                    "pressure_relative": abs(actual["p0_Pa"] - exp_p) / exp_p,
                }
                passed = max(errs.values()) <= 1e-12
                hertz_cases.append({"contact": "inner" if inner else "outer", "actual": actual,
                                    "oracle": ref, "expected_ellipse": {"a_m": exp_a, "b_m": exp_b, "p0_Pa": exp_p},
                                    "relative_errors": errs, "status": "passed" if passed else "failed"})
                _append(checks, "T02_HERTZ_AND_ELLIPSE", f"{family_name}:{'inner' if inner else 'outer'}",
                        "passed" if passed else "failed", {"relative_errors": errs})
                zero_load = tai.hertz_contact_properties(angle, params, inner, 0.0)
                zero_load_ok = (zero_load["semi_major_m"] == 0.0 and
                                zero_load["semi_minor_m"] == 0.0 and zero_load["p0_Pa"] == 0.0 and
                                all(math.isfinite(float(zero_load[k])) for k in ("K_N_m32", "kappa", "p0_Pa")))
                _append(checks, "T02_ZERO_LOAD_ELLIPSE", f"{family_name}:{'inner' if inner else 'outer'}",
                        "passed" if zero_load_ok else "failed", {
                            "semi_major_m": zero_load["semi_major_m"],
                            "semi_minor_m": zero_load["semi_minor_m"], "p0_Pa": zero_load["p0_Pa"]})
            family_out["nominal"]["hertz_ellipse"] = hertz_cases

            # Exact circular Hertz limit and principal-radius ordering using
            # the elliptic-integral reference API, independent of the F04 fit.
            ball_radius = float(params[tai.D]) / 2.0
            circle_load = 75.0
            exact_circle = tai.hertz_principal_radii(ball_radius, ball_radius,
                                                     float(params[tai.EPRIME]), circle_load)
            radius_pair_a = tai.hertz_principal_radii(0.65 * ball_radius, 1.4 * ball_radius,
                                                      float(params[tai.EPRIME]), circle_load)
            radius_pair_b = tai.hertz_principal_radii(1.4 * ball_radius, 0.65 * ball_radius,
                                                      float(params[tai.EPRIME]), circle_load)
            exact_a = (3.0 * circle_load * ball_radius / (2.0 * float(params[tai.EPRIME]))) ** (1.0 / 3.0)
            exact_k = (2.0 / 3.0) * float(params[tai.EPRIME]) * math.sqrt(ball_radius)
            exact_p0 = 3.0 * circle_load / (2.0 * math.pi * exact_a * exact_a)
            circle_errors = {
                "K_relative": abs(exact_circle["K_N_m32"] - exact_k) / exact_k,
                "semi_major_relative": abs(exact_circle["semi_major_m"] - exact_a) / exact_a,
                "semi_minor_relative": abs(exact_circle["semi_minor_m"] - exact_a) / exact_a,
                "pressure_relative": abs(exact_circle["p0_Pa"] - exact_p0) / exact_p0,
                "radius_swap_max_abs": max(abs(radius_pair_a[k] - radius_pair_b[k])
                                            for k in ("K_N_m32", "semi_major_m", "semi_minor_m", "p0_Pa")),
            }
            circle_ok = max(circle_errors.values()) <= 1e-12
            family_out["nominal"]["exact_circle_limit"] = {
                "radius_m": ball_radius, "load_N": circle_load, "production_reference": exact_circle,
                "independent_closed_form": {"K_N_m32": exact_k, "semi_axis_m": exact_a, "p0_Pa": exact_p0},
                "relative_errors": circle_errors,
            }
            _append(checks, "T02_EXACT_CIRCULAR_HERTZ_LIMIT", family_name,
                    "passed" if circle_ok else "failed", circle_errors)

            # T08 zero-angle rolling speed limit, run through the production
            # kinematics kernel and compare to its independent closed form.
            kin = []
            lam = float(params[tai.D]) / float(params[tai.DM])
            for control_name, control_code in tai.CONTROL_CODES.items():
                for omega in (-100.0, 100.0):
                    oe, ob, beta = tai.kinematics_kernel(0.0, 0.0, omega, params, control_code)
                    expected_oe = omega * (1.0 - lam) / 2.0
                    expected_ob = omega * (-(1.0 - lam * lam) / (2.0 * lam))
                    err = max(abs(oe - expected_oe), abs(ob - expected_ob))
                    passed = err <= 1e-11 * max(abs(omega), 1.0) and abs(beta) <= 1e-14
                    kin.append({"control": control_name, "omega_local_rad_s": omega,
                                "omega_E": oe, "omega_B": ob, "beta": beta,
                                "max_abs_error": err, "status": "passed" if passed else "failed"})
                    _append(checks, "T08_ZERO_ANGLE_KINEMATICS", f"{family_name}:{control_name}:{omega:+g}",
                            "passed" if passed else "failed", kin[-1])
            family_out["nominal"]["zero_angle_kinematics"] = kin

            # Local speed, force balance, contact state, and control candidates.
            speed_results = []
            for rpm in speeds:
                omega = rpm * 2 * math.pi / 60
                A0, B0 = A, B
                point = {"rpm_magnitude": rpm, "A_m": A0, "B_m": B0, "modes": {}}
                for mode in ("none", "centrifugal", "full"):
                    try:
                        candidates = tai.solve_ball_candidates(A0, B0, omega, params, mode=mode)
                        branches = {name: item for name, item in candidates.items() if name in tai.CONTROL_CODES}
                        point["modes"][mode] = {"classification": candidates["classification"], "branches": branches}
                        numeric_branch_count = 0
                        control_valid = []
                        for branch_name, ball in branches.items():
                            if ball.get("failed"):
                                _append(checks, "T05_LOCAL_SPEED_ROOT", f"{family_name}:{rpm:g}:{mode}:{branch_name}",
                                        "failed", message=ball.get("failure", "local solve failed"))
                                continue
                            ball["state_AB_m"] = (A0, B0)
                            numeric_branch_count += int(_check_ball_root(checks,
                                f"{family_name}:{rpm:g}:{mode}:{branch_name}", ball, params))
                            if ball["active_set"] == "dual":
                                outer_margin, inner_margin = _recompute_control_margins(ball, params)
                                selected = outer_margin if branch_name == "outer" else inner_margin
                                margin_matches = abs(selected - float(ball["outer_control_margin" if branch_name == "outer" else "inner_control_margin"])) <= 1e-10
                                reported = bool(ball["control_self_consistent"])
                                self_ok = margin_matches and (reported == (selected > 0.0))
                                control_valid.append((branch_name, selected > 0.0))
                                _append(checks, "T08_CONTROL_SELF_CONSISTENCY",
                                        f"{family_name}:{rpm:g}:{mode}:{branch_name}",
                                        "passed" if self_ok else "failed", {
                                            "outer_margin_independent": outer_margin,
                                            "inner_margin_independent": inner_margin,
                                            "selected_margin": selected,
                                            "production_flag": reported,
                                            "classification": candidates["classification"],
                                        })
                            local_omega = omega
                            cage_omega = local_omega * float(params[tai.CAGE_RATIO])
                            phase_error = (abs(float(ball["omega_E_rad_s"]) - cage_omega) / abs(cage_omega)
                                           if cage_omega else 0.0)
                            phase_error_reported = float(ball["cage_speed_relative_difference"])
                            phase_consistent = abs(phase_error - phase_error_reported) <= 1e-12
                            phase_status = "failed" if not phase_consistent else ("conditional" if phase_error > 0.02 else "passed")
                            _append(checks, "T08_CAGE_PHASE_SPEED_DIAGNOSTIC",
                                    f"{family_name}:{rpm:g}:{mode}:{branch_name}", phase_status, {
                                        "loaded_omega_E_rad_s": ball["omega_E_rad_s"],
                                        "prescribed_phase_cage_omega_rad_s": cage_omega,
                                        "relative_difference": phase_error,
                                        "reported_relative_difference": phase_error_reported,
                                        "trigger_threshold": 0.02,
                                        "requires_common_phase_sensitivity_if_triggered": phase_error > 0.02,
                                    })
                        if candidates["classification"] in ("mixed_or_undetermined", "both_candidates") and numeric_branch_count:
                            control_status = "conditional"
                        else:
                            control_status = "passed" if control_valid else "conditional"
                        _append(checks, "T08_CONTROL_CLASSIFICATION", f"{family_name}:{rpm:g}:{mode}",
                                control_status, {"classification": candidates["classification"],
                                                 "self_consistent_branches": [name for name, ok in control_valid if ok],
                                                 "evaluated_branches": list(branches)})
                    except Exception as exc:
                        point["modes"][mode] = {"failed": f"{type(exc).__name__}: {exc}"}
                        _append(checks, "T05_LOCAL_SPEED_ROOT", f"{family_name}:{rpm:g}:{mode}", "failed",
                                message=f"{type(exc).__name__}: {exc}")
                speed_results.append(point)
            family_out["nominal"]["speed_matrix"] = speed_results

            # N1/N2/N3 at three non-zero speeds, with the same geometry and
            # local branch so centrifugal and gyroscopic contributions remain separable.
            mechanism_rows = []
            for rpm in (3000.0, 7000.0, 10000.0):
                for control in ("outer", "inner"):
                    row: dict[str, Any] = {"rpm": rpm, "control": control}
                    for mode in ("none", "centrifugal", "full"):
                        try:
                            ball = tai.solve_ball(A, B, rpm * 2 * math.pi / 60, params,
                                                  mode=mode, control=control)
                            row[mode] = {k: ball[k] for k in ("Qi_N", "Qo_N", "Fc_N", "Mg_N_m",
                                "omega_E_rad_s", "omega_B_rad_s", "beta_rad", "active_set",
                                "control_self_consistent", "force_residual_scaled")}
                        except Exception as exc:
                            row[mode] = {"failed": f"{type(exc).__name__}: {exc}"}
                    if all(isinstance(row.get(m), dict) and "failed" not in row[m] for m in ("none", "centrifugal", "full")):
                        effects_ok = (abs(row["none"]["Fc_N"]) <= 1e-14 and abs(row["none"]["Mg_N_m"]) <= 1e-14 and
                                      row["centrifugal"]["Fc_N"] > 0 and abs(row["centrifugal"]["Mg_N_m"]) <= 1e-14 and
                                      row["full"]["Fc_N"] > 0 and math.isfinite(row["full"]["Mg_N_m"]) and
                                      abs(row["full"]["Mg_N_m"]) > 1e-18)
                        _append(checks, "T05_INERTIA_ABLATION", f"{family_name}:{rpm:g}:{control}",
                                "passed" if effects_ok else "failed", row)
                    else:
                        _append(checks, "T05_INERTIA_ABLATION", f"{family_name}:{rpm:g}:{control}",
                                "failed", row, "one or more inertia mode roots failed")
                    mechanism_rows.append(row)
            family_out["nominal"]["inertia_ablation"] = mechanism_rows

            # F10 inner-contact-loss branch. It deliberately leaves the nominal
            # preload domain and is marked as a local active-set check.
            a_i, a_o = (float(params[tai.FI]) - 0.5) * float(params[tai.D]), (float(params[tai.FO]) - 0.5) * float(params[tai.D])
            B0full = a_i + a_o
            A_loss, B_loss = 0.0, B0full - 2e-6
            omega_loss = 7000 * 2 * math.pi / 60
            loss_rows = []
            for sign in (-1.0, 1.0):
                try:
                    ball = tai.solve_ball(A_loss, B_loss, sign * omega_loss, params,
                                          mode="full", control="outer")
                    ball["state_AB_m"] = (A_loss, B_loss)
                    numeric = _check_ball_root(checks, f"{family_name}:inner-loss:{sign:+g}", ball, params)
                    expected_state = (ball["active_set"] == "outer_only" and
                                      abs(ball["Qi_N"]) <= _force_tolerance(params, ball["Fc_N"]) and
                                      abs(ball["Qo_N"] - ball["Fc_N"]) <= _force_tolerance(params, ball["Fc_N"]) and
                                      abs(ball["Mg_N_m"]) <= 1e-14)
                    _append(checks, "T06_INNER_CONTACT_LOSS", f"{family_name}:{sign:+g}",
                            "passed" if numeric and expected_state else "failed", {
                                "active_set": ball["active_set"], "Qi_N": ball["Qi_N"],
                                "Qo_N": ball["Qo_N"], "Fc_N": ball["Fc_N"], "Mg_N_m": ball["Mg_N_m"],
                                "omega_E_rad_s": ball["omega_E_rad_s"], "omega_B_rad_s": ball["omega_B_rad_s"],
                                "A_m": A_loss, "B_m": B_loss,
                            })
                    loss_rows.append({"sign": sign, "result": ball})
                except Exception as exc:
                    _append(checks, "T06_INNER_CONTACT_LOSS", f"{family_name}:{sign:+g}", "failed",
                            message=f"{type(exc).__name__}: {exc}")
                    loss_rows.append({"sign": sign, "failed": f"{type(exc).__name__}: {exc}"})
            family_out["nominal"]["inner_contact_loss"] = loss_rows

            # Reverse-spin mirror: OmegaE and OmegaB reverse, Fc and Mg retain
            # their sign because both local rates reverse together.
            mirror = {}
            for sign in (-1.0, 1.0):
                try:
                    mirror[str(int(sign))] = tai.solve_ball(A, B, sign * 7000 * 2 * math.pi / 60,
                                                            params, mode="full", control="outer")
                except Exception as exc:
                    mirror[str(int(sign))] = {"failed": f"{type(exc).__name__}: {exc}"}
            if all("failed" not in mirror[str(int(s))] for s in (-1.0, 1.0)):
                pos, neg = mirror["1"], mirror["-1"]
                sign_ok = (abs(pos["Fc_N"] - neg["Fc_N"]) <= 1e-10 and
                           abs(pos["Mg_N_m"] - neg["Mg_N_m"]) <= 1e-12 and
                           abs(pos["omega_E_rad_s"] + neg["omega_E_rad_s"]) <= 1e-10 and
                           abs(pos["omega_B_rad_s"] + neg["omega_B_rad_s"]) <= 1e-10)
                _append(checks, "T06_SIGNED_SPEED_MIRROR", family_name,
                        "passed" if sign_ok else "failed", {
                            "Fc_plus_N": pos["Fc_N"], "Fc_minus_N": neg["Fc_N"],
                            "Mg_plus_Nm": pos["Mg_N_m"], "Mg_minus_Nm": neg["Mg_N_m"],
                            "omega_E_plus_rad_s": pos["omega_E_rad_s"], "omega_E_minus_rad_s": neg["omega_E_rad_s"],
                            "omega_B_plus_rad_s": pos["omega_B_rad_s"], "omega_B_minus_rad_s": neg["omega_B_rad_s"],
                        })
            else:
                _append(checks, "T06_SIGNED_SPEED_MIRROR", family_name, "failed", mirror)

            # Call-order independence across perturbed local states. The same
            # points are solved forward and reverse; no warm-start cache exists
            # in the public API, so returned physical fields must be identical.
            order_points = [(A + da, B + db) for da, db in
                            ((0., 0.), (0., 1e-6), (0., -1e-6), (1e-6, 0.), (-1e-6, 0.))]
            forward, reverse = [], []
            for a, b in order_points:
                forward.append(tai.solve_ball(a, b, -7000 * 2 * math.pi / 60, params,
                                              mode="full", control="outer"))
            for a, b in reversed(order_points):
                reverse.append(tai.solve_ball(a, b, -7000 * 2 * math.pi / 60, params,
                                              mode="full", control="outer"))
            fields = [name for name in tai.BALL_NAMES if name not in ("iterations", "evaluations")]
            max_order_diff = max(abs(float(forward[i][field]) - float(reverse[-1 - i][field]))
                                 for i in range(len(order_points)) for field in fields)
            order_ok = max_order_diff <= 1e-14
            _append(checks, "T11_CALL_ORDER_INDEPENDENCE", family_name,
                    "passed" if order_ok else "failed", {"max_absolute_difference": max_order_diff,
                                                           "compared_fields": fields,
                                                           "states": order_points})

            # The complete per-bearing entry is also called in increasing,
            # decreasing and repeated time order, with a different geometry
            # interleaved. This catches hidden mutable warm-start/cache state.
            time_points = (0.00031, 0.00077, 0.00119)
            xy, tilt, orientation, omega_global = (0.35e-6, -0.22e-6), (1.7e-6, -1.1e-6), -1, -7000 * 2 * math.pi / 60
            def wrench_at(time_value, parameter_array=params):
                return tai.bearing_wrenches(time_value, xy, tilt, parameter_array, orientation,
                                            omega_global, mode="full", control="outer")
            forward_wrenches = {time_value: wrench_at(time_value) for time_value in time_points}
            alt_family = _family_at_curvature(family, 0.52, 0.52)
            alt_params = tai.make_bearing_parameters(alt_family, materials)
            _ = wrench_at(time_points[1], alt_params)
            reverse_wrenches = {time_value: wrench_at(time_value) for time_value in reversed(time_points)}
            max_wrench_order_diff = 0.0
            for time_value in time_points:
                for key in ("inner_global", "outer_global", "balls"):
                    a = np.nan_to_num(np.asarray(forward_wrenches[time_value][key], dtype=float), nan=0.0)
                    b = np.nan_to_num(np.asarray(reverse_wrenches[time_value][key], dtype=float), nan=0.0)
                    max_wrench_order_diff = max(max_wrench_order_diff, float(np.max(np.abs(a - b))))
            wrench_order_ok = max_wrench_order_diff <= 1e-14
            _append(checks, "T11_BEARING_CALL_ORDER_AND_PARAMETER_ISOLATION", family_name,
                    "passed" if wrench_order_ok else "failed", {
                        "max_absolute_difference": max_wrench_order_diff,
                        "time_order_forward": time_points,
                        "time_order_reverse": tuple(reversed(time_points)),
                        "interleaved_curvature": 0.52,
                    })

            # Independent four-equation SciPy solve against the Numba reduced
            # solver; selected representative cases, not used as the only oracle.
            for rpm, mode, control in ((3000.0, "centrifugal", "outer"),
                                       (7000.0, "full", "outer"),
                                       (10000.0, "full", "inner")):
                case_id = f"{family_name}:{rpm:g}:{mode}:{control}"
                try:
                    fast = tai.solve_ball(A, B, rpm * 2 * math.pi / 60, params, mode, control)
                    ref = tai.solve_ball_reference(A, B, rpm * 2 * math.pi / 60, params, mode, control)
                    diffs = {field: abs(float(fast[field]) - float(ref[field])) /
                             max(abs(float(ref[field])), 1e-30)
                             for field in ("U_m", "V_m", "delta_i_m", "delta_o_m", "Qi_N", "Qo_N")}
                    passed = max(diffs["Qi_N"], diffs["Qo_N"], diffs["delta_i_m"], diffs["delta_o_m"]) <= 1e-6
                    _append(checks, "T05_REFERENCE_SOLVER_CROSSCHECK", case_id,
                            "passed" if passed else "failed", {"relative_differences": diffs,
                                                               "reference_residuals": ref["residuals"]})
                except Exception as exc:
                    _append(checks, "T05_REFERENCE_SOLVER_CROSSCHECK", case_id, "failed",
                            message=f"{type(exc).__name__}: {exc}")

            # Full Python F12 oracle versus Numba bearing wrench for both DB
            # orientations. The local state is transformed so it is equivalent.
            _backend_compare(checks, tai, family, materials, params, family_name)

            # Pure +/- tilt probes at each local axis, transformed through both
            # DB orientations. These are local adapter checks; they are not a
            # full structural moment-balance test.
            tilt_rows = []
            local_omega = -7000 * 2 * math.pi / 60
            B0_geom = (float(params[tai.FI]) + float(params[tai.FO]) - 1.0) * float(params[tai.D])
            for axis in (0, 1):
                for sign in (-1.0, 1.0):
                    local_tilt = [0.0, 0.0]
                    local_tilt[axis] = sign * 25e-6
                    for orientation in (-1, 1):
                        global_tilt = (local_tilt[0], orientation * local_tilt[1])
                        try:
                            out = tai.bearing_wrenches(0.00041, (0.0, 0.0), global_tilt,
                                params, orientation, orientation * local_omega,
                                mode="full", control="outer")
                            balls = np.asarray(out["balls"], dtype=float)
                            geom = float(np.max(np.abs(balls[:, [tai.G1, tai.G2]])) / (B0_geom * B0_geom))
                            force = float(np.max(np.abs(balls[:, [tai.G3, tai.G4]])))
                            all_dual = bool(np.all(balls[:, tai.ACTIVE] == 2))
                            active_ok = all_dual and bool(np.all(balls[:, tai.QI] > 0)) and bool(np.all(balls[:, tai.QO] > 0))
                            passed = geom <= 1e-9 and force <= _force_tolerance(params) and active_ok
                            row = {"axis": "tx" if axis == 0 else "ty", "sign": sign,
                                   "orientation": orientation, "local_tilt_rad": local_tilt,
                                   "max_normalized_geometry_residual": geom,
                                   "max_force_residual_N": force, "all_dual_contact": all_dual}
                            tilt_rows.append({**row, "status": "passed" if passed else "failed"})
                            _append(checks, "T05_SIGNED_TILT_LOCAL_ROOT",
                                    f"{family_name}:{row['axis']}:{sign:+g}:s{orientation:+d}",
                                    "passed" if passed else "failed", row)
                        except Exception as exc:
                            row = {"axis": "tx" if axis == 0 else "ty", "sign": sign,
                                   "orientation": orientation, "local_tilt_rad": local_tilt,
                                   "error": f"{type(exc).__name__}: {exc}"}
                            tilt_rows.append({**row, "status": "failed"})
                            _append(checks, "T05_SIGNED_TILT_LOCAL_ROOT",
                                    f"{family_name}:{row['axis']}:{sign:+g}:s{orientation:+d}",
                                    "failed", row)
            family_out["nominal"]["signed_tilt_local_roots"] = tilt_rows

            # fi/fo local sensitivity at every contract speed. Each geometry
            # independently solves its own fixed-position reference preload.
            for f in (0.52, 0.54):
                changed_family = _family_at_curvature(family, f, f)
                changed = tai.make_bearing_parameters(changed_family, materials)
                changed_preload = _preload_oracle(changed_family, materials)
                changed_A, changed_B = _geometry_for(changed_family, changed)
                preload_diff = abs(float(changed[tai.A_PRE]) - changed_preload["a_pre_m"])
                preload_force_diff = abs(changed_preload["axial_load_N"] - changed_preload["target_load_N"])
                preload_sensitivity_ok = (preload_diff <= 1e-9 and
                                          preload_force_diff <= max(1e-6, 1e-7 * changed_preload["target_load_N"]))
                _append(checks, "T04_CURVATURE_PRELOAD_ORACLE",
                        f"{family_name}:fi_fo={f:.2f}",
                        "passed" if preload_sensitivity_ok else "failed", {
                            "a_pre_production_m": float(changed[tai.A_PRE]),
                            "a_pre_independent_m": changed_preload["a_pre_m"],
                            "a_pre_difference_m": preload_diff,
                            "preload_residual_N": preload_force_diff,
                            "preload_target_N": changed_preload["target_load_N"],
                        })
                for rpm in speeds:
                    omega = rpm * 2 * math.pi / 60
                    candidates = tai.solve_ball_candidates(changed_A, changed_B, omega, changed,
                                                           mode="full")
                    row = {"fi": f, "fo": f, "rpm_magnitude": rpm,
                           "a_pre_m": float(changed[tai.A_PRE]),
                           "independent_preload_oracle": changed_preload,
                           "classification": candidates["classification"], "branches": {}}
                    numeric = True
                    for branch in ("outer", "inner"):
                        ball = candidates.get(branch)
                        if ball is None or ball.get("failed"):
                            numeric = False
                            row["branches"][branch] = ball
                            continue
                        ball["state_AB_m"] = (changed_A, changed_B)
                        resid = np.asarray(ball["residuals"], dtype=float)
                        force_tol = _force_tolerance(changed, ball["Fc_N"])
                        geom = float(np.max(np.abs(resid[:2])) / ((float(changed[tai.FI]) + float(changed[tai.FO]) - 1) * float(changed[tai.D]))**2)
                        branch_ok = (geom <= 1e-9 and float(np.max(np.abs(resid[2:]))) <= force_tol and
                                     math.isfinite(float(ball["Qi_N"])) and math.isfinite(float(ball["Qo_N"])))
                        numeric = numeric and branch_ok
                        row["branches"][branch] = {k: ball[k] for k in ("active_set", "Qi_N", "Qo_N", "Fc_N",
                            "Mg_N_m", "alpha_i_rad", "alpha_o_rad", "Ki_N_m32", "Ko_N_m32",
                            "control_self_consistent", "outer_control_margin", "inner_control_margin",
                            "force_residual_scaled", "residuals")}
                    row["numerical_status"] = "passed" if numeric else "failed"
                    family_out["curvature_sensitivity"].append(row)
                    _append(checks, "T02_T05_CURVATURE_SPEED_SENSITIVITY",
                            f"{family_name}:fi_fo={f:.2f}:rpm={rpm:g}",
                            "passed" if numeric else "failed", row)
            # Give the saved sensitivity records cross-differences between f=.52 and .54.
            for rpm in speeds:
                rows = [r for r in family_out["curvature_sensitivity"] if r["rpm_magnitude"] == rpm]
                if len(rows) == 2:
                    by_f = {r["fi"]: r for r in rows}
                    if 0.52 in by_f and 0.54 in by_f:
                        for branch in ("outer", "inner"):
                            a, b = by_f[0.52]["branches"].get(branch), by_f[0.54]["branches"].get(branch)
                            if isinstance(a, dict) and isinstance(b, dict) and "failed" not in a and "failed" not in b:
                                by_f[0.54].setdefault("delta_from_fi_fo_0_52", {})[branch] = {
                                    key: float(b[key]) - float(a[key]) for key in
                                    ("Qi_N", "Qo_N", "Fc_N", "Mg_N_m", "alpha_i_rad", "alpha_o_rad", "Ki_N_m32", "Ko_N_m32")
                                }
        except Exception as exc:
            _append(checks, "FAMILY_SETUP", family_name, "failed", message=f"{type(exc).__name__}: {exc}")
            family_out["setup_error"] = f"{type(exc).__name__}: {exc}"
        result["families"][family_name] = family_out

    # The system stage is deliberately limited to static balance and a two-turn
    # backend replay; it does not launch the full experiment matrix.
    try:
        system = _run_system_validation(system_config_path)
    except Exception as exc:
        failure = _error_record(exc)
        system = {"schema": "spindle_v9_system_integration_validation_v1",
                  "status": "failed", "config_path": str(system_config_path.resolve()),
                  "checks": [{"check_id": "T12_SYSTEM_VALIDATION_ENTRY", "case_id": "system",
                              "status": "failed", "metrics": failure,
                              "message": "system validation entry raised before producing checks"}],
                  "summary": {"checks_total": 1, "passed": 0, "conditional": 0, "failed": 1,
                              "machine_validated": False}}
    result["system_integration"] = system
    checks.extend(system.get("checks", []))

    statuses = [item["status"] for item in checks]
    if any(status == "failed" for status in statuses):
        result["status"] = "failed"
    elif any(status == "conditional" for status in statuses):
        result["status"] = "conditional_physics"
    else:
        result["status"] = "numerical_pass"
    result["summary"] = {
        "checks_total": len(checks),
        "passed": statuses.count("passed"),
        "conditional": statuses.count("conditional"),
        "failed": statuses.count("failed"),
        "machine_validated": False,
        "full_spindle_integrated": False,
        "short_system_integration_status": system["status"],
        "system_checks": system.get("summary", {}).get("checks_total", 0),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--system-config", type=Path, default=DEFAULT_SYSTEM_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = _run_validation(args.contract, args.output_dir, args.system_config)
    _save_results(args.output_dir, result)
    print(json.dumps(_jsonable({"status": result["status"], "summary": result.get("summary"),
                                "blocker": result.get("blocker"), "output_dir": str(args.output_dir.resolve())}), ensure_ascii=False))
    return 0 if result["status"] == "numerical_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
