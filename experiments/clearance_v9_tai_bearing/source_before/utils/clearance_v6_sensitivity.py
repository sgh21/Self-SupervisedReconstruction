"""Frozen-tangent V6 parameter screening; not nonlinear time-domain evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from spindle import dynamics_model as model


def scenarios(parameters: dict) -> list[tuple[str, str, dict]]:
    """One assumption at a time. Do not recompute U or other coefficients."""
    result = [("baseline", "baseline", deepcopy(parameters))]
    for factor in (0.5, 2.0):
        p = deepcopy(parameters)
        p["Kf"] *= factor
        p["Kf2"] *= factor
        result.append((f"Kfit_x{factor:g}", "Kfit", p))
        p = deepcopy(parameters)
        p["cb_total"] *= factor
        result.append((f"cb_front_x{factor:g}", "cb_front", p))
    for factor in (0.75, 1.25):
        p = deepcopy(parameters)
        p["Jt"] *= factor
        p["Jp"] *= factor
        result.append((f"inertia_x{factor:g}", "inertia", p))
    p = deepcopy(parameters)
    center = np.mean([s["z_m"] for s in p["rear_supports"]])
    for s, sign in zip(p["rear_supports"], (-1, 1)):
        s["z_m"] = float(center + sign * 0.028 / 2)
    result.append(("rear_span_28mm", "rear_span", p))
    p = deepcopy(parameters)
    p["rear_elastic_force_factor"] = 0.5
    result.append(("rear_elastic_x0.5", "rear_elastic", p))
    for mass in (0.06, 0.14):
        p = deepcopy(parameters)
        p["mh"] += p["mO"] + p["mO2"] - 2 * mass
        p["mO"] = p["mO2"] = mass
        result.append((f"outer_mass_{mass:g}kg", "outer_mass", p))
    for factor in (0.8, 1.2):
        p = deepcopy(parameters)
        p["normal_overlap"] *= factor ** (2 / 3)
        p["preload_reference_N"] *= factor
        p["radial_tangent_reference_N_m"] *= factor ** (1 / 3)
        result.append((f"front_preload_x{factor:g}", "front_preload", p))
    return result


def frozen_response(p: dict, gap: float) -> tuple[dict, dict]:
    p = model.front_group_parameters(p, gap)
    q, residual = model.equilibrium_continuation(p)
    mass, damping_gyro, stiffness = model.linear_matrices(p, 0, q)
    omega = abs(p["rpm"]) * 2 * np.pi / 60
    f = model.generalized_point_force(np.array([1., -1j * np.sign(p["rpm"])]), p["z_u"], 12)
    response = np.linalg.solve(stiffness - omega ** 2 * mass + 1j * omega * damping_gyro, f)
    measured = model.rotor_section(response, p["z_m"])
    static = np.column_stack([
        model.rotor_section(np.linalg.solve(stiffness, model.generalized_point_force(axis, p["z_u"], 12)), p["z_m"])
        for axis in np.eye(2)])
    eigen = eigh((stiffness + stiffness.T) / 2, mass, eigvals_only=True)
    frequencies = np.sqrt(np.maximum(eigen, 0)) / (2 * np.pi)
    state_matrix = np.block([[np.zeros((12, 12)), np.eye(12)],
                             [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping_gyro)]])
    poles = np.linalg.eigvals(state_matrix)
    row = {
        "gap_um": gap,
        "E_dir_linear_um_per_N": float(2e6 * np.max(np.abs(measured))),
        "Ex_linear_um_per_N": float(2e6 * abs(measured[0])),
        "Ey_linear_um_per_N": float(2e6 * abs(measured[1])),
        "Hx_linear_vrms_mm_s_per_N": float(1e3 * omega * abs(response[4]) / np.sqrt(2)),
        "Hy_linear_vrms_mm_s_per_N": float(1e3 * omega * abs(response[5]) / np.sqrt(2)),
        "max_static_compliance_um_N": float(np.linalg.svd(static, compute_uv=False).max() * 1e6),
        "first_mode_Hz": float(frequencies.min()),
        "max_mode_Hz": float(frequencies.max()),
        "max_frozen_pole_real_s": float(poles.real.max()),
        "equilibrium_scaled_residual_N": float(residual),
        "spindle_mass_kg": float(p["ms"] + p["mh"] + p["mO"] + p["mO2"]),
    }
    detail = {"parameters": p, "equilibrium_q": q.tolist(),
              "all_frozen_modes_Hz": frequencies.tolist(),
              "static_compliance_xy_um_N": (static * 1e6).tolist(),
              "M_complex_response_um_per_N": [[float(z.real*1e6), float(z.imag*1e6)] for z in measured]}
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/clearance_v6_config.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/clearance_v6_GMN/sensitivity"))
    args = parser.parse_args()
    p = json.loads(args.config.read_text(encoding="utf-8"))["parameters"]
    rows, details, baseline = [], {}, {}
    for name, family, parameters in scenarios(p):
        for gap in (2.0, 10.0):
            row, detail = frozen_response(parameters, gap)
            row.update(scenario=name, family=family)
            if name == "baseline":
                baseline[gap] = row.copy()
            for feature in ("E_dir_linear_um_per_N", "Hx_linear_vrms_mm_s_per_N", "Hy_linear_vrms_mm_s_per_N"):
                row[feature + "_change_pct"] = 100 * (row[feature] / baseline[gap][feature] - 1)
            rows.append(row)
            details[f"{name}/gap_{gap:g}um"] = detail
    ranking = {}
    for row in rows:
        if row["family"] != "baseline":
            key = row["family"]
            score = abs(row["E_dir_linear_um_per_N_change_pct"])
            if key not in ranking or score > ranking[key]["score"]:
                ranking[key] = {"score": score, "scenario": row["scenario"], "gap_um": row["gap_um"]}
    ranked = sorted(ranking.items(), key=lambda item: item[1]["score"], reverse=True)
    selected = [{"family": key, **value} for key, value in ranked[:2]]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "local_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence = {"status": "frozen_tangent_screen_only_not_time_domain_validation",
                "ranking_basis": "maximum absolute E_dir per unit rotating force change across c=2 and 10",
                "selected_for_time_domain": selected, "all_family_ranking": ranked,
                "rows": rows, "details": details,
                "limits": "t=0 cage phase and static-contact tangent; no calibrated U used; neither nonlinear steady state nor domain/coverage certified"}
    (args.output / "local_sensitivity.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# V6 参数局部筛查", "", "冻结 t=0 保持架相位和静平衡切线，7000 rpm 每 1 N 旋转力计算局部频响；包含有符号陀螺项。结果只用于选少量敏感性工况，不是非线性稳态预测，也不用于标定 U。", "",
             "每次只改一个假设；外圈质量变化由 mh 补偿，Pa 变化保持 Keq 固定并按 Pa^(2/3) 改 δ0。Kfit 或质量变更时 Df 固定，避免多参数混改。", "",
             "| 假设族 | 最大局部 E 变化绝对值 % | 代表情景 | 间隙 μm |", "|---|---:|---|---:|"]
    lines += [f"| {key} | {value['score']:.4f} | {value['scenario']} | {value['gap_um']:g} |" for key, value in ranked]
    lines += ["", "按上述固定排序选择前两项做健康/退化成对时域检查：" + "、".join(v["scenario"] for v in selected) + "。完整的0.5/2、0.75/1.25、72/108 N等两侧筛查保存在CSV及JSON，不把最软情景并入主曲线。", "",
              "所有表中的模态和极点只属于冻结局部系统；保持架周期变化、接触开闭及有限幅运动必须另由时域结果验证。倾角、球载、强度及高速预紧适用性仍按正式验证报告判断。"]
    (args.output / "local_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
