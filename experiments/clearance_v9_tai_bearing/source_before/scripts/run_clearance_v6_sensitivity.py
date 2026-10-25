"""Run the predeclared V6 assumption/holdout cases with one frozen U.

Use only after healthy calibration has been independently accepted. This entry
does not calibrate U and does not certify these scenarios with h/N refinements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scripts.run_clearance_v3 import write_csv, write_json
from scripts.run_clearance_v6 import case_config, execute
from utils.clearance_v6_sensitivity import scenarios


def changed_fields(base, new, prefix=""):
    """Plain field comparison; no hashes or hidden physics matching."""
    changes = {}
    for key in sorted(set(base) | set(new)):
        name = prefix + key
        a, b = base.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.update(changed_fields(a, b, name + "."))
        elif a != b:
            changes[name] = {"baseline": a, "scenario": b}
    return changes


def cases(baseline):
    parameter_cases = {name: p for name, _, p in scenarios(baseline["parameters"])}
    for name in ("rear_elastic_x0.5", "rear_span_28mm", "front_preload_x0.8", "front_preload_x1.2"):
        config = deepcopy(baseline)
        config["parameters"] = parameter_cases[name]
        for gap in (2., 10.):
            yield name, gap, config
    for rpm in (6000, 8000):
        config = deepcopy(baseline)
        config["parameters"]["rpm"] = -float(rpm)
        for gap in (2., 10.):
            yield f"rpm_{rpm}", gap, config
    yield "health_gap_1um_fixed_U", 1., deepcopy(baseline)


def summarize(root):
    rows = []
    nominal = {}
    for gap in (2., 3., 10.):
        path = root / "data" / f"gap_{gap:g}um" / "metrics.json"
        if path.exists():
            nominal[gap] = json.loads(path.read_text(encoding="utf-8"))
    keys = ("E_dir_um", "E_x_um", "E_y_um", "mean_x_um", "mean_y_um",
            "H_v_rms_x_mm_s", "H_v_rms_y_mm_s", "H_v_rms_xy_mm_s",
            "H_v_1x_x_mm_s", "H_v_1x_y_mm_s", "H_v_residual_x_mm_s", "H_v_residual_y_mm_s",
            "fit_contact_fraction", "fit2_contact_fraction", "fit_peak_N", "fit2_peak_N")
    for folder in sorted((root / "sensitivity").glob("*/gap_*um")):
        path = folder / "metrics.json"
        row = {"scenario": folder.parent.name, "gap_um": float(folder.name[4:-2]),
               "status": "not_completed", "U_kg_m": None, "force_N": None, "rpm": None,
               **{key: None for key in keys}, "steady": False, "domain_screen": False,
               "sampling": False, "independent_time_angular_certification": "not_tested",
               "physical_validation": "conditional_not_real_machine_validation", "reason": "",
               "nominal_reference_gap_um": None, "nominal_reference_E_um": None,
               "E_change_vs_nominal_um": None, "E_change_vs_nominal_pct": None}
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            config = json.loads((folder / "actual_config.json").read_text(encoding="utf-8"))
            row.update(status="completed", U_kg_m=config["excitation"]["U_kg_m"],
                       force_N=config["excitation"]["peak_force_N"], rpm=config["parameters"]["rpm"],
                       steady=result["steady_pass"], domain_screen=result["domain_pass"],
                       sampling=result["sampling_pass"])
            row.update({key: result.get(key) for key in keys})
            reference_gap = 2. if row["gap_um"] == 1. else row["gap_um"]
            if reference_gap in nominal:
                reference = nominal[reference_gap]["E_dir_um"]
                row.update(nominal_reference_gap_um=reference_gap,
                           nominal_reference_E_um=reference,
                           E_change_vs_nominal_um=result["E_dir_um"]-reference,
                           E_change_vs_nominal_pct=100*(result["E_dir_um"]/reference-1))
            if not result["steady_pass"]:
                row["reason"] = "full export window did not meet stationarity thresholds"
            if not result["domain_pass"]:
                row["reason"] += "; model applicability screen failed"
        else:
            for file in ("skipped.json", "failed.json"):
                if (folder / file).exists():
                    row.update(status=file[:-5], reason=json.loads((folder / file).read_text(encoding="utf-8")).get("reason", ""))
        rows.append(row)
    write_csv(root / "sensitivity_summary.csv", rows)
    write_json(root / "sensitivity" / "time_domain_summary.json", {
        "status": "actual_time_domain_scenarios_not_independently_h_N_certified",
        "rows": rows, "U_policy": "identical frozen U in every case; F changes with rpm squared",
        "force_ceiling_N": 30.0,
        "parameter_scenarios_are_not_main_gap_curve": True,
    })
    lines = ["# V6 固定 U 的参数与速度情景", "",
             "全部情景沿用已标定的同一 U；未在新健康点或退化点重新加力。非主线情景不能计入主间隙曲线的范围。下表只列实际完成量，域内表示小角度/非拉力筛查通过，接触强度和实机适用性仍未独立验证。", "",
             "| 情景 | c μm | 力 N | E μm | 对名义同c的变化 % | 稳态 | 域筛查 |",
             "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        if row["status"] != "completed":
            continue
        change = "未知" if row["E_change_vs_nominal_pct"] is None else f"{row['E_change_vs_nominal_pct']:+.4f}"
        lines.append(f"| {row['scenario']} | {row['gap_um']:g} | {row['force_N']:.6f} | {row['E_dir_um']:.8f} | {change} | {row['steady']} | {row['domain_screen']} |")
    lines += ["", "c1正常配合假设行的参照是主线健康c2，其余行参照名义7000 rpm同间隙主线。速度留出保留旋向符号；力按Ω²变化，并未保持恒定力。", "",
              "这些情景保留完整时序、实际配置及输出采样比较；未逐个实施独立h/2和N1024，故数值误差未知，不凭这些情景宣称严格20 μm通过或微弱变化已可分辨。", ""]
    if 3. in nominal:
        lines += [f"另直接复用主线c3结果作为正常配合3 μm假设：E={nominal[3.]['E_dir_um']:.8f} μm，稳态={nominal[3.]['steady_pass']}。没有重复运行或为该情景重标定U。", ""]
    if 10. in nominal:
        for scenario in ("rear_span_28mm", "front_preload_x1.2"):
            matching = [r for r in rows if r["scenario"] == scenario and r["gap_um"] == 10 and r["status"] == "completed"]
            if matching:
                row = matching[0]
                lines += [f"{scenario} 的c10：E相对主线变化{row['E_change_vs_nominal_pct']:+.4f}%；O2配合接触率从名义{nominal[10.]['fit2_contact_fraction']:.6f}变为{row['fit2_contact_fraction']:.6f}，O2配合峰力变为{row['fit2_peak_N']:.6f} N。这说明前配合开闭分支随未知后跨度或工作预紧改变；健康约2 μm拟合不能排除这类退化幅值不确定性。", ""]
    for row in rows:
        if row["status"] == "completed" and not row["steady"]:
            lines += [f"{row['scenario']} c{row['gap_um']:g} 在约定最大热身后完整窗仍未通过，最后窗口E={row['E_dir_um']:.8f} μm；响应类别未认证，不能称为已证发散或已证统计稳态，也不计7000 rpm主线范围。本轮未追加240/480转留出统计。", ""]
    (root / "sensitivity" / "time_domain_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/clearance_v6_GMN/parameters_v6_calibrated.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/clearance_v6_GMN"))
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output)
        return
    baseline = json.loads(args.config.read_text(encoding="utf-8"))
    if baseline["excitation"]["U_kg_m"] is None:
        raise ValueError("A frozen calibrated U is required; no calibration is performed here.")
    jobs = []
    for name, gap, config in cases(baseline):
        if args.scenarios and name not in args.scenarios:
            continue
        folder = args.output / "sensitivity" / name / f"gap_{gap:g}um"
        force = config["excitation"]["U_kg_m"] * (config["parameters"]["rpm"] * 2*np.pi/60) ** 2
        if force > 30 + 1e-10:
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "skipped.json", {"reason": "Frozen U would exceed the user 30 N force ceiling", "force_N": force})
            continue
        actual = case_config(config, gap)
        actual["sensitivity"] = {"scenario": name, "frozen_U": True,
                                 "independent_time_angular_certification": "not_tested",
                                 "changes_from_healthy_baseline": changed_fields(baseline["parameters"], actual["parameters"])}
        jobs.append((actual, folder))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, config, folder): folder for config, folder in jobs}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
                print(json.dumps({"folder": str(folder), "E_um": result["E_dir_um"],
                                  "steady": result["steady_pass"], "domain": result["domain_pass"]}), flush=True)
            except Exception as error:
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "failed.json", {"reason": str(error)})
                print(json.dumps({"folder": str(folder), "failed": str(error)}), flush=True)
            summarize(args.output)
    summarize(args.output)


if __name__ == "__main__":
    main()
