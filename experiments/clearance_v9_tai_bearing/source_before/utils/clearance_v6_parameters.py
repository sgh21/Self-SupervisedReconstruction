"""Translate the V6 parameter contract into the existing solver configuration.

No calibration or dynamics are performed here. U remains unset until a run
explicitly chooses a candidate. Values labelled assumed remain assumptions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path


def derive_bearing(spec: dict) -> dict:
    """Fit the fixed-angle Hertz law to preload and the radial tangent."""
    pa, kr = spec["preload_N"], spec["radial_tangent_single_N_m"]
    alpha, balls = spec["contact_angle_rad"], spec["balls"]
    delta = 3 * pa * math.cos(alpha) ** 2 / (4 * kr * math.sin(alpha))
    keq = pa / (balls * math.sin(alpha) * delta ** 1.5)
    return {
        "balls": balls,
        "ball_diameter": spec["ball_diameter_m"],
        "pitch_diameter": spec["pitch_diameter_m"],
        "contact_angle": alpha,
        "normal_overlap": delta,
        "Keq": keq,
        "cb_total": spec["cb_total_N_s_m"],
        "cage_phase": 0.0,
        "preload_reference_N": pa,
        "radial_tangent_reference_N_m": kr,
        "designation": spec["designation"],
    }


def build_config(contract: dict, old_config: dict) -> dict:
    config = deepcopy(old_config)
    mass, geom, fitting = (contract[k] for k in ("mass", "geometry", "fitting"))
    config.update(schema_version=6, scenario_id="JD130_GMN_conditional_V6_health2um")
    p = config["parameters"]
    removed = {key: p.pop(key) for key in
               ("Kother", "Cother", "Kb2", "Cb2", "healthy_supports", "ki", "ko")
               if key in p}
    p.update(derive_bearing(contract["bearings"]["front"]))
    p.update({
        "ms": mass["ms_kg"], "mt": mass["mt_kg"], "mh": mass["mh_kg"],
        "mO": mass["mO1_kg"], "mO2": mass["mO2_kg"],
        "Jt": mass["Jt_kg_m2"], "Jp": mass["Jp_kg_m2"],
        "nonlinear_bearings_v6": True,
        "rear_bearing": derive_bearing(contract["bearings"]["rear"]),
        "rear_supports": [{"name": "R1", "z_m": geom["z_r1_m"]},
                          {"name": "R2", "z_m": geom["z_r2_m"]}],
        "rear_elastic_force_factor": 1.0,
        "rear_outer_fit_model": "healthy_outer_to_H_fixed_assumption",
        "z_f": geom["z_f1_m"], "z_f2": geom["z_f2_m"],
        "z_m": geom["z_m_m"], "z_T": geom["z_T_m"], "z_u": geom["z_u_m"],
        "L_ref": geom["L_ref_m"],
        "Kf": fitting["Kf1_N_m"], "Kf2": fitting["Kf2_N_m"],
        "Df": fitting["Df1_N_s_m"], "Df2": fitting["Df2_N_s_m"],
        "N": fitting["N_initial"],
        "fit_clearance_um": fitting["common_front_health_radial_gap_um"],
        "fit2_clearance_um": fitting["common_front_health_radial_gap_um"],
        "front_group_clearance_um": fitting["common_front_health_radial_gap_um"],
        "healthy_error_reported_um": 2.0,
        "healthy_error_definition": "approximate user reference; direction unspecified; conditional E_dir calibration",
        "machine_model": "JD130S-24-HA40; GMN 6009 front pair / 6007 rear pair; 12 DOF",
        "parameter_status": "mixed A/B/C/D; approximate machine mass, catalogue reference and explicit assumptions",
        "fit_parameter_status": "Kf inherited; Df=2*.02*sqrt(.1*Kf); c_h=2um assumed, not measured",
        "model_status": "conditionally_calibrated_candidate_not_independently_validated",
    })
    config["deprecated_v5_parameters_not_active"] = removed
    config["excitation"].update(
        U_kg_m=None, peak_force_N=None,
        source="unset before healthy calibration; only U is fitted; force ceiling 30 N at 7000 rpm")
    config["calibration"] = deepcopy(contract["calibration"])
    config["clearances_um"] = fitting["main_radial_clearances_um"]
    config["ideal_control_gap_um"] = fitting["ideal_control_gap_um"]
    config["accepted_gap_range_um"] = [0.0, 10.0]
    config["trial_design"] = {
        "only_free_calibration_parameter": "U_kg_m",
        "healthy_gap_um": 2.0, "force_ceiling_N_at_7000": 30.0,
        "calibration_force_candidates_N": [0, 2, 5, 10, 20, 30],
        "gap_sweep_free_parameter": "common_front_radial_clearance_um",
        "sensitivity": deepcopy(contract["sensitivity"]),
    }
    config["parameter_provenance"] = {
        "status": "mixed_machine_information_catalogue_conditional_derivations_and_assumptions",
        "contract": "docs/reports/clearance_v6_parameter_contract.md",
        "source_contract": "GPTOrder/spindle_v6_plan/config/parameter_contract_v6.json",
        "inherited_config": "config/clearance_v5_config.json",
        "mass_ledger": deepcopy(mass),
        "materials_reference_only": deepcopy(contract["materials"]),
        "bearing_references": deepcopy(contract["bearings"]),
        "geometry_status": geom["status"],
        "observation": "z_m=.165m rotor section relative fixed ground/probe approximation; E_dir retains 1X",
        "excitation_status": "conditional healthy calibration; not measured residual unbalance",
    }
    config["constitutive_domains"] = {
        "maximum_tilt_rad": 0.001,
        "status": "small-angle screen only; real contact pressure and high-speed preload validity not established",
        "report_each_bearing": ["delta_over_ball_diameter", "delta_over_delta0", "maximum_ball_force_N"],
        "removed_legacy_limits": "F2/R1 1um linear tangent limits and 2delta0 strength proxy do not apply to new nonlinear branches",
    }
    config["processing"]["passband_hz"] = [10, 900]
    config["processing"]["auxiliary_passband_hz"] = [10, 2048]
    config["numerical_thresholds"].update(
        position_um=0.005, velocity_mm_s=1e-4, contact_rate=0.02)
    config["physical_parameter_support"] = "conditional; see provenance levels"
    config["real_machine_validation"] = "not_validated"
    config.pop("allocation_v4", None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("GPTOrder/spindle_v6_plan/config/parameter_contract_v6.json"))
    parser.add_argument("--baseline", type=Path, default=Path("config/clearance_v5_config.json"))
    parser.add_argument("--output", type=Path, default=Path("config/clearance_v6_config.json"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/clearance_v6_GMN"))
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config = build_config(contract, json.loads(args.baseline.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.experiment.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    (args.experiment / "parameters_v6_before_calibration.json").write_text(encoded, encoding="utf-8")
    print(f"Wrote {args.output}; U remains null, no dynamics run")


if __name__ == "__main__":
    main()
