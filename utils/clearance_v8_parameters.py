"""Build V8 runtime parameters from the recorded V6 healthy configuration.

This module only constructs configuration. It does not allocate supports,
calibrate U, or run the dynamics model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "experiments/clearance_v6_GMN/data/gap_2um/actual_config.json"
DEFAULT_OUTPUT = ROOT / "config/clearance_v8_config.json"
CONTACT_MODELS = {"radial_only_v6", "relative_tilt_v8"}

_CACHE_KEYS = (
    "_v8_bearing_context",
    "_v8_bearing_context_key",
    "_tilt_constants",
    "_tilt_constants_key",
    "_front2_constants",
    "_front2_constants_key",
    "_rear_bearing_parameters",
    "_rear_bearing_parameters_key",
    "_bearing_constants",
    "_bearing_constants_key",
)

_ARRANGEMENTS = {
    "DB_DB": {"front": "DB", "rear": "DB"},
    "DB_DF": {"front": "DB", "rear": "DF"},
    "DF_DB": {"front": "DF", "rear": "DB"},
    "DF_DF": {"front": "DF", "rear": "DF"},
}

_SIGN_BY_BEARING = {
    "DB": {"F1": -1, "F2": 1, "R1": 1, "R2": -1},
    "DF": {"F1": 1, "F2": -1, "R1": -1, "R2": 1},
}


def _layout_entries(parameters: dict[str, Any], layout: str) -> list[dict[str, Any]]:
    if layout not in _ARRANGEMENTS:
        raise ValueError(f"Unknown bearing layout {layout!r}; choose one of {tuple(_ARRANGEMENTS)}")

    front_pair = _ARRANGEMENTS[layout]["front"]
    rear_pair = _ARRANGEMENTS[layout]["rear"]
    positions = {
        "F1": (parameters["z_f"], "O1", "front", parameters["pitch_diameter"] / 2),
        "F2": (parameters["z_f2"], "O2", "front", parameters["pitch_diameter"] / 2),
    }
    for item in parameters["rear_supports"]:
        name = item["name"]
        if name not in {"R1", "R2"}:
            raise ValueError(f"Unexpected rear support {name!r}")
        positions[name] = (
            item["z_m"],
            "H",
            "rear",
            parameters["rear_bearing"]["pitch_diameter"] / 2,
        )

    entries = []
    for name in ("F1", "F2", "R1", "R2"):
        z_m, outer_source, material, rho_m = positions[name]
        pair_arrangement = front_pair if material == "front" else rear_pair
        entries.append(
            {
                "name": name,
                "z_m": float(z_m),
                "orientation_sign": _SIGN_BY_BEARING[pair_arrangement][name],
                "rho_m": float(rho_m),
                "outer_translation_source": outer_source,
                "outer_tilt_constraint": "fixed",
                "axial_increment_constraint": "fixed",
                "arrangement_basis": f"{pair_arrangement}_pair_{layout}_assumed_not_machine_verified",
                "material_parameters": material,
            }
        )
    return entries


def _parameters(config: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = config["parameters"]
    except KeyError as exc:
        raise ValueError("Expected a V6/V8 configuration with a 'parameters' object") from exc
    if not isinstance(parameters, dict):
        raise ValueError("config['parameters'] must be an object")
    return parameters


def layout_config(config: dict[str, Any], layout: str) -> dict[str, Any]:
    """Return a copy with the requested arrangement and stale bearing caches removed."""
    updated = deepcopy(config)
    p = _parameters(updated)
    if layout not in _ARRANGEMENTS:
        raise ValueError(f"Unknown bearing layout {layout!r}; choose one of {tuple(_ARRANGEMENTS)}")
    prior_layout = p.get("bearing_layout_name")
    if prior_layout and prior_layout != layout:
        prior_calibration = deepcopy(updated.get("calibration", {}))
        if prior_calibration and prior_calibration.get("status") not in {"not_run", "not_run_layout_changed"}:
            updated.setdefault("previous_layout_calibrations", []).append(
                {"layout": prior_layout, "record": prior_calibration}
            )
        if isinstance(updated.get("calibration"), dict):
            updated["calibration"].update(
                status="not_run_layout_changed",
                layout=layout,
                U_star_kg_m=None,
                F_star_N_at_7000=None,
            )
            updated["calibration"].pop("calibration_pass", None)
        p["calibration_status"] = "not_run_layout_changed"
    p["bearing_layout_name"] = layout
    p["bearing_layout_status"] = "assumed_not_machine_verified"
    p["bearing_layout_basis"] = f"{layout}_assumed_not_machine_verified"
    p["bearing_layout"] = _layout_entries(p, layout)
    p["bearing_position_basis"] = "geometric_center_assumed"
    p["front_pair_arrangement_assumption"] = f"{_ARRANGEMENTS[layout]['front']}_assumed_not_machine_verified"
    p["rear_pair_arrangement_assumption"] = f"{_ARRANGEMENTS[layout]['rear']}_assumed_not_machine_verified"
    if isinstance(updated.get("calibration"), dict):
        updated["calibration"]["layout"] = layout
    for key in _CACHE_KEYS:
        p.pop(key, None)
    return updated


def build_config(
    layout: str = "DB_DB",
    contact_model: str = "relative_tilt_v8",
    base_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Build a V8 runtime configuration while retaining V6 physical parameters.

    ``base_config`` may be a configuration object or a path. If omitted, the
    recorded V6 c=2 um healthy run is used, preserving its old calibrated U for
    fixed-U mechanism comparisons.
    """
    if contact_model not in CONTACT_MODELS:
        raise ValueError(f"Unknown bearing contact model {contact_model!r}")
    if base_config is None:
        source = DEFAULT_BASE_CONFIG
        config = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(base_config, (str, Path)):
        source = Path(base_config)
        if not source.is_absolute():
            source = ROOT / source
        config = json.loads(source.read_text(encoding="utf-8"))
    else:
        source = None
        config = deepcopy(base_config)

    p = _parameters(config)
    if int(p.get("model_dof", -1)) != 12:
        raise ValueError("V8 requires the recorded 12-DOF V6 runtime configuration")
    if config.get("excitation", {}).get("U_kg_m") is None:
        raise ValueError("V8 experiment-A config requires the calibrated V6 U from an actual healthy run")

    old_calibration = deepcopy(config.get("calibration", {}))
    old_excitation = deepcopy(config.get("excitation", {}))
    old_model_status = p.get("model_status")
    legacy_health_result = {}
    v6_acceptance_path = ROOT / "experiments/clearance_v6_GMN/acceptance_result.json"
    if v6_acceptance_path.exists():
        v6_acceptance = json.loads(v6_acceptance_path.read_text(encoding="utf-8"))
        legacy_health_result = deepcopy(v6_acceptance.get("health", {}))
    config["legacy_calibration"] = {
        "model_version": "v6",
        "record": old_calibration,
        "excitation": old_excitation,
        "health_result": legacy_health_result,
        "health_result_source": "experiments/clearance_v6_GMN/acceptance_result.json",
        "model_status": old_model_status,
        "note": "Conditional V6 health calibration retained as the experiment-A U source; not a measured residual unbalance.",
    }

    # Keep the V6 calibrated U and force for experiment A. V8 calibration has
    # not started when this builder runs, so do not carry forward a pass state.
    config["calibration"] = {
        "status": "not_run",
        "contact_model": contact_model,
        "layout": layout,
        "rpm_signed": p["rpm"],
        "phase_rad": old_excitation.get("phase_rad"),
        "target_error_um": 2.0,
        "acceptance_range_um": [1.9, 2.1],
        "only_free_calibration_parameter": "excitation.U_kg_m",
        "U_star_kg_m": None,
        "F_star_N_at_7000": None,
        "measured_unbalance_claim_allowed": False,
    }
    config["schema_version"] = 8
    config["analysis_version"] = "v8"
    if source is None:
        config["base_config_source"] = "caller_supplied_actual_v6_config"
    else:
        try:
            config["base_config_source"] = source.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            config["base_config_source"] = str(source.resolve())
    p["bearing_contact_model"] = contact_model
    p["model_status"] = "v8_not_yet_calibrated_or_independently_validated"
    p["calibration_status"] = "not_run"
    p["nonlinear_bearings_v6"] = True
    p.pop("calibration_pass", None)

    config = layout_config(config, layout)
    config["calibration"]["layout"] = layout
    config["calibration"]["contact_model"] = contact_model
    config["simulation"]["maximum_warmup_s"] = 20.0
    config["excitation"]["source"] = (
        "legacy V6 conditional calibration retained for experiment A; V8 health recalibration not run"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layout", choices=tuple(_ARRANGEMENTS), default="DB_DB")
    parser.add_argument(
        "--contact-model", choices=tuple(sorted(CONTACT_MODELS)), default="relative_tilt_v8"
    )
    args = parser.parse_args()
    config = build_config(layout=args.layout, contact_model=args.contact_model, base_config=args.base)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; U={config['excitation']['U_kg_m']:.16g} kg·m; calibration={config['calibration']['status']}")


if __name__ == "__main__":
    main()
