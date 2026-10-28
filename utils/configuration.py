"""Complete runtime configurations and explicit physical parameter changes."""

from copy import deepcopy
import numpy as np

from utils.io import load_json
from spindle.installation_dynamics import InstallationModel

PARAMETER_NAMES = ("kHC_N_m", "kCP_N_m", "kPGx_N_m", "kPGy_N_m", "zeta_link", "zeta_foot")


def load_config(path):
    config = load_json(path)
    model = InstallationModel(config)
    if config["structural_dof"] != model.structural_dof:
        raise ValueError("Configured total DOF differs from mechanical topology")
    if config["probe_reference"] != "platform":
        raise ValueError("Current runout contract requires M relative to platform P")
    parameters = config["base_v8"]["parameters"]
    gaps = config["comparison_identity"]
    front = parameters["fit_clearance_um"]
    if any(
        value != front
        for value in (
            parameters["fit2_clearance_um"],
            parameters["front_group_clearance_um"],
            gaps["c_F1_um"],
            gaps["c_F2_um"],
        )
    ):
        raise ValueError("Front pair clearance aliases disagree")
    for name in ("R1", "R2"):
        expected = (
            parameters["rear_outer_rings"][name]["clearance_um"]
            if model.structural_dof == 20
            else None
        )
        if gaps["c_" + name + "_um"] != expected:
            raise ValueError("Rear clearance metadata disagrees with physical parameters")
    return config


def with_clearances(config, front_um, rear_um=None):
    """Change the front pair; 20 DOF requires two explicit rear clearances."""
    if config["structural_dof"] == 20 and rear_um is None:
        raise ValueError("20 DOF requires explicit rear_um=(R1, R2); rear gaps never follow implicitly")
    result = deepcopy(config)
    p = result["base_v8"]["parameters"]
    front_um = float(front_um)
    if front_um < 0 or not np.isfinite(front_um):
        raise ValueError("Clearance must be finite and nonnegative")
    p.update(
        front_group_clearance_um=front_um, fit_clearance_um=front_um, fit2_clearance_um=front_um
    )
    result["comparison_identity"].update(c_F1_um=front_um, c_F2_um=front_um)
    if result["structural_dof"] == 20:
        rear = tuple(rear_um)
        if len(rear) != 2 or any(not np.isfinite(x) or x < 0 for x in rear):
            raise ValueError("Supply two finite nonnegative rear clearances")
        for name, gap in zip(("R1", "R2"), rear):
            p["rear_outer_rings"][name]["clearance_um"] = float(gap)
            if "c_" + name + "_um" in p:
                p["c_" + name + "_um"] = float(gap)
            result["comparison_identity"]["c_" + name + "_um"] = float(gap)
    elif rear_um is not None:
        raise ValueError("The 16-DOF model binds rear rings to H and has no rear fit gaps")
    result["case_id"] = f"model{result['structural_dof']}_F_{front_um:g}um"
    if result["structural_dof"] == 20:
        result["case_id"] += f"_R1_{rear[0]:g}um_R2_{rear[1]:g}um"
    return result


def update_installation(config, parameters):
    """Recompute actual K/C when fitting; untouched frozen configurations retain their exact matrices."""
    result = deepcopy(config)
    values = parameters if isinstance(parameters, dict) else dict(zip(PARAMETER_NAMES, parameters))
    inst = result["installation"]
    for name, value in values.items():
        low, high = inst["bounds"][name]
        if not low <= value <= high:
            raise ValueError(f"{name} outside [{low}, {high}]")
        inst["parameters"][name] = float(value)
    for name in ("actual_link_stiffness_N_m", "actual_link_damping_N_s_m", "actual_matrix_policy"):
        inst.pop(name, None)
    model = InstallationModel(result)
    inst.update(
        actual_link_stiffness_N_m=model.link_stiffness.tolist(),
        actual_link_damping_N_s_m=model.link_damping.tolist(),
        actual_matrix_policy="recomputed_from_candidate_parameters",
    )
    return result


def decode_unit(unit, config):
    physical = np.asarray([config["installation"]["bounds"][name] for name in PARAMETER_NAMES])
    bounds = np.log(physical)
    values = np.exp(bounds[:, 0] + np.clip(unit, 0, 1) * np.diff(bounds, axis=1)[:, 0])
    return np.clip(values, physical[:, 0], physical[:, 1])


def encode_unit(config):
    bounds = np.log([config["installation"]["bounds"][name] for name in PARAMETER_NAMES])
    values = [config["installation"]["parameters"][name] for name in PARAMETER_NAMES]
    return (np.log(values) - bounds[:, 0]) / np.diff(bounds, axis=1)[:, 0]
