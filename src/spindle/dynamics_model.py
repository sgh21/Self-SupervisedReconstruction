"""Four-bearing rotor mechanics for the 16/20-DOF installation models.

Small-angle A(z) kinematics, relative-tilt fixed-angle Hertz contacts,
unilateral outer-ring fits, and rotor gyroscopic moment. SI units internally.
The 12/16 mechanical coordinates become 16/20 after adding Cxy and Pxy.
This conditional model omits ball inertia and axial/outer-ring rotations.
"""

from copy import deepcopy
import numpy as np
from scipy.optimize import root

S, T, H, O = (slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8))
O2 = slice(10, 12)
OR1, OR2 = (slice(12, 14), slice(14, 16))
ROTOR = np.array([0, 1, 8, 9])


def is_rear_outer_fit(p):
    """The explicit four-outer-ring topology; zero rear gap is not binding."""
    mode = p.get("rear_outer_fit_model", "rear_bound_to_H")
    if mode not in ("rear_bound_to_H", "healthy_outer_to_H_fixed_assumption", "independent_xy"):
        raise ValueError("Unknown rear_outer_fit_model")
    independent = mode == "independent_xy"
    if independent and (
        p.get("model_dof") != 16
        or p.get("bearing_contact_model") != "relative_tilt_v8"
        or p.get("damping_policy") == "common_structural_ring"
    ):
        raise ValueError("independent_xy requires the 16-DOF relative_tilt_v8 topology")
    if p.get("model_dof") == 16 and (not independent):
        raise ValueError("The 16-DOF base requires rear_outer_fit_model=independent_xy")
    return independent


def base_mechanical_dof(p):
    if p.get("model_dof") not in (12, 16):
        raise ValueError("Only 12/16 mechanical DOF plus 4 installation DOF are supported")
    return 16 if is_rear_outer_fit(p) else 12


def base_state_names(p):
    names = (
        "Sx",
        "Sy",
        "Tx",
        "Ty",
        "Hx",
        "Hy",
        "O1x",
        "O1y",
        "theta_x",
        "theta_y",
        "O2x",
        "O2y",
        "OR1x",
        "OR1y",
        "OR2x",
        "OR2y",
    )
    return names[: base_mechanical_dof(p)]


def base_translation_pairs(p):
    return tuple((i for i in (0, 2, 4, 6, 10, 12, 14) if i < base_mechanical_dof(p)))


def rear_fit_parameters(p, name):
    """Adapt one independent rear fit to the existing unilateral contact law."""
    if not is_rear_outer_fit(p) or name not in ("R1", "R2"):
        raise ValueError("Rear fit parameters require independent_xy and R1 or R2")
    ring = p["rear_outer_rings"][name]
    values = (
        ring["fit_direction_count"],
        ring["fit_stiffness_N_m"],
        ring["fit_damping_N_s_m"],
        ring["clearance_um"],
    )
    alias = p.get(f"c_{name}_um", ring["clearance_um"])
    if alias != ring["clearance_um"]:
        raise ValueError(f"Conflicting {name} clearance aliases")
    key = "_rear_fit_" + name
    if _cache_enabled(p) and p.get(key + "_key") == values:
        return p[key]
    count, k, d, gap = values
    if int(count) != count or count < 4 or count % 2 or (k <= 0) or (d < 0) or (gap < 0):
        raise ValueError(f"{name} requires even N>=4, K>0, D>=0 and gap>=0")
    count = int(count)
    phi = 2 * np.pi * np.arange(count) / count
    fit = dict(
        N=count,
        Kf=k,
        Df=d,
        fit_clearance_um=gap,
        _fit_directions=np.column_stack((np.cos(phi), np.sin(phi))),
    )
    if _cache_enabled(p):
        p[key], p[key + "_key"] = (fit, values)
    return fit


def rear_fit_forces(q, v, p, summary=False):
    """R1/R2 fit loads transmitted to H, with independent local diagnostics."""
    q, v = (np.asarray(q), np.asarray(v))
    contact = fit_contact_summary if summary else fit_contact_force
    forces, diagnostics = ([], {})
    for name, outer in (("R1", OR1), ("R2", OR2)):
        force, state = contact(
            q[..., outer] - q[..., H], v[..., outer] - v[..., H], rear_fit_parameters(p, name)
        )
        forces.append(force)
        diagnostics[name] = state
    return (np.stack(forces, axis=-2), diagnostics)


def _cache_enabled(p):
    return bool(p.get("_derived_cache_enabled", False))


def rear_bearing_parameters(p):
    """6007 constitutive data; spin is shared, elastic scenario factor is explicit."""
    source = p["rear_bearing"]
    key = (
        p["rpm"],
        p.get("rear_elastic_force_factor", 1.0),
        tuple(sorted(((name, repr(value)) for name, value in source.items()))),
    )
    if _cache_enabled(p) and p.get("_rear_bearing_parameters_key") == key:
        return p["_rear_bearing_parameters"]
    rear = dict(p["rear_bearing"], rpm=p["rpm"])
    rear["Keq"] *= p.get("rear_elastic_force_factor", 1.0)
    if _cache_enabled(p):
        p["_rear_bearing_parameters"] = rear
        p["_rear_bearing_parameters_key"] = key
    return rear


def front_group_parameters(p, clearance_um):
    """One public degradation parameter; the single-front option is diagnostic."""
    local = dict(
        p,
        front_group_clearance_um=float(clearance_um),
        fit_clearance_um=float(clearance_um),
        fit2_clearance_um=0.0 if p.get("diagnostic_single_front", False) else float(clearance_um),
    )
    local.pop("_front2_constants", None)
    local.pop("_front2_constants_key", None)
    return local


def bearing_contact_model(p):
    base_mechanical_dof(p)
    if p.get("bearing_contact_model") != "relative_tilt_v8" or not p.get(
        "nonlinear_bearings_v6", False
    ):
        raise ValueError("Only the four-bearing relative_tilt_v8 model is supported")
    return "relative_tilt_v8"


def _bearing_layout_signature(p):
    layout = p.get("bearing_layout")
    if not isinstance(layout, (list, tuple)):
        raise ValueError("Explicit bearing contact models require bearing_layout")
    required = (
        "name",
        "z_m",
        "orientation_sign",
        "rho_m",
        "outer_translation_source",
        "outer_tilt_constraint",
        "axial_increment_constraint",
        "arrangement_basis",
    )
    values = []
    for item in layout:
        if not isinstance(item, dict) or any((key not in item for key in required)):
            raise ValueError(
                "Each bearing_layout entry needs the complete V8 geometry and constraint fields"
            )
        values.append(tuple(((key, repr(item[key])) for key in required)))
    return tuple(values)


def _bearing_parameter_signature(p):
    return (
        p.get("rpm"),
        p.get("balls"),
        p.get("ball_diameter"),
        p.get("pitch_diameter"),
        p.get("contact_angle"),
        p.get("normal_overlap"),
        bearing_keq(p),
        p.get("cb_total"),
        p.get("cage_phase"),
    )


def _bearing_context(p):
    """Validated four-bearing layout and material parameters, cached by their inputs."""
    model = bearing_contact_model(p)
    if model is None:
        raise ValueError("bearing_wrenches requires an explicit 12-DOF bearing contact model")
    layout_signature = _bearing_layout_signature(p)
    if p.get("bearing_position_basis") not in (
        "geometric_center_assumed",
        "geometric_center_assumed_not_identified_pressure_center",
    ):
        raise ValueError(
            "bearing_position_basis must explicitly identify geometric-center coordinates"
        )
    by_name = {item["name"]: item for item in p["bearing_layout"]}
    names = ("F1", "F2", "R1", "R2")
    if len(by_name) != 4 or set(by_name) != set(names):
        raise ValueError("bearing_layout must contain F1, F2, R1 and R2 exactly once")
    front = p
    rear = rear_bearing_parameters(p)
    front_signature = _bearing_parameter_signature(front)
    rear_signature = _bearing_parameter_signature(rear)
    key = (
        model,
        p.get("rear_outer_fit_model"),
        p.get("model_dof"),
        p["bearing_position_basis"],
        layout_signature,
        front_signature,
        rear_signature,
        p.get("rear_elastic_force_factor", 1.0),
    )
    if _cache_enabled(p) and p.get("_v8_bearing_context_key") == key:
        return p["_v8_bearing_context"]
    expected_sources = {"F1": "O1", "F2": "O2", "R1": "H", "R2": "H"}
    if is_rear_outer_fit(p):
        expected_sources.update(R1="OR1", R2="OR2")
    context = {}
    for name in names:
        spec = by_name[name]
        source = spec["outer_translation_source"]
        if source != expected_sources[name]:
            raise ValueError(f"{name} outer_translation_source must be {expected_sources[name]}")
        if (
            spec["outer_tilt_constraint"] != "fixed"
            or spec["axial_increment_constraint"] != "fixed"
        ):
            raise ValueError(f"{name} requires fixed outer tilt and axial increment constraints")
        if spec["orientation_sign"] not in (-1, 1) or spec["rho_m"] <= 0:
            raise ValueError(f"{name} requires orientation_sign of +/-1 and rho_m > 0")
        if not spec["arrangement_basis"]:
            raise ValueError(f"{name} requires an explicit arrangement_basis")
        params = dict(front) if name[0] == "F" else rear
        expected_material = "front" if name[0] == "F" else "rear"
        if spec.get("material_parameters", expected_material) != expected_material:
            raise ValueError(f"{name} must use the {expected_material} bearing parameters")
        context[name] = (spec, params)
    expected_z = {"F1": p["z_f"], "F2": p["z_f2"]}
    expected_z.update({s["name"]: s["z_m"] for s in p["rear_supports"]})
    for name, (spec, _) in context.items():
        if not np.isclose(spec["z_m"], expected_z[name], rtol=0, atol=1e-12):
            raise ValueError(f"{name} bearing_layout z_m must match the frozen 12-DOF geometry")
    if _cache_enabled(p):
        p["_v8_bearing_context"] = context
        p["_v8_bearing_context_key"] = key
    return context


def section_matrix(z):
    """Small-angle section kinematics u(z)=A(z)[Sx,Sy,theta_x,theta_y]."""
    return np.array([[1.0, 0.0, 0.0, z], [0.0, 1.0, -z, 0.0]])


def rotor_section(q, z):
    return np.asarray(q)[..., ROTOR] @ section_matrix(z).T


def generalized_point_force(force, z=0.0, n=10):
    """Virtual-work dual of section_matrix, including the physical moment arm."""
    force = np.asarray(force)
    result = np.zeros(force.shape[:-1] + (n,), dtype=np.result_type(force.dtype, float))
    if n in (10, 12, 16, 20):
        result[..., ROTOR] = force @ section_matrix(z)
    else:
        result[..., S] = force
    return result


def gyro_matrix(p):
    """Left-hand-side G for signed spin; G.T=-G and v.T@G@v=0."""
    n = base_mechanical_dof(p)
    matrix = np.zeros((n, n))
    value = p["Jp"] * p["rpm"] * 2 * np.pi / 60
    matrix[8, 9], matrix[9, 8] = (value, -value)
    return matrix


def _tilt_constants(p):
    """Constants are cached on the local parameter copy during integration."""
    model = bearing_contact_model(p)
    layout = _bearing_context(p)
    f1_z = layout["F1"][0]["z_m"]
    rear_supports = [layout[name][0] for name in ("R1", "R2")]
    key = (
        p.get("z_T"),
        f1_z,
        model,
        p.get("model_dof"),
        p.get("rpm"),
        p.get("Jp"),
        p.get("kappa"),
        p.get("contact_diameter"),
        p.get("contact_length"),
        p.get("kbend"),
        tuple((float(s["z_m"]) for s in rear_supports)),
        bool(p.get("nonlinear_bearings_v6", False)),
    )
    if _cache_enabled(p) and p.get("_tilt_constants_key") == key:
        return p["_tilt_constants"]
    nonlinear = p.get("nonlinear_bearings_v6", False)
    supports = rear_supports
    constants = (
        section_matrix(f1_z),
        section_matrix(p["z_T"]),
        np.array([section_matrix(s["z_m"]) for s in supports]),
        np.zeros((len(supports), 2, 2)) if nonlinear else np.array([s["K"] for s in supports]),
        np.zeros((len(supports), 2, 2)) if nonlinear else np.array([s["C"] for s in supports]),
        gyro_matrix(p),
        area_and_link(p, 0)[1],
    )
    if _cache_enabled(p):
        p["_tilt_constants"] = constants
        p["_tilt_constants_key"] = key
    return constants


def _tilt_forces(t, q, v, p, constants):
    af, at, ah, kh, ch, _, kl = constants
    qr, vr = (q[ROTOR], v[ROTOR])
    contact_model = bearing_contact_model(p)
    wrench_set = None
    wrench_set = bearing_wrenches(t, q, v, p)
    bearing = wrench_set["F1"]["force_N"]
    link = kl * (at @ qr - q[T]) + p["cl"] * (at @ vr - v[T])
    healthy = np.stack([wrench_set[name]["force_N"] for name in ("R1", "R2")], axis=-2)
    fit, _ = fit_contact_force(q[O] - q[H], v[O] - v[H], p)
    return (bearing, link, healthy, fit, wrench_set)


def _front2_constants(p):
    model = bearing_contact_model(p)
    z_f2 = _bearing_context(p)["F2"][0]["z_m"]
    key = (
        z_f2,
        p["N"],
        p["Kf2"],
        p["Df2"],
        p["fit2_clearance_um"],
        model,
        tuple(np.asarray(p.get("Kb2", np.zeros((2, 2)))).ravel()),
        tuple(np.asarray(p.get("Cb2", np.zeros((2, 2)))).ravel()),
    )
    if _cache_enabled(p) and p.get("_front2_constants_key") == key:
        return p["_front2_constants"]
    fit = dict(N=p["N"], Kf=p["Kf2"], Df=p["Df2"], fit_clearance_um=p["fit2_clearance_um"])
    if "_fit_directions" in p:
        fit["_fit_directions"] = p["_fit_directions"]
    fit["_fit_coefficients"] = (4 * p["Kf2"] / p["N"], 4 * p["Df2"] / p["N"])
    constants = (section_matrix(z_f2), None, None, fit)
    if _cache_enabled(p):
        p["_front2_constants"] = constants
        p["_front2_constants_key"] = key
    return constants


def front2_forces(q, v, p, summary=False, t=0.0, wrench_set=None):
    """F2 inner reaction and independent O2-H fit; also supports saved arrays."""
    a, k, c, fit_p = _front2_constants(p)
    relative = np.asarray(q)[..., ROTOR] @ a.T - q[..., O2]
    velocity = np.asarray(v)[..., ROTOR] @ a.T - v[..., O2]
    b = (
        wrench_set["F2"]["force_N"]
        if wrench_set is not None
        else bearing_wrench_at(t, q, v, p, "F2")["force_N"]
    )
    contact = fit_contact_summary if summary else fit_contact_force
    fit, diagnostics = contact(q[..., O2] - q[..., H], v[..., O2] - v[..., H], fit_p)
    return (b, fit, diagnostics)


def area_and_link(p, lost_area):
    """m² -> N/m. Loss is axisymmetric equivalent area, not a local pit."""
    area = np.pi * p["contact_diameter"] * p["contact_length"]
    if not 0 <= lost_area < area:
        raise ValueError("lost_area must be in [0, nominal taper area)")
    kj = p["kappa"] * (area - lost_area)
    return (area, 1.0 / (1.0 / p["kbend"] + 1.0 / kj))


def cage_speed(p):
    omega = p["rpm"] * 2 * np.pi / 60
    return omega / 2 * (1 - p["ball_diameter"] / p["pitch_diameter"] * np.cos(p["contact_angle"]))


def bearing_keq(p):
    """Explicit V6 equivalent coefficient; old series-contact input is preserved."""
    return p["Keq"] if "Keq" in p else (p["ki"] ** (-2 / 3) + p["ko"] ** (-2 / 3)) ** (-1.5)


def bearing_wrench(
    t,
    relative_xy,
    relative_velocity_xy,
    relative_tilt_xy,
    relative_tilt_rate_xy,
    bearing_parameters,
    orientation_sign,
    rho_m,
):
    """Hertz restoring force and local moment from one relative-contact projection.

    Inputs support a leading batch/time shape and end in two radial or angular
    components. Returned force and moment also end in two components; the
    diagnostics retain one value per ball on the last axis.
    """
    if orientation_sign not in (-1, 1) or rho_m <= 0:
        raise ValueError("orientation_sign must be +/-1 and rho_m must be positive")
    relative_xy = np.asarray(relative_xy)
    relative_velocity_xy = np.asarray(relative_velocity_xy)
    relative_tilt_xy = np.asarray(relative_tilt_xy)
    relative_tilt_rate_xy = np.asarray(relative_tilt_rate_xy)
    if any(
        (
            value.shape[-1] != 2
            for value in (
                relative_xy,
                relative_velocity_xy,
                relative_tilt_xy,
                relative_tilt_rate_xy,
            )
        )
    ):
        raise ValueError("Bearing relative translations and tilts must end in XY pairs")
    key = (_bearing_parameter_signature(bearing_parameters), int(orientation_sign), float(rho_m))
    cache_enabled = _cache_enabled(bearing_parameters)
    constants = bearing_parameters.get("_bearing_wrench_constants") if cache_enabled else None
    if constants is None or bearing_parameters.get("_bearing_wrench_constants_key") != key:
        balls = int(bearing_parameters["balls"])
        angle = bearing_parameters["contact_angle"]
        ca, sa = (np.cos(angle), np.sin(angle))
        constants = (
            cage_speed(bearing_parameters),
            np.arange(balls) * (2 * np.pi / balls),
            ca,
            sa,
            bearing_keq(bearing_parameters),
            bearing_parameters["cb_total"] / (balls / 2 * ca**2),
            balls,
        )
        if cache_enabled:
            bearing_parameters["_bearing_wrench_constants"] = constants
            bearing_parameters["_bearing_wrench_constants_key"] = key
    cage, ball_angles, ca, sa, keq, cball, balls = constants
    phi = cage * np.asarray(t)[..., None] + bearing_parameters["cage_phase"] + ball_angles
    cosine, sine = (np.cos(phi), np.sin(phi))
    sign = float(orientation_sign)
    tilt_scale = sign * float(rho_m) * sa
    tilt_compression = tilt_scale * (
        relative_tilt_xy[..., 0, None] * sine - relative_tilt_xy[..., 1, None] * cosine
    )
    delta = (
        bearing_parameters["normal_overlap"]
        + ca * (relative_xy[..., 0, None] * cosine + relative_xy[..., 1, None] * sine)
        + tilt_compression
    )
    normal_velocity = ca * (
        relative_velocity_xy[..., 0, None] * cosine + relative_velocity_xy[..., 1, None] * sine
    ) + tilt_scale * (
        relative_tilt_rate_xy[..., 0, None] * sine - relative_tilt_rate_xy[..., 1, None] * cosine
    )
    compression = np.maximum(delta, 0)
    elastic = keq * compression**1.5
    normal = np.where(delta > 0, np.maximum(elastic + cball * normal_velocity, 0), 0)
    force = np.stack(
        (np.sum(normal * ca * cosine, axis=-1), np.sum(normal * ca * sine, axis=-1)), axis=-1
    )
    moment = np.stack(
        (
            np.sum(normal * tilt_scale * sine, axis=-1),
            np.sum(-normal * tilt_scale * cosine, axis=-1),
        ),
        axis=-1,
    )
    potential = np.sum(2 / 5 * keq * compression**2.5, axis=-1)
    dissipation = np.sum((normal - elastic) * normal_velocity, axis=-1)
    axial_restoring = np.sum(normal * sign * sa, axis=-1)
    diagnostics = dict(
        delta_m=delta,
        normal_force_N=normal,
        elastic_force_N=elastic,
        normal_velocity_m_s=normal_velocity,
        tilt_compression_m=tilt_compression,
        axial_restoring_N=axial_restoring,
        potential_energy_J=potential,
        dissipation_W=dissipation,
        active_ball_count=np.sum(normal > 0, axis=-1),
        delta_over_ball_diameter=delta / bearing_parameters["ball_diameter"],
        delta_over_overlap=delta / bearing_parameters["normal_overlap"],
        maximum_ball_force_N=np.max(normal, axis=-1),
        eta_tilt=float(rho_m)
        * sa
        * np.linalg.norm(relative_tilt_xy, axis=-1)
        / bearing_parameters["normal_overlap"],
    )
    return (force, moment, diagnostics)


def _bearing_wrench_from_context(t, q, v, name, context, model):
    spec, params = context[name]
    z = spec["z_m"]
    outer_index = {"O1": O, "O2": O2, "H": H, "OR1": OR1, "OR2": OR2}[
        spec["outer_translation_source"]
    ]
    relative_xy = rotor_section(q, z) - np.asarray(q)[..., outer_index]
    relative_velocity_xy = rotor_section(v, z) - np.asarray(v)[..., outer_index]
    tilt = np.asarray(q)[..., 8:10]
    tilt_rate = np.asarray(v)[..., 8:10]
    force, moment, diagnostics = bearing_wrench(
        t,
        relative_xy,
        relative_velocity_xy,
        tilt,
        tilt_rate,
        params,
        spec["orientation_sign"],
        spec["rho_m"],
    )
    return dict(force_N=force, moment_Nm=moment, diagnostics=diagnostics)


def bearing_wrench_at(t, q, v, p, name):
    """Evaluate one named bearing, suitable for chunked native contact export."""
    model = bearing_contact_model(p)
    context = _bearing_context(p)
    if name not in context:
        raise KeyError(f"Unknown bearing {name!r}")
    return _bearing_wrench_from_context(t, q, v, name, context, model)


def bearing_wrenches(t, q, v, p):
    """Evaluate F1/F2/R1/R2, returning named radial force, local moment and state."""
    model = bearing_contact_model(p)
    context = _bearing_context(p)
    return {
        name: _bearing_wrench_from_context(t, q, v, name, context, model)
        for name in ("F1", "F2", "R1", "R2")
    }


def fit_contact_force(relative, velocity, p):
    """Force transmitted to H by O, N; radial gap c is not bearing overlap.

    Directions are fixed to H. Supports (..., 2) arrays for saved diagnostics.
    At c=0, Df=0 and even N, Fc=Kf*relative exactly. Damping is unilateral;
    its effective global coefficient is not a constant Kelvin--Voigt element.
    """
    count = p["N"]
    if count < 4 or count % 2 or p["Kf"] <= 0 or (p["Df"] < 0) or (p["fit_clearance_um"] < 0):
        raise ValueError("Fit requires even N >= 4, Kf > 0, Df >= 0 and radial clearance >= 0")
    directions = p.get("_fit_directions")
    if directions is None or len(directions) != count:
        phi = 2 * np.pi * np.arange(count) / count
        directions = np.column_stack((np.cos(phi), np.sin(phi)))
    signed_gap = np.asarray(relative) @ directions.T - p["fit_clearance_um"] * 1e-06
    compression = np.maximum(signed_gap, 0)
    normal_velocity = np.asarray(velocity) @ directions.T
    coefficients = (4 * p["Kf"] / count, 4 * p["Df"] / count)
    elastic = coefficients[0] * compression
    normal = np.where(signed_gap > 0, np.maximum(elastic + coefficients[1] * normal_velocity, 0), 0)
    return (
        normal @ directions,
        dict(
            normal_force_N=normal,
            compression_m=compression,
            signed_gap_m=signed_gap,
            normal_velocity_m_s=normal_velocity,
            elastic_force_N=elastic,
            active_count=np.sum(normal > 0, axis=-1),
        ),
    )


def fit_contact_summary(relative, velocity, p):
    """Export the required contact diagnostics in blocks, without N-by-time storage."""
    forces = []
    active = []
    compression = []
    for first in range(0, len(relative), 8192):
        force, diag = fit_contact_force(
            relative[first : first + 8192], velocity[first : first + 8192], p
        )
        forces.append(force)
        active.append(diag["active_count"])
        compression.append(diag["compression_m"].max(axis=1))
    return (
        np.concatenate(forces),
        dict(active_count=np.concatenate(active), max_compression_m=np.concatenate(compression)),
    )


def mass_and_gravity(p):
    independent_rear = is_rear_outer_fit(p)
    if "platform_mass" in p:
        raise ValueError("Tilt rotor model cannot include platform_mass")
    if min(p["ms"], p["mt"], p["mh"], p["mO"], p["Jt"], p["Jp"]) <= 0:
        raise ValueError("Masses and rotor inertias must be positive")
    masses = np.r_[np.repeat([p["ms"], p["mt"], p["mh"], p["mO"]], 2), p["Jt"], p["Jt"]]
    gravity = np.r_[masses[:8] * np.tile(p["gravity_radial_m_s2"], 4), 0.0, 0.0]
    if p["mO2"] <= 0:
        raise ValueError("O2 mass must be positive")
    masses = np.r_[masses, [p["mO2"]] * 2]
    gravity = np.r_[gravity, p["mO2"] * np.asarray(p["gravity_radial_m_s2"])]
    if independent_rear:
        for name in ("R1", "R2"):
            mass = p["rear_outer_rings"][name]["mass_kg"]
            if mass <= 0:
                raise ValueError(f"{name} outer-ring mass must be positive")
            rear_fit_parameters(p, name)
            masses = np.r_[masses, mass, mass]
            gravity = np.r_[gravity, mass * np.asarray(p["gravity_radial_m_s2"])]
    static_load = [0.0, p.get("radial_load_y", 0.0)]
    gravity += generalized_point_force(static_load, p.get("radial_load_z_m", 0.0), len(masses))
    return (masses, gravity)


def internal_force(t, q, v, p, lost_area):
    """Restoring force; positive sign on left of Mq''+R=F."""
    independent_rear = is_rear_outer_fit(p)
    if lost_area != 0 or any(
        (
            p.get(key, 0) != 0
            for key in ("outer_pit_area_m2", "inner_pit_area_m2", "ball_deformation_amplitude_m")
        )
    ):
        raise ValueError("Tilt rotor model varies target fit clearance only")
    constants = _tilt_constants(p)
    bearing, link, healthy, fit, wrench_set = _tilt_forces(t, q, v, p, constants)
    af, at, ah, _, _, gyro, _ = constants
    force = np.zeros(len(gyro))
    force[ROTOR] = af.T @ bearing + at.T @ link + np.einsum("kij,ki->j", ah, healthy)
    force[T] = -link
    force[H] = -healthy.sum(axis=0) - fit + p["kh"] * q[H] + p["ch"] * v[H]
    if independent_rear:
        rear_fit, _ = rear_fit_forces(q, v, p)
        force[H] = -rear_fit.sum(axis=0) - fit + p["kh"] * q[H] + p["ch"] * v[H]
        force[OR1] = -healthy[0] + rear_fit[0]
        force[OR2] = -healthy[1] + rear_fit[1]
    force[O] = -bearing + fit
    b2, fit2, _ = front2_forces(q, v, p, t=t, wrench_set=wrench_set)
    force[ROTOR] += _front2_constants(p)[0].T @ b2
    force[H] -= fit2
    force[O2] = -b2 + fit2
    if wrench_set is not None:
        force[8:10] += sum((wrench_set[name]["moment_Nm"] for name in ("F1", "F2", "R1", "R2")))
    force += gyro @ v
    return force


def equilibrium(p, lost_area, initial_q=None):
    """Frozen t=0 equilibrium; positive gaps default to zero-gap continuation."""
    positive_gap = "mO" in p and p["fit_clearance_um"] > 0
    if is_rear_outer_fit(p):
        positive_gap = (
            positive_gap
            or p["fit2_clearance_um"] > 0
            or any((p["rear_outer_rings"][name]["clearance_um"] > 0 for name in ("R1", "R2")))
        )
    if positive_gap and initial_q is None:
        return equilibrium_continuation(p, lost_area)
    masses, gravity = mass_and_gravity(p)
    n = len(masses)
    qscale, fscale = (np.full(n, 1e-06), np.ones(n))
    qscale[8:10] /= p["L_ref"]
    fscale[8:10] = p["L_ref"]
    residual_function = lambda z: (
        (internal_force(0, z * qscale, np.zeros(n), p, lost_area) - gravity) / fscale / 100
    )
    jacobian = None
    if is_rear_outer_fit(p):
        difference_step = 1e-05
        basis = np.eye(n) * difference_step

        def jacobian(z):
            return np.column_stack(
                [
                    (residual_function(z + axis) - residual_function(z - axis))
                    / (2 * difference_step)
                    for axis in basis
                ]
            )

    solution = root(
        residual_function,
        np.zeros(n) if initial_q is None else np.asarray(initial_q) / qscale,
        tol=1e-10,
        jac=jacobian,
    )
    q = solution.x * qscale
    residual = np.linalg.norm((internal_force(0, q, np.zeros(n), p, lost_area) - gravity) / fscale)
    if not np.isfinite(residual) or residual > 1e-06:
        raise RuntimeError(f"equilibrium residual {residual:.3g} N")
    return (q, residual)


def equilibrium_continuation(
    p, lost_area=0, initial_q=None, start_clearance_um=0.0, max_increment_um=1.0
):
    """Continue the frozen loaded equilibrium; halve failed gap increments."""
    if is_rear_outer_fit(p):
        target = np.array(
            [
                p["fit_clearance_um"],
                p["fit2_clearance_um"],
                p["rear_outer_rings"]["R1"]["clearance_um"],
                p["rear_outer_rings"]["R2"]["clearance_um"],
            ],
            dtype=float,
        )
        start = (
            np.zeros(4)
            if initial_q is None
            else np.r_[
                start_clearance_um,
                0.0 if p.get("diagnostic_single_front", False) else start_clearance_um,
                target[2:],
            ]
        )

        def at_vector(values):
            local = dict(
                p,
                fit_clearance_um=float(values[0]),
                front_group_clearance_um=float(values[0]),
                fit2_clearance_um=float(values[1]),
                rear_outer_rings=deepcopy(p["rear_outer_rings"]),
            )
            for name, gap in zip(("R1", "R2"), values[2:]):
                local["rear_outer_rings"][name]["clearance_um"] = float(gap)
                if f"c_{name}_um" in local:
                    local[f"c_{name}_um"] = float(gap)
            return local

        q, residual = equilibrium(
            at_vector(start), lost_area, np.zeros(16) if initial_q is None else initial_q
        )
        history = [
            dict(fraction=0.0, gaps_um=start.tolist(), residual_equivalent_N=float(residual))
        ]
        size = np.max(abs(target - start))
        step = 1.0 if size == 0 else min(1.0, max_increment_um / size)
        fraction = 0.0
        while fraction < 1.0:
            trial = min(1.0, fraction + step)
            values = start + trial * (target - start)
            try:
                next_q, residual = equilibrium(at_vector(values), lost_area, q)
            except RuntimeError:
                history.append(
                    dict(fraction=float(trial), gaps_um=values.tolist(), status="retry_half_step")
                )
                step /= 2
                if step * max(size, 1.0) < 1e-06:
                    p["_equilibrium_continuation_history"] = history
                    raise
                continue
            q, fraction = (next_q, trial)
            history.append(
                dict(
                    fraction=float(fraction),
                    gaps_um=values.tolist(),
                    residual_equivalent_N=float(residual),
                )
            )
            step = min(1.0 if size == 0 else max_increment_um / size, step * 2)
        p["_equilibrium_continuation_history"] = history
        return (q, residual)
    target = p["front_group_clearance_um"]

    def at_gap(value):
        return front_group_parameters(p, value)

    current = start_clearance_um
    q, residual = equilibrium(at_gap(current), lost_area, initial_q)
    increment = max_increment_um
    while current < target:
        trial = min(target, current + increment)
        try:
            q_next, residual = equilibrium(at_gap(trial), lost_area, q)
        except RuntimeError:
            increment /= 2
            if increment < 1e-06:
                raise
            continue
        q, current = (q_next, trial)
        increment = min(max_increment_um, increment * 2)
    return (q, residual)
