"""Small radial spindle model; all mechanical equations are in this file.

SI units, fixed frame: Z spindle axis, X/Y radial. q = [shaft XY,
holder-section XY, front-housing XY], optionally followed by platform Y or
outer-ring XY. The latter eight-DOF branch is enabled by mO and cannot include
the platform. It uses one fixed front equivalent bearing, S--O rolling contact,
O--H radial fit contact, and a fixed healthy S--H support allocated at baseline.
Inner ring is fixed to shaft; in the legacy branch the outer ring
is fixed to the housing node: one equivalent radial support, not four real
bearings. No shaft-inner-ring fit gap is modeled. H is never the shaft.
T is the holder-section mass node. Legacy branches have translations only.
The explicit model_dof=10 branch adds rotor theta_x/theta_y at indices 8/9,
spaced healthy supports and gyroscopic coupling; it has no local housing
modes, ball inertia or axial preload solve. S is then the rotor mass centre.
This is a documented reduction, NOT a reproduction of full SpindlePro.

Primary sources (local PDFs in ../../docs/references/papers/):
Tai/Altintas 2024 Part I, 081004-3 Eq(1): Hertz inner/outer contact;
081004-5 Eqs(21),(22): ball/cage phase; 081004-8 Eqs(41),(42):
lost taper contact area removes spring energy. Namazi et al. 2007,
p1335 Eqs(6),(8): k_r*dA and contact energy. Cao/Altintas 2004,
p1094 Eq(43): assemble actual nonlinear restoring forces.

Optional localized race pits: Tai I p5 Eqs(19),(20),(23)-(25),
with a documented full-angle convention cross-checked against Liu/Shao 2017
p89 Eqs(10),(11). Pit area is separate from lost taper contact area.
Single-ball second-order deformation: Tai I p7 Eq(33), retaining w=2 only;
the two antipodal contact offsets add in this massless, equal-angle reduction.
Spin speed uses the fixed-angle, no-slip approximation in Tai I p16 Eq(53).

Reductions derived here:
massless series contacts: delta=(Q/Ki)**(2/3)+(Q/Ko)**(2/3), hence
Keq=(Ki**(-2/3)+Ko**(-2/3))**(-3/2). Equal, fixed contact angles assumed.
delta_k=delta0+cos(alpha)*n_k dot(qs-qh), Q=max(0,Keq*delta+**1.5
+ cball*cos(alpha)*n_k dot(vs-vh)) on active contact, B=sum(Q*cos(alpha)*n_k).
The damping uses projected relative vibration velocity, NOT the complete time
derivative of rotating-contact compression (which also contains n_dot dot r).
delta0 is ASSUMED normal overlap, NOT identified axial preload or clearance.
Uniform relative taper translation gives Uj=0.5*kappa*Aeff*|qs-qt|².
Aj=A0-A_loss; Kj=kappa*Aj. Series bending/contact stiffness:
Kl=(Kbend**-1+Kj**-1)**-1. Link damping is a positive phenomenological
constant (not inferred microscopic damping). All internal reactions balance.
ms*qs''=F+ms*g-B-L; mt*qt''=mt*g+L;
mh*qh''=mh*g+B-Kh*qh-Ch*qh'. L=Kl*(qs-qt)+Cl*(qs'-qt').
With platform_mass supplied, housing Y is supported relative to platform Y;
the equal opposite support force loads the platform's ground spring/damper.
Housing X retains its original ground support. The platform has translation
only; no rocking or local plate deformation is represented.

Eight-DOF extension: Cao et al. 2019 PDF p4 section 2.3 Eq(7) supports
projected outer-ring/housing approach minus radial fit clearance. The discrete
weights 4*Kf/N, damping clipping, and healthy-support allocation are this
project's implementation choices, not the paper's equations. Tai I PDF p8
Eqs(49),(50) gives rotating unbalance U*omega**2*[cos(theta),sin(theta)].
S relative to H is a fixed-section center trajectory including 1X, not the
fundamental-removed error-motion metric used by Cao on PDF p7.

10-DOF extension (model_dof=10 or Jt supplied): S is rotor COM and the final
two coordinates are theta_x/theta_y (rad). A(z)=[[1,0,0,z],[0,1,-z,0]] maps
rotor coordinates to each section; A.T maps forces back. Jt/Jp (kg m²) include
only S's body. Healthy supports have fixed {z_m,K,C}; their tight or ideal
zero-clearance fits have no opening law in this demonstration. z_f, z_T,
z_u and z_m (m) locate target bearing, holder link, force and measurement.
L_ref (m) scales equilibrium moments and angular increments. Radial gravity
is explicitly gravity_radial_m_s2; radial_load_y acts at radial_load_z_m
(default COM). probe_reference selects ground or housing; measurement_kind
selects rotor_section or holder. H has constrained orientation. These are
rigid-body/virtual-work reductions, not equations copied from Cao or Tai.

V6 is explicitly enabled by nonlinear_bearings_v6 on the 12-DOF branch.
Both front bearings share the same top-level Hertz parameters and explicit
Keq; each rear_supports position uses rear_bearing parameters against H.
The two rear contacts add no states. Old healthy_supports/Kother/Kb2 linear
paths are inactive in V6. Fixed contact angle, equal inner/outer contact load,
prescribed overlap and a housing-fixed rear outer race remain reductions.
Neither the supplied Keq nor the split outer-ring masses are manufacturer
measurements of the actual assembled spindle.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import CubicSpline

S, T, H, O = slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)
O2 = slice(10, 12)
ROTOR = np.array([0, 1, 8, 9])


def is_tilt_model(p):
    return p.get('model_dof') in (10, 12) or 'Jt' in p


def is_front_group(p):
    return p.get('model_dof') == 12


def _cache_enabled(p):
    return bool(p.get('_derived_cache_enabled', False))


def rear_bearing_parameters(p):
    """6007 constitutive data; spin is shared, elastic scenario factor is explicit."""
    source = p['rear_bearing']
    key = (p['rpm'], p.get('rear_elastic_force_factor', 1.0),
           tuple(sorted((name, repr(value)) for name, value in source.items())))
    if _cache_enabled(p) and p.get('_rear_bearing_parameters_key') == key:
        return p['_rear_bearing_parameters']
    rear = dict(p['rear_bearing'], rpm=p['rpm'])
    rear['Keq'] *= p.get('rear_elastic_force_factor', 1.0)
    if _cache_enabled(p):
        p['_rear_bearing_parameters'] = rear
        p['_rear_bearing_parameters_key'] = key
    return rear


def rear_bearing_forces(t, q, v, p):
    """R1/R2 forces transmitted to H, shape (..., 2 supports, 2 directions)."""
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        from .tai_system import bearing_wrenches as separate_wrenches
        wr = separate_wrenches(t, q, v, p)
        return np.stack([-wr[name]['outer_restoring_6'][..., :2] for name in ('R1','R2')], axis=-2)
    if p.get('bearing_contact_model') == 'relative_tilt_v8':
        values = [bearing_wrench_at(t, q, v, p, name)['force_N'] for name in ('R1', 'R2')]
        return np.stack(values, axis=-2)
    rear = rear_bearing_parameters(p)
    return np.stack([bearing_force(t, rotor_section(q, s['z_m'])-q[..., H],
                                  rotor_section(v, s['z_m'])-v[..., H], rear)
                     for s in p['rear_supports']], axis=-2)


def front_group_parameters(p, clearance_um):
    """One public degradation parameter; the single-front option is diagnostic."""
    local = dict(p, front_group_clearance_um=float(clearance_um),
                 fit_clearance_um=float(clearance_um),
                 fit2_clearance_um=0. if p.get('diagnostic_single_front', False) else float(clearance_um))
    local.pop('_front2_constants', None)
    local.pop('_front2_constants_key', None)
    return local


def bearing_contact_model(p):
    """Return the explicit 12-DOF contact switch, or None for older branches."""
    model = p.get('bearing_contact_model')
    if model is None:
        return 'radial_only_v6' if is_front_group(p) and p.get('nonlinear_bearings_v6', False) else None
    if model not in ('radial_only_v6', 'relative_tilt_v8', 'quasistatic_tai_v9'):
        raise ValueError('Unknown explicit bearing contact model')
    if not is_front_group(p) or not p.get('nonlinear_bearings_v6', False):
        raise ValueError('Explicit bearing contact models require the nonlinear 12-DOF bearing branch')
    return model


def _bearing_layout_signature(p):
    layout = p.get('bearing_layout')
    if not isinstance(layout, (list, tuple)):
        raise ValueError('Explicit bearing contact models require bearing_layout')
    required = ('name', 'z_m', 'orientation_sign', 'rho_m',
                'outer_translation_source', 'outer_tilt_constraint',
                'axial_increment_constraint', 'arrangement_basis')
    values = []
    for item in layout:
        if not isinstance(item, dict) or any(key not in item for key in required):
            raise ValueError('Each bearing_layout entry needs the complete V8 geometry and constraint fields')
        values.append(tuple((key, repr(item[key])) for key in required))
    return tuple(values)


def _bearing_parameter_signature(p):
    return (p.get('rpm'), p.get('balls'), p.get('ball_diameter'),
            p.get('pitch_diameter'), p.get('contact_angle'), p.get('normal_overlap'),
            bearing_keq(p), p.get('cb_total'), p.get('cage_phase'))


def _bearing_context(p):
    """Validated four-bearing layout and material parameters, cached by their inputs."""
    model = bearing_contact_model(p)
    if model is None:
        raise ValueError('bearing_wrenches requires an explicit 12-DOF bearing contact model')
    layout_signature = _bearing_layout_signature(p)
    if p.get('bearing_position_basis') not in (
            'geometric_center_assumed', 'geometric_center_assumed_not_identified_pressure_center'):
        raise ValueError('bearing_position_basis must explicitly identify geometric-center coordinates')
    by_name = {item['name']: item for item in p['bearing_layout']}
    names = ('F1', 'F2', 'R1', 'R2')
    if len(by_name) != 4 or set(by_name) != set(names):
        raise ValueError('bearing_layout must contain F1, F2, R1 and R2 exactly once')
    front = p
    rear = rear_bearing_parameters(p)
    front_signature = _bearing_parameter_signature(front)
    rear_signature = _bearing_parameter_signature(rear)
    key = (model, p['bearing_position_basis'], layout_signature,
           front_signature, rear_signature, p.get('rear_elastic_force_factor', 1.0))
    if _cache_enabled(p) and p.get('_v8_bearing_context_key') == key:
        return p['_v8_bearing_context']
    expected_sources = {'F1': 'O1', 'F2': 'O2', 'R1': 'H', 'R2': 'H'}
    context = {}
    for name in names:
        spec = by_name[name]
        source = spec['outer_translation_source']
        if source != expected_sources[name]:
            raise ValueError(f'{name} outer_translation_source must be {expected_sources[name]}')
        if spec['outer_tilt_constraint'] != 'fixed' or spec['axial_increment_constraint'] != 'fixed':
            raise ValueError(f'{name} requires fixed outer tilt and axial increment constraints')
        if spec['orientation_sign'] not in (-1, 1) or spec['rho_m'] <= 0:
            raise ValueError(f'{name} requires orientation_sign of +/-1 and rho_m > 0')
        if not spec['arrangement_basis']:
            raise ValueError(f'{name} requires an explicit arrangement_basis')
        params = dict(front) if name[0] == 'F' else rear
        expected_material = 'front' if name[0] == 'F' else 'rear'
        if spec.get('material_parameters', expected_material) != expected_material:
            raise ValueError(f'{name} must use the {expected_material} bearing parameters')
        context[name] = (spec, params)
    expected_z = {'F1': p['z_f'], 'F2': p['z_f2']}
    expected_z.update({s['name']: s['z_m'] for s in p['rear_supports']})
    for name, (spec, _) in context.items():
        if not np.isclose(spec['z_m'], expected_z[name], rtol=0, atol=1e-12):
            raise ValueError(f'{name} bearing_layout z_m must match the frozen 12-DOF geometry')
    if _cache_enabled(p):
        p['_v8_bearing_context'] = context
        p['_v8_bearing_context_key'] = key
    return context


def section_matrix(z):
    """Small-angle section kinematics u(z)=A(z)[Sx,Sy,theta_x,theta_y]."""
    return np.array([[1., 0., 0., z], [0., 1., -z, 0.]])


def rotor_section(q, z):
    return np.asarray(q)[..., ROTOR] @ section_matrix(z).T


def generalized_point_force(force, z=0.0, n=10):
    """Virtual-work dual of section_matrix, including the physical moment arm."""
    force = np.asarray(force)
    result = np.zeros(force.shape[:-1]+(n,), dtype=np.result_type(force.dtype, float))
    if n in (10, 12):
        result[..., ROTOR] = force @ section_matrix(z)
    else:
        result[..., S] = force
    return result


def gyro_matrix(p):
    """Left-hand-side G for signed spin; G.T=-G and v.T@G@v=0."""
    n = 12 if is_front_group(p) else 10
    matrix = np.zeros((n, n))
    value = p['Jp'] * p['rpm'] * 2*np.pi/60
    matrix[8, 9], matrix[9, 8] = value, -value
    return matrix


def _tilt_constants(p):
    """Constants are cached on the local parameter copy during integration."""
    model = bearing_contact_model(p)
    if model == 'relative_tilt_v8':
        layout = _bearing_context(p)
        f1_z = layout['F1'][0]['z_m']
        rear_supports = [layout[name][0] for name in ('R1', 'R2')]
    else:
        f1_z = p['z_f']
        rear_supports = p['rear_supports'] if p.get('nonlinear_bearings_v6', False) else p['healthy_supports']
    key = (p.get('z_T'), f1_z, model, p.get('model_dof'),
           p.get('rpm'), p.get('Jp'), p.get('kappa'),
           p.get('contact_diameter'), p.get('contact_length'), p.get('kbend'),
           tuple(float(s['z_m']) for s in rear_supports),
           bool(p.get('nonlinear_bearings_v6', False)))
    if _cache_enabled(p) and p.get('_tilt_constants_key') == key:
        return p['_tilt_constants']
    nonlinear = p.get('nonlinear_bearings_v6', False)
    supports = rear_supports
    constants = (section_matrix(f1_z), section_matrix(p['z_T']),
            np.array([section_matrix(s['z_m']) for s in supports]),
            np.zeros((len(supports), 2, 2)) if nonlinear else np.array([s['K'] for s in supports]),
            np.zeros((len(supports), 2, 2)) if nonlinear else np.array([s['C'] for s in supports]),
            gyro_matrix(p), area_and_link(p, 0)[1])
    if _cache_enabled(p):
        p['_tilt_constants'] = constants
        p['_tilt_constants_key'] = key
    return constants


def _tilt_forces(t, q, v, p, constants):
    af, at, ah, kh, ch, _, kl = constants
    qr, vr = q[ROTOR], v[ROTOR]
    contact_model = bearing_contact_model(p)
    wrench_set = None
    if contact_model == 'relative_tilt_v8':
        wrench_set = bearing_wrenches(t, q, v, p)
        bearing = wrench_set['F1']['force_N']
    else:
        bearing = bearing_force(t, af@qr-q[O], af@vr-v[O], p)
    link = kl*(at@qr-q[T]) + p['cl']*(at@vr-v[T])
    if contact_model == 'relative_tilt_v8':
        healthy = np.stack([wrench_set[name]['force_N'] for name in ('R1', 'R2')], axis=-2)
    elif p.get('nonlinear_bearings_v6', False):
        healthy = rear_bearing_forces(t, q, v, p)
    else:
        healthy = (np.einsum('kij,kj->ki', kh, ah@qr-q[H])
                   + np.einsum('kij,kj->ki', ch, ah@vr-v[H]))
    fit, _ = fit_contact_force(q[O]-q[H], v[O]-v[H], p)
    return bearing, link, healthy, fit, wrench_set


def _front2_constants(p):
    model = bearing_contact_model(p)
    z_f2 = _bearing_context(p)['F2'][0]['z_m'] if model == 'relative_tilt_v8' else p['z_f2']
    key = (z_f2, p['N'], p['Kf2'], p['Df2'], p['fit2_clearance_um'], model,
           tuple(np.asarray(p.get('Kb2', np.zeros((2, 2)))).ravel()),
           tuple(np.asarray(p.get('Cb2', np.zeros((2, 2)))).ravel()))
    if _cache_enabled(p) and p.get('_front2_constants_key') == key:
        return p['_front2_constants']
    fit = dict(N=p['N'], Kf=p['Kf2'], Df=p['Df2'], fit_clearance_um=p['fit2_clearance_um'])
    if '_fit_directions' in p:
        fit['_fit_directions'] = p['_fit_directions']
    fit['_fit_coefficients'] = (4*p['Kf2']/p['N'], 4*p['Df2']/p['N'])
    if p.get('nonlinear_bearings_v6', False):
        constants = (section_matrix(z_f2), None, None, fit)
    else:
        constants = (section_matrix(z_f2), np.asarray(p['Kb2']), np.asarray(p['Cb2']), fit)
    if _cache_enabled(p):
        p['_front2_constants'] = constants
        p['_front2_constants_key'] = key
    return constants


def front2_forces(q, v, p, summary=False, t=0.0, wrench_set=None):
    """F2 inner reaction and independent O2-H fit; also supports saved arrays."""
    a, k, c, fit_p = _front2_constants(p)
    relative = np.asarray(q)[..., ROTOR]@a.T-q[..., O2]
    velocity = np.asarray(v)[..., ROTOR]@a.T-v[..., O2]
    if bearing_contact_model(p) in ('relative_tilt_v8','quasistatic_tai_v9'):
        b = (wrench_set['F2']['force_N'] if wrench_set is not None
             else bearing_wrench_at(t, q, v, p, 'F2')['force_N'])
    elif p.get('nonlinear_bearings_v6', False):
        b = bearing_force(t, relative, velocity, p)
    else:
        b = relative@k.T + velocity@c.T
    contact = fit_contact_summary if summary else fit_contact_force
    fit, diagnostics = contact(q[..., O2]-q[..., H], v[..., O2]-v[..., H], fit_p)
    return b, fit, diagnostics


def force_components(t, q, v, p):
    """10/12-DOF physical forces, restoring sign; H orientation is constrained.

    Healthy support forces act on H; the opposite acts on the rotor. The
    returned constraint moment balances their and the fit force's moments
    about the S axial plane. No local housing rotation is integrated.
    """
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        from .tai_system import force_components as separate_components
        return separate_components(t,q,v,p)
    bearing, link, healthy, fit, wrench_set = _tilt_forces(t, q, v, p, _tilt_constants(p))
    contact_model = bearing_contact_model(p)
    explicit_contact_model = 'bearing_contact_model' in p
    if explicit_contact_model and wrench_set is None:
        wrench_set = bearing_wrenches(t, q, v, p)
    f1_z = (_bearing_context(p)['F1'][0]['z_m'] if explicit_contact_model
            else p['z_f'])
    housing_moment = np.array([-f1_z*fit[1], f1_z*fit[0]])
    supports = ([ _bearing_context(p)[name][0] for name in ('R1', 'R2') ]
                if contact_model == 'relative_tilt_v8' else
                p['rear_supports'] if p.get('nonlinear_bearings_v6', False)
                else p['healthy_supports'])
    for item, force in zip(supports, healthy):
        housing_moment += [-item['z_m']*force[1], item['z_m']*force[0]]
    result = dict(bearing_N=bearing, healthy_support_forces_N=healthy, link_N=link,
                fit_N=fit,
                housing_ground_N=p['kh']*q[H]+p['ch']*v[H],
                housing_constraint_torque_Nm=-housing_moment)
    if is_front_group(p):
        b2, fit2, _ = front2_forces(q, v, p, t=t, wrench_set=wrench_set)
        result.update(bearing2_N=b2, fit2_N=fit2)
        f2_z = (_bearing_context(p)['F2'][0]['z_m'] if explicit_contact_model
                else p['z_f2'])
        housing_moment += np.array([-f2_z*fit2[1], f2_z*fit2[0]])
    if p.get('nonlinear_bearings_v6', False):
        result['rear_bearing_forces_N'] = healthy
    result['housing_constraint_torque_Nm'] = -housing_moment
    if explicit_contact_model:
        local_moments = {name: wrench_set[name]['moment_Nm']
                         for name in ('F1', 'F2', 'R1', 'R2')}
        axial_reactions = {name: -wrench_set[name]['diagnostics']['axial_restoring_N']
                           for name in ('F1', 'F2', 'R1', 'R2')}
        result.update(bearing_wrenches=wrench_set,
            bearing_local_moments_Nm=local_moments,
            outer_rotation_constraint_moment_Nm_by_bearing={
                name: -moment for name, moment in local_moments.items()},
            axial_constraint_reaction_N_by_bearing=axial_reactions)
        for name, moment in local_moments.items():
            housing_moment += moment
        result['housing_constraint_torque_Nm'] = -housing_moment
    return result


def default_parameters(fit_clearance=False):
    """Defaults are NOT identified machine parameters.
    ms/mt/mh: shaft/holder/housing effective mass, kg.
    kh/kbend: ground and shaft-holder bending stiffness, N/m; ch/cl: N s/m.
    kappa: taper stiffness per area, N/m³; contact diameter/length: m.
    ki/ko: Hertz constants, N/m^(3/2); cb_total: radial damping N s/m.
    normal_overlap: m (effective fixed-speed preload placeholder, not clearance).
    balls: count; ball/pitch diameter: m; contact_angle/cage_phase: rad.
    rpm: rev/min; radial_load_y: N (+Y); t_section_from_face: m.
    Optional platform_mass (kg), platform_ky (N/m), platform_cy (N s/m)
    enable platform Y. Its mass excludes the separately represented masses.
    fit_clearance=True instead enables S/T/H/O, allocating a demonstration
    baseline once: radial fit_clearance_um, outer-ring mO (kg), Kf (N/m),
    Df (N s/m), even fixed-direction count N, 2x2 Kother (N/m), Cother (N s/m).
    This option cannot be combined with platform_mass or other faults.
    As for the original nodes, the same lumped mass sets inertia and gravity.
    Ball count/diameter/angle borrow Tai I Table1 p10 (7014), NOT JD bearings.
    Optional outer_pit_area_m2 defaults to zero when absent. Nonzero pits also
    require outer_pit_width_m, outer_pit_depth_m, outer_pit_center_rad (fixed
    housing frame, measured counterclockwise from +X). Only small localized
    defects, length < ball diameter, are implemented; width spans the track.
    inner_pit_* uses the same units; center_rad is at t=0 and rotates with shaft.
    ball_deformation_amplitude_m is a single surface's w=2 waviness amplitude;
    ball_deformation_index is zero-based (default 0), phase_rad defaults to 0.
    This represents ball out-of-roundness, not cylindrical roller deformation.
    User: section ~15 mm; healthy synchronous peak-to-peak ~3 µm. No runout
    waveform or zero healthy error is imposed. All other mechanics provisional.
    """
    p = dict(ms=2.5, mt=0.7, mh=8.0, kh=8.0e6, ch=600.0,
                kbend=20.0e6, kappa=1.1e9, contact_diameter=0.030,
                contact_length=0.040, cl=80.0, ki=1.0e10, ko=1.0e10,
                normal_overlap=3.0e-6, cb_total=100.0,
                balls=20, ball_diameter=0.011906, pitch_diameter=0.090,
                contact_angle=np.deg2rad(15.0), rpm=7000.0,
                cage_phase=0.0, radial_load_y=0.0, t_section_from_face=0.015,
                machine_model='JD130S-24-H40A (user supplied)',
                healthy_error_reported_um=3.0,
                healthy_error_definition="user: synchronous peak-to-peak, approximate scale; not imposed on model",
                parameter_status="provisional except rpm and reported error scale")
    # Opt in explicitly: existing six/seven-DOF callers retain their parameters.
    return allocate_fit_support(p)[0] if fit_clearance else p


def area_and_link(p, lost_area):
    """m² -> N/m. Loss is axisymmetric equivalent area, not a local pit."""
    area = np.pi * p['contact_diameter'] * p['contact_length']
    if not 0 <= lost_area < area:
        raise ValueError('lost_area must be in [0, nominal taper area)')
    kj = p['kappa'] * (area - lost_area)
    return area, 1.0 / (1.0 / p['kbend'] + 1.0 / kj)


def cage_speed(p):
    omega = p['rpm'] * 2 * np.pi / 60
    return omega / 2 * (1 - p['ball_diameter'] / p['pitch_diameter']
                        * np.cos(p['contact_angle']))


def ball_spin_speed(p):
    """rad/s; fixed-angle, no-slip spin from Tai I p16 Eq(53), not RPM in Hz."""
    ratio = p['ball_diameter'] / p['pitch_diameter']
    omega = p['rpm'] * 2*np.pi / 60
    return omega / (2*ratio) * (1 - (ratio*np.cos(p['contact_angle']))**2)


def _race_pit_geometry(p, race):
    """Rectangular pit: SI area = transverse width * circumferential chord length.

    Tai/Altintas 2024 Part I, p5 Eqs(19),(20),(23): small-defect sag and
    half-sine displacement. The FULL opening is 2*asin(L/(2*R)); Tai p4
    Eq(13) prints a half-angle but uses it as a full interval. We explicitly
    use the chord geometry, consistent with Liu/Shao 2017 p89 Eqs(10),(11).
    Width must span the ball contact patch (not resolved by this model).
    R is the selected contact-track radius, not groove cross-section curvature.
    """
    prefix = race + '_pit_'
    area = p.get(prefix + 'area_m2', 0.0)
    if area < 0:
        raise ValueError(prefix + 'area_m2 must be nonnegative')
    if area == 0:
        return dict(area_m2=0.0, length_m=0.0, maximum_drop_m=0.0, span_rad=0.0)
    width, depth = p[prefix + 'width_m'], p[prefix + 'depth_m']
    if width <= 0 or depth <= 0:
        raise ValueError('Pit width and depth must be positive')
    length = area / width
    sign = 1 if race == 'outer' else -1
    radius = (p['pitch_diameter'] + sign*p['ball_diameter']*np.cos(p['contact_angle'])) / 2
    ball_radius = p['ball_diameter'] / 2
    if not 0 < length < min(p['ball_diameter'], 2*radius):
        raise ValueError('Small-defect model requires 0 < length < ball diameter and track diameter')
    # Rationalized sag formula avoids subtracting nearly equal radii.
    sag = (length/2)**2 / (ball_radius + np.sqrt(ball_radius**2-(length/2)**2))
    return dict(area_m2=area, width_m=width, depth_m=depth, length_m=length,
                track_radius_m=radius, maximum_drop_m=min(depth, sag),
                span_rad=2*np.arcsin(length/(2*radius)),
                center_rad=p[prefix + 'center_rad'])


def outer_race_pit_geometry(p):
    """Fixed outer contact-track geometry; see _race_pit_geometry."""
    return _race_pit_geometry(p, 'outer')


def inner_race_pit_geometry(p):
    """Inner track Ri=(Dm-D*cos(alpha))/2; center_rad is shaft-fixed at t=0."""
    return _race_pit_geometry(p, 'inner')


def _pit_displacement(phi, g):
    phi = np.asarray(phi)
    if g['area_m2'] == 0:
        return np.zeros_like(phi, dtype=float)
    distance = (phi-g['center_rad']+np.pi) % (2*np.pi)-np.pi
    position = np.clip(distance/g['span_rad'] + 0.5, 0.0, 1.0)
    return np.where(np.abs(distance) < g['span_rad']/2,
                    g['maximum_drop_m']*np.sin(np.pi*position), 0.0)


def outer_race_pit_displacement(phi, p, geometry=None):
    """Normal-direction loss of approach, zero outside the stationary outer pit.

    Tai p3 Eq(7) projects the race defect along the contact normal. With fixed,
    equal inner/outer contact angles and massless series contacts, subtract
    this drop ONCE from their total normal approach. No appended impact signal.
    """
    g = outer_race_pit_geometry(p) if geometry is None else geometry
    return _pit_displacement(phi, g)


def inner_race_pit_displacement(t, phi, p, geometry=None):
    """Tai I Eq(25): pit travels at shaft speed; evaluate shaft-relative angle."""
    g = inner_race_pit_geometry(p) if geometry is None else geometry
    return _pit_displacement(np.asarray(phi)-p['rpm']*2*np.pi/60*t, g)


def ball_deformation_displacement(t, p):
    """Signed TOTAL approach loss, single ball: Eq(33), w=2, Hi+Ho=2*C*cos(2*phi+gamma).

    C is the surface amplitude, not diameter error or two-contact amplitude.
    Signed waviness is intentional; clipping it would invent a local defect.
    Odd orders cancel under the equal-angle antipodal series simplification.
    """
    amplitude = p.get('ball_deformation_amplitude_m', 0.0)
    if amplitude < 0:
        raise ValueError('ball_deformation_amplitude_m must be nonnegative')
    loss = np.zeros(p['balls'])
    index = p.get('ball_deformation_index', 0)
    if not 0 <= index < p['balls']:
        raise ValueError('ball_deformation_index must identify an existing ball')
    phase = 2*ball_spin_speed(p)*t + p.get('ball_deformation_phase_rad', 0.0)
    loss[index] = 2*amplitude*np.cos(phase)
    return loss


def _race_passage_boundaries(p, end_time, g, speed):
    """Explicit entry/exit times prevent an adaptive step skipping a narrow pit."""
    if g['area_m2'] == 0 or speed == 0:
        return np.array([0.0, end_time])
    pitch = 2*np.pi/p['balls']
    boundaries = [0.0, end_time]
    for edge in (g['center_rad']-g['span_rad']/2, g['center_rad']+g['span_rad']/2):
        first, last = sorted(((p['cage_phase']-edge)/pitch,
                              (p['cage_phase']+speed*end_time-edge)/pitch))
        k = np.arange(np.ceil(first), np.floor(last)+1)
        t = (edge+k*pitch-p['cage_phase'])/speed
        boundaries.extend(t[(t > 0) & (t < end_time)])
    return np.unique(boundaries)


def outer_race_passage_boundaries(p, end_time):
    return _race_passage_boundaries(p, end_time, outer_race_pit_geometry(p), cage_speed(p))


def inner_race_passage_boundaries(p, end_time):
    speed = cage_speed(p) - p['rpm']*2*np.pi/60
    return _race_passage_boundaries(p, end_time, inner_race_pit_geometry(p), speed)


def bearing_keq(p):
    """Explicit V6 equivalent coefficient; old series-contact input is preserved."""
    return p['Keq'] if 'Keq' in p else (p['ki']**(-2/3) + p['ko']**(-2/3))**(-1.5)


def bearing_contact_state(t, relative, velocity, p):
    """Shared front/rear fixed-angle Hertz law, scalar or saved-array diagnostics.

    t broadcasts against the leading relative/velocity axes. Delta is signed
    normal approach; no new contact-pressure or 2*overlap validity limit is used.
    """
    key = (cage_speed(p), p['balls'], p['contact_angle'], bearing_keq(p),
           p['cb_total'], p['cage_phase'])
    cached = p.get('_bearing_constants') if _cache_enabled(p) else None
    if cached is None or p.get('_bearing_constants_key') != key:
        speed = cage_speed(p)
        angles = np.arange(p['balls']) * (2*np.pi/p['balls'])
        ca = np.cos(p['contact_angle'])
        keq = bearing_keq(p)
        cball = p['cb_total'] / (p['balls']/2 * ca**2)
        cached = (speed, angles, ca, keq, cball)
        if _cache_enabled(p):
            p['_bearing_constants'] = cached
            p['_bearing_constants_key'] = key
    else:
        speed, angles, ca, keq, cball = cached
    phi = speed * np.asarray(t)[..., None] + p['cage_phase'] + angles
    directions = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    delta = p['normal_overlap'] + ca * np.einsum('...ji,...i->...j', directions, relative)
    if p.get('outer_pit_area_m2', 0.0) != 0:
        delta -= outer_race_pit_displacement(phi, p, p.get('_pit_geometry'))
    if p.get('inner_pit_area_m2', 0.0) != 0:
        delta -= inner_race_pit_displacement(t, phi, p, p.get('_inner_pit_geometry'))
    if p.get('ball_deformation_amplitude_m', 0.0) != 0:
        delta -= ball_deformation_displacement(t, p)
    compression = np.maximum(delta, 0)
    normal = keq * compression**1.5 + cball * ca * np.einsum('...ji,...i->...j', directions, velocity)
    normal = np.where(delta > 0, np.maximum(normal, 0), 0)
    force = ca * np.einsum('...j,...ji->...i', normal, directions)
    return force, dict(delta_m=delta, normal_force_N=normal,
                      delta_over_ball_diameter=delta/p['ball_diameter'],
                      delta_over_overlap=delta/p['normal_overlap'])


def bearing_force(t, relative, velocity, p):
    return bearing_contact_state(t, relative, velocity, p)[0]


def bearing_wrench(t, relative_xy, relative_velocity_xy, relative_tilt_xy,
                   relative_tilt_rate_xy, bearing_parameters,
                   orientation_sign, rho_m):
    """Hertz restoring force and local moment from one relative-contact projection.

    Inputs support a leading batch/time shape and end in two radial or angular
    components. Returned force and moment also end in two components; the
    diagnostics retain one value per ball on the last axis.
    """
    if orientation_sign not in (-1, 1) or rho_m <= 0:
        raise ValueError('orientation_sign must be +/-1 and rho_m must be positive')
    relative_xy = np.asarray(relative_xy)
    relative_velocity_xy = np.asarray(relative_velocity_xy)
    relative_tilt_xy = np.asarray(relative_tilt_xy)
    relative_tilt_rate_xy = np.asarray(relative_tilt_rate_xy)
    if any(value.shape[-1] != 2 for value in
           (relative_xy, relative_velocity_xy, relative_tilt_xy, relative_tilt_rate_xy)):
        raise ValueError('Bearing relative translations and tilts must end in XY pairs')
    key = (_bearing_parameter_signature(bearing_parameters), int(orientation_sign), float(rho_m))
    cache_enabled = _cache_enabled(bearing_parameters)
    constants = bearing_parameters.get('_bearing_wrench_constants') if cache_enabled else None
    if constants is None or bearing_parameters.get('_bearing_wrench_constants_key') != key:
        balls = int(bearing_parameters['balls'])
        angle = bearing_parameters['contact_angle']
        ca, sa = np.cos(angle), np.sin(angle)
        constants = (cage_speed(bearing_parameters),
                     np.arange(balls)*(2*np.pi/balls), ca, sa,
                     bearing_keq(bearing_parameters),
                     bearing_parameters['cb_total']/(balls/2*ca**2), balls)
        if cache_enabled:
            bearing_parameters['_bearing_wrench_constants'] = constants
            bearing_parameters['_bearing_wrench_constants_key'] = key
    cage, ball_angles, ca, sa, keq, cball, balls = constants
    phi = cage*np.asarray(t)[..., None] + bearing_parameters['cage_phase'] + ball_angles
    cosine, sine = np.cos(phi), np.sin(phi)
    sign = float(orientation_sign)
    tilt_scale = sign*float(rho_m)*sa
    tilt_compression = tilt_scale*(relative_tilt_xy[..., 0, None]*sine
                                    - relative_tilt_xy[..., 1, None]*cosine)
    delta = (bearing_parameters['normal_overlap']
             + ca*(relative_xy[..., 0, None]*cosine+relative_xy[..., 1, None]*sine)
             + tilt_compression)
    normal_velocity = (ca*(relative_velocity_xy[..., 0, None]*cosine
                           + relative_velocity_xy[..., 1, None]*sine)
                       + tilt_scale*(relative_tilt_rate_xy[..., 0, None]*sine
                                     - relative_tilt_rate_xy[..., 1, None]*cosine))
    compression = np.maximum(delta, 0)
    elastic = keq*compression**1.5
    normal = np.where(delta > 0, np.maximum(elastic+cball*normal_velocity, 0), 0)
    force = np.stack((np.sum(normal*ca*cosine, axis=-1),
                      np.sum(normal*ca*sine, axis=-1)), axis=-1)
    moment = np.stack((np.sum(normal*tilt_scale*sine, axis=-1),
                       np.sum(-normal*tilt_scale*cosine, axis=-1)), axis=-1)
    potential = np.sum((2/5)*keq*compression**2.5, axis=-1)
    dissipation = np.sum((normal-elastic)*normal_velocity, axis=-1)
    axial_restoring = np.sum(normal*sign*sa, axis=-1)
    diagnostics = dict(delta_m=delta, normal_force_N=normal,
        elastic_force_N=elastic, normal_velocity_m_s=normal_velocity,
        tilt_compression_m=tilt_compression, axial_restoring_N=axial_restoring,
        potential_energy_J=potential, dissipation_W=dissipation,
        active_ball_count=np.sum(normal > 0, axis=-1),
        delta_over_ball_diameter=delta/bearing_parameters['ball_diameter'],
        delta_over_overlap=delta/bearing_parameters['normal_overlap'],
        maximum_ball_force_N=np.max(normal, axis=-1),
        eta_tilt=(float(rho_m)*sa*np.linalg.norm(relative_tilt_xy, axis=-1)
                  /bearing_parameters['normal_overlap']))
    return force, moment, diagnostics


def _bearing_wrench_from_context(t, q, v, name, context, model):
    spec, params = context[name]
    z = spec['z_m']
    outer_index = {'O1': O, 'O2': O2, 'H': H}[spec['outer_translation_source']]
    relative_xy = rotor_section(q, z)-np.asarray(q)[..., outer_index]
    relative_velocity_xy = rotor_section(v, z)-np.asarray(v)[..., outer_index]
    tilt = np.asarray(q)[..., 8:10]
    tilt_rate = np.asarray(v)[..., 8:10]
    if model == 'relative_tilt_v8':
        force, moment, diagnostics = bearing_wrench(t, relative_xy,
            relative_velocity_xy, tilt, tilt_rate, params,
            spec['orientation_sign'], spec['rho_m'])
    else:
        force, old = bearing_contact_state(t, relative_xy, relative_velocity_xy, params)
        delta, normal = old['delta_m'], old['normal_force_N']
        ca, sa = np.cos(params['contact_angle']), np.sin(params['contact_angle'])
        phi = cage_speed(params)*np.asarray(t)[..., None]+params['cage_phase']+np.arange(params['balls'])*(2*np.pi/params['balls'])
        normal_velocity = ca*(relative_velocity_xy[..., 0, None]*np.cos(phi)
                              + relative_velocity_xy[..., 1, None]*np.sin(phi))
        elastic = bearing_keq(params)*np.maximum(delta, 0)**1.5
        diagnostics = dict(delta_m=delta, normal_force_N=normal,
            elastic_force_N=elastic, normal_velocity_m_s=normal_velocity,
            tilt_compression_m=np.zeros_like(delta),
            axial_restoring_N=np.sum(normal*spec['orientation_sign']*sa, axis=-1),
            potential_energy_J=np.sum((2/5)*bearing_keq(params)*np.maximum(delta, 0)**2.5, axis=-1),
            dissipation_W=np.sum((normal-elastic)*normal_velocity, axis=-1),
            active_ball_count=np.sum(normal > 0, axis=-1),
            delta_over_ball_diameter=old['delta_over_ball_diameter'],
            delta_over_overlap=old['delta_over_overlap'],
            maximum_ball_force_N=np.max(normal, axis=-1),
            eta_tilt=np.zeros(np.shape(np.sum(normal, axis=-1))))
        moment = np.zeros_like(force)
    return dict(force_N=force, moment_Nm=moment, diagnostics=diagnostics)


def bearing_wrench_at(t, q, v, p, name):
    """Evaluate one named bearing, suitable for chunked native contact export."""
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        from .tai_system import bearing_wrenches as separate_wrenches
        return separate_wrenches(t,q,v,p)[name]
    model = bearing_contact_model(p)
    context = _bearing_context(p)
    if name not in context:
        raise KeyError(f'Unknown bearing {name!r}')
    return _bearing_wrench_from_context(t, q, v, name, context, model)


def bearing_wrenches(t, q, v, p):
    """Evaluate F1/F2/R1/R2, returning named radial force, local moment and state."""
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        from .tai_system import bearing_wrenches as separate_wrenches
        return separate_wrenches(t,q,v,p)
    model = bearing_contact_model(p)
    context = _bearing_context(p)
    return {name: _bearing_wrench_from_context(t, q, v, name, context, model)
            for name in ('F1', 'F2', 'R1', 'R2')}


def fit_contact_force(relative, velocity, p):
    """Force transmitted to H by O, N; radial gap c is not bearing overlap.

    Directions are fixed to H. Supports (..., 2) arrays for saved diagnostics.
    At c=0, Df=0 and even N, Fc=Kf*relative exactly. Damping is unilateral;
    its effective global coefficient is not a constant Kelvin--Voigt element.
    """
    count = p['N']
    if count < 4 or count % 2 or p['Kf'] <= 0 or p['Df'] < 0 or p['fit_clearance_um'] < 0:
        raise ValueError('Fit requires even N >= 4, Kf > 0, Df >= 0 and radial clearance >= 0')
    directions = p.get('_fit_directions')
    if directions is None or len(directions) != count:
        phi = 2*np.pi*np.arange(count)/count
        directions = np.column_stack((np.cos(phi), np.sin(phi)))
    signed_gap = np.asarray(relative) @ directions.T - p['fit_clearance_um']*1e-6
    compression = np.maximum(signed_gap, 0)
    normal_velocity = np.asarray(velocity) @ directions.T
    coefficients = p.get('_fit_coefficients')
    if coefficients is None:
        coefficients = (4*p['Kf']/count, 4*p['Df']/count)
    elastic = coefficients[0] * compression
    normal = np.where(signed_gap > 0,
                      np.maximum(elastic + coefficients[1]*normal_velocity, 0), 0)
    return normal @ directions, dict(normal_force_N=normal, compression_m=compression,
        signed_gap_m=signed_gap, normal_velocity_m_s=normal_velocity,
        elastic_force_N=elastic, active_count=np.sum(normal > 0, axis=-1))


def fit_contact_summary(relative, velocity, p):
    """Export the required contact diagnostics in blocks, without N-by-time storage."""
    forces=[];active=[];compression=[]
    for first in range(0,len(relative),8192):
        force,diag=fit_contact_force(relative[first:first+8192],velocity[first:first+8192],p)
        forces.append(force);active.append(diag['active_count'])
        compression.append(diag['compression_m'].max(axis=1))
    return np.concatenate(forces),dict(active_count=np.concatenate(active),max_compression_m=np.concatenate(compression))


def mass_and_gravity(p):
    if is_tilt_model(p):
        if 'platform_mass' in p:
            raise ValueError('Tilt rotor model cannot include platform_mass')
        if min(p['ms'], p['mt'], p['mh'], p['mO'], p['Jt'], p['Jp']) <= 0:
            raise ValueError('Masses and rotor inertias must be positive')
        masses = np.r_[np.repeat([p['ms'], p['mt'], p['mh'], p['mO']], 2),
                       p['Jt'], p['Jt']]
        gravity = np.r_[masses[:8]*np.tile(p['gravity_radial_m_s2'], 4), 0., 0.]
        if is_front_group(p):
            if p['mO2'] <= 0:
                raise ValueError('O2 mass must be positive')
            masses = np.r_[masses, [p['mO2']]*2]
            gravity = np.r_[gravity, p['mO2']*np.asarray(p['gravity_radial_m_s2'])]
        static_load = [0., p.get('radial_load_y', 0.)]
        gravity += generalized_point_force(static_load, p.get('radial_load_z_m', 0.), len(masses))
        return masses, gravity
    masses = np.repeat([p['ms'], p['mt'], p['mh']], 2)
    gravity = masses * np.tile([0.0, -9.80665], 3)
    gravity[1] += p['radial_load_y']
    if 'mO' in p:
        if 'platform_mass' in p:
            raise ValueError('Eight-DOF fit model cannot include platform_mass')
        if min(p['ms'], p['mt'], p['mh'], p['mO']) <= 0:
            raise ValueError('All eight-DOF node masses must be positive')
        return np.r_[masses, [p['mO']]*2], np.r_[gravity, 0.0, -9.80665*p['mO']]
    if 'platform_mass' in p:
        masses = np.r_[masses, p['platform_mass']]
        gravity = np.r_[gravity, -9.80665*p['platform_mass']]
    return masses, gravity


def internal_force(t, q, v, p, lost_area):
    """Restoring force; positive sign on left of Mq''+R=F."""
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        if lost_area != 0:
            raise ValueError('V9 currently varies fit clearance only')
        from .tai_system import internal_force as separate_internal
        return separate_internal(t,q,v,p)
    if is_tilt_model(p):
        if lost_area != 0 or any(p.get(key, 0) != 0 for key in
                ('outer_pit_area_m2', 'inner_pit_area_m2', 'ball_deformation_amplitude_m')):
            raise ValueError('Tilt rotor model varies target fit clearance only')
        constants = _tilt_constants(p)
        bearing, link, healthy, fit, wrench_set = _tilt_forces(t, q, v, p, constants)
        af, at, ah, _, _, gyro, _ = constants
        force = np.zeros(len(gyro))
        force[ROTOR] = af.T@bearing + at.T@link + np.einsum('kij,ki->j', ah, healthy)
        force[T] = -link
        force[H] = -healthy.sum(axis=0)-fit+p['kh']*q[H]+p['ch']*v[H]
        force[O] = -bearing+fit
        if is_front_group(p):
            b2, fit2, _ = front2_forces(q, v, p, t=t, wrench_set=wrench_set)
            force[ROTOR] += _front2_constants(p)[0].T@b2
            force[H] -= fit2
            force[O2] = -b2+fit2
        if wrench_set is not None:
            force[8:10] += sum((wrench_set[name]['moment_Nm']
                                for name in ('F1', 'F2', 'R1', 'R2')))
        force += gyro@v
        return force
    _, kl = area_and_link(p, lost_area)
    if 'mO' in p:
        if 'platform_mass' in p:
            raise ValueError('Eight-DOF fit model cannot include platform_mass')
        if lost_area != 0 or any(p.get(key, 0) != 0 for key in
                ('outer_pit_area_m2', 'inner_pit_area_m2', 'ball_deformation_amplitude_m')):
            raise ValueError('Eight-DOF fit model accepts radial fit clearance only; other faults must be zero')
        b = bearing_force(t, q[S]-q[O], v[S]-v[O], p)
        link = kl*(q[S]-q[T]) + p['cl']*(v[S]-v[T])
        other = np.asarray(p['Kother'])@(q[S]-q[H]) + np.asarray(p['Cother'])@(v[S]-v[H])
        fit, _ = fit_contact_force(q[O]-q[H], v[O]-v[H], p)
        return np.r_[b+link+other, -link,
                     -other-fit+p['kh']*q[H]+p['ch']*v[H], -b+fit]
    b = bearing_force(t, q[:2]-q[4:6], v[:2]-v[4:6], p)
    link = kl*(q[:2]-q[2:4]) + p['cl']*(v[:2]-v[2:4])
    housing = -b+p['kh']*q[4:6]+p['ch']*v[4:6]
    if 'platform_mass' in p:
        support = p['kh']*(q[5]-q[6]) + p['ch']*(v[5]-v[6])
        housing[1] = -b[1]+support
        return np.r_[b+link, -link, housing,
                     p['platform_ky']*q[6]+p['platform_cy']*v[6]-support]
    return np.r_[b+link, -link, housing]


def equilibrium(p, lost_area, initial_q=None):
    """Frozen t=0 equilibrium; positive gaps default to zero-gap continuation."""
    if 'mO' in p and p['fit_clearance_um'] > 0 and initial_q is None:
        return equilibrium_continuation(p, lost_area)
    if p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring':
        if lost_area != 0:
            raise ValueError('V9 static solve requires the unchanged holder contact')
        from .tai_system import equilibrium as separate_equilibrium
        return separate_equilibrium(p,initial_q)
    masses, gravity = mass_and_gravity(p)
    n = len(masses)
    # Scale unknowns/residuals to avoid treating micron translations as O(1).
    qscale, fscale = np.full(n, 1e-6), np.ones(n)
    if is_tilt_model(p):
        qscale[8:10] /= p['L_ref']
        fscale[8:10] = p['L_ref']
    solution = root(lambda z: (internal_force(0, z*qscale, np.zeros(n), p,
                                              lost_area)-gravity)/fscale/100,
                    np.zeros(n) if initial_q is None else np.asarray(initial_q)/qscale,
                    tol=1e-10)
    q = solution.x*qscale
    residual = np.linalg.norm((internal_force(0, q, np.zeros(n), p, lost_area)-gravity)/fscale)
    if not np.isfinite(residual) or residual > (1e-6 if is_front_group(p) else 1e-5):
        raise RuntimeError(f'equilibrium residual {residual:.3g} N')
    return q, residual


def equilibrium_continuation(p, lost_area=0, initial_q=None,
                             start_clearance_um=0.0, max_increment_um=1.0):
    """Continue the frozen loaded equilibrium; halve failed gap increments."""
    target = p['front_group_clearance_um'] if is_front_group(p) else p['fit_clearance_um']
    def at_gap(value):
        return front_group_parameters(p, value) if is_front_group(p) else dict(p, fit_clearance_um=value)
    current = start_clearance_um
    q, residual = equilibrium(at_gap(current), lost_area, initial_q)
    increment = max_increment_um
    while current < target:
        trial = min(target, current+increment)
        try:
            q_next, residual = equilibrium(at_gap(trial), lost_area, q)
        except RuntimeError:
            increment /= 2
            if increment < 1e-6:
                raise
            continue
        q, current = q_next, trial
        increment = min(max_increment_um, increment*2)
    return q, residual


def linear_matrices(p, lost_area, initial_q=None):
    """Frozen tangent M,C_total,K; C_total includes G exactly once for 10 DOF.

    This is a local derivative, not a global constitutive law R=Kq. For 10 DOF
    the position step is 1e-11 m, velocity step 1e-9 m/s; angular steps divide
    these by L_ref. Legacy branches retain 1e-9 for both. At an exact contact
    switch a unique tangent need not exist, regardless of difference step.
    """
    q, _ = equilibrium(p, lost_area, initial_q)
    n = len(q)
    basis = np.eye(n)
    zero = np.zeros(n)
    qstep, vstep = np.full(n, 1e-9), np.full(n, 1e-9)
    if is_tilt_model(p):
        qstep[:] = 1e-11
        qstep[8:10] /= p['L_ref']
        vstep[8:10] /= p['L_ref']
    k = np.column_stack([(internal_force(0,q+basis[i]*qstep[i],zero,p,lost_area)
                          -internal_force(0,q-basis[i]*qstep[i],zero,p,lost_area))
                         /(2*qstep[i]) for i in range(n)])
    c = np.column_stack([(internal_force(0,q,basis[i]*vstep[i],p,lost_area)
                          -internal_force(0,q,-basis[i]*vstep[i],p,lost_area))
                         /(2*vstep[i]) for i in range(n)])
    m = np.diag(mass_and_gravity(p)[0])
    return m, c, k


def _bearing_tangent(relative, p):
    step, basis, zero = 1e-9, np.eye(2), np.zeros(2)
    return np.column_stack([(bearing_force(0, relative+axis*step, zero, p)
                             -bearing_force(0, relative-axis*step, zero, p))/(2*step)
                            for axis in basis])


def allocate_fit_support(old_p, mass_fraction=0.01, zeta=0.02):
    """One-time DEMONSTRATION allocation from a six-DOF healthy reference.

    Split outer-ring mass from housing, preserve old total bearing tangent by
    series/parallel allocation, then freeze it for the clearance sweep. This
    does not preserve the old full frequency response or identify a machine.
    """
    if 'mO' in old_p or 'platform_mass' in old_p:
        raise ValueError('Fit allocation requires a six-DOF reference without mO or platform_mass')
    q_old, old_residual = equilibrium(old_p, 0)
    kbase = _bearing_tangent(q_old[S]-q_old[H], old_p)
    kfit = np.eye(2)*np.trace(kbase)/2
    target_old = kbase-kbase@np.linalg.solve(kbase+kfit, kbase)
    other = kbase-target_old
    p = dict(old_p, mO=old_p['mh']*mass_fraction, mh=old_p['mh']*(1-mass_fraction),
             fit_clearance_um=0.0, Kf=float(kfit[0, 0]), N=32,
             Kother=other.tolist(), Cother=np.zeros((2, 2)).tolist())
    p['Df'] = float(2*zeta*np.sqrt(p['mO']*p['Kf']))
    p['fit_parameter_status'] = ('Demonstration: outer-ring mass split from housing; '
        'Kf from old frozen bearing tangent; fixed Kother from series/parallel allocation; '
        'Df from assumed zeta=%.4g; Cother=0. None identified on the machine.' % zeta)
    q, residual = equilibrium(p, 0, np.r_[q_old, q_old[H]])
    for updates in range(2):
        kb = _bearing_tangent(q[S]-q[O], p)
        target = kb-kb@np.linalg.solve(kb+kfit, kb)
        error = np.linalg.norm(other+target-kbase)/np.linalg.norm(kbase)
        if error <= 0.05 or updates == 1:
            break
        other = kbase-target
        p['Kother'] = other.tolist()
        q, residual = equilibrium(p, 0, q)
    symmetric_error = np.linalg.norm(other-other.T)/np.linalg.norm(other)
    eigenvalues = np.linalg.eigvalsh(other)
    if symmetric_error > 1e-8 or eigenvalues.min() <= 0 or error > 0.05:
        raise RuntimeError('Baseline healthy-support allocation failed symmetry, positivity or 5% tangent check')
    allocation = dict(reference_dof=6, new_dof=8, reference_equilibrium_m=q_old.tolist(),
        reference_equilibrium_residual_N=old_residual, baseline_equilibrium_m=q.tolist(),
        baseline_equilibrium_residual_N=residual, Kbase_N_m=kbase.tolist(),
        Kb_new_N_m=kb.tolist(), Kfit_N_m=kfit.tolist(), Ktarget_N_m=target.tolist(),
        Kother_N_m=other.tolist(), Kother_eigenvalues_N_m=eigenvalues.tolist(),
        Kother_symmetry_relative=float(symmetric_error), tangent_relative_error=float(error),
        baseline_updates=updates, mO_fraction_of_original_mh=mass_fraction,
        nominal_fit_damping_ratio=zeta, parameter_status=p['fit_parameter_status'])
    return p, allocation


def harmonic_response(p, lost_area, frequencies, force_coefficients):
    """Complex force convention F=Re(sum(c_f*exp(i*2πft))). Returns q amplitudes."""
    m,c,k = linear_matrices(p, lost_area)
    response = []
    for f, coeff in zip(frequencies,force_coefficients):
        omega = 2*np.pi*f
        force = generalized_point_force(coeff, p.get('z_u', 0.), len(m))
        response.append(np.linalg.solve(k-omega**2*m+1j*omega*c, force))
    return np.asarray(response)


def inverse_housing_force(p, frequencies, housing_velocity_amplitudes):
    """Conditional force estimate with fixed assumed M/K/C, NOT unique identification.
    Full 2x2 response is used; observability regularization bounds weak transfer.
    Input velocity amplitudes: m/s, X then Y. Returned amplitudes: complex N.
    """
    coefficients=[]
    for f, amp in zip(frequencies,housing_velocity_amplitudes):
        q1=harmonic_response(p,0,[f],[[1.0,0.0]])[0,4:6]
        q2=harmonic_response(p,0,[f],[[0.0,1.0]])[0,4:6]
        transfer=1j*2*np.pi*f*np.column_stack([q1,q2])
        lam=1e-4*np.max(np.linalg.svd(transfer,compute_uv=False))
        force=np.linalg.solve(transfer.conj().T@transfer+lam**2*np.eye(2),
                              transfer.conj().T@amp)
        coefficients.append(force)
    return np.asarray(coefficients)


def simulate(p, lost_area, frequencies, coefficients, duration=0.5, warmup=0.1,
             fs=8192, max_step=1/32768, deadline=None, pit_steps_per_passage=4,
             q_initial=None, rtol=2e-7, atol=None, unbalance=None,
             t_start=0.0, state_initial=None, method='DOP853'):
    """Nonlinear integration with common forcing, preserving absolute phase.

    unbalance=(U_kg_m, phase_rad) selects analytic circular force at z_u for
    10/12 DOF (on S otherwise). Its signed rpm sets rotation direction. For continued
    runs, t_start and state_initial must be the preceding final time/state;
    warmup is relative to t_start. q_initial is the static equilibrium guess.
    """
    if method != 'DOP853':
        raise ValueError('Only DOP853 is supported by this integration/export path')
    if is_front_group(p):
        p = front_group_parameters(p, p['front_group_clearance_um'])
    geometry = outer_race_pit_geometry(p)
    inner_geometry = inner_race_pit_geometry(p)
    p = dict(p, _pit_geometry=geometry, _inner_pit_geometry=inner_geometry,
             _derived_cache_enabled=True)
    if 'mO' in p:
        phi = 2*np.pi*np.arange(p['N'])/p['N']
        p['_fit_directions'] = np.column_stack((np.cos(phi), np.sin(phi)))
    if is_tilt_model(p):
        p['_tilt_constants'] = _tilt_constants(p)
        ca = np.cos(p['contact_angle'])
        p['_bearing_constants'] = (cage_speed(p),
            np.arange(p['balls'])*(2*np.pi/p['balls']), ca,
            bearing_keq(p),
            p['cb_total']/(p['balls']/2*ca**2))
        p['_bearing_constants_key'] = (cage_speed(p), p['balls'],
            p['contact_angle'], bearing_keq(p), p['cb_total'], p['cage_phase'])
        if p.get('nonlinear_bearings_v6', False):
            p['_rear_bearing_parameters'] = rear_bearing_parameters(p)
        if 'bearing_contact_model' in p:
            _bearing_context(p)
        p['_fit_coefficients'] = (4*p['Kf']/p['N'], 4*p['Df']/p['N'])
        if is_front_group(p):
            p['_front2_constants'] = _front2_constants(p)
    masses, gravity = mass_and_gravity(p)
    n = len(masses)
    actual_atol = np.r_[np.full(n, 1e-12), np.full(n, 1e-8)] if atol is None else np.broadcast_to(atol, (2*n,)).copy()
    if is_tilt_model(p) and atol is None:
        actual_atol[8:10] = 1e-12/p['L_ref']
        actual_atol[n+8:n+10] = 1e-8/p['L_ref']
    load_map = np.zeros((n, 2))
    if is_tilt_model(p):
        load_map[ROTOR] = section_matrix(p['z_u']).T
    else:
        load_map[S] = np.eye(2)
    q0, residual = equilibrium(p, lost_area, q_initial)
    angular = 2*np.pi*np.asarray(frequencies)
    # Dense measured-spectrum forcing is evaluated once, then smoothly interpolated.
    # It is common to all cases; no fault-dependent signal/noise is appended.
    # Force bandwidth is independent of the requested mechanical output rate.
    drive_fs = min(fs, 8192)*8
    drive_times=t_start+np.arange(int(np.ceil((warmup+duration)*drive_fs))+2)/drive_fs
    frequencies = np.asarray(frequencies)
    # Integer-Hz grids have an exact 1 s Fourier representation. Synthesis by
    # irFFT avoids a multi-GB dense exponential matrix for broadband calibration.
    bins = np.rint(frequencies).astype(int)
    if unbalance is not None:
        imbalance, phase = unbalance
        omega = p['rpm']*2*np.pi/60
        def forcing(t):
            angle = omega*t+phase
            return imbalance*omega**2*np.array([np.cos(angle), np.sin(angle)])
    elif t_start == 0 and np.allclose(frequencies, bins, rtol=0, atol=1e-10) and np.all((bins > 0) & (bins < drive_fs/2)):
        spectrum = np.zeros((int(drive_fs)//2+1, 2), dtype=complex)
        np.add.at(spectrum, bins, np.asarray(coefficients)*drive_fs/2)
        period = np.fft.irfft(spectrum, n=int(drive_fs), axis=0)
        drive = period[np.arange(len(drive_times)) % len(period)]
        forcing=CubicSpline(drive_times,drive,axis=0)
    else:
        drive = np.empty((len(drive_times), 2))
        for begin in range(0, len(drive_times), 1024):
            block = drive_times[begin:begin+1024]
            drive[begin:begin+1024] = np.real(np.exp(1j*block[:,None]*angular)@coefficients)
        forcing=CubicSpline(drive_times,drive,axis=0)
    calls=0
    backend = p.get('rhs_backend', 'python')
    compiled_rhs = None
    if backend == 'numba':
        if lost_area != 0:
            raise ValueError('numba backend accepts zero taper loss only')
        from .fast_rhs import prepare_fast_rhs
        compiled_rhs = prepare_fast_rhs(p, unbalance)
    elif backend != 'python':
        raise ValueError('rhs_backend must be python or numba')
    def acceleration(t,q,v):
        if compiled_rhs is not None:
            return compiled_rhs(t, np.r_[q, v])[n:]
        force=load_map@forcing(t)
        return (gravity+force-internal_force(t,q,v,p,lost_area))/masses
    def rhs(t,state):
        nonlocal calls
        calls+=1
        if calls%256==0 and deadline is not None and time.perf_counter()>deadline:
            raise TimeoutError('Adjust --max-seconds or --duration; no long optimization started')
        if compiled_rhs is not None:
            return compiled_rhs(t, state)
        return np.r_[state[n:],acceleration(t,state[:n],state[n:])]
    times=t_start+np.arange(int(round((warmup+duration)*fs))+1)/fs
    boundaries = outer_race_passage_boundaries(p, times[-1])
    if inner_geometry['area_m2']:
        boundaries = np.union1d(boundaries, inner_race_passage_boundaries(p, times[-1]))
    boundaries = np.r_[t_start, boundaries[boundaries > t_start]]
    states = np.empty((2*n, len(times)))
    state = np.r_[q0, np.zeros(n)] if state_initial is None else np.asarray(state_initial)
    states[:, 0] = state
    speed = abs(cage_speed(p))
    dwell = geometry['span_rad']/speed if speed else 0.0
    inner_speed = abs(cage_speed(p)-p['rpm']*2*np.pi/60)
    inner_dwell = inner_geometry['span_rad']/inner_speed if inner_speed else 0.0
    dwells = [d for d in (dwell, inner_dwell) if d > 0]
    accepted_steps = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # At least four integration steps inside each defect passage; outside,
        # retain the original step bound. Endpoint states carry across segments.
        inside_passage = any(end-start <= d*1.001 for d in dwells)
        local_step = min(max_step, min(dwells)/pit_steps_per_passage) if inside_passage else max_step
        solution = solve_ivp(rhs, (start, end), state, dense_output=True,
                             method=method, rtol=rtol, atol=actual_atol,
                             max_step=local_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        accepted_steps.append(np.diff(solution.t))
        begin = np.searchsorted(times, start, side='right')
        finish = np.searchsorted(times, end, side='right')
        if finish > begin:
            states[:, begin:finish] = solution.sol(times[begin:finish])
        state = solution.y[:, -1]
    use=times>=t_start+warmup
    t=times[use];q=states[:n,use].T;v=states[n:,use].T
    acc=np.asarray([acceleration(ti,qi,vi) for ti,qi,vi in zip(t,q,v)])
    if not np.isfinite(states).all():raise RuntimeError('Nonfinite state')
    accepted_steps = np.concatenate(accepted_steps)
    step_statistics = dict(count=len(accepted_steps),
        min_s=float(np.min(accepted_steps)), median_s=float(np.median(accepted_steps)),
        max_s=float(np.max(accepted_steps)),
        max_to_requested_max_step_ratio=float(np.max(accepted_steps)/max_step),
        fraction_ge_0p99_max_step=float(np.mean(accepted_steps >= .99*max_step)))
    result = dict(t=t,q=q,v=v,acc=acc,q_equilibrium=q0,
                equilibrium_residual_N=residual,nfev=calls,
                H_velocity_m_s=v[:,4:6],H_acceleration_m_s2=acc[:,4:6],
                T_center_m=q[:,2:4],integration_max_step_s=max_step,output_fs_hz=fs,
                integration_segments=len(boundaries)-1, pit_dwell_s=dwell,
                inner_pit_dwell_s=inner_dwell,
                pit_steps_per_passage=pit_steps_per_passage, state_final=state.copy(),
                integration_rtol=rtol, integration_start_s=t_start,
                integration_atol=actual_atol, integration_method=method,
                integration_accepted_steps_s=accepted_steps,
                integration_accepted_step_statistics=step_statistics,
                actual_solver_settings=dict(method=method, rtol=rtol, rhs_backend=backend,
                    atol=actual_atol.tolist(), max_step_s=max_step,
                    output_fs_hz=fs, fit_direction_count=p.get('N'),
                    accepted_steps=step_statistics,
                    velocity_matrix_convention='C_total=C+G; G included once' if is_tilt_model(p) else 'C'))
    if 'mO' in p:
        contact = fit_contact_summary if is_front_group(p) else fit_contact_force
        fit, diagnostics = contact(q[:,O]-q[:,H], v[:,O]-v[:,H], p)
        result.update(S_relative_H_m=q[:,S]-q[:,H], O_relative_H_m=q[:,O]-q[:,H],
                      O_center_m=q[:,O], O_velocity_m_s=v[:,O],
                      fit_force_N=fit, fit_active_count=diagnostics['active_count'])
    if is_front_group(p):
        separate = p.get('bearing_contact_model') == 'quasistatic_tai_v9' or p.get('damping_policy') == 'common_structural_ring'
        if separate:
            fit2,diag2 = fit_contact_summary(q[:,O2]-q[:,H],v[:,O2]-v[:,H],_front2_constants(p)[3])
            result.update(fit2_force_N=fit2,fit2_active_count=diag2['active_count'],
                          bearing_diagnostics='separate ring diagnostics exported by V9 runner')
        else:
            b2, fit2, diag2 = front2_forces(q, v, p, summary=True, t=t)
            result.update(bearing2_force_N=b2, fit2_force_N=fit2,
                          fit2_active_count=diag2['active_count'])
        if p.get('nonlinear_bearings_v6', False) and not separate:
            result['rear_bearing_forces_N'] = rear_bearing_forces(t, q, v, p)
    if is_tilt_model(p):
        section = q[:, T] if p['measurement_kind'] == 'holder' else rotor_section(q, p['z_m'])
        if p['measurement_kind'] not in ('rotor_section', 'holder'):
            raise ValueError('measurement_kind must be rotor_section or holder')
        if p['probe_reference'] not in ('housing', 'ground'):
            raise ValueError('probe_reference must be housing or ground')
        probe = q[:, H] if p['probe_reference'] == 'housing' else np.zeros_like(section)
        result.update(measured_section_xy_m=section, probe_reference_xy_m=probe,
                      measured_relative_xy_m=section-probe,
                      rotor_tilt_rad=q[:, 8:10],
                      equilibrium_residual_equivalent_N=residual)
    if 'platform_mass' in p:
        result.update(P_displacement_m=q[:,6], P_velocity_m_s=v[:,6],
                      P_acceleration_m_s2=acc[:,6])
    return result


def displacement_metrics(center):
    """Dynamic-center metrics only; subtract mean, retain a nonzero healthy trajectory.
    Neither metric is a validated ISO error-motion measurement. Geometric runout
    and probe-support motion are unknown and are not synthesized as 3 µm waves.
    """
    motion=center-center.mean(axis=0)
    return dict(dynamic_radial_rms_um=float(np.sqrt(np.mean(np.sum(motion**2,axis=1)))*1e6),
                dynamic_max_radius_um=float(np.max(np.linalg.norm(motion,axis=1))*1e6),
                dynamic_x_pp_um=float(np.ptp(motion[:,0])*1e6),
                dynamic_y_pp_um=float(np.ptp(motion[:,1])*1e6),
                mean_center_um=(center.mean(axis=0)*1e6).tolist(),
                label='simulation_dynamic_displacement_not_measured_error_motion')


def modes_and_stability(p,lost_area):
    m,c,k=linear_matrices(p,lost_area)
    n = len(m)
    a=np.block([[np.zeros((n,n)),np.eye(n)],[-np.linalg.solve(m,k),-np.linalg.solve(m,c)]])
    poles=np.linalg.eigvals(a)
    return dict(mode_frequencies_hz=sorted((np.imag(poles[np.imag(poles)>0])/(2*np.pi)).tolist()),
                maximum_pole_real_s_inv=float(np.max(np.real(poles))),
                tangent_symmetry_relative=float(np.linalg.norm(k-k.T)/np.linalg.norm(k)))
