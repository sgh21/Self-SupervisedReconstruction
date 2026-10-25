"""Mechanical acceptance and frozen-linear feasibility for the 10-DOF branch.

All quantities below are diagnostics of the declared model. Frozen poles and
linear envelopes do not establish nonlinear stability or machine validation.
"""
import csv
import json
from pathlib import Path

import numpy as np

from spindle.dynamics_model import (
    area_and_link, bearing_force, equilibrium, fit_contact_force,
    force_components, generalized_point_force, gyro_matrix, internal_force,
    linear_matrices, mass_and_gravity, rotor_section, section_matrix,
)
from utils.clearance_validation import validate_mechanics as validate_legacy_mechanics


ROTOR = np.array([0, 1, 8, 9])


def audit_legacy_saved_runout(directory):
    """Read-only reconstruction from V1 q, independent of its metric helper."""
    directory = Path(directory)
    rows = []
    with (directory/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        summaries = list(csv.DictReader(stream))
    for summary in summaries:
        clearance = float(summary['clearance_um'])
        path = directory/f'clearance_{clearance:g}um.npz'
        with np.load(path) as saved:
            time = saved['t_raw_s']
            relative_um = (saved['q_raw_m'][:, :2]-saved['q_raw_m'][:, 4:6])*1e6
            peaks = []
            for begin, end in zip(saved['turn_start_s'], saved['turn_end_s']):
                inside = relative_um[(time > begin) & (time < end)]
                boundaries = np.column_stack([np.interp([begin, end], time, relative_um[:, axis])
                                              for axis in range(2)])
                peaks.append(np.ptp(np.vstack((inside, boundaries)), axis=0))
            peaks = np.asarray(peaks)
            correct = float(np.max(np.mean(peaks, axis=0)))
            alternative = float(np.mean(np.max(peaks, axis=1)))
            reported = float(summary['E_dir_um'])
            point_error = float(np.max(np.abs(peaks-saved['turn_pp_um'])))
            metadata = json.loads(str(saved['metadata_json']))
            rows.append(dict(clearance_um=clearance, source=str(path),
                passed=bool(abs(correct-reported) < 1e-10 and point_error < 1e-10),
                reported_E_dir_um=reported, recomputed_max_of_means_E_dir_um=correct,
                mean_of_per_turn_maximum_um=alternative,
                correct_minus_reported_um=correct-reported,
                alternative_minus_correct_um=alternative-correct,
                maximum_turn_reconstruction_error_um=point_error,
                complete_turns=len(peaks), signed_rpm=metadata['config']['parameters']['rpm']))
    return dict(passed=bool(all(row['passed'] for row in rows)), cases=rows,
                definition='E_dir=max(mean(Ex),mean(Ey)); raw S-H and stored turn boundaries',
                scope='saved V1 metric audit only; old stationarity/numerical claims not revalidated')


def _torque(z, force):
    """Transverse components of z*e_z cross F, in the stated XYZ frame."""
    return z*np.array([-force[1], force[0]])


def observation_matrix(p):
    """The exact configured section/reference; no common translation alias."""
    obs = np.zeros((2, 10))
    if p['measurement_kind'] == 'holder':
        obs[:, 2:4] = np.eye(2)
    else:
        obs[:, ROTOR] = section_matrix(p['z_m'])
    if p['probe_reference'] == 'housing':
        obs[:, 4:6] -= np.eye(2)
    return obs


def _scaled_stiffness(k, length):
    scale = np.ones(len(k))
    scale[8:10] = 1/length
    return k*scale[:, None]*scale[None, :]


def healthy_constraint(p):
    """Healthy supports alone, with H fixed; no target support or tool link."""
    k = np.zeros((4, 4))
    for support in p['healthy_supports']:
        a = section_matrix(support['z_m'])
        k += a.T@np.asarray(support['K'])@a
    scale = np.diag([1., 1., 1/p['L_ref'], 1/p['L_ref']])
    normalized = scale.T@k@scale
    eig = np.linalg.eigvalsh((normalized+normalized.T)/2)
    return dict(matrix_translation_rotation=k.tolist(),
                equivalent_length_eigenvalues_N_m=eig.tolist(),
                full_rank=bool(np.all(eig > max(eig)*1e-10)),
                condition_number=float(np.linalg.cond(normalized)),
                coordinate_scaling='[x, y, L_ref*theta_x, L_ref*theta_y]',
                scope='bilateral healthy tangent; absence of target fit')


def equilibrium_evidence(p, q):
    """Report forces and moments separately; angles never count as metres."""
    mass, gravity = mass_and_gravity(p)
    residual = internal_force(0., q, np.zeros(10), p, 0.)-gravity
    components = force_components(0., q, np.zeros(10), p)
    force_error = float(np.linalg.norm(residual[:8]))
    moment_error = float(np.linalg.norm(residual[8:]))
    scaled = np.r_[residual[:8], residual[8:]/p['L_ref']]
    force_scale = max(float(np.linalg.norm(gravity[:8])), 1.)
    return dict(
        passed=bool(force_error <= 1e-5 and moment_error <= 1e-5*p['L_ref']),
        q_translation_m=q[:8].tolist(), q_tilt_rad=q[8:].tolist(),
        force_residual_N=force_error, moment_residual_Nm=moment_error,
        moment_threshold_Nm=1e-5*p['L_ref'],
        equivalent_force_residual_N=float(np.linalg.norm(scaled)),
        dimensionless_residual=float(np.linalg.norm(scaled)/force_scale),
        force_scale_N=force_scale, reference_length_m=p['L_ref'],
        bearing_force_N=np.asarray(components['bearing_N']).tolist(),
        healthy_support_forces_N=np.asarray(components['healthy_support_forces_N']).tolist(),
        link_force_N=np.asarray(components['link_N']).tolist(),
        fit_force_N=np.asarray(components['fit_N']).tolist(),
        housing_ground_restoring_N=np.asarray(components['housing_ground_N']).tolist(),
        housing_constraint_torque_Nm=np.asarray(components['housing_constraint_torque_Nm']).tolist(),
        total_translating_mass_kg=float(mass[:8:2].sum()),
    )


def validate_mechanics_v3(p, old8p, old6p=None):
    """Independent assembly identities plus the existing contact/label checks."""
    checks = {}
    # A deliberately nonsymmetric state exercises both angular force paths.
    q = np.r_[np.array([1., -2., .2, .5, -1., -4., 2., -8.])*1e-6,
              [3e-6, -7e-6]]
    v = np.r_[np.array([.2, -.3, -.7, .4, .8, .1, -.4, .6])*1e-3,
              [2e-3, -3e-3]]
    dq = np.array([.2, -.4, .3, -.1])
    force = np.array([2.3, -4.1])
    errors = []
    virtual_work = []
    for z in [p['z_f'], p['z_T'], p['z_u'], p['z_m'], -.23, .17]:
        expected_a = np.array([[1., 0., 0., z], [0., 1., -z, 0.]])
        a = section_matrix(z)
        applied = generalized_point_force(force, z)
        expected = np.zeros(10)
        expected[ROTOR] = np.r_[force, _torque(z, force)]
        errors += [float(np.max(np.abs(a-expected_a))),
                   float(np.max(np.abs(applied-expected))),
                   float(np.max(np.abs(rotor_section(q, z)-expected_a@q[ROTOR])))]
        virtual_work.append(float(abs(dq@(a.T@force)-(a@dq)@force)))
    checks['section_mapping_virtual_work_and_point_moment'] = dict(
        passed=bool(max(errors) < 1e-13 and max(virtual_work) < 1e-13),
        maximum_mapping_error=max(errors), maximum_virtual_work_error=max(virtual_work),
        virtual_work_relative_error=max(virtual_work)/max(abs(dq@(section_matrix(p['z_u']).T@force)), 1e-30),
        unit_force_x_moment_Nm=_torque(p['z_u'], np.array([1., 0.])).tolist(),
        unit_force_y_moment_Nm=_torque(p['z_u'], np.array([0., 1.])).tolist())

    masses, gravity = mass_and_gravity(p)
    old_masses, _ = mass_and_gravity(old8p)
    radial_g = np.asarray(p['gravity_radial_m_s2'])
    expected_g = np.r_[np.concatenate([m*radial_g for m in masses[:8:2]]), 0., 0.]
    expected_g += generalized_point_force([0., p.get('radial_load_y', 0.)],
                                          p.get('radial_load_z_m', 0.))
    mass_error = float(abs(masses[:8:2].sum()-old_masses[:8:2].sum()))
    gravity_error = float(np.max(np.abs(gravity-expected_g)))
    checks['mass_inertia_and_gravity_budget'] = dict(
        passed=bool(len(masses) == 10 and np.all(masses > 0)
                    and mass_error < 1e-12 and gravity_error < 1e-12
                    and np.array_equal(masses[8:], [p['Jt'], p['Jt']])
                    and p['Jp'] > 0 and p['Jp'] <= 2*p['Jt']),
        total_mass_kg=float(masses[:8:2].sum()), mass_difference_from_v1_kg=mass_error,
        node_masses_kg=masses[:8:2].tolist(), Jt_kg_m2=p['Jt'], Jp_kg_m2=p['Jp'],
        maximum_gravity_error_N=gravity_error, angular_generalized_gravity_Nm=gravity[8:].tolist(),
        scope='budget/units/rigid-body inertia inequality; geometry provenance remains separate')

    g = gyro_matrix(p)
    g_negative = gyro_matrix(dict(p, rpm=-p['rpm']))
    g_zero = gyro_matrix(dict(p, rpm=0.))
    expected_gyr = np.zeros((10, 10))
    expected_gyr[8, 9] = p['Jp']*p['rpm']*2*np.pi/60
    expected_gyr[9, 8] = -expected_gyr[8, 9]
    checks['gyroscopic_sign_and_zero_power'] = dict(
        passed=bool(np.allclose(g, expected_gyr, rtol=0, atol=1e-14)
                    and np.max(np.abs(g+g.T)) < 1e-14
                    and abs(v@g@v) < 1e-14
                    and np.max(np.abs(g_negative+g)) < 1e-14
                    and np.max(np.abs(g_zero)) == 0.),
        skew_symmetry_error=float(np.max(np.abs(g+g.T))),
        power_W=float(v@g@v), rpm_reversal_error=float(np.max(np.abs(g_negative+g))),
        angular_block_Nm_s_rad=g[8:, 8:].tolist())

    # Assemble independent expected forces from the original reusable laws.
    bearing = bearing_force(.013, rotor_section(q, p['z_f'])-q[6:8],
                            rotor_section(v, p['z_f'])-v[6:8], p)
    _, kl = area_and_link(p, 0.)
    link = kl*(rotor_section(q, p['z_T'])-q[2:4]) + p['cl']*(rotor_section(v, p['z_T'])-v[2:4])
    healthy = [np.asarray(s['K'])@(rotor_section(q, s['z_m'])-q[4:6])
               +np.asarray(s['C'])@(rotor_section(v, s['z_m'])-v[4:6])
               for s in p['healthy_supports']]
    fit, _ = fit_contact_force(q[6:8]-q[4:6], v[6:8]-v[4:6], p)
    ground = p['kh']*q[4:6]+p['ch']*v[4:6]
    expected = g@v
    expected[ROTOR] += section_matrix(p['z_f']).T@bearing + section_matrix(p['z_T']).T@link
    for support, load in zip(p['healthy_supports'], healthy):
        expected[ROTOR] += section_matrix(support['z_m']).T@load
    expected[2:4] = -link
    expected[4:6] = ground-fit-np.sum(healthy, axis=0)
    expected[6:8] = -bearing+fit
    actual = internal_force(.013, q, v, p, 0.)
    node_error = float(np.max(np.abs(actual-expected)))
    force_error = float(np.max(np.abs(actual[:8].reshape(4, 2).sum(axis=0)-ground)))
    transmitted_h_moment = sum((_torque(s['z_m'], load)
                               for s, load in zip(p['healthy_supports'], healthy)), np.zeros(2))
    transmitted_h_moment += _torque(p['z_f'], fit)
    # Physical rotor torque = minus restoring torque (excluding G). Physical
    # forces on independent T and O nodes act at z_T and z_f respectively.
    internal_moment = -(actual-g@v)[8:] + _torque(p['z_T'], link)
    internal_moment += _torque(p['z_f'], bearing-fit) + transmitted_h_moment
    comp = force_components(.013, q, v, p)
    housing_constraint = np.asarray(comp['housing_constraint_torque_Nm'])
    constraint_error = float(np.max(np.abs(housing_constraint+transmitted_h_moment)))
    moment_error = float(np.max(np.abs(internal_moment)))
    checks['internal_reactions_and_housing_constraint_moment'] = dict(
        passed=bool(node_error < 1e-9 and force_error < 1e-9
                    and moment_error < 1e-9 and constraint_error < 1e-9),
        node_assembly_error_mixed_N_Nm=node_error,
        node_assembly_relative_error=node_error/max(float(np.max(np.abs(expected))), 1e-30),
        internal_force_sum_minus_ground_N=force_error,
        internal_moment_sum_Nm=moment_error,
        housing_constraint_balance_error_Nm=constraint_error,
        housing_constraint_torque_Nm=housing_constraint.tolist(),
        constraint_definition='negative torque transmitted by healthy supports and target fit to rigid H')

    coincident = dict(p, z_f=0., z_T=0., z_u=0., z_m=0.,
                     healthy_supports=[dict(s, z_m=0.) for s in p['healthy_supports']])
    reduced = dict(old8p, **{key: p[key] for key in old8p if key in p})
    reduced['Kother'] = np.sum([np.asarray(s['K']) for s in p['healthy_supports']], axis=0)
    reduced['Cother'] = np.sum([np.asarray(s['C']) for s in p['healthy_supports']], axis=0)
    frozen_q, frozen_v = q.copy(), v.copy()
    frozen_q[8:] = frozen_v[8:] = 0.
    new_force = internal_force(.013, frozen_q, frozen_v, coincident, 0.)
    old_force = internal_force(.013, q[:8], v[:8], reduced, 0.)
    reduction_error = float(np.max(np.abs(new_force[:8]-old_force)))
    checks['coincident_supports_frozen_tilt_v1_reduction'] = dict(
        passed=bool(reduction_error < 1e-9 and np.max(np.abs(new_force[8:])) < 1e-12),
        maximum_translation_force_error_N=reduction_error,
        residual_torque_Nm=new_force[8:].tolist(),
        scope='algebraic constrained reduction, not a singular 10-DOF dynamic solve')

    qeq, _ = equilibrium(p, 0.)
    checks['static_force_and_moment_equilibrium'] = equilibrium_evidence(p, qeq)
    m, c_total, k = linear_matrices(p, 0., qeq)
    ks = _scaled_stiffness(k, p['L_ref'])
    cs = _scaled_stiffness(c_total-g, p['L_ref'])
    ksym = float(np.linalg.norm(ks-ks.T)/np.linalg.norm(ks))
    csym = float(np.linalg.norm(cs-cs.T)/max(np.linalg.norm(cs), 1.))
    keig = np.linalg.eigvalsh((ks+ks.T)/2)
    ceig = np.linalg.eigvalsh((cs+cs.T)/2)
    checks['elastic_energy_and_dissipative_tangent'] = dict(
        passed=bool(ksym < 1e-6 and csym < 1e-6 and keig.min() > 0
                    and ceig.min() >= -1e-6*max(ceig.max(), 1.)),
        stiffness_relative_asymmetry=ksym,
        stiffness_equivalent_length_eigenvalues_N_m=keig.tolist(),
        damping_relative_asymmetry_after_removing_G=csym,
        damping_equivalent_length_eigenvalues_N_s_m=ceig.tolist(),
        scope='local stored-energy Hessian at the loaded equilibrium; C_total includes G exactly once')
    checks['remaining_healthy_tilt_constraint'] = healthy_constraint(p)
    checks['remaining_healthy_tilt_constraint']['passed'] = checks['remaining_healthy_tilt_constraint']['full_rank']
    if old6p is not None:
        legacy = validate_legacy_mechanics(old8p, old6p)
        checks['reused_v1_contact_circle_and_legacy_dimensions'] = legacy
    return dict(passed=bool(all(item['passed'] for item in checks.values())), checks=checks,
                machine_physics_validated=False,
                scope='declared demonstration equations, not identification or experimental validation')


def _complex_record(value):
    value = np.asarray(value)
    return dict(real=value.real.tolist(), imag=value.imag.tolist(),
                magnitude=np.abs(value).tolist(), phase_rad=np.angle(value).tolist())


def preflight_v3(p, force_amplitude_N, gaps_um, phase_rad=0.):
    """Cheap static/tangent screening under one frozen structure and forcing.

    The fixed +frequency phasor [1,-i*sign(rpm)] preserves the signed rotation.
    A time-domain contact transition still needs separate verification.
    """
    omega = abs(p['rpm'])*2*np.pi/60
    sign = np.sign(p['rpm'])
    force_phase = np.exp(1j*sign*phase_rad)*np.array([1., -1j*sign])
    drive = np.column_stack([generalized_point_force(axis, p['z_u']) for axis in np.eye(2)])
    obs = observation_matrix(p)
    hobs = np.eye(10)[4:6]
    outer_obs = np.eye(10)[6:8]-hobs
    moment_drive = np.eye(10)[:, 8:10]
    healthy = healthy_constraint(p)
    baseline_q, _ = equilibrium(dict(p, fit_clearance_um=0.), 0.)
    rows = []
    previous = None
    for clearance in sorted(set(float(value) for value in gaps_um)):
        case = dict(p, fit_clearance_um=clearance)
        q, _ = equilibrium(case, 0., previous)
        previous = q
        evidence = equilibrium_evidence(case, q)
        m, c_total, k = linear_matrices(case, 0., q)
        state = np.block([[np.zeros((10, 10)), np.eye(10)],
                          [-np.linalg.solve(m, k), -np.linalg.solve(m, c_total)]])
        poles = np.linalg.eigvals(state)
        oscillatory = poles[poles.imag > 1e-8]
        oscillatory = oscillatory[np.argsort(oscillatory.imag)]
        slow = float(np.max(poles.real))
        z = k-omega**2*m+1j*omega*c_total
        dynamic_force = np.linalg.solve(z, drive)
        dynamic_moment = np.linalg.solve(z, moment_drive)
        static_force = np.linalg.solve(k, drive)
        static_moment = np.linalg.solve(k, moment_drive)
        response = dynamic_force@force_phase*force_amplitude_N
        motion = obs@response
        outer_motion = outer_obs@response
        healthy_motion = []
        for support in p['healthy_supports']:
            support_obs = np.zeros((2, 10))
            support_obs[:, ROTOR] = section_matrix(support['z_m'])
            support_obs[:, 4:6] -= np.eye(2)
            mean = support_obs@q
            baseline_mean = support_obs@baseline_q
            ac = support_obs@response
            peak = float(np.linalg.svd(np.column_stack((ac.real, -ac.imag)),
                                       compute_uv=False)[0])
            excursion_bound = float(np.linalg.norm(mean-baseline_mean)+peak)
            domain = support.get('applicability_excursion_from_baseline_m')
            healthy_motion.append(dict(name=support['name'], z_m=support['z_m'],
                static_relative_xy_um=(mean*1e6).tolist(),
                ideal_zero_gap_baseline_relative_xy_um=(baseline_mean*1e6).tolist(),
                dynamic_ellipse_maximum_radius_um=peak*1e6,
                static_plus_dynamic_radius_bound_um=float((np.linalg.norm(mean)+peak)*1e6),
                excursion_from_ideal_baseline_bound_um=excursion_bound*1e6,
                assumed_domain_excursion_limit_um=domain*1e6 if domain is not None else None,
                inside_assumed_linear_domain=bool(excursion_bound <= domain) if domain is not None else None,
                linear_mean_force_N=(np.asarray(support['K'])@mean).tolist(),
                linear_dynamic_force_peak_xy_N=np.abs((np.asarray(support['K'])
                    +1j*omega*np.asarray(support['C']))@ac).tolist()))
        # Largest radius of a single-frequency planar ellipse, exact for this
        # linear response; triangle bound adds the equilibrium displacement.
        outer_peak = float(np.linalg.svd(np.column_stack((outer_motion.real, -outer_motion.imag)),
                                            compute_uv=False)[0])
        static_outer = outer_obs@q
        fit, fit_diag = fit_contact_force(static_outer, np.zeros(2), case)
        predicted_pp = 2*np.abs(motion)*1e6
        observed_compliance = float(np.max(np.abs(motion))*1e6/force_amplitude_N)
        force_required = float(10/observed_compliance)
        relative_bearing = rotor_section(q, p['z_f'])-q[6:8]
        phi = p['cage_phase'] + np.arange(p['balls'])*2*np.pi/p['balls']
        normal = p['normal_overlap'] + np.cos(p['contact_angle'])*(np.column_stack((np.cos(phi), np.sin(phi)))@relative_bearing)
        rows.append(dict(
            fit_clearance_um=clearance, equilibrium=evidence,
            static_measurement_relative_um=(obs@q*1e6).tolist(),
            static_outer_relative_um=(static_outer*1e6).tolist(),
            static_fit_active_directions=int(fit_diag['active_count']),
            static_fit_force_N=fit.tolist(),
            static_maximum_fit_compression_um=float(np.max(fit_diag['compression_m'])*1e6),
            static_bearing_normal_approach_um=(normal*1e6).tolist(),
            healthy_support_envelopes=healthy_motion,
            tilt_static_rad=q[8:].tolist(),
            tilt_dynamic_peak_rad=np.abs(response[8:]).tolist(),
            frozen_poles_real_s_inv=poles.real.tolist(), frozen_poles_imag_rad_s=poles.imag.tolist(),
            frozen_mode_frequency_hz=(oscillatory.imag/(2*np.pi)).tolist(),
            frozen_mode_decay_s_inv=(-oscillatory.real).tolist(),
            frozen_stable=bool(slow < 0), slowest_pole_real_s_inv=slow,
            slowest_decay_time_s=float(-1/slow) if slow < 0 else None,
            stiffness_condition_equivalent_length=float(np.linalg.cond(_scaled_stiffness(k, p['L_ref']))),
            C_total_includes_gyro_once=True,
            measured_relative_static_force_compliance_um_N=(obs@static_force*1e6).tolist(),
            measured_relative_static_moment_compliance_um_Nm=(obs@static_moment*1e6).tolist(),
            measured_relative_dynamic_force_compliance_um_N=_complex_record(obs@dynamic_force*1e6),
            measured_relative_dynamic_moment_compliance_um_Nm=_complex_record(obs@dynamic_moment*1e6),
            housing_dynamic_force_displacement_um_N=_complex_record(hobs@dynamic_force*1e6),
            housing_dynamic_force_velocity_mm_s_N=_complex_record(1j*omega*hobs@dynamic_force*1e3),
            housing_dynamic_force_acceleration_m_s2_N=_complex_record(-omega**2*hobs@dynamic_force),
            housing_dynamic_moment_displacement_um_Nm=_complex_record(hobs@dynamic_moment*1e6),
            housing_dynamic_moment_velocity_mm_s_Nm=_complex_record(1j*omega*hobs@dynamic_moment*1e3),
            housing_dynamic_moment_acceleration_m_s2_Nm=_complex_record(-omega**2*hobs@dynamic_moment),
            common_drive_relative_pp_xy_um=predicted_pp.tolist(),
            common_drive_relative_position_phasor_um=_complex_record(motion*1e6),
            common_drive_housing_position_phasor_um=_complex_record(hobs@response*1e6),
            common_drive_E_dir_um=float(max(predicted_pp)),
            common_drive_housing_velocity_peak_mm_s=(np.abs(omega*hobs@response)*1e3).tolist(),
            common_drive_housing_acceleration_peak_m_s2=np.abs(omega**2*hobs@response).tolist(),
            predicted_rotating_force_peak_compliance_um_N=observed_compliance,
            linear_extrapolated_force_for_20um_N=force_required,
            amplitude_shortfall_factor_20um=20/float(max(predicted_pp)),
            outer_static_radius_um=float(np.linalg.norm(static_outer)*1e6),
            outer_dynamic_ellipse_maximum_radius_um=outer_peak*1e6,
            outer_static_plus_dynamic_bound_um=float((np.linalg.norm(static_outer)+outer_peak)*1e6),
            envelope_bound_above_clearance_um=float((np.linalg.norm(static_outer)+outer_peak)*1e6-clearance),
            linearization_scope='frozen t=0 tangent, not a nonlinear upper/lower bound or stability proof'))
    return dict(
        source='computed from the frozen 10-DOF demonstration configuration',
        observation=dict(measurement_kind=p['measurement_kind'], z_m=p['z_m'],
                         probe_reference=p['probe_reference']),
        rpm=p['rpm'], fixed_force_peak_N=force_amplitude_N,
        required_peak_compliance_for_20um_um_N=10/force_amplitude_N,
        healthy_only_constraint=healthy, cases=rows,
        warning='Extrapolated force and linear envelopes diagnose scale; they are not proposed inputs or validated nonlinear predictions.',
        physics_validated_pass=False)
