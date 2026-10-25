"""Reproducible force-law and label checks for the fixed-front-support model.

Synthetic positions here are validation inputs only, never simulation data.
Dynamic convergence and steady-state checks belong to the sweep entry point.
"""
import numpy as np

from spindle.dynamics_model import (
    area_and_link, bearing_force, fit_contact_force, internal_force,
    mass_and_gravity,
)
from utils.runout_labels import complete_turn_runout


def validate_mechanics(p, old_p):
    """Return JSON-ready evidence; do not alter parameters or integrate dynamics."""
    checks = {}
    masses, gravity = mass_and_gravity(p)
    old_masses, old_gravity = mass_and_gravity(old_p)
    total_mass = float(masses[::2].sum())
    original_mass = float(old_masses[::2].sum())
    gravity_error = float(np.max(np.abs(gravity.reshape(4, 2).sum(axis=0)
                                        - old_gravity.reshape(3, 2).sum(axis=0))))
    checks['mass_and_gravity_budget'] = dict(
        passed=bool(masses.shape == (8,) and np.all(masses > 0)
                    and abs(total_mass-original_mass) < 1e-12
                    and gravity_error < 1e-12),
        new_total_mass_kg=total_mass, old_total_mass_kg=original_mass,
        node_mass_kg=masses[::2].tolist(), gravity_difference_N=gravity_error)

    other_k = np.asarray(p['Kother'], dtype=float)
    other_c = np.asarray(p['Cother'], dtype=float)
    k_eigen = np.linalg.eigvalsh(other_k)
    c_eigen = np.linalg.eigvalsh(other_c)
    k_symmetry = float(np.linalg.norm(other_k-other_k.T)/np.linalg.norm(other_k))
    c_symmetry = float(np.linalg.norm(other_c-other_c.T))
    positive = ('ms', 'mt', 'mh', 'mO', 'kh', 'kbend', 'kappa', 'ki', 'ko', 'Kf')
    nonnegative = ('ch', 'cl', 'cb_total', 'Df', 'normal_overlap', 'fit_clearance_um')
    checks['parameter_signs'] = dict(
        passed=bool(all(p[name] > 0 for name in positive)
                    and all(p[name] >= 0 for name in nonnegative)
                    and p['N'] >= 4 and p['N'] % 2 == 0
                    and np.all(k_eigen > 0) and np.all(c_eigen >= -1e-12)
                    and k_symmetry < 1e-10 and c_symmetry < 1e-10),
        Kother_eigenvalues_N_m=k_eigen.tolist(),
        Cother_eigenvalues_N_s_m=c_eigen.tolist(),
        Kother_relative_asymmetry=k_symmetry,
        Cother_asymmetry_N_s_m=c_symmetry)

    # A nonzero state exercises every connection. The sum isolates ground forces;
    # explicit node balances also detect accidentally applying B directly to H.
    q = np.array([1., -2., .2, .5, -1., -4., 2., -8.])*1e-6
    v = np.array([.2, -.3, -.7, .4, .8, .1, -.4, .6])*1e-3
    actual = internal_force(.013, q, v, p, 0.)
    bearing = bearing_force(.013, q[:2]-q[6:8], v[:2]-v[6:8], p)
    _, link_k = area_and_link(p, 0.)
    link = link_k*(q[:2]-q[2:4])+p['cl']*(v[:2]-v[2:4])
    other = other_k@(q[:2]-q[4:6])+other_c@(v[:2]-v[4:6])
    contact, _ = fit_contact_force(q[6:8]-q[4:6], v[6:8]-v[4:6], p)
    ground = p['kh']*q[4:6]+p['ch']*v[4:6]
    expected = np.r_[bearing+link+other, -link, -other-contact+ground, -bearing+contact]
    sum_error = float(np.max(np.abs(actual.reshape(4, 2).sum(axis=0)-ground)))
    node_error = float(np.max(np.abs(actual-expected)))
    checks['internal_reactions'] = dict(
        passed=bool(sum_error < 1e-10 and node_error < 1e-10),
        internal_sum_minus_ground_maximum_N=sum_error,
        node_balance_maximum_error_N=node_error)

    contact_p = dict(p, fit_clearance_um=5.)
    zero = np.zeros(2)
    inside, inside_diag = fit_contact_force(np.array([2., 1.])*1e-6,
                                            np.array([.2, -.1]), contact_p)
    compressed, compressed_diag = fit_contact_force(np.array([7e-6, 0.]),
                                                     zero, contact_p)
    separating, separating_diag = fit_contact_force(np.array([7e-6, 0.]),
                                                     np.array([-1e3, 0.]), contact_p)
    direction_forces = np.r_[inside_diag['normal_force_N'],
                            compressed_diag['normal_force_N'],
                            separating_diag['normal_force_N']]
    checks['unilateral_contact'] = dict(
        passed=bool(np.linalg.norm(inside) < 1e-12 and compressed[0] > 0
                    and abs(compressed[1]) < 1e-10
                    and np.min(direction_forces) >= 0
                    and (p['Df'] == 0 or np.linalg.norm(separating) < 1e-12)),
        dead_zone_force_N=inside.tolist(),
        compressed_force_N=compressed.tolist(),
        rapid_unloading_force_N=separating.tolist(),
        minimum_direction_force_N=float(direction_forces.min()))

    linear_errors = {}
    for count in (32, 64):
        elastic_p = dict(p, fit_clearance_um=0., Df=0., N=count)
        errors = []
        for angle in np.linspace(0., 2*np.pi, 37, endpoint=False):
            relative = 7e-6*np.array([np.cos(angle), np.sin(angle)])
            force, _ = fit_contact_force(relative, zero, elastic_p)
            errors.append(np.linalg.norm(force-p['Kf']*relative)/(p['Kf']*7e-6))
        linear_errors[str(count)] = float(max(errors))
    checks['zero_clearance_elastic_limit'] = dict(
        passed=bool(max(linear_errors.values()) < 1e-12),
        maximum_relative_force_error_by_N=linear_errors)

    # Fc acts on H; dissipated power is +(Fc-Felastic) dot (vO-vH).
    # The mechanical power of the contact pair has the opposite sign.
    dissipation = []
    minimum_normal = 0.
    for angle in np.linspace(0., 2*np.pi, 17, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        for radius_um in (0., 2., 7., 15.):
            relative = radius_um*1e-6*direction
            elastic, _ = fit_contact_force(relative, zero, dict(contact_p, Df=0.))
            for velocity_angle in (angle, angle+1.1, angle+np.pi):
                velocity = .01*np.array([np.cos(velocity_angle), np.sin(velocity_angle)])
                force, diagnostics = fit_contact_force(relative, velocity, contact_p)
                dissipation.append(float((force-elastic)@velocity))
                minimum_normal = min(minimum_normal, float(np.min(diagnostics['normal_force_N'])))
    checks['contact_dissipation'] = dict(
        passed=bool(min(dissipation) >= -1e-12 and minimum_normal >= 0.),
        tested_states=len(dissipation), minimum_dissipation_W=min(dissipation),
        maximum_dissipation_W=max(dissipation),
        minimum_direction_force_N=minimum_normal,
        definition='(Fc - Felastic) dot (vO - vH); contact subsystem only')

    fs, rpm, phase = 65536, 7000., .37
    time_s = .003+np.arange(int(.2*fs)+1)/fs
    angle = 2*np.pi*rpm/60*time_s+phase
    position_um = 5.*np.column_stack((np.cos(angle), np.sin(angle)))
    turns = complete_turn_runout(time_s, position_um, rpm, phase_at_zero_rad=phase)
    translated = complete_turn_runout(time_s, position_um+[37., -81.], rpm,
                                      phase_at_zero_rad=phase)
    peaks = turns['peak_to_peak_um']
    peak_error = float(np.max(np.abs(peaks-10.)))
    translation_error = float(np.max(np.abs(peaks-translated['peak_to_peak_um'])))
    checks['known_circle_runout'] = dict(
        passed=bool(peak_error < .01 and translation_error < 1e-10),
        radius_um=5., expected_xy_peak_to_peak_um=[10., 10.],
        actual_xy_mean_um=peaks.mean(axis=0).tolist(),
        maximum_peak_to_peak_error_um=peak_error,
        translation_error_um=translation_error, complete_turns=int(len(peaks)),
        sample_rate_hz=fs, input_is_validation_only=True)

    rejected = {}
    for key, value in dict(platform_mass=1., outer_pit_area_m2=1e-8,
                           inner_pit_area_m2=1e-8,
                           ball_deformation_amplitude_m=1e-6).items():
        try:
            internal_force(0., np.zeros(8), np.zeros(8), dict(p, **{key: value}), 0.)
            rejected[key] = False
        except ValueError:
            rejected[key] = True
    try:
        internal_force(0., np.zeros(8), np.zeros(8), p, 1e-8)
        rejected['lost_area'] = False
    except ValueError:
        rejected['lost_area'] = True
    checks['unsupported_combinations_rejected'] = dict(
        passed=bool(all(rejected.values())), rejected=rejected)

    dimensions = []
    for legacy in (old_p, dict(old_p, platform_mass=12., platform_ky=2e6, platform_cy=300.)):
        legacy_m, legacy_g = mass_and_gravity(legacy)
        size = len(legacy_m)
        restoring = internal_force(0., np.zeros(size), np.zeros(size), legacy, 0.)
        dimensions.append(dict(mass=size, gravity=len(legacy_g), restoring=len(restoring)))
    checks['legacy_dimensions'] = dict(
        passed=dimensions == [dict(mass=6, gravity=6, restoring=6),
                              dict(mass=7, gravity=7, restoring=7)],
        actual=dimensions,
        scope='dimension compatibility only; no legacy time-domain regression')
    return dict(passed=bool(all(check['passed'] for check in checks.values())), checks=checks)
