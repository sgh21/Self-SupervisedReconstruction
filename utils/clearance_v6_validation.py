"""Reproducible V6 mechanical, contact and numerical-backend checks (P1).

The passive tests stop both spin and unbalance. They do not claim that the
rotating, moving-ball-contact system has an autonomous decreasing energy.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from spindle import dynamics_model as model
from spindle.fast_rhs import prepare_fast_rhs
from scripts.run_clearance_v3 import write_csv, write_json


def energy(q, v, p):
    """Mechanical energy including gravity; valid for the stopped-cage audit."""
    if p['rpm'] != 0:
        raise ValueError('This autonomous energy audit requires stopped spin/cages.')
    masses, gravity = model.mass_and_gravity(p)
    value = .5*np.sum(masses*v*v, axis=-1)-q@gravity
    value += .5*p['kh']*np.sum(q[..., 4:6]**2, axis=-1)
    link = model.rotor_section(q, p['z_T'])-q[..., 2:4]
    value += .5*model.area_and_link(p, 0)[1]*np.sum(link*link, axis=-1)
    rear = model.rear_bearing_parameters(p)
    bearings = [(p, p['z_f'], 6), (p, p['z_f2'], 10)]
    bearings += [(rear, s['z_m'], 4) for s in p['rear_supports']]
    for bearing, z, endpoint in bearings:
        phi = bearing['cage_phase']+np.arange(bearing['balls'])*2*np.pi/bearing['balls']
        relative = model.rotor_section(q, z)-q[..., endpoint:endpoint+2]
        delta = bearing['normal_overlap']+np.cos(bearing['contact_angle'])*(relative@np.array([np.cos(phi), np.sin(phi)]))
        value += .4*model.bearing_keq(bearing)*np.sum(np.maximum(delta, 0)**2.5, axis=-1)
    phi = np.arange(p['N'])*2*np.pi/p['N']
    directions = np.array([np.cos(phi), np.sin(phi)])
    for endpoint, stiffness, gap in ((6, p['Kf'], p['fit_clearance_um']),
                                     (10, p['Kf2'], p['fit2_clearance_um'])):
        # Keep diagnostic memory independent of the length of the saved run.
        relative = q[..., endpoint:endpoint+2]-q[..., 4:6]
        delta = np.maximum(relative@directions-gap*1e-6, 0)
        value += 2*stiffness/p['N']*np.sum(delta*delta, axis=-1)
    return value


def audit(config, folder, source_before):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p, settings = config['parameters'], config['simulation']
    checks, details = {}, {}
    masses, gravity = model.mass_and_gravity(p)
    checks['v6_explicit_12dof'] = p.get('nonlinear_bearings_v6') and p['model_dof'] == 12
    checks['no_legacy_linear_bypass'] = not any(k in p for k in ('healthy_supports', 'Kother', 'Cother', 'Kb2', 'Cb2'))
    checks['mass_positive'] = len(masses) == 12 and np.all(masses > 0)
    total = sum(p[k] for k in ('ms', 'mh', 'mO', 'mO2'))
    checks['mass_budget'] = abs(total-44.613) < 1e-9 and abs(total+p['mt']-45.217) < 1e-9
    checks['outer_mass_bounds'] = 0 < p['mO'] < .21 and 0 < p['mO2'] < .21
    checks['rotor_mass_and_inertia_updated'] = p['ms'] == 4 and np.isclose(p['Jt'], .02006870333454608*1.6, rtol=1e-12) and np.isclose(p['Jp'], .0026374066690921603*1.6, rtol=1e-12)
    details['mass_budget_kg'] = dict(internal_rotor=p['ms'], holder=p['mt'], housing=p['mh'], outer1=p['mO'], outer2=p['mO2'], spindle_sum=total, total_with_holder=total+p['mt'])
    rear = model.rear_bearing_parameters(p)
    single = []
    for name, bp in [('front', p), ('rear', rear)]:
        ca, sa = np.cos(bp['contact_angle']), np.sin(bp['contact_angle'])
        pa = bp['balls']*model.bearing_keq(bp)*bp['normal_overlap']**1.5*sa
        kr = .75*bp['balls']*model.bearing_keq(bp)*np.sqrt(bp['normal_overlap'])*ca**2
        tangent = model._bearing_tangent(np.zeros(2), bp)
        err = float(abs(tangent-kr*np.eye(2)).max()/kr)
        single.append(dict(name=name, balls=bp['balls'], Keq=model.bearing_keq(bp), delta0_m=bp['normal_overlap'], preload_computed_N=pa, preload_reference_N=bp['preload_reference_N'], radial_tangent_computed_N_m=kr, radial_tangent_reference_N_m=bp['radial_tangent_reference_N_m'], tangent_finite_difference_N_m=tangent.tolist(), tangent_relative_error=err, implied_axial_pair_tangent_N_m=3*bp['balls']*model.bearing_keq(bp)*np.sqrt(bp['normal_overlap'])*sa**2))
        checks[name+'_preload_and_tangent'] = abs(pa/bp['preload_reference_N']-1) < .005 and abs(kr/bp['radial_tangent_reference_N_m']-1) < .005 and err < .001
        _, diag = model.bearing_contact_state(np.array([0., .01]), np.array([[0., 0.], [-10e-6, 3e-6]]), np.array([[0., 0.], [-2., 2.]]), bp)
        checks[name+'_contact_no_tension'] = np.all(diag['normal_force_N'] >= 0) and np.all(diag['normal_force_N'][diag['delta_m'] <= 0] == 0)
    details['single_bearing'] = single
    qtest, vtest = np.zeros(12), np.zeros(12)
    qtest[6:8] = qtest[10:12] = [1.7e-6, -1.2e-6]
    vtest[6:8] = vtest[10:12] = [.001, -.002]
    f1 = model.bearing_force(.017, -qtest[6:8], -vtest[6:8], p)
    f2 = model.front2_forces(qtest, vtest, p, t=.017)[0]
    checks['fronts_identical_constitutive_law'] = np.array_equal(f1, f2)
    # Known point-force duality and whole-model action/reaction conservation.
    rng = np.random.default_rng(6009)
    qr, f = rng.normal(size=4), rng.normal(size=2)
    a = model.section_matrix(.127)
    checks['virtual_work_and_moment'] = abs(f@(a@qr)-(a.T@f)@qr) < 1e-12 and np.allclose((a.T@f)[2:], [-.127*f[1], .127*f[0]])
    q0, _ = model.equilibrium_continuation(p)
    q, v = q0+rng.normal(size=12)*1e-6, rng.normal(size=12)*.001
    internal = model.internal_force(.021, q, v, p, 0)
    gyro = model.gyro_matrix(p)
    checks['gyro_skew_zero_power'] = abs(gyro+gyro.T).max() == 0 and abs(v@gyro@v) < 1e-12
    without = model.internal_force(.021, q, v, dict(p, Jp=0), 0)
    checks['gyro_once'] = np.allclose(internal-without, gyro@v, atol=1e-10)
    summed = internal[:8].reshape(4, 2).sum(axis=0)+internal[10:12]
    residual = summed-p['kh']*q[4:6]-p['ch']*v[4:6]
    checks['translational_action_reaction'] = max(abs(residual)) < 1e-9
    forces = model.force_components(.021, q, v, p)
    expected_rotor = model.section_matrix(p['z_f']).T@forces['bearing_N']+model.section_matrix(p['z_f2']).T@forces['bearing2_N']+model.section_matrix(p['z_T']).T@forces['link_N']
    for support, force in zip(p['rear_supports'], forces['rear_bearing_forces_N']):
        expected_rotor += model.section_matrix(support['z_m']).T@force
    checks['all_support_moments'] = np.allclose(internal[model.ROTOR]-(gyro@v)[model.ROTOR], expected_rotor, atol=1e-10)
    rear_matrix = sum(model.section_matrix(s['z_m']).T@(rear['radial_tangent_reference_N_m']*np.eye(2))@model.section_matrix(s['z_m']) for s in p['rear_supports'])
    checks['rear_only_rotor_rank_four'] = np.linalg.matrix_rank(rear_matrix) == 4
    details['rear_only_tangent_eigenvalues_mixed_SI'] = np.linalg.eigvalsh(rear_matrix).tolist()
    details['rear_only_rank'] = int(np.linalg.matrix_rank(rear_matrix))
    for name, k, d in [('F1', p['Kf'], p['Df']), ('F2', p['Kf2'], p['Df2'])]:
        fit = dict(N=p['N'], Kf=k, Df=d, fit_clearance_um=2)
        f, _ = model.fit_contact_force(np.array([1e-6, 0]), np.array([20., 0]), fit)
        _, diag = model.fit_contact_force(np.array([3e-6, 0]), np.array([-20., 0]), fit)
        checks[name+'_fit_deadzone_and_no_tension'] = np.all(f == 0) and np.all(diag['normal_force_N'] >= 0)
        r = np.array([1.2e-6, -.8e-6])
        f, _ = model.fit_contact_force(r, np.zeros(2), dict(fit, Df=0, fit_clearance_um=0))
        checks[name+'_fit_zero_gap_elastic_limit'] = np.allclose(f, k*r, rtol=1e-12)
    static, preflight = [], []
    omega = p['rpm']*2*np.pi/60
    for gap in (2., 10.):
        local = model.front_group_parameters(p, gap)
        qe, res = model.equilibrium_continuation(local)
        M, C, K = model.linear_matrices(local, 0, qe)
        fc = model.force_components(0., qe, np.zeros(12), local)
        residual = model.internal_force(0., qe, np.zeros(12), local, 0)-gravity
        eigen = eigh((K+K.T)/2, M, eigvals_only=True)
        poles = np.linalg.eigvals(np.block([[np.zeros((12, 12)), np.eye(12)], [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]]))
        phasor = np.array([1., -1j])
        response = np.linalg.solve(K-omega**2*M+1j*omega*C, model.generalized_point_force(phasor, p['z_u'], 12))
        motion = model.section_matrix(p['z_m'])@response[model.ROTOR]
        freqs = np.sqrt(np.maximum(eigen, 0))/2/np.pi
        row = dict(gap_um=gap, static_max_force_residual_N=float(max(abs(residual[np.r_[0:8, 10:12]]))), static_max_moment_residual_Nm=float(max(abs(residual[8:10]))), equivalent_residual_N=res, minimum_frequency_Hz=float(freqs.min()), maximum_frequency_Hz=float(freqs.max()), maximum_pole_real_per_s=float(poles.real.max()), local_1N_frozen_E_um=float(2e6*max(abs(motion))), rear_only_rank=int(np.linalg.matrix_rank(rear_matrix)))
        preflight.append(row)
        static.append(dict(gap_um=gap, q=qe.tolist(), forces={k: np.asarray(value).tolist() for k, value in fc.items()}, modal_frequencies_Hz=freqs.tolist(), tangent_K=K.tolist(), tangent_C_including_G=C.tolist(), M=M.tolist()))
        checks[f'static_gap_{gap:g}'] = row['static_max_force_residual_N'] < 1e-6 and row['static_max_moment_residual_Nm'] < 1e-6
    write_json(folder/'static_equilibria.json', static)
    write_csv(folder/'preflight.csv', preflight)
    # Force and switching-state equivalence, with actual 18/16-ball parameters.
    backend = []
    for n in (512, 1024):
        for gap in (0., 2., 10.):
            local = model.front_group_parameters(dict(p, N=n), gap)
            qe, _ = model.equilibrium_continuation(local)
            compiled = prepare_fast_rhs(local, (5/omega**2, -.3661687497264663))
            maximum = 0.
            for index in range(60):
                q = qe+rng.normal(size=12)*2e-6
                q[8:10] = qe[8:10]+rng.normal(size=2)*1e-4
                if index < 6:
                    q[:] = 0
                    q[6] = gap*1e-6*(1+(index-3)*1e-9)
                    q[10] = -p['normal_overlap']/np.cos(p['contact_angle'])*(1+(index-3)*1e-9)
                v = rng.normal(size=12)*.005
                t = 0. if index < 6 else float(rng.uniform(0, 2))
                error = model.internal_force(t, q, v, local, 0)-compiled.internal_force(t, q, v)
                maximum = max(maximum, float(max(abs(error))))
            backend.append(dict(N=n, gap_um=gap, states=60, maximum_force_or_moment_difference=maximum))
    details['backend_force_comparisons'] = backend
    checks['python_numba_forces'] = all(b['maximum_force_or_moment_difference'] < 1e-8 for b in backend)
    # A short healthy trajectory also exercises parameter caching and exports.
    atols = np.r_[np.full(12, 1e-13), np.full(12, 1e-9)]
    atols[8:10] = 1e-12; atols[20:22] = 1e-8
    trajectories = []
    for backend_name in ('python', 'numba'):
        trajectories.append(model.simulate(dict(p, rhs_backend=backend_name), 0, [], [], duration=.012, warmup=0, fs=65536, max_step=settings['max_step_s'], rtol=2e-9, atol=atols, unbalance=(5/omega**2, -.3661687497264663)))
    trajectory_differences = {key: float(abs(trajectories[0][key]-trajectories[1][key]).max()) for key in ('q', 'v', 'acc', 'bearing2_force_N', 'rear_bearing_forces_N')}
    checks['python_numba_short_trajectory'] = trajectory_differences['q'] < 1e-10 and trajectory_differences['v'] < 1e-7 and trajectory_differences['acc'] < .01
    details['backend_short_trajectory_max_differences_SI'] = trajectory_differences
    np.savez_compressed(folder/'backend_short_trajectory.npz', t=trajectories[0]['t'], python_q=trajectories[0]['q'], numba_q=trajectories[1]['q'], python_v=trajectories[0]['v'], numba_v=trajectories[1]['v'], python_acc=trajectories[0]['acc'], numba_acc=trajectories[1]['acc'])
    # Legacy equations stay available and retain their numerical responses.
    source = Path(source_before)/'src/spindle/dynamics_model.py'
    spec = importlib.util.spec_from_file_location('before_v6', source)
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    legacy = json.loads(Path('config/clearance_v5_config.json').read_text(encoding='utf-8'))['parameters']
    old_q, _ = previous.equilibrium_continuation(legacy)
    legacy_max = 0.
    for _ in range(20):
        q = old_q+rng.normal(size=12)*1e-6; v = rng.normal(size=12)*.001
        legacy_max = max(legacy_max, float(max(abs(previous.internal_force(.123, q, v, legacy, 0)-model.internal_force(.123, q, v, legacy, 0)))))
    checks['legacy_v5_force_regression'] = legacy_max < 1e-8
    details['legacy_v5_max_force_difference'] = legacy_max
    # Spin is stopped: autonomous potential gradient and passive decay are valid.
    passive = []
    for gap in (2., 10.):
        stopped = model.front_group_parameters(dict(p, rpm=0), gap)
        qe, _ = model.equilibrium_continuation(stopped)
        qi = qe.copy(); qi[0] += 1e-7; qi[8] += 1e-6
        zero = np.zeros(12)
        gradient = []
        for i in range(12):
            dq = np.zeros(12); dq[i] = 1e-9 if i in (8, 9) else 1e-10
            gradient.append((energy(qi+dq, zero, stopped)-energy(qi-dq, zero, stopped))/(2*dq[i]))
        error = np.asarray(gradient)-(model.internal_force(0, qi, zero, stopped, 0)-model.mass_and_gravity(stopped)[1])
        error[8:10] /= p['L_ref']
        checks[f'energy_gradient_gap_{gap:g}'] = max(abs(error)) < 1e-4
        result = model.simulate(stopped, 0, [], [], duration=.03, warmup=0, fs=65536, max_step=settings['max_step_s'], q_initial=qe, state_initial=np.r_[qi, zero], unbalance=(0., 0.))
        e = energy(result['q'], result['v'], stopped)
        excess = e-energy(qe, zero, stopped)
        gain = float(np.diff(e).max()); tolerance = max(1e-12, abs(excess[0])*1e-7)
        checks[f'passive_decay_gap_{gap:g}'] = gain <= tolerance and excess[-1] < excess[0]
        passive.append(dict(gap_um=gap, initial_excess_J=float(excess[0]), final_excess_J=float(excess[-1]), max_step_energy_gain_J=gain, gain_tolerance_J=tolerance, gradient_max_equivalent_N=float(max(abs(error))), spin_rpm=0, unbalance_kg_m=0))
        np.savez_compressed(folder/f'passive_decay_gap_{gap:g}um.npz', t=result['t'], q=result['q'], v=result['v'], acc=result['acc'], energy_J=e, excess_energy_J=excess)
    details['passive_decay'] = passive
    checks = {k: bool(v) for k, v in checks.items()}
    report = dict(passed=all(checks.values()), checks=checks, details=details,
                  limitations='Frozen static/modal/FRF checks and stopped-spin passivity do not prove periodic stability or real-machine validity.')
    write_json(folder/'mechanics.json', report)
    print(json.dumps(dict(passed=report['passed'], checks=checks), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v6_config.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments/clearance_v6_GMN/validation'))
    parser.add_argument('--source-before', type=Path, default=Path('experiments/clearance_v6_GMN/source_before'))
    args = parser.parse_args()
    audit(json.loads(args.config.read_text(encoding='utf-8')), args.output, args.source_before)
