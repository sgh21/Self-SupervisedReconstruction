"""Audit V9's loaded-speed / prescribed-cage contact-loss boundary.

This is a reusable LOCAL verification entry point, not a spindle simulation.
It reads frozen parameter vectors and never changes the production model.
Run: python -m utils.clearance_v9_contact_boundary --output <validation-dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root

from spindle import tai_bearing as b


def outer_control_rates(alpha_i, alpha_o, omega, p):
    """F07 independently written for the explicit outer-control limit."""
    lam = p[b.D]/p[b.DM]
    beta = math.atan2(math.sin(alpha_o), math.cos(alpha_o)+lam)
    eta_e = (1-lam*math.cos(alpha_i))/(1+math.cos(alpha_i-alpha_o))
    eta_b = -1/(((math.cos(alpha_o)+math.tan(beta)*math.sin(alpha_o))/(1+lam*math.cos(alpha_o))
                 +(math.cos(alpha_i)+math.tan(beta)*math.sin(alpha_i))/(1-lam*math.cos(alpha_i)))
                *lam*math.cos(beta))
    oe, ob = omega*eta_e, omega*eta_b
    fc = .5*p[b.BALL_MASS]*p[b.DM]*oe*oe
    mg = p[b.BALL_INERTIA]*oe*ob*math.sin(beta)
    return oe, ob, beta, fc, mg


def local_wrenches(qi, qo, alpha_i, alpha_o, mg, p):
    """F12, one ball at local phi=0; local x axial, y radial, z transverse."""
    g = mg/p[b.D]
    pi = qi*math.cos(alpha_i)-g*math.sin(alpha_i)
    ai = qi*math.sin(alpha_i)+g*math.cos(alpha_i)
    po = qo*math.cos(alpha_o)-g*math.sin(alpha_o)
    ao = qo*math.sin(alpha_o)+g*math.cos(alpha_o)
    ric = p[b.DM]/2+(p[b.FI]-.5)*p[b.D]*math.cos(p[b.ALPHA0])
    roc = p[b.DM]/2-(p[b.FO]-.5)*p[b.D]*math.cos(p[b.ALPHA0])
    inner = [ai, pi, 0., 0., 0., -(ric*ai-p[b.FI]*mg)]
    outer = [-ao, -po, 0., 0., 0., roc*ao+p[b.FO]*mg]
    return inner, outer


def prescribed_loss_limit(A, omega, p):
    ai, ao = (p[b.FI]-.5)*p[b.D], (p[b.FO]-.5)*p[b.D]
    oe = omega*p[b.CAGE_RATIO]
    fc = .5*p[b.BALL_MASS]*p[b.DM]*oe*oe
    ko = b.hertz_kernel(0., p, False)[0]
    delta_o = (fc/ko)**(2/3)
    alpha_i = math.asin(A/ai)
    B = ao+delta_o+math.sqrt(ai*ai-A*A)
    inner, outer = local_wrenches(0., fc, alpha_i, 0., 0., p)
    return dict(A_m=A, B_m=B, Qi_N=0., Qo_N=fc, Fc_N=fc, Mg_Nm=0.,
                alpha_i_rad=alpha_i, alpha_o_rad=0., delta_i_m=0., delta_o_m=delta_o,
                omega_E_rad_s=oe, inner_local6=inner, outer_local6=outer)


def loaded_loss_limit(A, omega, p):
    ai, ao = (p[b.FI]-.5)*p[b.D], (p[b.FO]-.5)*p[b.D]
    if omega == 0:
        # At zero inertia the unloaded ball position is not unique. The actual
        # dual-contact limit is the line of centres, not the finite-speed ao=0.
        alpha_i = alpha_o = math.asin(A/(ai+ao))
        fc = oe = delta_o = 0.
        B = math.sqrt((ai+ao)**2-A*A)
    else:
        alpha_i, alpha_o = math.asin(A/ai), 0.
        oe, _, _, fc, _ = outer_control_rates(alpha_i, alpha_o, omega, p)
        delta_o = (fc/b.hertz_kernel(0., p, False)[0])**(2/3)
        B = ao+delta_o+math.sqrt(ai*ai-A*A)
    inner, outer = local_wrenches(0., fc, alpha_i, alpha_o, 0., p)
    return dict(A_m=A, B_m=B, Qi_N=0., Qo_N=fc, Fc_N=fc, Mg_Nm=0.,
                alpha_i_rad=alpha_i, alpha_o_rad=alpha_o, delta_i_m=0., delta_o_m=delta_o,
                omega_E_rad_s=oe, inner_local6=inner, outer_local6=outer)


def trace_dual_branch(A, qi, omega, p):
    """Set a positive Qi and solve for B: avoids the production active-set switch.

    Unknowns are alpha_i, alpha_o, and Qo. Hertz supplies both indentations;
    the axial centre-distance equation and both F08 balances supply three
    equations. B is then reconstructed, so this traces the actual closed branch
    even when the production switch would already select the open branch.
    """
    ai, ao = (p[b.FI]-.5)*p[b.D], (p[b.FO]-.5)*p[b.D]
    length_scale = ai+ao
    force_scale = max(p[b.P_REF]/(p[b.Z]*math.sin(p[b.ALPHA0])), 1.)
    if omega == 0:
        def axial(angle):
            ki = b.hertz_kernel(angle, p, True)[0]
            ko = b.hertz_kernel(angle, p, False)[0]
            return (ai+ao+(qi/ki)**(2/3)+(qi/ko)**(2/3))*math.sin(angle)-A
        angle = 0. if A == 0 else brentq(axial, 0., math.pi/2, xtol=1e-15)
        alpha_i = alpha_o = angle
        qo = qi
        residual = np.zeros(3)
        solved = True
    elif A == 0:
        alpha_i = alpha_o = 0.
        fc = outer_control_rates(0., 0., omega, p)[3]
        qo = qi+fc
        residual = np.zeros(3)
        solved = True
    else:
        limit = loaded_loss_limit(A, omega, p)
        def equations(x):
            alpha_i, alpha_o, qo_scaled = x
            qo = qo_scaled*force_scale
            ki = b.hertz_kernel(alpha_i, p, True)[0]
            ko = b.hertz_kernel(alpha_o, p, False)[0]
            di, do = (qi/ki)**(2/3), (max(qo, 0.)/ko)**(2/3)
            _, _, _, fc, mg = outer_control_rates(alpha_i, alpha_o, omega, p)
            g = mg/p[b.D]
            return np.array([((ai+di)*math.sin(alpha_i)+(ao+do)*math.sin(alpha_o)-A)/length_scale,
                (qo*math.cos(alpha_o)-g*math.sin(alpha_o)-qi*math.cos(alpha_i)+g*math.sin(alpha_i)-fc)/force_scale,
                (qo*math.sin(alpha_o)+g*math.cos(alpha_o)-qi*math.sin(alpha_i)-g*math.cos(alpha_i))/force_scale])
        initial = [limit['alpha_i_rad'], 0., (limit['Fc_N']+qi)/force_scale]
        solution = root(equations, initial, method='hybr', options={'xtol': 1e-11})
        alpha_i, alpha_o, qo_scaled = solution.x
        qo = qo_scaled*force_scale
        residual = equations(solution.x)
        solved = (abs(residual[0]) <= 1e-9 and np.max(np.abs(residual[1:]))*force_scale <= 1e-6
                  and qo >= 0 and alpha_o >= -1e-9)
    ki, ko = b.hertz_kernel(alpha_i, p, True)[0], b.hertz_kernel(alpha_o, p, False)[0]
    di, do = (qi/ki)**(2/3), (max(qo, 0.)/ko)**(2/3)
    B = (ai+di)*math.cos(alpha_i)+(ao+do)*math.cos(alpha_o)
    oe, ob, beta, fc, mg = outer_control_rates(alpha_i, alpha_o, omega, p)
    inner, outer = local_wrenches(qi, qo, alpha_i, alpha_o, mg, p)
    return dict(A_m=A, B_m=B, Qi_N=qi, Qo_N=qo, Fc_N=fc, Mg_Nm=mg,
                alpha_i_rad=alpha_i, alpha_o_rad=alpha_o, delta_i_m=di, delta_o_m=do,
                omega_E_rad_s=oe, omega_B_rad_s=ob, beta_rad=beta,
                trace_pass=bool(solved), scaled_residual=residual.tolist(),
                inner_local6=inner, outer_local6=outer)


def production_sample(A, B, omega, p):
    state = b.solve_ball_kernel(A, B, omega, p, 2, 0, 1.)
    inner, outer = local_wrenches(state[b.QI], state[b.QO], state[b.ANGLE_I], state[b.ANGLE_O], state[b.MG], p)
    return dict(A_m=A, B_m=B, Qi_N=state[b.QI], Qo_N=state[b.QO], Fc_N=state[b.FC], Mg_Nm=state[b.MG],
                alpha_i_rad=state[b.ANGLE_I], alpha_o_rad=state[b.ANGLE_O],
                delta_i_m=state[b.DI], delta_o_m=state[b.DO], omega_E_rad_s=state[b.OMEGA_E],
                active_code=int(state[b.ACTIVE]), status_code=int(state[b.STATUS]),
                force_residual_N=float(max(abs(state[b.G3]), abs(state[b.G4]))),
                iterations=int(state[b.ITERATIONS]), inner_local6=inner, outer_local6=outer)


def difference(left, right):
    return {key: float(right[key]-left[key]) for key in ('Qi_N', 'Qo_N', 'Fc_N', 'Mg_Nm', 'B_m')} | {
        'inner_local6': (np.asarray(right['inner_local6'])-left['inner_local6']).tolist(),
        'outer_local6': (np.asarray(right['outer_local6'])-left['outer_local6']).tolist()}


def analyze_case(family, A_label, A, rpm, p):
    omega = rpm*math.pi/30
    prescribed, loaded = prescribed_loss_limit(A, omega, p), loaded_loss_limit(A, omega, p)
    limit_difference = difference(prescribed, loaded)
    traces = [trace_dual_branch(A, qi, omega, p) for qi in (1e-8, 1e-6, 1e-4, .01, .1, 1.)]
    for row in traces:
        row['production_at_same_AB'] = production_sample(A, row['B_m'], omega, p)
    samples = []
    pairs = []
    for boundary_name, boundary in (('F10_prescribed', prescribed), ('F07_loaded', loaded)):
        for epsilon in (1e-9, 1e-11, 1e-13):
            left = production_sample(A, boundary['B_m']-epsilon, omega, p)
            right = production_sample(A, boundary['B_m']+epsilon, omega, p)
            for side, row in (('below', left), ('above', right)):
                row.update(sample=f'{boundary_name}:{side}', distance_m=epsilon)
                samples.append(row)
            pairs.append(dict(boundary=boundary_name, epsilon_m=epsilon,
                              both_roots_accepted=left['status_code']>=0 and right['status_code']>=0,
                              below=left, above=right, jump=difference(left, right)))
    lower, upper = sorted((prescribed['B_m'], loaded['B_m']))
    if omega != 0 and upper-lower > 1e-14:
        for fraction in np.linspace(.1, .9, 9):
            row = production_sample(A, lower+fraction*(upper-lower), omega, p)
            row.update(sample='between_boundaries', fraction=float(fraction))
            samples.append(row)
    failures = [row for row in samples if row['status_code'] < 0]
    force_tolerance = max(1e-6, 1e-7*p[b.P_REF]/(p[b.Z]*math.sin(p[b.ALPHA0])))
    if omega == 0:
        continuity_pass = all(row['both_roots_accepted'] and max(abs(row['jump']['Qo_N']), abs(row['jump']['Qi_N'])) <= force_tolerance
                              for row in pairs if row['boundary']=='F07_loaded' and row['epsilon_m']==1e-13)
        verdict = 'zero_speed_geometric_open_set_continuous' if continuity_pass else 'zero_speed_boundary_failed'
    else:
        continuity_pass = abs(limit_difference['Qo_N']) <= force_tolerance and not failures
        verdict = 'incompatible_loaded_and_prescribed_cage_limits' if not continuity_pass else 'continuous_within_tolerance'
    return dict(family=family, A_label=A_label, rpm_magnitude=rpm, parameters=p.tolist(),
                F10_prescribed_limit=prescribed, F07_loaded_limit=loaded,
                loaded_minus_prescribed_limit=limit_difference,
                independent_positive_Qi_trace=traces, boundary_pairs=pairs,
                production_samples=samples, root_failure_count=len(failures),
                failed_B_min_m=min((row['B_m'] for row in failures), default=None),
                failed_B_max_m=max((row['B_m'] for row in failures), default=None),
                force_acceptance_N=force_tolerance, boundary_continuity_pass=bool(continuity_pass),
                trace_all_pass=all(row['trace_pass'] for row in traces), verdict=verdict)


def run(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    cases = []
    for family, spec in config['parameters']['tai_bearing']['families'].items():
        p = np.asarray(spec['initialized_parameter_vector'], dtype=float)
        b0 = (p[b.FI]+p[b.FO]-1)*p[b.D]
        nominal_A = b0*math.sin(p[b.ALPHA0])+p[b.A_PRE]
        for label, A in (('zero_axial_distance', 0.), ('nominal_preloaded_axial_distance', nominal_A)):
            for rpm in (0, 3000, 7000, 10000):
                cases.append(analyze_case(family, label, A, rpm, p))
    report = dict(scope='Local boundary audit only; no new spindle trajectory and no change to production physics or parameters.',
        config=str(config_path), source_module='src/spindle/tai_bearing.py',
        independent_trace='Positive Qi is prescribed, and contact angles/Qo solve axial geometry plus both F08 balances; B is reconstructed. No production active-set switch is used in that trace.',
        wrench_convention='Single ball at local phi=0, [Fx axial,Fy radial,Fz,Mx,My,Mz], left-hand restoring quantities.',
        source_contract=dict(F10='失去内接触时控环表无法再由内接触角确定全部球运动；脱离分支ΩE沿用规定保持架速度，须标记该运动学近似。',
            F11='Omega_c0 = Omega*(1-lambda*cos(alpha0))/2',
            F07_limit='For Omega!=0 and Qi->0: alpha_o->0, Mg->0, alpha_i=asin(A/a_i), eta_E=(1-lambda*cos(alpha_i))/(1+cos(alpha_i)).',
            incompatibility='For general A, the F07 loaded limit and fixed F11 speed are unequal. F08 gives Qo=Fc at Qi=0, so exact force continuity and both specified speed closures cannot all hold. This is a closure mismatch, not a residual tolerance failure.',
            zero_speed='With Fc=0 the no-force ball position is not unique; the dual-contact limit is L=B0 and the geometric open set must be used, as F10 explicitly requires.'),
        cases=cases,
        summary=dict(case_count=len(cases), independent_traces_all_pass=all(case['trace_all_pass'] for case in cases),
            zero_speed_cases_pass=all(case['boundary_continuity_pass'] for case in cases if case['rpm_magnitude']==0),
            finite_speed_boundary_continuity_pass=all(case['boundary_continuity_pass'] for case in cases if case['rpm_magnitude']!=0),
            root_failure_samples=sum(case['root_failure_count'] for case in cases),
            verdict='The already-detached F10 branch can pass algebra tests; the general finite-speed transition-continuity requirement does not pass. Always-dual trajectories require separate domain assessment.'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'contact_boundary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = []
    for case in cases:
        shared = {key:case[key] for key in ('family','A_label','rpm_magnitude')}
        for kind, samples in (('production',case['production_samples']),('independent_dual',case['independent_positive_Qi_trace']),
                              ('F10_limit',[case['F10_prescribed_limit']]),('F07_limit',[case['F07_loaded_limit']])):
            for sample in samples:
                row = dict(shared, record_type=kind)
                row.update({key:sample.get(key) for key in ('sample','distance_m','A_m','B_m','Qi_N','Qo_N','Fc_N','Mg_Nm','alpha_i_rad','alpha_o_rad','delta_i_m','delta_o_m','omega_E_rad_s','active_code','status_code','force_residual_N','trace_pass')})
                for side in ('inner','outer'):
                    row.update({f'{side}_{axis}':value for axis,value in zip(('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm'),sample[f'{side}_local6'])})
                rows.append(row)
    with (output/'contact_boundary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v9_config.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments/clearance_v9_tai_bearing/validation'))
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    for case in report['cases']:
        print(case['family'],case['A_label'],case['rpm_magnitude'],
              'dB_um=',case['loaded_minus_prescribed_limit']['B_m']*1e6,
              'dQo_N=',case['loaded_minus_prescribed_limit']['Qo_N'],
              'failed_samples=',case['root_failure_count'])


if __name__ == '__main__':
    main()
