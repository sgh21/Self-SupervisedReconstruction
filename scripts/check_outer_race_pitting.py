"""Persistent mechanics and numerical convergence checks for the pit model."""
import argparse
import json
from pathlib import Path

import numpy as np

from spindle.dynamics_model import (bearing_force, cage_speed, default_parameters, internal_force,
                                    outer_race_pit_displacement, outer_race_pit_geometry,
                                    outer_race_passage_boundaries, simulate)
from utils.data_io import save_json
from utils.pitting_analysis import process_pit_signals


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    generation = json.loads((out/'generation.json').read_text(encoding='utf-8'))
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    representative = [c for c in generation['cases'] if c['group'] == 'phase_0']
    p = representative[-1]['parameters']
    g = outer_race_pit_geometry(p)
    phi = g['center_rad'] + g['span_rad']*np.array([-.51, -.5, 0, .5, .51])
    drops = outer_race_pit_displacement(phi, p)
    assert np.allclose(drops, [0, 0, g['maximum_drop_m'], 0, 0], rtol=0, atol=1e-16)
    wrapped = dict(p, outer_pit_center_rad=0)
    assert np.allclose(outer_race_pit_displacement(np.array([0, 2*np.pi]), wrapped), g['maximum_drop_m'])
    assert np.isclose(g['area_m2'], g['width_m']*g['length_m'])
    sag = p['ball_diameter']/2-np.sqrt((p['ball_diameter']/2)**2-(g['length_m']/2)**2)
    assert np.isclose(g['maximum_drop_m'], min(g['depth_m'], sag))
    period = 1/generation['bpfo_hz']
    boundaries = outer_race_passage_boundaries(p, .1)
    intervals = np.diff(boundaries)[1:-1]
    dwell = g['span_rad']/abs(cage_speed(p))
    assert np.all(np.isclose(intervals, dwell) | np.isclose(intervals, period-dwell))
    # Contact force must vanish for negative approach, even with a large velocity.
    uncontacted = dict(default_parameters(), normal_overlap=-1e-3)
    assert np.array_equal(bearing_force(0, np.zeros(2), np.ones(2), uncontacted), [0, 0])
    q, v = np.arange(1,7)*1e-6, np.arange(1,7)*1e-4
    restoring = internal_force(.01, q, v, p, 0).reshape(3,2)
    balance = np.linalg.norm(restoring.sum(0)-p['kh']*q[4:]-p['ch']*v[4:])
    assert balance < 1e-10
    checks = dict(geometry_passed=True, passage_period_s=period, pit_dwell_s=dwell,
                  no_tension_passed=True, reaction_balance_N=float(balance), convergence=[])
    # Compare against preserved pre-change equations, not a second copy of the new formula.
    old = {}
    exec(compile((out/'evidence/dynamics_model_before.txt').read_text(encoding='utf-8'),
                 'preserved_original_dynamics', 'exec'), old)
    healthy = default_parameters()
    drive_f, drive_c = np.array([100.]), np.array([[.1, .15]])
    before = old['simulate'](healthy, 0, drive_f, drive_c, duration=.01, warmup=.01)
    after = simulate(healthy, 0, drive_f, drive_c, duration=.01, warmup=.01)
    checks['healthy_regression_state_relative_l2'] = relative(after['q'], before['q'])
    assert checks['healthy_regression_state_relative_l2'] < 1e-10
    with np.load(out/'signals/phase_0_forcing.npz', allow_pickle=False) as drive:
        frequencies, coefficients = drive['frequencies_hz'], drive['coefficients_N']
    sim = cfg['simulation']
    processing = cfg.get('processing')
    guard = processing['guard_s'] if processing else 0
    akey = 'H_acceleration_band_m_s2' if processing else 'H_acceleration_band_m_s2_25600'
    for case in (representative[1], representative[-1]):
        print('Convergence:', case['case'], flush=True)
        duration = .125
        fine = simulate(case['parameters'], 0, frequencies, coefficients, duration=duration+2*guard,
                        warmup=sim['warmup_s'], fs=2*sim['output_fs_hz'], max_step=sim['max_step_s']/2,
                        pit_steps_per_passage=8)
        n = round((duration+2*guard)*sim['output_fs_hz'])+1
        crop = slice(round(guard*sim['output_fs_hz']), round((guard+duration)*sim['output_fs_hz'])+1)
        with np.load(out/'signals'/f"{case['case']}.npz", allow_pickle=False) as original:
            velocity_error = relative(original['H_velocity_m_s'][:n][crop], fine['H_velocity_m_s'][::2][crop])
            tc, tf = original['T_center_m'][:n][crop], fine['T_center_m'][::2][crop]
            position_error = relative(tc-tc.mean(0), tf-tf.mean(0))
            coarse = {k: original[k][:n] for k in ('t','H_acceleration_m_s2', 'H_velocity_m_s')}
            saved_processed = original[akey][:,:round(duration*sim['comparison_fs_hz'])]
        fine_sim = dict(sim, duration_s=duration, output_fs_hz=2*sim['output_fs_hz'])
        coarse_sim = dict(sim, duration_s=duration)
        fine_processed = process_pit_signals(fine, fine_sim, processing)[akey]
        coarse_processed = process_pit_signals(coarse, coarse_sim, processing)[akey]
        sampled = {k: fine[k][::2] for k in ('t','H_acceleration_m_s2', 'H_velocity_m_s')}
        sampled_processed = process_pit_signals(sampled, coarse_sim, processing)[akey]
        result = dict(case=case['case'], compared_s=duration,
                      H_velocity_relative_l2=velocity_error, T_centered_relative_l2=position_error,
                      H_acc_band_combined_relative_l2=relative(coarse_processed, fine_processed),
                      H_acc_band_sampling_only_relative_l2=relative(sampled_processed, fine_processed))
        if processing:
            result['guarded_crop_relative_l2'] = relative(coarse_processed, saved_processed)
        checks['convergence'].append(result)
        np.savez_compressed(out/f"convergence_{case['area_mm2']:.1f}mm2.npz", **fine)
        print(json.dumps(result), flush=True)
        save_json(out/'verification.json', checks)
    checks['passed'] = all(c['H_velocity_relative_l2'] < .005 and c['T_centered_relative_l2'] < .005
                           and c['H_acc_band_combined_relative_l2'] < .03
                           and c.get('guarded_crop_relative_l2',0) < 1e-6 for c in checks['convergence'])
    save_json(out/'verification.json', checks)
    assert checks['passed'], 'Convergence tolerance exceeded; review native sampling/integration before reporting.'
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
