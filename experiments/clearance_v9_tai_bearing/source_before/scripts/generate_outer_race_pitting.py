"""Generate matched outer-race pit areas; retain SI H/T and real-reference provenance."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from scripts.run_prototype import calibrate_forcing
from spindle.dynamics_model import (cage_speed, default_parameters, displacement_metrics,
                                    outer_race_pit_geometry, simulate)
from utils.data_io import bandlimit, save_json
from utils.pitting_analysis import (read_raw_references, process_pit_signals, acceleration_metrics,
                                    prepare_broadband_references, audit_steady_references)
from utils.healthy_reference import read_healthy_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/outer_race_pitting.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.output
    (out/'signals').mkdir(parents=True, exist_ok=False)
    data_cfg = json.loads(Path(cfg['healthy_data_config']).read_text(encoding='utf-8'))['data']
    cfg['healthy_data'] = data_cfg
    save_json(out/'config.json', cfg)
    healthy, provenance = read_healthy_runs(data_cfg)
    np.savez_compressed(out/'healthy_velocity_reference.npz', **healthy)
    save_json(out/'healthy_velocity_provenance.json', provenance)
    p = default_parameters()
    processing = cfg.get('processing')
    raw, raw_provenance = read_raw_references(data_cfg, cfg['raw_root'], provenance,
                       count=cfg.get('raw_segments_per_run',3), full_segments=processing is not None)
    save_json(out/'steady_state_audit.json', audit_steady_references(raw_provenance, data_cfg))
    if processing:
        acceleration, velocity, calibration_data = prepare_broadband_references(raw, processing)
        train = np.array([r['split']=='train' for r in raw_provenance])
        np.savez_compressed(out/'healthy_velocity_reference.npz', train=velocity[train], validation=velocity[~train])
        np.savez_compressed(out/'healthy_calibration_reference.npz', acceleration_m_s2=calibration_data[train])
        for r in raw_provenance:
            r.update(comparison_start_s=r['start_s']+4.5, comparison_duration_s=1.0,
                     comparison_fs_hz=processing['fs_hz'])
        frequencies, coefficients, calibration = calibrate_forcing(
            calibration_data[train], processing['fs_hz'], p,
            band=cfg['calibration']['band_hz'], nperseg=cfg['calibration']['nperseg'],
            quantity=cfg['calibration']['quantity'])
        save_json(out/'healthy_velocity_provenance.json', raw_provenance)
    else:
        acceleration = bandlimit(raw,25600,*cfg['simulation']['acceleration_band_hz'])
        frequencies, coefficients, calibration = calibrate_forcing(healthy['train']*1e-3, 2048, p)
    np.savez_compressed(out/'healthy_acceleration_reference.npz', raw_m_s2=raw, band_m_s2=acceleration)
    save_json(out/'healthy_acceleration_provenance.json', raw_provenance)
    save_json(out/'calibration.json', calibration)
    sim, pit = cfg['simulation'], cfg['pit']
    p.update(outer_pit_width_m=pit['width_mm']*1e-3, outer_pit_depth_m=pit['depth_mm']*1e-3,
             outer_pit_center_rad=float(np.deg2rad(pit['center_deg'])))
    bpfo = abs(cage_speed(p))*p['balls']/(2*np.pi)
    cases, rows = [], []
    start = time.perf_counter()
    for group in sim['groups']:
        rng = np.random.default_rng(group['seed'])
        drive = coefficients*np.exp(1j*rng.uniform(-np.pi, np.pi, (len(frequencies), 1)))
        gp = dict(p, cage_phase=float(rng.uniform(0, 2*np.pi)))
        np.savez_compressed(out/'signals'/f"{group['id']}_forcing.npz",
                            frequencies_hz=frequencies, coefficients_N=drive)
        for label, area in enumerate(pit['areas_mm2']):
            name = f"{group['id']}_area_{area:.1f}mm2"
            parameters = dict(gp, outer_pit_area_m2=area*1e-6)
            print(f'Integrating {name}', flush=True)
            guard = processing['guard_s'] if processing else 0
            result = simulate(parameters, 0, frequencies, drive, duration=sim['duration_s']+2*guard,
                              warmup=sim['warmup_s'], fs=sim['output_fs_hz'], max_step=sim['max_step_s'])
            processed = process_pit_signals(result, sim, processing)
            result.update(processed, parameters_json=json.dumps(parameters), group_id=group['id'],
                          split=group['split'], outer_pit_area_mm2=area, taper_lost_area_m2=0.0)
            np.savez_compressed(out/'signals'/f'{name}.npz', **result)
            begin = round(guard*sim['output_fs_hz'])
            end = begin+round(sim['duration_s']*sim['output_fs_hz'])
            displacement = displacement_metrics(result['T_center_m'][begin:end])
            akey = 'H_acceleration_band_m_s2' if processing else 'H_acceleration_band_m_s2_25600'
            vkey = 'H_velocity_band_mm_s' if processing else 'H_velocity_band_mm_s_2048'
            acceleration = acceleration_metrics(processed[akey], bpfo, sim['comparison_fs_hz'])
            vrms = np.sqrt(np.mean(processed[vkey]**2, axis=-1))
            row = dict(case=name, group=group['id'], split=group['split'], label=label, area_mm2=area,
                       H_acc_x_rms_m_s2=acceleration['rms_xy_m_s2'][0], H_acc_y_rms_m_s2=acceleration['rms_xy_m_s2'][1],
                       H_bpfo_x_rms_m_s2=acceleration['bpfo_band_rms_xy_m_s2'][0],
                       H_bpfo_y_rms_m_s2=acceleration['bpfo_band_rms_xy_m_s2'][1],
                       H_vel_x_rms_mm_s=float(vrms[0]), H_vel_y_rms_mm_s=float(vrms[1]),
                       T_dynamic_radial_rms_um=displacement['dynamic_radial_rms_um'],
                       nfev=result['nfev'], integration_segments=result['integration_segments'])
            rows.append(row)
            cases.append(dict(**row, parameters=parameters, geometry=outer_race_pit_geometry(parameters),
                              displacement=displacement, equilibrium_residual_N=result['equilibrium_residual_N']))
            save_json(out/'generation.json', dict(bpfo_hz=bpfo, cases=cases, completed=False))
            print(f"  acc XY={acceleration['rms_xy_m_s2']}; vel XY={vrms}; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    with (out/'severity_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out/'generation.json', dict(bpfo_hz=bpfo, cases=cases, completed=True,
              elapsed_seconds=time.perf_counter()-start, native_fs_hz=sim['output_fs_hz'],
              note='Phase families share a healthy-calibrated spectrum. This is a mechanism sweep, not independent machine validation.'))
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
