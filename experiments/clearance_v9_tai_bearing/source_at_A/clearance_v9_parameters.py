"""V9 conditional parameter contract, derived from the frozen V8 configuration."""
from copy import deepcopy
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/clearance_v9_tai_bearing'
MODES = {'B0': None, 'B0d': None, 'N1': 'none', 'N2': 'centrifugal', 'N3': 'full'}


def build_config(variant='N3', base=None, contract=None):
    if variant not in MODES:
        raise ValueError(f'Unknown V9 mechanism variant: {variant}')
    if base is None:
        base = json.loads((EXPERIMENT/'config/parameters_v8_frozen.json').read_text(encoding='utf-8'))
    if contract is None:
        contract = json.loads((ROOT/'GPTOrder/spindle_v9_tai_bearing_plan/templates/v9_contract.json').read_text(encoding='utf-8'))
    config = deepcopy(base)
    p = config['parameters']
    if p['model_dof'] != 12 or p['bearing_layout_name'] != 'DB_DB':
        raise ValueError('V9 requires the frozen 12-DOF DB/DB V8 configuration.')
    config.update(schema_version=9, analysis_version='v9', mechanism_variant=variant,
                  base_config_source='config/parameters_v8_frozen.json')
    p['bearing_contact_model'] = ('relative_tilt_v8' if variant.startswith('B0')
                                  else 'quasistatic_tai_v9')
    p['damping_policy'] = ('v8_clipped_contact' if variant == 'B0'
                           else 'common_structural_ring')
    p['model_status'] = 'conditional_geometry_preload_control_phase_and_damping'
    p['calibration_status'] = 'fixed_old_U_mechanism_comparison'
    p['rhs_backend'] = 'numba'
    p['tai_bearing'] = dict(
        ball_inertia_mode=MODES[variant] or 'none',
        ring_control_policy='conditional_outer_with_candidate_audit',
        control_branch='outer', hertz_policy='angle_dependent_geometry_hertz',
        preload_mode='fixed_position_reference', phase_policy='tai_uniform_phase',
        phase_speed_factor={'front': 1., 'rear': 1.},
        local_tolerance_scale=1.,
        materials=deepcopy(contract['nominal_materials']),
        families=deepcopy(contract['families']),
        failed_local_solve='stop_and_save_state_no_fallback',
    )
    config['excitation']['source'] = 'V8 verified calibrated U; frozen unchanged for V9 mechanism A'
    config['legacy_v8_calibration'] = deepcopy(base.get('calibration', {}))
    config['calibration'] = dict(status='not_run_v9', target_E_dir_um=[1.9, 2.1],
        healthy_clearance_um=2., only_parameter='excitation.U_kg_m',
        U_old_kg_m=base['excitation']['U_kg_m'], U_new_kg_m=None)
    config['simulation'].update(contract['numerics']['baseline_simulation'])
    config['simulation']['workers'] = 2
    config['numerical_thresholds'].update(position_um=.01, velocity_mm_s=.005,
        relative_main=.01, contact_rate=.02, relative_contact_peak=.05,
        relative_contact_impulse=.05, relative_residual_convergence=.05)
    config['v9_acceptance'] = dict(
        local_geometry_normalized=1e-9, local_force_absolute_N=1e-6,
        local_force_relative=1e-7, residual_velocity_absolute_mm_s=.002,
        wrench_rms_relative=.02, wrench_peak_relative=.05,
        wrench_force_absolute_N=1e-5, wrench_moment_absolute_Nm=1e-7,
        phase_speed_warning_fraction=.02,
        source_traceability='CRC32 and field comparison; no SHA256 per user rule',
    )
    return config


def initialize_preload(config):
    """Initialize each family once at rest; the actual run never retunes it."""
    from spindle.tai_bearing import make_bearing_parameters
    from spindle.tai_bearing import PARAMETER_NAMES
    local = deepcopy(config)
    tai = local['parameters']['tai_bearing']
    for family in tai['families'].values():
        vector = make_bearing_parameters(family, tai['materials'])
        named = dict(zip(PARAMETER_NAMES, (float(x) for x in vector)))
        family['initialized_parameter_vector'] = vector.tolist()
        family['initialized_parameter_names'] = list(PARAMETER_NAMES)
        family['initialization'] = named
        # The vector and explicit names are the runtime contract, not a cache.
        family['preload_initialization_status'] = 'solved_at_rest_fixed_afterwards'
    return local


def case_config(baseline, gap, U=None, rpm=None, directions=None, step_factor=1.,
                root_factor=1., turns=None, warmup=None):
    from spindle.dynamics_model import front_group_parameters
    config = deepcopy(baseline)
    config['parameters'] = front_group_parameters(config['parameters'], float(gap))
    p, sim, drive = config['parameters'], config['simulation'], config['excitation']
    if rpm is not None:
        p['rpm'] = float(rpm)
    if U is not None:
        drive['U_kg_m'] = float(U)
    if drive['U_kg_m'] is None or drive['U_kg_m'] < 0:
        raise ValueError('V9 requires a nonnegative, explicitly recorded U.')
    old_U = config['calibration']['U_old_kg_m']
    if drive['U_kg_m'] > 2*old_U+1e-15:
        raise ValueError('U exceeds the predeclared V9 search domain [0,2 U_old].')
    drive['peak_force_N'] = drive['U_kg_m']*(p['rpm']*2*np.pi/60)**2
    if directions is not None:
        p['N'] = int(directions)
    sim['max_step_s'] *= step_factor
    p['tai_bearing']['local_tolerance_scale'] *= root_factor
    sim['retained_turns'] = int(turns if turns is not None else (480 if gap >= 7.5 else 120))
    if warmup is not None:
        sim['warmup_s'] = float(warmup)
    return config


def parameter_delta(base, updated):
    rows = []
    def walk(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a)|set(b)):
                walk(a.get(key), b.get(key), f'{path}.{key}' if path else key)
        elif a != b:
            rows.append(dict(parameter=path, old=json.dumps(a,ensure_ascii=False),
                             new=json.dumps(b,ensure_ascii=False)))
    walk(base, updated)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'config/clearance_v9_config.json')
    args = parser.parse_args()
    config = initialize_preload(build_config())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    base = json.loads((EXPERIMENT/'config/parameters_v8_frozen.json').read_text(encoding='utf-8'))
    with (EXPERIMENT/'parameter_delta.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['parameter','old','new'])
        writer.writeheader(); writer.writerows(parameter_delta(base,config))
    print(args.output)


if __name__ == '__main__':
    main()
