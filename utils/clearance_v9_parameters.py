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
    # All selected positive-clearance cases can switch their second front fit.
    # Use the prescribed 480-turn window throughout the final campaign.
    sim['retained_turns'] = int(turns if turns is not None else 480)
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


def local_workpoints(config, output):
    """P2/C local table; each curvature family keeps its rest preload at all speeds."""
    from spindle import tai_bearing as tb
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    tai = config['parameters']['tai_bearing']
    rows=[]
    for name,family in tai['families'].items():
        for curvature in (.54,.52):
            current = dict(family,fi=curvature,fo=curvature)
            p = (np.asarray(family['initialized_parameter_vector']) if curvature==.54
                 else tb.make_bearing_parameters(current,tai['materials']))
            for rpm in (0,3000,7000,10000):
                omega = -rpm*2*np.pi/60
                inner,outer,states = tb.bearing_wrenches_kernel(0.,0.,0.,0.,0.,p,1.,omega,0.,2,0)
                delta=1e-9
                plus=tb.bearing_wrenches_fast_kernel(0.,delta,0.,0.,0.,p,1.,omega,0.,2,0)[0]
                minus=tb.bearing_wrenches_fast_kernel(0.,-delta,0.,0.,0.,p,1.,omega,0.,2,0)[0]
                b=states[0]
                rows.append(dict(family=name,fi=curvature,fo=curvature,rpm_magnitude=rpm,
                    a_pre_um=p[tb.A_PRE]*1e6,ball_mass_kg=p[tb.BALL_MASS],
                    ball_inertia_kg_m2=p[tb.BALL_INERTIA],
                    alpha_i_deg=np.degrees(b[tb.ANGLE_I]),alpha_o_deg=np.degrees(b[tb.ANGLE_O]),
                    Qi_N=b[tb.QI],Qo_N=b[tb.QO],Fc_N=b[tb.FC],Mg_Nm=b[tb.MG],
                    Ki_N_m32=b[tb.KI],Ko_N_m32=b[tb.KO],
                    axial_inner_restoring_N=inner[2],axial_outer_restoring_N=outer[2],
                    radial_tangent_N_per_um=(plus[0]-minus[0])/(2*delta)*1e-6,
                    cage_phase_factor=b[tb.OMEGA_E]/(omega*p[tb.CAGE_RATIO]) if rpm else 1.,
                    Fc_over_Qi=b[tb.FC]/b[tb.QI],abs_Mg_over_DQi=abs(b[tb.MG])/(p[tb.D]*b[tb.QI]),
                    p0_i_GPa=b[tb.PRESSURE_I]/1e9,p0_o_GPa=b[tb.PRESSURE_O]/1e9,
                    outer_control_margin=b[tb.OUTER_MARGIN],inner_control_margin=b[tb.INNER_MARGIN],
                    force_residual_N=max(abs(b[tb.G3]),abs(b[tb.G4]))))
    with (output/'local_workpoints.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sensitivity={}
    for name in tai['families']:
        a,b=[next(r for r in rows if r['family']==name and r['fi']==f and r['rpm_magnitude']==7000)
             for f in (.54,.52)]
        sensitivity[name]=dict(radial_tangent_relative_change=b['radial_tangent_N_per_um']/a['radial_tangent_N_per_um']-1,
            nominal_common_phase_factor=a['cage_phase_factor'])
    record=dict(workpoints=rows,sensitivity=sensitivity,
        curvature_system_comparison_required=any(abs(r['radial_tangent_relative_change'])>.05 for r in sensitivity.values()),
        curvature_screen='More than 5% change in the 7000 rpm local radial tangent triggers planned c2/c10 comparisons; this is a declared engineering screen, not a material limit.',
        preload='90 N front and 70 N rear per bearing at rest; fixed a_pre across all speeds.',
        scope='Axisymmetric local bearing workpoints; no system trajectory or machine calibration claim.')
    (output/'local_workpoints.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'config/clearance_v9_config.json')
    parser.add_argument('--local-workpoints',action='store_true')
    args = parser.parse_args()
    config = initialize_preload(build_config())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    base = json.loads((EXPERIMENT/'config/parameters_v8_frozen.json').read_text(encoding='utf-8'))
    with (EXPERIMENT/'parameter_delta.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['parameter','old','new'])
        writer.writeheader(); writer.writerows(parameter_delta(base,config))
    print(args.output)
    if args.local_workpoints:
        result=local_workpoints(config,EXPERIMENT/'validation')
        print(json.dumps(result['sensitivity']))


if __name__ == '__main__':
    main()
