"""Rebuild V4 metrics and comparisons from saved native arrays, without simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.run_clearance_v3 import write_json,write_csv
from utils.clearance_v4_analysis import analyze,compare,keys,absolute_floor


def replay(path):
    with np.load(path) as data:
        meta=json.loads(str(data['metadata_json']))
        result={k:data[v] for k,v in dict(t='t_raw',q='q_raw_m',v='v_raw_m_s',acc='a_raw_m_s2',
            measured_relative_xy_m='r_m_raw_m',fit_force_N='fit_force_N',fit_active_count='fit_active_count',
            fit2_force_N='fit2_force_N',fit2_active_count='fit2_active_count').items()}
    metrics,turns,_,steady=analyze(result,meta['config'],meta['export_start_s'],meta['export_end_s'])
    coarse,_,_,_=analyze(result,meta['config'],meta['export_start_s'],meta['export_end_s'],stride=2)
    sampling=compare(coarse,metrics,meta['config'])
    saved=json.loads(path.with_name('metrics.json').read_text(encoding='utf-8'))
    differences={k:abs(metrics[k]-saved[k]) for k in keys(metrics)}
    c=meta['config'];s=c['simulation'];p=c['parameters']
    aq=np.full(12,s['atol_translation_m']);aq[8:10]=s['atol_angle_rad']
    av=np.full(12,s['atol_velocity_m_s']);av[8:10]=s['atol_angle_rate_rad_s']
    solver=[]
    for settings in meta['actual_solver_history']:
        solver.append(bool(settings['method']==s['method'] and settings['rtol']==s['rtol']
            and settings['max_step_s']==s['max_step_s'] and settings['fit_direction_count']==p['N']
            and settings['output_fs_hz']==s['fs_raw_hz'] and np.array_equal(settings['atol'],np.r_[aq,av])
            and settings['accepted_steps']['max_s']<=s['max_step_s']*(1+1e-8)))
    domain=json.loads(path.with_name('domain.json').read_text(encoding='utf-8'))
    return dict(metrics=metrics,steady=steady,sampling=sampling,config=meta['config'],
                saved_replay_max_difference=max(differences.values()),source=str(path),
                solver_pass=all(solver) and result['q'].shape[1]==12,
                actual_max_steps_s=[x['accepted_steps']['max_s'] for x in meta['actual_solver_history']],domain=domain)


def rebuild(root,output,base_group=None,integration_group='validation/h_half',angular_group='validation/n256'):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data_root=root/(base_group or ('pilots' if (root/'pilots').exists() else 'data'))
    data={f.parent.name:replay(f) for f in sorted(data_root.glob('gap_*/native.npz'))}
    refined={label:{f.parent.name:replay(f) for f in sorted((root/path).glob('gap_*/native.npz'))}
             for label,path in (('h_half',integration_group),('N_double',angular_group))}
    comparisons={};eps={};valid={};physical_checks={};status={};observation_checks={}
    frozen=json.loads((root/'frozen_physics.json').read_text(encoding='utf-8'))
    allowed={'front_group_clearance_um','fit_clearance_um','fit2_clearance_um','N'}
    common_N=next(iter(data.values()))['config']['parameters']['N']
    baseline=dict(frozen['parameters']);baseline['diagnostic_single_front']=False
    for name,case in data.items():
        config=case['config'];p=config['parameters']
        changed=[k for k in set(p)|set(baseline) if k not in allowed and p.get(k)!=baseline.get(k)]
        physical_checks[name]=dict(passed=not changed and config['excitation']==frozen['excitation'] and p['N']==common_N,
                                   non_gap_physical_changes=changed,common_numerical_N=common_N)
        domain=json.loads((data_root/name/'domain.json').read_text(encoding='utf-8'))
        observation_path=data_root/name/'observation_checks.json'
        observation=json.loads(observation_path.read_text(encoding='utf-8')) if observation_path.exists() else {}
        observation_checks[name]=observation
        comparisons[name]={label:compare(case['metrics'],values[name]['metrics'],config) for label,values in refined.items() if name in values}
        for label,comparison in comparisons[name].items():
            fine=refined[label][name];fp=fine['config']['parameters'];fs=fine['config']['simulation']
            changes={k for k in set(fp)|set(p) if fp.get(k)!=p.get(k)}
            correct=(changes=={'N'} and fp['N']==2*p['N'] and fs['max_step_s']==config['simulation']['max_step_s']) if label=='N_double' else (
                not changes and fp['N']==p['N'] and fs['max_step_s']==config['simulation']['max_step_s']/2
                and max(fine['actual_max_steps_s'])<.75*max(case['actual_max_steps_s']))
            comparison['independent_refinement_applied']=bool(correct)
            comparison['passed']=bool(comparison['passed'] and correct and fine['solver_pass'] and fine['sampling']['passed']
                and fine['domain']['passed'] and fine['config']['excitation']==config['excitation'])
        enough=len(comparisons[name])==2
        valid[name]=bool(enough and case['solver_pass'] and case['saved_replay_max_difference']<1e-9
                         and case['steady']['passed'] and case['sampling']['passed'] and domain['passed']
                         and physical_checks[name]['passed'] and all(x['passed'] for x in comparisons[name].values())
                         and all(refined[label][name]['steady']['passed'] for label in comparisons[name])
                         and observation.get('O2H_drift_pass',False) and observation.get('outer_dynamic_equilibrium_pass',False))
        status[name]='pass' if valid[name] else ('not_evaluated' if not enough else 'fail')
        eps[name]={k:max(absolute_floor(k,config),sum(item['metrics'][k]['difference'] for item in comparisons[name].values())
                        +case['sampling']['metrics'][k]['difference']+case['steady']['checks'][k]['difference']) for k in keys(case['metrics'])} if enough else {}
    health=data.get('gap_0um');accepted=[(name,case) for name,case in data.items() if valid[name]]
    maximum=max(accepted,key=lambda x:x[1]['metrics']['E_dir_um']) if accepted else None
    degraded=[(name,case) for name,case in accepted if case['metrics']['gap_um']>0]
    maxdeg=max(degraded,key=lambda x:x[1]['metrics']['E_dir_um']) if degraded else None
    pairs=[]
    if maxdeg and valid.get('gap_0um'):
        name,case=maxdeg
        for k in keys(health['metrics']):
            if not k.startswith('H_v_'):continue
            a,b=health['metrics'][k],case['metrics'][k];e0,ec=eps['gap_0um'][k],eps[name][k]
            pairs.append(dict(case=name,feature=k,healthy=a,degraded=b,absolute_change=b-a,
                 relative_change_percent=100*(b-a)/abs(a) if abs(a)>5*e0 else None,
                 epsilon_healthy=e0,epsilon_degraded=ec,epsilon_sum=e0+ec,
                 resolvable=bool(abs(b-a)>5*(e0+ec))))
    cover=dict(healthy=bool(health and valid.get('gap_0um') and health['metrics']['E_dir_um']<=2))
    for name,low,high in (('level5',4,6),('level10',9,11),('level15',14,16)):
        cover[name]=any(low<=case['metrics']['E_dir_um']<=high for _,case in accepted)
    cover['level20']=any(case['metrics']['E_dir_um']-eps[name]['E_dir_um']>=20 for name,case in accepted)
    cover['at_least_five']=len(accepted)>=5
    acceptance=dict(mechanics=json.loads((root/'validation/mechanics/mechanics.json').read_text(encoding='utf-8'))['passed'],
        base_group=str(data_root.relative_to(root)),integration_group=integration_group,angular_group=angular_group,
        numerical_case_pass=valid,numerical_case_status=status,physics_freeze=physical_checks,range_checks=cover,range_pass=all(cover.values()),
        observation_checks=observation_checks,
        max_valid_case=maximum[0] if maximum else None,max_valid_E_um=maximum[1]['metrics']['E_dir_um'] if maximum else None,
        max_valid_degraded_case=maxdeg[0] if maxdeg else None,
        vibration_difference_pass=any(r['resolvable'] for r in pairs) if pairs else None,
        engineering_status='assumed_demo',real_machine_detectability='not_validated',
        replay_max_difference=max((c['saved_replay_max_difference'] for c in data.values()),default=0))
    write_csv(output/'independent_metrics.csv',[dict(case=name,**case['metrics']) for name,case in data.items()])
    if pairs:write_csv(output/'vibration_differences.csv',pairs)
    write_json(output/'acceptance.json',acceptance);write_json(output/'convergence.json',comparisons)
    write_json(output/'numerical_error_estimates.json',eps)
    print(json.dumps(acceptance,ensure_ascii=False),flush=True)
    return data,refined,acceptance,eps,pairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-group');parser.add_argument('--integration-group',default='validation/h_half')
    parser.add_argument('--angular-group',default='validation/n256')
    args=parser.parse_args();rebuild(args.root,args.output or args.root/'recomputed',args.base_group,args.integration_group,args.angular_group)


if __name__=='__main__':main()
