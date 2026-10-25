"""Fixed-physics V4 pilots and independent h/N refinements, with real warmup extension."""
import argparse
import copy
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from spindle.dynamics_model import (front_group_parameters, equilibrium_continuation, simulate,
    rotor_section, bearing_force, front2_forces, fit_contact_summary)
from scripts.run_clearance_v3 import write_json, write_csv, export_window
from utils.clearance_v4_analysis import analyze, compare


def run_case(config, folder, analyze_fn=analyze, compare_fn=compare):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    p,s=config['parameters'],config['simulation']
    from utils.clearance_v4_validation import validate_config, domain_check
    validate_config(config)
    write_json(folder/'actual_config.json',config)
    q0,_=equilibrium_continuation(p)
    atol_q=np.full(12,s['atol_translation_m']);atol_q[8:10]=s['atol_angle_rad']
    atol_v=np.full(12,s['atol_velocity_m_s']);atol_v[8:10]=s['atol_angle_rate_rad_s']
    fs=s['fs_raw_hz']; result=None; history=[]; solver_history=[]
    began=time.perf_counter()
    while True:
        start,end=export_window(config)
        keep=np.floor((start-s['guard_s'])*fs)/fs
        finish=np.ceil((end+s['guard_s'])*fs)/fs
        previous=0 if result is None else float(result['t'][-1])
        new=simulate(p,0,[],[],warmup=keep if result is None else 0,
            duration=finish-keep if result is None else finish-previous,
            fs=fs,max_step=s['max_step_s'],rtol=s['rtol'],atol=np.r_[atol_q,atol_v],
            method=s['method'],q_initial=q0,t_start=previous,
            state_initial=None if result is None else result['state_final'],
            unbalance=(config['excitation']['U_kg_m'],config['excitation']['phase_rad']))
        solver_history.append(new['actual_solver_settings'])
        if result is not None:
            for key in ('t','q','v','acc','measured_relative_xy_m','measured_section_xy_m',
                        'probe_reference_xy_m','fit_force_N','fit_active_count',
                        'fit2_force_N','fit2_active_count','bearing2_force_N'):
                new[key]=np.concatenate((result[key],new[key][1:]),axis=0)
            new['nfev']+=result['nfev']
        result=new
        metrics,turns,output,steady=analyze_fn(result,config,start,end)
        history.append(dict(warmup_s=s['warmup_s'],start_s=start,end_s=end,stationarity=steady))
        print(json.dumps(dict(folder=str(folder),warmup=s['warmup_s'],E_um=metrics['E_dir_um'],
                              steady=steady['passed'])),flush=True)
        if steady['passed'] or s['warmup_s']>=s['maximum_warmup_s']: break
        s['warmup_s']=min(s['maximum_warmup_s'],s['warmup_s']+s['extension_s'])
    coarse,_,_,_=analyze_fn(result,config,start,end,stride=2)
    sampling=compare_fn(coarse,metrics,config)
    # Discard old extension windows; keep true guards for independent replay.
    cut=np.searchsorted(result['t'],np.floor((start-s['guard_s'])*fs)/fs)
    cut-=cut%int(fs/s['fs_out_hz']) # retain the same antialias/8192-Hz phase after extension
    use=np.arange(len(result['t']))>=cut
    t,q,v,a=(result[k][use] for k in ('t','q','v','acc'))
    valid=(t>=start)&(t<end)
    b1=np.array([bearing_force(ti,ri,vi,p) for ti,ri,vi in zip(t,
         rotor_section(q,p['z_f'])-q[:,6:8],rotor_section(v,p['z_f'])-v[:,6:8])])
    b2,f2,d2=front2_forces(q,v,p,summary=True)
    f1,d1=fit_contact_summary(q[:,6:8]-q[:,4:6],v[:,6:8]-v[:,4:6],p)
    rear=p['healthy_supports'][0]
    rear_relative=rotor_section(q,rear['z_m'])-q[:,4:6]
    rear_force=rear_relative@np.array(rear['K']).T+(rotor_section(v,rear['z_m'])-v[:,4:6])@np.array(rear['C']).T
    domain=domain_check(t[valid],q[valid],config)
    metrics.update(domain_pass=domain['passed'],sampling_pass=sampling['passed'],
                   wall_seconds=time.perf_counter()-began,nfev=result['nfev'])
    metadata=dict(config=config,export_start_s=start,export_end_s=end,
        state_order=['Sx','Sy','Tx','Ty','Hx','Hy','O1x','O1y','theta_x','theta_y','O2x','O2y'],
        coordinate_units=['m']*8+['rad']*2+['m']*2,
        actual_solver_history=solver_history,main_metric='max(mean(turn_PP_x),mean(turn_PP_y)); includes 1X',
        reference='M at z_m relative fixed ground; assumed demo',
        processing='Native SI, no normalization; guarded common FIR at 8192Hz, 10-2048Hz half-amplitude edges')
    tout,hv,ha=output
    arrays=dict(t_raw=t,q_raw_m=q,v_raw_m_s=v,a_raw_m_s2=a,
        r_m_raw_m=result['measured_relative_xy_m'][use],M_relative_H_m=rotor_section(q,p['z_m'])-q[:,4:6],
        S_relative_H_m=q[:,:2]-q[:,4:6],rO1_raw_m=q[:,6:8]-q[:,4:6],rO2_raw_m=q[:,10:12]-q[:,4:6],
        fit_force_N=f1,fit2_force_N=f2,fit_active_count=d1['active_count'],fit2_active_count=d2['active_count'],
        fit_max_compression_m=d1['max_compression_m'],fit2_max_compression_m=d2['max_compression_m'],
        bearing1_force_N=b1,bearing2_force_N=b2,rear_force_N=rear_force,rear_relative_m=rear_relative,
        bearing1_relative_m=rotor_section(q,p['z_f'])-q[:,6:8],bearing2_relative_m=rotor_section(q,p['z_f2'])-q[:,10:12],
        q_equilibrium=q0,t_out=tout,H_velocity_mm_s=hv,H_acceleration_m_s2=ha,
        metadata_json=json.dumps(metadata,ensure_ascii=False),**{'turn_'+k:val for k,val in turns.items()})
    np.savez_compressed(folder/'native.npz',**arrays)
    write_json(folder/'metrics.json',metrics);write_json(folder/'stationarity.json',steady)
    write_json(folder/'warmup_history.json',history);write_json(folder/'domain.json',domain)
    write_json(folder/'sampling_convergence.json',sampling)
    write_json(folder/'actual_solver_settings.json',solver_history)
    write_json(folder/'actual_config.json',config)
    rows=[]
    for i in range(len(turns['turn_id'])):
        row={k:float(val[i]) for k,val in turns.items() if val.ndim==1}
        row.update(Ex_um=float(turns['peak_to_peak_um'][i,0]),Ey_um=float(turns['peak_to_peak_um'][i,1]))
        rows.append(row)
    write_csv(folder/'turns.csv',rows)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v4_config.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gaps',type=float,nargs='+')
    parser.add_argument('--directions',type=int)
    parser.add_argument('--step-factor',type=float,default=1)
    parser.add_argument('--warmup',type=float)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--single-front-diagnostic',action='store_true')
    args=parser.parse_args(); config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.directions:config['parameters']['N']=args.directions
    config['parameters']['diagnostic_single_front']=args.single_front_diagnostic
    config['simulation']['max_step_s']*=args.step_factor
    if args.warmup is not None:config['simulation']['warmup_s']=args.warmup
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'run_settings.json',config)
    jobs=[]
    for gap in config['clearances_um'] if args.gaps is None else args.gaps:
        local=copy.deepcopy(config);local['parameters']=front_group_parameters(local['parameters'],gap)
        jobs.append((local,args.output/f'gap_{gap:g}um'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[f.result() for f in as_completed([pool.submit(run_case,*job) for job in jobs])]
    write_csv(args.output/'summary.csv',sorted(rows,key=lambda r:r['gap_um']))


if __name__=='__main__':main()
