"""Run a bounded first experiment; no GUI, training, or application changes."""
import argparse
import time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from utils.data_io import DEFAULT_PACKAGE,load_health,bandlimit,save_json,capacitance_reference
from utils.metrics import comparison_metrics
from spindle.dynamics_model import (default_parameters,area_and_link,inverse_housing_force,
                            harmonic_response,simulate,displacement_metrics,
                            modes_and_stability,internal_force)
from utils.plot_report import make_plots,write_report


def calibrate_forcing(train,fs,p,band=(10,900),nperseg=1024,quantity='velocity'):
    """Only equivalent excitation is fitted; M, contacts, stiffness are NOT identified.
    Band spectrum selected from TRAIN only. Measured bin energy supplies amplitude;
    first training window supplies arbitrary common time origin (no keyphase).
    A 3x3 gain grid compensates retained-line RMS; no long optimizer.
    """
    if quantity not in ('velocity', 'acceleration'):
        raise ValueError('Calibration quantity must be velocity or acceleration, in SI units')
    f,psd=welch(train,fs=fs,nperseg=nperseg,axis=-1)
    psd=psd.mean(axis=0)
    selected=np.flatnonzero((f>=band[0])&(f<=band[1]))
    frequencies=f[selected]
    amplitudes=[]
    reference=train[0]
    tt=np.arange(reference.shape[-1])/fs
    for i,frequency in zip(selected,frequencies):
        # sqrt(2*PSD*df) gives the sinusoidal peak in the selected SI quantity.
        peak=np.sqrt(2*psd[:,i]*(f[1]-f[0]))
        phase=np.angle(reference@np.exp(-2j*np.pi*frequency*tt))
        amplitudes.append(peak*np.exp(1j*phase))
    omega = 2*np.pi*frequencies[:,None]
    velocity_amplitudes = np.asarray(amplitudes)
    if quantity == 'acceleration':
        velocity_amplitudes = velocity_amplitudes/(1j*omega)
    coefficients=inverse_housing_force(p,frequencies,velocity_amplitudes)
    response=harmonic_response(p,0,frequencies,coefficients)
    multiplier = 1j*omega if quantity == 'velocity' else -omega**2
    predicted_rms=np.sqrt(np.sum(abs(multiplier*response[:,4:6])**2,axis=0)/2)
    target_rms=np.sqrt(np.mean(train**2,axis=(0,2)))
    search=[]
    for gx in [0.8,1.0,1.2]:
        for gy in [0.8,1.0,1.2]:
            gain=np.array([gx,gy])*target_rms/predicted_rms
            r=harmonic_response(p,0,frequencies,coefficients*gain)
            rms=np.sqrt(np.sum(abs(multiplier*r[:,4:6])**2,axis=0)/2)
            objective=float(np.mean((np.log(rms/target_rms))**2))
            search.append(dict(gains=gain.tolist(),relative_grid=[gx,gy],objective=objective))
    best=min(search,key=lambda x:x['objective'])
    coefficients*=np.asarray(best['gains'])
    info=dict(identified='conditional equivalent radial excitation only',
              structural_parameters_identified=False,
              forcing_assumption='external multisine on shaft, unchanged across faults',
              frequencies_hz=frequencies.tolist(),peak_force_amplitudes_N=abs(coefficients).tolist(),
              force_phase_rad=np.angle(coefficients).tolist(),
              phase='relative training-window phase; no mechanical keyphase or phase validation',
              search=search,best=best,calibration_quantity=quantity,band_hz=list(band),
              spectral_bins=len(frequencies),noise_added=False,
              interpretation='calibrated equivalent force spectrum, not measured forces; may absorb sensor/environment noise')
    info['training_rms_mm_s' if quantity=='velocity' else 'training_rms_m_s2'] = (
        target_rms*(1000 if quantity=='velocity' else 1)).tolist()
    return frequencies,coefficients,info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package',type=Path,default=DEFAULT_PACKAGE)
    ap.add_argument('--output',type=Path,required=True,help='New experiment directory')
    ap.add_argument('--duration',type=float,default=0.5)
    ap.add_argument('--max-seconds',type=float,default=240)
    ap.add_argument('--max-step',type=float,default=1/32768)
    ap.add_argument('--output-fs',type=int,default=8192)
    args=ap.parse_args()
    if args.duration<0.1:raise ValueError('Use at least 0.1 s for spectral comparisons')
    start=time.perf_counter();deadline=start+args.max_seconds
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    train,holdout,data=load_health(args.package)
    reference,contour=capacitance_reference()
    p=default_parameters();p['rpm']=data['actual_speed_median_rpm']
    print(f'Data: {len(train)} training + {len(holdout)} holdout windows; {p["rpm"]:.3f} rpm',flush=True)
    frequencies,coefficients,calibration=calibrate_forcing(train,data['fs_hz'],p)
    area,_=area_and_link(p,0)
    cases=[];signals=[]
    for fraction in [0.0,0.1,0.3]:
        print(f'Solving contact area loss {fraction:.0%}',flush=True)
        result=simulate(p,area*fraction,frequencies,coefficients,duration=args.duration,
                        fs=args.output_fs,max_step=args.max_step,deadline=deadline)
        hv=bandlimit(result['H_velocity_m_s'].T,args.output_fs).T
        metric=displacement_metrics(result['T_center_m'])
        row=dict(case=f'area_loss_{fraction:.2f}',lost_area_fraction=fraction,
                 lost_area_mm2=area*fraction*1e6,nominal_area_mm2=area*1e6,
                 mode_check=modes_and_stability(p,area*fraction),displacement_metrics=metric,
                 housing_radial_rms_mm_s=float(np.sqrt(np.mean(np.sum(hv**2,axis=1)))*1000),
                 equilibrium_residual_N=result['equilibrium_residual_N'],nfev=result['nfev'])
        cases.append(row);signals.append(result)
        columns=np.column_stack([result['t'],np.full(len(hv),p['rpm']),
                                 np.full(len(hv),area*fraction*1e6),
                                 result['H_acceleration_m_s2'],hv*1000,result['T_center_m']*1e6])
        np.savetxt(out/(row['case']+'.csv'),columns,delimiter=',',
                   header='time_s,rpm,lost_taper_area_mm2,H_ax_m_s2,H_ay_m_s2,H_vx_band_mm_s,H_vy_band_mm_s,T_center_x_um,T_center_y_um',comments='')
        np.savez(out/(row['case']+'.npz'),**result)
    print('Checking worst-case integration with half maximum step',flush=True)
    fine=simulate(p,area*0.3,frequencies,coefficients,duration=args.duration,
                  fs=args.output_fs,max_step=args.max_step/2,deadline=deadline)
    worst=signals[-1]
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20))
    # Center before displacement convergence so large static offsets cannot mask error.
    coarse_t=worst['T_center_m']-worst['T_center_m'].mean(0)
    fine_t=fine['T_center_m']-fine['T_center_m'].mean(0)
    step_check=dict(H_velocity_relative_l2=relative(worst['H_velocity_m_s'],fine['H_velocity_m_s']),
                    T_dynamic_relative_l2=relative(coarse_t,fine_t),
                    coarse_step_s=args.max_step,fine_step_s=args.max_step/2)
    tiny=harmonic_response(p,area*1e-8,frequencies,coefficients)
    healthy_linear=harmonic_response(p,0,frequencies,coefficients)
    print('Checking near-zero area loss in nonlinear time integration',flush=True)
    tiny_time=simulate(p,area*1e-8,frequencies,coefficients,duration=min(0.15,args.duration),
                       fs=args.output_fs,max_step=args.max_step,deadline=deadline)
    nz=len(tiny_time['t'])
    zero_time_relative=relative(tiny_time['q']-tiny_time['q_equilibrium'],
                               signals[0]['q'][:nz]-signals[0]['q_equilibrium'])
    test_q=np.array([1,2,3,4,5,6])*1e-6;test_v=np.zeros(6)
    r=internal_force(0,test_q,test_v,p,area*0.3).reshape(3,2)
    reaction_residual=r.sum(axis=0)-p['kh']*test_q[4:]
    checks=dict(zero_fault_limit_relative_linear_response=relative(tiny,healthy_linear),
                zero_fault_limit_relative_nonlinear_dynamic_state=zero_time_relative,
                internal_reaction_balance_N=float(np.linalg.norm(reaction_residual)),
                finite_all_cases=all(np.isfinite(s['q']).all() for s in signals),
                stable_loaded_linearizations=all(c['mode_check']['maximum_pole_real_s_inv']<0 for c in cases),
                step_convergence=step_check,
                expected_fault='taper area loss changes link stiffness and coupled poles; not bearing BPFO',
                monotonicity_required=False,fault_experimentally_validated=False)
    comparison=comparison_metrics(signals[0]['H_velocity_m_s'],holdout,args.output_fs,data['fs_hz'])
    results=dict(data=data,parameters=p,parameter_origins={
                     'rpm':'measured stable telemetry','t_section_from_face':'user approximate 15 mm',
                     'healthy_error_reported_um':'user synchronous P-P approximate scale, not imposed',
                     'balls/ball_diameter/contact_angle':'Tai I Table1 p10 example 7014; not confirmed JD bearing',
                     'other_mechanical_parameters':'provisional assumptions, not identified'},
                 capacitance_reference=reference,calibration=calibration,cases=cases,health_holdout=comparison,
                 checks=checks,elapsed_seconds=time.perf_counter()-start,
                 route='measured-data calibrated physical response; no measured-output superposition',
                 holder_validation='CSH1 audited at 6000 rpm on different date; no synchronous 7000 rpm holder validation',
                 bearing_lookup=dict(user_model='JD130S-24-H40A',verified_bearing_model=None,
                                     status='no exact verifiable bearing designation found in this search',
                                     related_manufacturer_model='JD130SC-24-HA40(CTS), not substituted',
                                     manufacturer_catalog='https://usmobile.jingdiao.com/resource/General-Catalog.pdf'),
                 mechanical_parameters='assumptions; bearing model not confirmed for JD130S-24-H40A')
    save_json(out/'experiment.json',results)
    make_plots(out,train,holdout,signals,cases,data['fs_hz'],contour)
    results['elapsed_seconds']=time.perf_counter()-start
    save_json(out/'experiment.json',results)
    write_report(out,results)
    print('Completed:',out,flush=True)
    print('Health holdout:',comparison,flush=True)
    print('Checks:',checks,flush=True)


if __name__=='__main__':main()
