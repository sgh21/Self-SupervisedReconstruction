"""Add compact raw/processed-reference evidence without changing native results."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import firwin,freqz
from utils.pitting_analysis import resample_to
from scripts.run_clearance_v3 import write_json


def enrich(path):
    path=Path(path)
    # Earlier warmup-extension exports cropped guards without retaining the
    # antialias sampling phase. Align using saved t_out; only guard samples go.
    with np.load(path) as d:
        meta=json.loads(str(d['metadata_json']));fs=meta['config']['simulation']['fs_raw_hz']
        ratio=int(fs/meta['config']['simulation']['fs_out_hz'])
        offset=int(round((d['t_out'][0]-d['t_raw'][0])*fs))%ratio
        if offset:
            count=len(d['t_raw']);arrays={k:d[k] for k in d.files}
    if offset:
        for key,value in arrays.items():
            if value.ndim and len(value)==count:arrays[key]=value[offset:]
        meta['guard_phase_alignment_removed_samples']=offset
        arrays['metadata_json']=json.dumps(meta,ensure_ascii=False)
        replacement=path.with_name('native_aligned.npz')
        np.savez_compressed(replacement,**arrays);replacement.replace(path)
    with np.load(path) as d:
        meta=json.loads(str(d['metadata_json']));c=meta['config'];p=c['parameters'];s=c['simulation']
        t=d['t_raw'];q=d['q_raw_m'];v=d['v_raw_m_s'];a=d['a_raw_m_s2']
        valid=(t>=meta['export_start_s'])&(t<meta['export_end_s'])
        hv=resample_to(v[:,4:6].T,s['fs_raw_hz'],s['fs_out_hz']).T*1000
        ha=resample_to(a[:,4:6].T,s['fs_raw_hz'],s['fs_out_hz']).T
        tout=t[0]+np.arange(len(hv))/s['fs_out_hz']
        keep=(tout>=meta['export_start_s']-1e-12)&(tout<meta['export_end_s']-1e-12)
        torque=np.zeros((len(t),2))
        for z,force in ((p['z_f'],d['fit_force_N']),(p['z_f2'],d['fit2_force_N']),(p['healthy_supports'][0]['z_m'],d['rear_force_N'])):
            torque[:,0]+=z*force[:,1];torque[:,1]-=z*force[:,0]
        units=dict(q=['m']*8+['rad']*2+['m']*2,v=['m/s']*8+['rad/s']*2+['m/s']*2,
                   a=['m/s2']*8+['rad/s2']*2+['m/s2']*2)
        np.savez_compressed(path.with_name('observation_auxiliary.npz'),valid_raw_mask=valid,
            angle_rad=p['rpm']*2*np.pi/60*t+c['excitation']['phase_rad'],
            t_out=tout[keep],H_velocity_unbandpassed_mm_s=hv[keep],H_acceleration_unbandpassed_m_s2=ha[keep],
            housing_constraint_torque_Nm=torque,housing_ground_force_N=p['kh']*q[:,4:6]+p['ch']*v[:,4:6],
            state_units_json=json.dumps(units),state_order_json=json.dumps(meta['state_order']))
        edges=np.linspace(meta['export_start_s'],meta['export_end_s'],5)
        means=np.array([(q[(t>=l)&(t<r),10:12]-q[(t>=l)&(t<r),4:6]).mean(axis=0)*1e6 for l,r in zip(edges[:-1],edges[1:])])
        m=json.loads(path.with_name('metrics.json').read_text(encoding='utf-8'))
        limit=max(c['numerical_thresholds']['position_um'],.01*m['E_dir_um'])
        forcebalance=[]
        for index,mi,bkey,pkey in ((6,p['mO'],'bearing1_force_N','fit_force_N'),(10,p['mO2'],'bearing2_force_N','fit2_force_N')):
            residual=mi*a[:,index:index+2]-(mi*np.array(p['gravity_radial_m_s2'])+d[bkey]-d[pkey])
            forcebalance.append(float(np.max(abs(residual[valid]))))
        write_json(path.with_name('observation_checks.json'),dict(O2H_subwindow_means_um=means.tolist(),
            O2H_drift_tolerance_um=limit,O2H_drift_pass=bool(np.max(np.ptp(means,axis=0))<=limit),
            outer_dynamic_equilibrium_max_residual_N=forcebalance,outer_dynamic_equilibrium_pass=max(forcebalance)<1e-8,
            original_name_warning='q_raw_m/v_raw_m_s/a_raw_m_s2 retain legacy names; indices8,9 have rad/rad_s/rad_s2 units',
            valid_raw_samples=int(valid.sum()),raw_guard_left_s=meta['export_start_s']-t[0],raw_guard_right_s=t[-1]-meta['export_end_s']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    for path in args.root.rglob('native.npz'):
        if path.with_name('metrics.json').exists():enrich(path)
    c=json.loads((args.root/'frozen_physics.json').read_text(encoding='utf-8'));p=c['processing'];fs=p['fs_hz']
    freq=np.array([10,abs(c['parameters']['rpm'])/60,abs(c['parameters']['rpm'])/30,abs(c['parameters']['rpm'])/20,900,2048])
    taps=firwin(p['fir_taps'],p['passband_hz'],fs=fs,pass_zero=False,window=('kaiser',p['kaiser_beta']))
    _,gain=freqz(taps,worN=freq,fs=fs)
    aa=firwin(161,1/8,window=('kaiser',5));_,again=freqz(aa,worN=freq,fs=65536)
    write_json(args.root/'validation/processing_gain.json',dict(frequency_Hz=freq.tolist(),FIR_amplitude_gain=abs(gain).tolist(),
        antialias_amplitude_gain=abs(again).tolist(),combined_amplitude_gain=abs(gain*again).tolist(),
        required_guard_s=.25+80/65536,actual_configured_guard_s=c['simulation']['guard_s'],
        edges='FIR nominal cutoff half-amplitude; not an ideal rectangular passband'))


if __name__=='__main__':main()
