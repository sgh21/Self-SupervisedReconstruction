"""Common physical axes for V4 range, contact, trajectory and H comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram


def plot(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    data_root=root/('pilots' if (root/'pilots').exists() else 'data')
    cases=[]
    for file in data_root.glob('gap_*/native.npz'):
        m=json.loads(file.with_name('metrics.json').read_text(encoding='utf-8'))
        cases.append((m['gap_um'],file,m))
    cases.sort();plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.grid':True,'grid.alpha':.25})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for key,label in [('E_x_um','X PP'),('E_y_um','Y PP'),('E_dir_um','E_dir')]:
        axes[0].plot([x[0] for x in cases],[x[2][key] for x in cases],'o-',label=label)
    axes[0].set(xlabel='Common radial gap (um)',ylabel='Mean per-turn PP (um)');axes[0].legend()
    for i in (1,2):
        prefix='fit_' if i==1 else 'fit2_'
        axes[1].plot([x[0] for x in cases],[x[2][prefix+'contact_fraction'] for x in cases],'o-',label=f'Fit {i}')
    axes[1].set(xlabel='Common radial gap (um)',ylabel='Contact time fraction',ylim=(-.03,1.03));axes[1].legend()
    fig.tight_layout();fig.savefig(out/'range_contact.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for gap,file,m in cases:
        with np.load(file) as d:
            meta=json.loads(str(d['metadata_json']));t=d['t_raw'];end=meta['export_end_s']
            use=(t>=end-3*60/abs(meta['config']['parameters']['rpm']))&(t<end)
            xy=d['r_m_raw_m'][use]*1e6
            axes[0].plot(xy[:,0],xy[:,1],label=f'{gap:g} um')
            xy=xy-xy.mean(axis=0);axes[1].plot(xy[:,0],xy[:,1],label=f'{gap:g} um')
    for ax in axes:ax.set(xlabel='M X (um)',ylabel='M Y (um)');ax.set_aspect('equal',adjustable='datalim');ax.legend()
    axes[0].set_title('Absolute center, last 3 turns');axes[1].set_title('Mean removed for display; 1X retained')
    fig.tight_layout();fig.savefig(out/'trajectories.png');plt.close(fig)
    # Compare health with largest simulated degraded E; validity is stated in report.
    chosen=[cases[0],max(cases[1:],key=lambda x:x[2]['E_dir_um'])]
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex='col')
    figc,axc=plt.subplots(2,2,figsize=(11,6),sharex=True)
    phase_details={}
    for gap,file,m in chosen:
        with np.load(file) as d:
            meta=json.loads(str(d['metadata_json']));end=meta['export_end_s'];rpm=meta['config']['parameters']['rpm']
            t=d['t_out'];hv=d['H_velocity_mm_s'];use=t>=end-3*60/abs(rpm)
            f,psd=periodogram(hv,fs=8192,window='hann',axis=0,scaling='density')
            for i in range(2):
                axes[i,0].plot((t[use]-end)*1000,hv[use,i],label=f'c={gap:g} um')
                axes[i,1].semilogy(f,np.maximum(psd[:,i],1e-22),label=f'c={gap:g} um')
            t=d['t_raw'];use=(t>=end-3*60/abs(rpm))&(t<end)
            for i,key in enumerate(('fit_force_N','fit2_force_N')):
                axc[i,0].plot((t[use]-end)*1000,np.linalg.norm(d[key][use],axis=1),label=f'c={gap:g} um')
                rkey='rO1_raw_m' if i==0 else 'rO2_raw_m'
                axc[i,1].plot((t[use]-end)*1000,np.linalg.norm(d[rkey][use],axis=1)*1e6,label=f'c={gap:g} um')
            keep=(t>=meta['export_start_s'])&(t<end);angle=rpm*2*np.pi/60*t[keep]+meta['config']['excitation']['phase_rad']
            design=np.column_stack((np.ones(sum(keep)),np.cos(angle),np.sin(angle)))
            phase_details[str(gap)]={}
            for key,signal in [('M',d['r_m_raw_m']),('H',d['q_raw_m'][:,4:6]),('M_minus_H',d['M_relative_H_m'])]:
                coeff=np.linalg.lstsq(design,signal[keep]*1e6,rcond=None)[0]
                phase_details[str(gap)][key]=dict(mean_um=coeff[0].tolist(),amplitude_um=np.linalg.norm(coeff[1:],axis=0).tolist(),
                    phase_rad=np.arctan2(-coeff[2],coeff[1]).tolist(),cos_um=coeff[1].tolist(),sin_um=coeff[2].tolist())
    for i in range(2):
        axes[i,0].set(ylabel=f'H {"XY"[i]} velocity (mm/s)');axes[i,1].set(ylabel='PSD ((mm/s)^2/Hz)',xlim=(0,2048))
        axc[i,0].set(ylabel=f'Fit {i+1} force magnitude (N)');axc[i,1].set(ylabel=f'O{i+1}-H radius (um)')
        for ax in (*axes[i],*axc[i]):ax.legend()
    axes[1,0].set_xlabel('Time relative window end (ms)');axes[1,1].set_xlabel('Frequency (Hz)')
    for ax in axc[1]:ax.set_xlabel('Time relative window end (ms)')
    fig.tight_layout();fig.savefig(out/'housing_velocity_psd.png');plt.close(fig)
    figc.tight_layout();figc.savefig(out/'two_fit_time.png');plt.close(figc)
    (out/'phase_decomposition.json').write_text(json.dumps(phase_details,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);plot(parser.parse_args().root)
