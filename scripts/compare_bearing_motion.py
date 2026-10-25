"""Four matched bearing conditions: XY orbit, displacement, velocity, acceleration.

Reuses the ideal-force generator and convergence checks. No mechanical equations
are changed, and no measured data or inverse forces are used.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.generate_bearing_faults_ideal import ROOT, generate, spectrum
from scripts.check_bearing_faults_ideal import convergence
from utils.data_io import save_json

XY_COLORS = ['#2868A4', '#C77632']
NODES = [('H', slice(4,6), '前轴承外壳 H｜对应 ACC2-X、ACC1-Y 的仿真节点'),
         ('T', slice(2,4), '刀柄代表截面中心 T｜中心平移，不含表面几何跳动')]


def label(row):
    if row['kind']=='healthy':
        return '健康｜无故障'
    if row['kind']=='ball':
        return f"滚动体二阶失圆｜{row['severity']:g} μm"
    return ('外圈点蚀' if row['kind']=='outer' else '内圈点蚀')+f"｜{row['severity']:g} mm²"


def read_records(out, rows):
    records=[]
    for row in rows:
        with np.load(out/f"{row['case']}.npz") as saved:
            data={key:saved[key] for key in ('display_time_s','display_force_N','display_q',
                                            'display_v','display_acc','q_equilibrium')}
        # Subtract a nearby reference first to retain precision at micron scales.
        relative=data['display_q']-data['display_q'][0]
        data['mean_q_m']=data['display_q'][0]+relative.mean(0)
        data['dynamic_q_um']=(relative-relative.mean(0))*1e6
        records.append(data)
    return records


def node_panel(out,cfg,rows,records,node,indices,title):
    fig,axes=plt.subplots(len(rows),4,figsize=(17.6,3.5*len(rows)+.8),layout='constrained')
    span_q=max(np.abs(r['dynamic_q_um'][:,indices]).max() for r in records)*1.12
    span_v=max(np.abs(r['display_v'][:,indices]*1000).max() for r in records)*1.12
    span_a=max(np.abs(r['display_acc'][:,indices]).max() for r in records)*1.12
    window=cfg['plot']['time_window_ms']
    for row,(case,r) in enumerate(zip(rows,records)):
        time_ms=(r['display_time_s']-r['display_time_s'][0])*1000
        shown=time_ms<=window
        q=r['dynamic_q_um'][:,indices]
        axes[row,0].plot(*q.T,color='#168C97',lw=.65)
        axes[row,0].plot(0,0,'+',color='.4')
        orbit_name='节点 X–Y 轨迹' if node=='H' else '中心 X–Y 轨迹'
        axes[row,0].set(xlabel='ΔX (μm)',ylabel='ΔY (μm)',xlim=(-span_q,span_q),
                        ylim=(-span_q,span_q),title=label(case)+'\n'+orbit_name)
        axes[row,0].set_aspect('equal',adjustable='box')
        quantities=[(q,'动态位移 (μm)',span_q),
                    (r['display_v'][:,indices]*1000,'速度 (mm/s)',span_v),
                    (r['display_acc'][:,indices],'加速度 (m/s²)',span_a)]
        for column,(values,ylabel,span) in enumerate(quantities,1):
            for axis,color in enumerate(XY_COLORS):
                axes[row,column].plot(time_ms[shown],values[shown,axis],color=color,lw=.85,label='XY'[axis])
            axes[row,column].set(xlim=(0,window),ylim=(-span,span),xlabel='时间 (ms)',ylabel=ylabel)
            axes[row,column].legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    d,s=cfg['drive'],cfg['simulation']
    fig.suptitle(title+f"\n{s['rpm']:g} rpm；Fx={d['amplitude_N']:g}cos(2π·{d['frequency_hz']:g}t) N，"
                 f"Fy={d['amplitude_N']:g}sin(2π·{d['frequency_hz']:g}t) N；同列统一量程，位移减去各自平均位置"
                 f"\n轨迹显示 {s['duration_s']:g} s；时域显示前 {window:g} ms；重力沿 −Y",fontsize=13)
    name='housing_H' if node=='H' else 'holder_center_T'
    fig.savefig(out/f'{name}.png',dpi=160)
    fig.savefig(out/f'{name}.svg')
    plt.close(fig)
    return dict(displacement_and_orbit_limit_um=float(span_q),velocity_limit_mm_s=float(span_v),
                acceleration_limit_m_s2=float(span_a))


def common_center_figure(out,rows,records):
    fig,axes=plt.subplots(2,len(rows),figsize=(16,8),layout='constrained')
    for node_row,(node,indices,_) in enumerate(NODES):
        reference=records[0]['q_equilibrium'][indices]
        centers=[(r['display_q'][:,indices]-reference)*1e6 for r in records]
        span=max(np.abs(center).max() for center in centers)*1.12
        for ax,case,r,center in zip(axes[node_row],rows,records,centers):
            ax.plot(*center.T,color='#168C97',lw=.55)
            ax.plot(*((r['mean_q_m'][indices]-reference)*1e6),'o',color='#BC345A',ms=5,
                    markeredgecolor='white',zorder=4)
            ax.plot(0,0,'+',color='black',ms=8,zorder=5)
            ax.set(xlim=(-span,span),ylim=(-span,span),aspect='equal',xlabel='X − 健康静态位置 (μm)',
                   ylabel='Y − 健康静态位置 (μm)',title=label(case)+'\n'+('前外壳 H' if node=='H' else '刀柄中心 T'))
            ax.grid(alpha=.18)
    fig.suptitle('使用共同位置参考的轨迹｜红点为各组平均中心，黑十字为健康静态中心\n保留平均位置变化；同一行统一尺度')
    fig.savefig(out/'common_reference_centers.png',dpi=160)
    plt.close(fig)


def spectral_figure(out,cfg,rows,records):
    fig,axes=plt.subplots(len(rows),2,figsize=(11,2.4*len(rows)+.7),layout='constrained')
    for column,(node,indices,_) in enumerate(NODES):
        spectra=[spectrum(r['display_acc'][:,indices],cfg['simulation']['display_fs_hz']) for r in records]
        ymax=max(a[f<=3500].max() for f,a in spectra)*1.12
        for row,(case,(f,a)) in enumerate(zip(rows,spectra)):
            ax=axes[row,column]
            for axis,color in enumerate(XY_COLORS):
                ax.plot(f,a[:,axis],color=color,lw=.8,label='XY'[axis])
            ax.axvline(cfg['drive']['frequency_hz'],color='.5',ls='--',lw=.6)
            ax.set(xlim=(0,3500),ylim=(0,ymax),xlabel='频率 (Hz)',ylabel='加速度峰值 (m/s²)',
                   title=label(case)+'｜'+node)
            ax.grid(alpha=.18)
            ax.legend(frameon=False,ncol=2,fontsize=9)
    fig.suptitle('频谱核对｜H、T同步加速度；同列统一量程，虚线为外加激励频率')
    fig.savefig(out/'acceleration_spectra.png',dpi=160)
    plt.close(fig)


def analyze(out,cfg,rows,records):
    summary=dict(measured_data_read=False,mechanical_equations_changed=False,
                 same_force=all(np.array_equal(records[0]['display_force_N'],r['display_force_N']) for r in records),
                 same_time=all(np.array_equal(records[0]['display_time_s'],r['display_time_s']) for r in records),
                 finite=all(np.isfinite(r[k]).all().item() for r in records for k in ('display_q','display_v','display_acc')),
                 coordinate_convention='Z spindle axis; X horizontal radial; Y upward radial; gravity -Y',
                 time_window_ms=cfg['plot']['time_window_ms'],orbit_duration_s=cfg['simulation']['duration_s'],cases=[])
    assert summary['same_force'] and summary['same_time'] and summary['finite']
    flat=[]
    for row,r in zip(rows,records):
        item=dict(case=row['case'],label=label(row),nodes={})
        for node,indices,_ in NODES:
            q=r['dynamic_q_um'][:,indices]
            item['nodes'][node]=m=dict(mean_xy_um=(r['mean_q_m'][indices]*1e6).tolist(),
                mean_offset_from_healthy_static_xy_um=((r['mean_q_m'][indices]-records[0]['q_equilibrium'][indices])*1e6).tolist(),
                displacement_pp_xy_um=np.ptp(q,axis=0).tolist(),
                dynamic_radial_rms_um=float(np.sqrt(np.mean(np.sum(q*q,axis=1)))),
                dynamic_max_radius_um=float(np.linalg.norm(q,axis=1).max()),
                velocity_rms_xy_mm_s=(np.sqrt(np.mean(r['display_v'][:,indices]**2,axis=0))*1000).tolist(),
                acceleration_rms_xy_m_s2=np.sqrt(np.mean(r['display_acc'][:,indices]**2,axis=0)).tolist())
            # Descriptive check only: deterministic beating can change short-window RMS.
            windows=np.array_split(r['display_acc'][:,indices],8)
            m['acceleration_rms_125ms_windows_xy']=[np.sqrt(np.mean(w*w,axis=0)).tolist() for w in windows]
            flat.append([row['case'],node,*m['mean_xy_um'],*m['mean_offset_from_healthy_static_xy_um'],
                         *m['displacement_pp_xy_um'],m['dynamic_radial_rms_um'],m['dynamic_max_radius_um'],
                         *m['velocity_rms_xy_mm_s'],*m['acceleration_rms_xy_m_s2']])
        summary['cases'].append(item)
    np.savetxt(out/'motion_metrics.csv',np.asarray(flat,dtype=object),fmt='%s',delimiter=',',comments='',
               header='case,node,mean_x_um,mean_y_um,mean_offset_x_um,mean_offset_y_um,q_pp_x_um,q_pp_y_um,'
                      'radial_rms_um,max_radius_um,v_rms_x_mm_s,v_rms_y_mm_s,a_rms_x_m_s2,a_rms_y_m_s2')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/bearing_motion_250hz.json')
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plots-only',action='store_true')
    mode.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    out=args.output
    if args.plots_only or args.check_only:
        saved=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        cfg,rows=saved['config'],saved['cases']
    else:
        if (out/'summary.json').exists():
            raise FileExistsError('Use a new --output directory to preserve existing results')
        out.mkdir(parents=True,exist_ok=True)
        cfg=json.loads(args.config.read_text(encoding='utf-8'))
        save_json(out/'config.json',cfg)
        rows=generate(out,cfg)
    if args.check_only:
        convergence(out,cfg)
        return
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    records=read_records(out,rows)
    summary=analyze(out,cfg,rows,records)
    summary['plot_limits']={node:node_panel(out,cfg,rows,records,node,indices,title) for node,indices,title in NODES}
    common_center_figure(out,rows,records)
    spectral_figure(out,cfg,rows,records)
    save_json(out/'motion_summary.json',summary)
    print('Saved matched H/T motion figures and metrics',flush=True)


if __name__=='__main__':
    main()
