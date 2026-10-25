"""Small scientific plots and report; no mechanical formulas here."""
from pathlib import Path
from os.path import relpath
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import PROJECT_ROOT
from .data_io import bandlimit


def relative_link(out,target):
    return Path(relpath(target,Path(out).resolve())).as_posix()


def make_plots(out,train,holdout,signals,cases,fs,capacitance_contour=None):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,
                         'grid.alpha':0.25,'figure.dpi':130})
    healthy=signals[0];n=len(healthy['t']);t=healthy['t']-healthy['t'][0]
    model_fs=int(healthy['output_fs_hz']);measured_n=int(round(n/model_fs*fs))
    measured_time=np.arange(measured_n)/fs
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for ch in range(2):
        axes[ch,0].plot(measured_time,holdout[0,ch,:measured_n]*1000,label='Held-out measured',alpha=0.65)
        v=bandlimit(healthy['H_velocity_m_s'][:,ch],model_fs)*1000
        axes[ch,0].plot(t,v,label='Healthy model',lw=0.8)
        axes[ch,0].set(xlabel='Time (s)',ylabel=f'H {"XY"[ch]} velocity (mm/s)',
                       title='Time origin not phase synchronized')
        fd,pd=welch(holdout[:,ch,:]*1000,fs,nperseg=measured_n,axis=-1)
        _,pt=welch(train[:,ch,:]*1000,fs,nperseg=measured_n,axis=-1)
        fm,pm=welch(v,model_fs,nperseg=n)
        kd=(fd>=10)&(fd<=900);km=(fm>=10)&(fm<=900)
        axes[ch,1].semilogy(fd[kd],pd.mean(0)[kd],label='Held-out measured')
        axes[ch,1].semilogy(fd[kd],pt.mean(0)[kd],label='Training measured',alpha=.6)
        axes[ch,1].semilogy(fm[km],pm[km],label='Healthy model')
        axes[ch,1].set(xlim=(10,900),xlabel='Frequency (Hz)',ylabel='PSD ((mm/s)^2/Hz)')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.savefig(out/'health_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for s,c in zip(signals,cases):
        label=f'{100*c["lost_area_fraction"]:.0f}% area loss'
        t=s['t']-s['t'][0];q=s['T_center_m']-s['T_center_m'].mean(0)
        axes[0].plot(t,bandlimit(s['H_velocity_m_s'][:,1],model_fs)*1000,label=label,lw=.9)
        axes[1].plot(q[:,0]*1e6,q[:,1]*1e6,label=label,lw=.8)
        f,ps=welch(bandlimit(s['H_velocity_m_s'][:,1],model_fs)*1000,model_fs,nperseg=len(t))
        keep=(f>=10)&(f<=900);axes[2].semilogy(f[keep],ps[keep],label=label)
    axes[0].set(xlabel='Time (s)',ylabel='Housing Y velocity (mm/s)')
    axes[1].set(xlabel='T dynamic X (um)',ylabel='T dynamic Y (um)',title='Center displacement; not probe runout')
    axes[1].set_aspect('equal',adjustable='datalim')
    axes[2].set(xlim=(10,900),xlabel='Frequency (Hz)',ylabel='Housing Y PSD')
    axes[0].legend(fontsize=8)
    fig.savefig(out/'paired_cases.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.3),layout='constrained')
    for c in cases:
        ax.plot(c['mode_check']['mode_frequencies_hz'],np.full(6,100*c['lost_area_fraction']),
                'o',label=f'{100*c["lost_area_fraction"]:.0f}%')
    ax.set(xlabel='Loaded coupled natural frequency (Hz)',ylabel='Taper area loss (%)')
    ax.legend();fig.savefig(out/'mode_shift.png');plt.close(fig)
    if capacitance_contour is not None:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        axes[0].plot(capacitance_contour[:,0],capacitance_contour[:,1])
        axes[0].set(xlabel='Probe X (um)',ylabel='Probe Y (um)',
                    title='Measured 6000 rpm, Sep 20\nSynchronous surface contour')
        q=healthy['T_center_m']-healthy['T_center_m'].mean(0)
        axes[1].plot(q[:,0]*1e6,q[:,1]*1e6,lw=.7)
        axes[1].set(xlabel='Center X (um)',ylabel='Center Y (um)',
                    title='Model 7000 rpm\nDynamic center trajectory')
        for a in axes:a.set_aspect('equal',adjustable='datalim')
        fig.savefig(out/'holder_reference_limits.png');plt.close(fig)


def write_report(out,r):
    c=r['health_holdout'];data=r['data'];checks=r['checks']
    rows='\n'.join(f'| {100*x["lost_area_fraction"]:.0f}% | {x["lost_area_mm2"]:.1f} | '
                    f'{x["housing_radial_rms_mm_s"]:.5f} | '
                    f'{x["displacement_metrics"]["dynamic_radial_rms_um"]:.5f} | '
                    f'{x["displacement_metrics"]["dynamic_max_radius_um"]:.5f} |' for x in r['cases'])
    reference=r.get('capacitance_reference')
    fmt=lambda values,pattern: ' / '.join(format(v,pattern) for v in values)
    reference_text=('无可读取的电容参考。' if reference is None else
                    f"电容原始记录：`{reference['source_csv']}`；3906.25 Hz、µm，2026-09-20、6000 rpm。复用既有分析得到X/Y同步轮廓峰峰值{reference['synchronous_pp_um']} µm，缓存重算一致。X=Distance2、Y=−Distance1，两路共用估计相位。每转约39原始点，插值512点不增加带宽；表面形状与中心运动尚未分离。与7月7000rpm振动不同日期/转速，未用于7000rpm校准或配对验证。用户15mm位置为近似信息。")
    text=f'''# 小规模主轴动力学原型：首轮实际运行

选择实测约束物理模型的等效激励路线，未唯一辨识结构参数；不叠加外壳实测残差，不任意加噪声或缩放健康波形表示退化。首轮为刀柄锥面等效均匀接触面积损失，非轴承点蚀或轴—内圈间隙。运行积分、检查与绘图耗时 {r['elapsed_seconds']:.2f} s；无长时间优化。

## 数据与输出

- 输入包：`{data['package']}`。原始加速度25.6 kHz/g；本次复用2048 Hz、10–900 Hz、mm/s速度，进入模型换算m/s。
- ACC2为前轴承外壳X，ACC1为前轴承外壳Y；ACC3是台面Y，未当作第三径向通道。转速遥测中位数{r['parameters']['rpm']:.3f} rpm，空转；额外切削载荷0 N，重力保留。
- 校准：{data['train_start_s']:.0f}–{data['train_end_s']:.0f} s，{len(data['train_window_indices'])}个10秒窗；留出：{data['validation_start_s']:.0f}–{data['validation_end_s']:.0f} s，{len(data['validation_window_indices'])}窗，中间20秒不用。同一采集的时间留出，不称跨采集验证。
- 同一非线性积分同步输出H外壳绝对加速度/速度和T刀柄截面中心X/Y位移；T为距靠近主轴端面约15mm的集中节点，尚无实际几何插值与转角。H为刚体前外壳节点，不表示任意局部测点。
- 仿真原生输出8192 Hz，避免最高约1084 Hz模型模态越过2048 Hz输出的Nyquist；与实测分别按各自采样率处理，在相同10–900 Hz频带比较。未把原生高频输出直接降采样冒充实测。
- 无键相；驱动力的相位来自校准窗相对时间零点，仿真三工况共用，不声称实测逐点相位吻合。

## 方程、原文与推导

核心数学全部在[dynamics_model.py](<{relative_link(out,PROJECT_ROOT/'src/spindle/dynamics_model.py')}>)。q含轴、刀柄截面、外壳各两径向平移。轴承力B和接口力L双侧反向装配：

`ms*s_ddot=F+ms*g-B-L; mt*T_ddot=mt*g+L; mh*h_ddot=mh*g+B-Kh*h-Ch*h_dot`。

Tai–Altintas2024 I p3式1给内外圈Hertz接触；本原型假设球无独立惯性、两侧等力，由两个串联接触的压缩量相加推导等效系数。每球正压缩接触、开接触无拉力，固定接触角与暂定重叠量；阻尼用相对振动速度的法向投影，不是旋转接触压缩量的完整时间导数。未实现完整高速角接触四未知局部解。

Namazi2007 p1335式6/8、Tai I p8式41/42给单位面积接触弹簧与失接触面积的势能。这里假设相对平移在有效面积上均匀：`U=0.5*kappa*(A0-A_loss)*|s-T|²`，所以`Kj=kappa*(A0-A_loss)`；再与暂定弯曲弹簧串联。不是原论文5DOF分布梁的完整复现，不预测局部磨损的倾斜/旋转调制。

原文：[Tai I](<{relative_link(out,PROJECT_ROOT/'docs/references/papers/A4HLRH4Z.pdf')}>)、[Namazi2007](<{relative_link(out,PROJECT_ROOT/'docs/references/papers/Namazi2007_interface_publisher.pdf')}>)、[Cao2004](<{relative_link(out,PROJECT_ROOT/'docs/references/papers/NV2AMLQH.pdf')}>)。实际读过相应全文/原页；原式疑义未照抄。

## 校准范围与假设

固定暂定质量、刚度、接触角与等效预紧重叠；只校准给定这些结构参数下的等效径向多频激励，频率和能量只取校准数据。9组幅值增益搜索。它不唯一识别真实载荷、轴承预紧或接口参数；未知输入与传递路径存在混淆。

完整暂定参数和力幅值/相位在[experiment.json](experiment.json)。7014算例的20球/11.906mm/15°作为文献几何占位，节圆90mm为暂定；它们未确认为JD130S-24-H40A轴承。正常同步回转峰峰值约3µm是用户提供的信息，不是0；未强加成3µm圆轨迹。这里计算的中心动态位移指标不包含尚未分离的几何跳动和探头支架运动。

{reference_text}

球离心/陀螺、接触角变化未求解；单转速等效径向支承是首版近似，不能由健康RMS吻合证明高转速轴承机理或故障幅值正确。
公开检索核到厂家目录的近似型号JD130SC-24-HA40(CTS)，尚未取得用户JD130S-24-H40A的可核验轴承型号；未把近似型号或BT30型号的轴承代入。真实球数、尺寸、沟曲率和预紧仍待厂家/拆检资料。[厂家目录索引](https://usmobile.jingdiao.com/resource/General-Catalog.pdf)

## 留出检查与故障结果

H X/Y模型RMS：{fmt(c['model_rms_mm_s'],'.5f')} mm/s；留出实测：{fmt(c['measured_rms_mm_s'],'.5f')} mm/s；相对误差：{fmt(c['relative_rms_error'],'.2%')}。
频谱对数RMSE：{fmt(c['log_psd_rmse_db'],'.2f')} dB。训练谱的{r['calibration']['spectral_bins']}个频率格点构成共同等效多频激励，不独立优化这些结构参数；相位取训练记录相对时间。未知测量噪声/环境振动可能被吸收到等效力，这不是实测力或唯一辨识。时域图按各自时间起点展示，无机械相位真值，不把匹配RMS等同模型真实。
图中第一留出短片段RMS为{fmt(c['first_displayed_holdout_segment_rms_mm_s'],'.5f')} mm/s，与全留出均值不同；当前未准确重现突发波动或逐点波形。

| 面积损失 | 损失面积mm² | H径向速度RMS mm/s | T动态径向RMS µm | T最大动态半径µm |
|---|---|---|---|---|
{rows}

指标为去时间平均中心后的二维径向RMS/最大半径，不用振动RMS或故障等级替代位移指标；不宣称等同真实同步回转误差。

零故障极限的线性响应相对差：{checks['zero_fault_limit_relative_linear_response']:.3g}；内部作用反作用残差：{checks['internal_reaction_balance_N']:.3g} N。
近零面积损失的非线性动态状态相对差：{checks['zero_fault_limit_relative_nonlinear_dynamic_state']:.3g}。
三组在本次短序列内状态有限：{checks['finite_all_cases']}；加载平衡点瞬时线性化极点稳定：{checks['stable_loaded_linearizations']}。后者不是整个时变非线性系统的稳定性证明。
最重故障积分最大步长减半后H速度相对L2差：{checks['step_convergence']['H_velocity_relative_l2']:.3g}；T动态位移相对L2差：{checks['step_convergence']['T_dynamic_relative_l2']:.3g}。见JSON具体设置；不是任意模型/步长的保证。
预期特征是接口刚度降低后耦合模态与传递响应改变，不期待轴承BPFO；结果不强求单调。本轮只有健康振动统计的时间留出一致性检查，故障工况未获实验验证，T尚未有同工况同步验证。

![健康留出比较](health_comparison.png)
![同模型配对响应](paired_cases.png)
![耦合模态变化](mode_shift.png)
![电容参照与模型量的区别](holder_reference_limits.png)

## 运行与后续

```powershell
& D:/Softwares/miniconda3/envs/ZMSoftware/python.exe -B -m scripts.run_prototype --duration 0.5 --max-seconds 240 --output experiments/new_prototype
```

从仓库根目录运行上式，每次更换输出目录名称。依赖NumPy、SciPy、Matplotlib、h5py，均复用现有环境。三个CSV逐行存时间、转速、故障面积、H与T；NPZ保留全部原始状态。CLI调整时长、积分步长和墙钟限时；超时会停止，不能把未完部分标为完成。

待解决：实际轴承型号/预紧和几何；测力锤击FRF；7000rpm与振动同步的刀柄双向位移；局部壳体/转角/陀螺与接口磨损空间分布。电容数据已核，但日期/转速不同，仅作参照，不能拼成实测配对数据。
'''
    if (Path(out)/'cross_run_validation_20261012/report.md').exists():
        text+='\n## 跨采集补充检查\n\n'
        text+='原图只比较同次采集的留出片段。[冻结模型跨组报告](cross_run_validation_20261012/report.md)已补充另三次采集。'
        text+='同日另一组X方向RMS偏差约95%，未保持原约2%的吻合程度；此结论对应补充报告所冻结的模型，不因再次运行本脚本自动更新。\n'
    (Path(out)/'report.md').write_text(text,encoding='utf-8')
