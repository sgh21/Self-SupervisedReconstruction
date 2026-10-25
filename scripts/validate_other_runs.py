"""Frozen-result validation on independently selected acquisitions. No fitting.

Reads saved healthy model output and compares ALL valid stable windows of each
new acquisition. Does not call calibrate_forcing/simulate or alter results.
Selection is by metadata/chronological order, never by numerical match quality.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import PROJECT_ROOT
from utils.data_io import load_velocity_record,load_health,bandlimit,save_json
from utils.metrics import comparison_metrics
from utils.plot_report import relative_link

PACKAGES=Path('D:/WorkSpace/ZMProject/SoftwareTools_PyQt/data/processed/spindle_sample_packages')
SELECTED=[
    ('same_day','20260723','run_20260723_095732_7000rpm_20min_0723_morning2.zip'),
    ('cross_day','20260724','run_20260724_101140_7000rpm_0724_morning1.zip'),
    ('tool_remount','20260726','run_20260726_092536_7000rpm_0726_morning1_reset.zip')]


def comparison_plot(path,velocity,model,fs,title):
    mfs=int(model['output_fs_hz']);n=len(model['t'])
    mn=int(round(n/mfs*fs));tm=model['t']-model['t'][0];td=np.arange(mn)/fs
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for ch in range(2):
        v=bandlimit(model['H_velocity_m_s'][:,ch],mfs)*1000
        axes[ch,0].plot(td,velocity[0,ch,:mn]*1000,label='Other acquisition: first segment',alpha=.65)
        axes[ch,0].plot(tm,v,label='Frozen healthy model',lw=.8)
        axes[ch,0].set(xlabel='Time (s)',ylabel=f'H {"XY"[ch]} (mm/s)',title='No keyphase; no pointwise alignment')
        fd,pd=welch(velocity[:,ch,:]*1000,fs,nperseg=mn,axis=-1)
        fm,pm=welch(v,mfs,nperseg=n)
        kd=(fd>=10)&(fd<=900);km=(fm>=10)&(fm<=900)
        axes[ch,1].semilogy(fd[kd],pd.mean(0)[kd],label='Other acquisition: all stable windows')
        axes[ch,1].semilogy(fm[km],pm[km],label='Frozen healthy model')
        axes[ch,1].set(xlim=(10,900),xlabel='Frequency (Hz)',ylabel='PSD ((mm/s)^2/Hz)')
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
    fig.suptitle(title,fontsize=11);fig.savefig(path);plt.close(fig)


def write_cross_report(out,result,base):
    old=result['original_split'];frozen=result['frozen_model'];records=result['records']
    names={'same_run_holdout':'原采集留出','same_day':'7/23 morning2：同日另次',
           'cross_day':'7/24 morning1：跨日','tool_remount':'7/26 morning1_reset：重装刀具'}
    lines=[]
    for r in records:
        m=r['metrics'];v=m['measured_rms_mm_s'];e=m['relative_rms_error'];p=m['log_psd_rmse_db']
        lines.append(f'| {names[r["kind"]]} | {r["windows"]} | {v[0]:.5f} / {v[1]:.5f} | '
                     f'{e[0]:+.2%} / {e[1]:+.2%} | {p[0]:.2f} / {p[1]:.2f} |')
    details=[]
    for r in records[1:]:
        i=r['data'];run=i['run']
        details.append(f'- **{names[r["kind"]]}**：`{run["run_id"]}`，采于{run["captured_at"]}；'
                       f'实际转速中位数{i["actual_speed_median_rpm"]:.3f} rpm，取全部{r["windows"]}个'
                       f'有效处理后稳态窗（{i["stable_start_s"]:.0f}–{i["stable_end_s"]:.0f}s），'
                       f'条件`{run["condition"]}`。温度范围NTC1/RTD1：{i["temperature_ranges_C"]} °C。')
    plots='\n'.join(f'![{names[r["kind"]]}]({r["plot"]})' for r in records[1:])
    same=records[1]['metrics']['relative_rms_error']
    cross=records[2]['metrics']['relative_rms_error']
    report=f'''# 冻结原模型后的跨采集健康数据比较（2026-10-12）

**当前不能认为其他组也保持原来的吻合程度。** 同日另次X/Y RMS误差为{same[0]:+.2%} / {same[1]:+.2%}；
7/24跨日为{cross[0]:+.2%} / {cross[1]:+.2%}。原来的约2%误差只针对同次采集留出部分。
本次没有重新校准来消除这些差异。

## 原来的图用了哪组

原校准来源为 `{old['run_id']}`（2026-07-23 09:30:18，7000 rpm空转）。
前28个10秒窗670–950s校准等效激励；950–970s不用；后29窗970–1260s只比较。
所以原“留出”图未重复使用校准窗口，但仍是同一次采集，不是独立另一组。
原图的时域示例是第一留出窗970s开始的前0.5秒；原频谱为全部29窗的平均功率谱，RMS为全部29窗的样本均方根。

## 本次怎样补其他组

读取[原experiment.json](<{relative_link(out,base/'experiment.json')}>)和[原健康仿真状态](<{relative_link(out,base/'area_loss_0.00.npz')}>)，冻结机械参数、转速、等效驱动力幅值/相位和原仿真结果。
没有重新估计激励、调参、缩放输出或重新跑一套拟合。三份外部记录在看RMS之前按采集顺序/元数据选定，未挑选最吻合的记录。
每份均使用全部有效的已处理稳态窗；没有称为全部原始20分钟。7000 rpm空转标签、传感器配置与2048 Hz/mm/s/10–900 Hz预处理与校准记录相符。

{chr(10).join(details)}

7/23和7/24的重装/密封气压字段未知；不能由空值证明安装条件完全一致。7/26明确重装刀具，单列工况变化检查。
磁吸传感器配置相同不证明真实安装位置未改变；相近温度读数也不证明热史一致。

## 结果

冻结模型X/Y速度RMS为 **{frozen['rms_mm_s'][0]:.5f} / {frozen['rms_mm_s'][1]:.5f} mm/s**。
误差定义为 `(模型RMS/实测RMS−1)`，正值表示模型偏大；所有比较同为10–900 Hz。
功率谱分别按8192/2048 Hz采样率估计，以相同约0.5秒窗得到约2 Hz频率分辨率，按频率匹配。
对数频谱误差是带内功率谱dB差的均方根，不能当作结构参数误差。
该指标对频率格点等权，大幅主峰偏差可能被其他频段稀释；需要同时看RMS和谱图，不能只凭数dB平均差称吻合。

| 比较数据 | 10秒窗数 | 实测RMS X/Y mm/s | 冻结模型RMS误差 X/Y | 对数功率谱误差 X/Y dB |
|---|---:|---|---|---|
{chr(10).join(lines)}

![各组RMS与冻结模型](rms_across_runs.png)
{plots}

时域图固定使用每份数据的第一个有效窗前0.5秒，不按吻合程度挑片段。每份图的功率谱使用整组所有有效稳态窗。
没有键相，时间零点各自归零仅为显示；没有对齐机械相位，也没有逐点波形预测成功的声明。
首个短片段RMS和整组RMS的区别记录在[完整结果JSON](validation.json)。

## 可以与不可以得出的结论

本次检查的是：**一组正常数据标定的等效激励与冻结响应，在另次/跨日正常采集中的统计可重复性**。
原模型的质量、刚度、轴承/预紧参数仍为暂定；此前实际拟合的是条件性等效输入，不是唯一识别整机物理参数。
不同采集的未知激励或安装状态可能变化；差异不能直接归为故障，也不能只靠吻合证明结构模型正确。
本次没有新的同步刀柄位移，仍不能验证H–T关系或故障预测。未进行外部数据再校准。

运行（从仓库根目录，输出使用新的试验名称）：

```powershell
& D:/Softwares/miniconda3/envs/ZMSoftware/python.exe -B -m scripts.validate_other_runs --results "{base.resolve().as_posix()}" --output experiments/new_cross_run
```

结果不改写原实验；另存本目录。检查耗时{result['elapsed_seconds']:.2f}s。
'''
    (out/'report.md').write_text(report,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=PROJECT_ROOT/'experiments/legacy_prototype')
    ap.add_argument('--output',type=Path,required=True,help='New comparison directory')
    args=ap.parse_args();start=time.perf_counter();base=args.results
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':130})
    experiment_bytes=(base/'experiment.json').read_bytes()
    model_bytes=(base/'area_loss_0.00.npz').read_bytes()
    exp=json.loads(experiment_bytes)
    with np.load(base/'area_loss_0.00.npz') as z:model={k:z[k] for k in z.files}
    _,holdout,original_data=load_health(Path(exp['data']['package']))
    original_metrics=comparison_metrics(model['H_velocity_m_s'],holdout,
                                         int(model['output_fs_hz']),original_data['fs_hz'])
    for key in ['model_rms_mm_s','measured_rms_mm_s','relative_rms_error','log_psd_rmse_db']:
        assert np.allclose(original_metrics[key],exp['health_holdout'][key],rtol=1e-12,atol=1e-12),key
    records=[dict(kind='same_run_holdout',windows=len(holdout),metrics=original_metrics)]
    for kind,day,name in SELECTED:
        package=PACKAGES/day/name
        assert str(package)!=str(Path(exp['data']['package'])),'Calibration acquisition must not be an external run'
        v,starts,info=load_velocity_record(package)
        assert info['run']['expected_source_health']=='normal' and info['run']['operation']=='idle'
        assert info['run']['speed_rpm']==7000 and info['preprocessing']==exp['data']['preprocessing']
        assert info['source_metadata']['configuration']['groups'][0]==exp['data']['source_metadata']['configuration']['groups'][0]
        metrics=comparison_metrics(model['H_velocity_m_s'],v,int(model['output_fs_hz']),info['fs_hz'])
        plot=kind+'.png';comparison_plot(out/plot,v,model,info['fs_hz'],f'{day} {kind} | fixed model, no recalibration')
        records.append(dict(kind=kind,windows=len(v),data=info,metrics=metrics,plot=plot))
        print(kind,metrics,flush=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    model_rms=original_metrics['model_rms_mm_s']
    for ch,ax in enumerate(axes):
        ax.bar(np.arange(4),[r['metrics']['measured_rms_mm_s'][ch] for r in records])
        ax.axhline(model_rms[ch],color='tab:orange',label='Frozen model')
        ax.set_xticks(np.arange(4),['7/23 holdout','7/23 other','7/24 other','7/26 remount'],rotation=15)
        ax.set(ylabel=f'H {"XY"[ch]} RMS (mm/s)');ax.legend()
    fig.savefig(out/'rms_across_runs.png');plt.close(fig)
    assert experiment_bytes==(base/'experiment.json').read_bytes()
    assert model_bytes==(base/'area_loss_0.00.npz').read_bytes()
    result=dict(original_split=dict(run_id=exp['data']['run']['run_id'],train_indices=exp['data']['train_window_indices'],
                                   holdout_indices=exp['data']['validation_window_indices']),
                frozen_model=dict(parameters=exp['parameters'],calibration=exp['calibration'],
                                  model_file=str(base/'area_loss_0.00.npz'),rms_mm_s=model_rms),
                records=records,source_files_unchanged_byte_for_byte=True,recalibration_performed=False,
                new_simulation_performed=False,model_or_force_rescaled=False,
                selection_rule='chronological metadata before inspecting signals; remount separated',
                elapsed_seconds=time.perf_counter()-start)
    save_json(out/'validation.json',result)
    flat=[]
    for record in records:
        m=record['metrics']
        for ch in range(2):flat.append(dict(group=record['kind'],axis='XY'[ch],windows=record['windows'],
                                           model_rms_mm_s=m['model_rms_mm_s'][ch],measured_rms_mm_s=m['measured_rms_mm_s'][ch],
                                           relative_rms_error=m['relative_rms_error'][ch],log_psd_rmse_db=m['log_psd_rmse_db'][ch]))
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write_cross_report(out,result,base)
    print('Supplementary report:',out/'report.md')


if __name__=='__main__':main()
