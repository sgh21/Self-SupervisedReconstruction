"""Build V3 delivery tables/figures/report from data and caller-owned decisions."""
import argparse
import csv
import json
from os.path import relpath
from pathlib import Path

import numpy as np

from utils.clearance_v3_plots import plot_v3


def finalize(folder, formal_paths: list[Path], convergence: dict, acceptance: dict):
    """Publish supplied acceptance unchanged; never infer a pass from a plot."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [Path(p)/'native.npz' if Path(p).is_dir() else Path(p) for p in formal_paths]
    rows = [json.loads(p.with_name('metrics.json').read_text(encoding='utf-8')) for p in paths]
    ordered = sorted(zip(rows, paths), key=lambda pair: pair[0]['gap_um'])
    rows, paths = map(list, zip(*ordered))
    baseline = rows[0]['E_dir_um']
    for row, path in zip(rows, paths):
        row['delta_E_from_reference_um'] = row['E_dir_um']-baseline
        row['case_id'] = path.parent.name
    validation = folder/'validation'
    validation.mkdir(exist_ok=True)
    for filename, value in [('acceptance_v3.json', acceptance), ('convergence_v3.json', convergence)]:
        (validation/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_v3(folder/'figures', paths, rows)

    def link(path):
        return Path(relpath(path, folder)).as_posix()

    def fmt(value):
        if value is None:
            return '未提供'
        return f'{value:.8g}' if isinstance(value, (float, int)) else str(value)

    checks = acceptance.get('checks', {})
    representatives = {item['case_id']: item for item in acceptance.get('representative_cases', [])
                       if isinstance(item, dict) and 'case_id' in item}
    case_details = []
    for row, path in zip(rows, paths):
        item = representatives.get(row['case_id'], {})
        row['numerical_validation'] = item.get('numerical_validation', 'not_validated')
        row['error_estimate_um'] = item.get('error_estimate_um')
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            p = meta['config']['parameters']
            q0 = data['q_equilibrium']
            static = (q0[2:4].copy() if p['measurement_kind'] == 'holder' else
                      q0[:2] + p['z_m']*np.array([q0[9], -q0[8]]))
            if p['probe_reference'] == 'housing':
                static -= q0[4:6]
            keep = ((data['t_raw'] >= meta['export_start_s']) &
                    (data['t_raw'] < meta['export_end_s']))
            case_details.append(dict(static_xy_um=static*1e6,
                fit_mean_y_N=float(data['fit_force_N'][keep, 1].mean()),
                healthy_mean_y_N=data['healthy_support_forces_N'][keep, :, 1].mean(axis=0)))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (folder/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    check_labels = {
        'parameter_contract': '支承/观测定义', 'mechanics_implementation': '力学实现',
        'stationarity': '整个导出窗稳态', 'numerical_validation': '数值可信度',
        'time_integration_convergence': '积分精度', 'contact_direction_convergence': '周向离散',
        'native_sampling_convergence': '原生输出漏峰',
        'housing_acceleration_convergence': '壳体加速度', 'contact_force_convergence': '接触力',
        'single_parameter_scan': '唯一间隙扫描', 'dual_output_effect': '双输出可分辨',
        'model_twenty_um_reached': '演示模型达到20 µm', 'twenty_um_reached': '有本机依据地达到20 µm',
        'bin_coverage_0_20': '四分箱覆盖', 'physical_parameter_support': '本机参数支持',
        'real_machine_validation': '实机定量验证'}
    check_table = '\n'.join(f'| {label} | {checks.get(key, "not_validated")} |'
                            for key, label in check_labels.items())
    case_table = '\n'.join(
        f'| {r["case_id"]} | {r["gap_um"]:g} | {r["E_x_um"]:.8g} | {r["E_y_um"]:.8g} | '
        f'{r["E_dir_um"]:.8g} | {r["delta_E_from_reference_um"]:.8g} | '
        f'{r["H_v_rms_x_mm_s"]:.8g} / {r["H_v_rms_y_mm_s"]:.8g} | '
        f'{r["fit_contact_fraction"]:.5%} | {r["fit_peak_N"]:.8g} | '
        f'{"pass" if r["steady_pass"] else "fail"} | {r["numerical_validation"]} | '
        f'{fmt(r["error_estimate_um"])} |' for r in rows)
    static_table = '\n'.join(
        f'| {r["case_id"]} | {d["static_xy_um"][0]:.8g} / {d["static_xy_um"][1]:.8g} | '
        f'{r["mean_x_um"]:.8g} / {r["mean_y_um"]:.8g} | {d["fit_mean_y_N"]:.8g} | '
        f'{" / ".join(f"{v:.8g}" for v in d["healthy_mean_y_N"])} |'
        for r, d in zip(rows, case_details))
    extent = acceptance.get('range', {})
    range_table = '\n'.join(f'| {key} | {fmt(extent.get(key))} |' for key in (
        'healthy_reference_um', 'min_absolute_um', 'max_absolute_um',
        'min_baseline_increment_um', 'max_baseline_increment_um',
        'nearest_20_um_case', 'distance_of_nearest_case_to_20_um', 'largest_internal_gap_um',
        'absolute_zero_endpoint', 'continuous_coverage'))
    bin_table = '\n'.join(f'| {r["interval"]} | {fmt(r.get("valid_case_count"))} | '
                          f'{fmt(r.get("resolved_level_count"))} | {fmt(r.get("case_ids"))} |'
                          for r in extent.get('bins', []))
    evidence = '\n'.join(f'- [{p.parent.name} 原生数据]({link(p)})；'
        f'[实际配置]({link(p.with_name("actual_config.json"))})；'
        f'[逐转表]({link(p.with_name("turns.csv"))})；'
        f'[全窗验稳]({link(p.with_name("stationarity.json"))})；'
        f'[实际求解设置]({link(p.with_name("actual_solver_settings.json"))})。' for p in paths)
    lo, hi = min(r['E_dir_um'] for r in rows), max(r['E_dir_um'] for r in rows)
    low_case = min(rows, key=lambda r: r['E_dir_um'])
    high_case = max(rows, key=lambda r: r['E_dir_um'])
    status_names = {'pass': '通过', 'fail': '未通过', 'not_validated': '未验证',
                    'boundary_uncertain': '边界不确定'}
    status = lambda key: status_names.get(checks.get(key), '未验证')
    h_ranges = '；'.join(f'H{axis.upper()}为{min(r[f"H_v_rms_{axis}_mm_s"] for r in rows):.8g}'
        f'–{max(r[f"H_v_rms_{axis}_mm_s"] for r in rows):.8g} mm/s' for axis in 'xy')
    differences = np.diff([r['E_dir_um'] for r in rows])
    trend = ('所采样点的E_dir随c不增' if np.all(differences <= 0) else
             '所采样点的E_dir随c不减' if np.all(differences >= 0) else
             '保存值存在升降转折，是否超出数值误差由可分辨性检查判断')
    open_cases = [r for r in rows if r['fit_contact_fraction'] == 0]
    switch_cases = [r for r in rows if 0 < r['fit_contact_fraction'] < 1]
    contact_text = (f'实际开闭接触点为c={", ".join(str(r["gap_um"]) for r in switch_cases)} µm。'
                    if switch_cases else '这些正式点未观测到整圈接触/分离交替。')
    if len(open_cases) == 1:
        contact_text += f'c={open_cases[0]["gap_um"]:g} µm完全脱开；只有一个脱开代表点，未通过重复平台区采样证明多个c等价。'
    elif open_cases:
        contact_text += (f'完全脱开点为c={", ".join(str(r["gap_um"]) for r in open_cases)} µm，'
                         f'其E_dir为{min(r["E_dir_um"] for r in open_cases):.8g}–'
                         f'{max(r["E_dir_um"] for r in open_cases):.8g} µm；去重等级见范围表。')
    else:
        contact_text += '本批正式点均未完全脱开。'
    gaps = ' '.join(f'{r["gap_um"]:g}' for r in rows)
    report = f'''# V3 配合间隙模型：实际数据与分项验收

本轮实现10自由度 `[Sx,Sy,Tx,Ty,Hx,Hy,Ox,Oy,theta_x,theta_y]`，固定目标前支承F1（z=0.12 m），其余健康前F2（0.09 m）和后R1（−0.12 m）保持 `assumed_closed_linear`。只改变F1外圈—壳体单侧径向间隙。主截面M为z=0.165 m，参考为固定台面/探头支架演示；没有把未知健康间隙都设为正间隙。

正式保存工况的主指标实际为 **{lo:.8g}–{hi:.8g} µm**，分别出现在c={low_case['gap_um']:g}/{high_case['gap_um']:g} µm；最低间隙理想参考为 **{baseline:.8g} µm**。壳体速度RMS：{h_ranges}。机械实现**{status('mechanics_implementation')}**，数值可信度**{status('numerical_validation')}**，双输出可分辨性**{status('dual_output_effect')}**，演示模型20 µm到达**{status('model_twenty_um_reached')}**；有本机参数依据的20 µm到达**{status('twenty_um_reached')}**。完整分项证据见[验收JSON](validation/acceptance_v3.json)。

{contact_text} {trend}。支承反力转移通过不同轴向位置的力臂改变倾摆，壳体共同平动与M相对运动叠加/抵消，因此不要求间隙越大主幅值越大。静偏移单列，未拿来补动态PP；未进行正反续接的数据不宣称已排除滞回。

三个主要限制：内部支承布局/惯量与健康配合仍是明确的演示假设；固定支架省略真实台面与探头支架动态；轴向预紧、高速球平衡、单只轴承本体倾覆刚度和轴/壳体局部模态未独立验证。因此代码与数值通过不能转换为实机定量预测通过。

下一轮最少需要：本机内部支承位置/型号与配合测量、分件转子质量和惯量；主截面和探头支架的实装坐标及同步运动；健康已知力柔度/多点频响和同工况H/轴心同步测量。完整待补项见参数合同。

## 实施与验收

| 项目 | 状态 |
|---|---|
{check_table}

机械、稳态、独立离散比较的证据位于 `validation/` 及各工况目录。`not_validated`不是通过，接触/加速度单独状态不由位移收敛替代。

## 实际工况

| case_id | c/µm | Ex/µm | Ey/µm | E_dir/µm | ΔE/µm | Hx/Hy速度RMS/(mm/s) | 接触率 | 配合峰力/N | 全窗稳态 | 数值状态 | E误差估计/µm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
{case_table}

`E_dir=max(mean Ex,mean Ey)`保留1×；ΔE相对表中最低间隙参考，负增量也保留。静平衡、均值、1×和去1×诊断、每转指标见原生元数据与CSV。相同解不自动算成不同严重度。

| case_id | 静平衡M的X/Y / µm | 正式窗M均值X/Y / µm | 配合传H的平均Y力/N | 健康F2/R1传H的平均Y力/N |
|---|---:|---:|---:|---:|
{static_table}

静平衡冻结t=0保持架相位；力表为完整正式窗的带符号平均力。误差估计与数值状态缺失时保留未验证，不由位移看起来相似代替加密证据。

## 范围声明

| 项目 | 实际范围记录 |
|---|---|
{range_table}

| 幅值箱/µm | 有效点数 | 可分辨去重幅值数 | case_id |
|---|---:|---:|---|
{bin_table}

范围身份：`{acceptance.get('range_claim_scope', 'not_validated')}`。缺少本机参数和允许故障范围证据时，演示数值结果不等于本机0–20 µm覆盖。少量离散工况亦不证明数学上的连续覆盖。

## 物理解释和可核查数据

目标配合传力随c改变，健康前后支承在各自位置分担固定载荷；空间力矩通过A/Aᵀ参与，H的共同平动与M相对运动可能发生叠加或抵消，不强求主指标单调。接触转变、平台或未观测到的状态按数据判断，不预贴轻/中/重标签。量化柔度缺口与预检假设见随包的可行性报告；静态阈值不代替动态接触率。

{evidence}

[汇总CSV](summary.csv)、[独立加密记录](validation/convergence_v3.json)、[图例与处理说明](figures/README.md)。完整状态同时包含m与rad，各量单位按元数据解释；NPZ原生数组含滤波保护区，主导出窗起止值明确保存。

![轨迹](figures/v3_trajectories.png)
![H时序](figures/v3_housing_waveforms.png)
![H频谱](figures/v3_housing_spectra.png)
![间隙比较](figures/v3_gap_comparison.png)
![接触与支承力](figures/v3_support_forces.png)

## 复算入口

从仓库或解包后的完整源码根目录运行，输出使用新目录。包内 `config/clearance_v3_config.json` 为正式冻结配置，各工况 `actual_config.json` 与实际求解设置另存供核对；命令只覆盖目标间隙：

```powershell
python -B -m scripts.run_clearance_v3 --config config/clearance_v3_config.json --output experiments/clearance_v3_replay --gaps {gaps} --workers 3
```

依赖版本、源码修改前后快照、全部加密命令与日志随交付包保存。验收由实际数据及明确的工程门槛判定，运行成功不自动意味着20 µm目标或实机验证完成。
'''
    (folder/'report.md').write_text(report, encoding='utf-8')
    return folder/'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--convergence', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.output, args.cases,
             json.loads(args.convergence.read_text(encoding='utf-8')),
             json.loads(args.acceptance.read_text(encoding='utf-8')))
