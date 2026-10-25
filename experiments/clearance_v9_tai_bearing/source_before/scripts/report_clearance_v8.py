"""Write the V8 Chinese report from saved experiment and validation evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from utils.clearance_v8_acceptance import DEFAULT_ROOT, build_acceptance


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _fmt(value, digits=6):
    if value is None or value == '':
        return '未执行/未知'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def _pair(x, y, digits=4):
    return f'{_fmt(x, digits)}/{_fmt(y, digits)}'


def _table(headers, rows):
    if not rows:
        rows = [['未执行'] + ['']*(len(headers)-1)]
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    lines.extend('| '+' | '.join(str(value).replace('|', '/') for value in row)+' |' for row in rows)
    return lines


def _figure(name, caption):
    return f'![{caption}](../figures/{name})'


def _figure_link(name):
    return f'[{name}](../figures/{name})'


def _case_map(data, group):
    cases = data.get('run_groups', {}).get(group, {})
    return cases if isinstance(cases, dict) else {}


def _bearing_norm(case, bearing, quantity):
    row = (case or {}).get('wrench', {}).get('bearings', {}).get(bearing, {})
    values = row.get(quantity, {}).get('rms')
    if values is None:
        return None
    return float((values[0]**2+values[1]**2)**.5)


def _bearing_pair(case, bearing, quantity, statistic='rms'):
    row = (case or {}).get('wrench', {}).get('bearings', {}).get(bearing, {})
    values = row.get(quantity, {}).get(statistic)
    return _pair(values[0], values[1]) if values is not None else '未执行/未知'


def _bearing_diag(case, bearing, key):
    return (case or {}).get('bearing_diagnostics', {}).get(bearing, {}).get(key)


def _metric(case, name):
    return (case or {}).get('metrics', {}).get(name)


def _a_relative(old, new, key):
    a, b = _metric(old, key), _metric(new, key)
    if a is None or b is None or abs(a) < 1e-15:
        return None
    return 100*(b-a)/abs(a)


def _report(root, result):
    root = Path(root)
    config = _read('config/clearance_v8_config.json', {})
    p = config.get('parameters', {})
    source = _read(root/'validation/source_checks.json', {})
    preflight = _read(root/'validation/static_layout_preflight.json', {})
    p1 = _read(root/'validation/production_mechanics.json', {})
    checks = result['checks']
    groups = result.get('run_groups', {})
    a = groups.get('A_same_U', {})
    aold, anew = a.get('old', {}), a.get('new', {})
    b = groups.get('B_recalibrated', {})
    selected = (result.get('layout_evidence') or {}).get('selected_max_sensitivity_layout')
    dynamic_layouts = groups.get('layout_sensitivity', [])
    preflight_layouts = preflight.get('layouts', {})
    calibration = groups.get('calibration', {})
    history = calibration.get('history') or []
    convergence = groups.get('convergence', {})
    waveform_audit = groups.get('B_waveform_certification', {})
    inventory = groups.get('dynamic_case_inventory', {})

    lines = ['# V8回传报告', '', '## 1. 一页结论', '']
    lines += [
        f'- 代码来源：V8源码快照 `source_after`，文件清单 `validation/source_code_manifest_crc32.json`；Python {result.get("runtime", {}).get("python", "见 environment.json")}；运行入口 `scripts.run_clearance_v8`，冷态认证入口 `scripts.certify_clearance_v8`，健康激励标定入口 `scripts.calibrate_clearance_v8`；报告汇总入口另为 `scripts.report_clearance_v8`。',
        f'- 状态为12自由度、24维积分状态；名义布局{result.get("nominal_layout")}，仅为几何中心假设。',
        f'- 旧U固定机制对照：U_old={_fmt(result.get("U_old_kg_m"))} kg·m；A同U验收为{checks["same_U_comparison"]["status"]}。',
        f'- 新健康标定：U_new={_fmt(result.get("U_new_kg_m"))} kg·m，7000 rpm等效峰力={_fmt(result.get("F_new_N_at_7000"))} N；状态{checks["new_health_calibration"]["status"]}。',
        f'- U_new与U_old相同={_fmt(calibration.get("U_new_matches_U_old"))}；差值={_fmt(calibration.get("U_new_minus_U_old_kg_m"))} kg·m。',
        f'- B组全波形认证最大平均E_dir={_fmt(result["range_coverage"].get("mean_Edir_max_um"))} μm；认证最大单转={_fmt(result["range_coverage"].get("maximum_single_turn_um"))} μm；平均20 μm达到={_fmt(result["range_coverage"].get("mean_20um_reached"))}。',
        f'- 数值验收：全窗{checks["whole_window_stationarity"]["status"]}，全波形(H与全部轴承特征){checks["full_waveform_certification"]["status"]}，h/2 {checks["time_step_convergence"]["status"]}，N1024 {checks["contact_discretization_convergence"]["status"]}，采样加密{checks["sampling_convergence"]["status"]}。',
        f'- 保存态完整方程力/矩审计：{checks["saved_state_force_moment_balance"]["status"]}（A-new 3点与B 5点分别核算）。',
        '- 三项核心限制：真实配对方向未确认；外圈固定倾转/轴向约束承载未实测；没有同工况H/轴心/支架实测配对。',
        '- 频带与位移口径：未另注的Hx/Hy速度RMS、1×和非1×特征均为10–900 Hz；10–2048 Hz和900–2048 Hz仅用于单独标出的辅助字段。轴心E由未滤波原生位移计算。',
        '- 不给总体验收盖一个总“通过”；各项按证据分别列状态。', '']

    lines += ['## 2. 来源与实施差异', '',
        '来源核查保存在 `validation/source_checks.json`，原文副本和页码索引位于 `references/`。本轮按计划将Tai/Altintas Part I的滚动体内外接触关系、倾角投影与Cao/Altintas反力装配用于生产函数；前后支承对的方向和跨距按DB/DF几何假设处理。V8仍是固定接触角、等效Hertz接触的简化模型，不能称作原文完整轴承动力学复现。',
        'V8配置由V6实际健康配置构造，非V8物理参数叶与V6配置一致；保存的V6模型快照与计划基线逐文件核对。机制A组在同U、同绝对测量窗下比较，c10使用`old_matched`补跑；原始旧c10另存为未匹配来源。CRC32仅作来源核对，未计算SHA-256。', '']

    lines += ['## 3. 参数与接口', '',
        '四只轴承的`bearing_wrench`返回恢复径向力(N)、局部恢复矩M(N·m)和状态诊断。转子实际载荷为其相反数；每只固定外圈的角约束反力矩是该轴承局部矩的相反数，轴向约束反力同样取轴向恢复力的相反数。H节点的`H_bearing_load_constraint`则是另一项整体账目：V8-16按`−(前配合力臂矩之和 + 后轴承力臂矩之和 + 全部轴承局部矩之和)`计算，作用参考点是H。它与逐只外圈转动约束矩属于不同受力体/参考点，不能相加。CSV保存的轴承力臂矩按轴承几何中心z计算；压力中心z_pc只用于B09广义力矩等价核验，不作为额外独立力臂矩输出。', '']
    layout = p.get('bearing_layout', [])
    lines += _table(['轴承','z / m','s','ρ / m','外圈平移节点','外圈角/轴向','方向假设'], [[
        row.get('name'), _fmt(row.get('z_m')), row.get('orientation_sign'), _fmt(row.get('rho_m')),
        row.get('outer_translation_source'), f'{row.get("outer_tilt_constraint")}/{row.get("axial_increment_constraint")}',
        row.get('arrangement_basis')]
        for row in layout])
    front = p
    rear = p.get('rear_bearing', {})
    lines += ['', '采用V6实际质量/惯量，外圈质量分到O1/O2；支承质量不重复计入。', '']
    lines += _table(['对象/参数','S','T','H','O1','O2','转子惯量'], [[
        '质量 kg', _fmt(p.get('ms')), _fmt(p.get('mt')), _fmt(p.get('mh')),
        _fmt(p.get('mO')), _fmt(p.get('mO2')), f'Jt={_fmt(p.get("Jt"))}; Jp={_fmt(p.get("Jp"))} kg·m²']])
    lines += ['', '']
    lines += _table(['参数','前组F1/F2','后组R1/R2','单位/说明'], [[
        '球数', _fmt(front.get('balls')), _fmt(rear.get('balls')), '个'], [
        'Db / Dm', _pair(front.get('ball_diameter'), front.get('pitch_diameter')),
        _pair(rear.get('ball_diameter'), rear.get('pitch_diameter')), 'm'], [
        'α / Keq', f'{_fmt(front.get("contact_angle"))} rad / {_fmt(front.get("Keq"))}',
        f'{_fmt(rear.get("contact_angle"))} rad / {_fmt(rear.get("Keq"))}', 'rad / N·m⁻³ᐟ²'], [
        'δ0 / 阻尼', f'{_fmt(front.get("normal_overlap"))} m / {_fmt(front.get("cb_total"))} N·s/m',
        f'{_fmt(rear.get("normal_overlap"))} m / {_fmt(rear.get("cb_total"))} N·s/m', '每组相同本构']])
    lines += ['', '']

    lines += ['## 4. 单测和力学', '', 'P1生产代码测试完整结果见`validation/production_mechanics.json`；旧失败轮次仅保留为过程证据，不作为最终状态。初始测试记录有两处说明性修正：B01实际是完整`radial_only_v6`开关对旧径向力的回归，名称改正以免误称为ρ=0试验；B02预紧相对误差分母改用`abs(reference)`，修正F1参考轴向量为负时的符号报告。两处是测试标签/误差表达修正，没有改生产方程。', '']
    p1_rows = []
    for item in p1.get('checks', []):
        details = item.get('details', {})
        name = item.get('name')
        if name == 'B01_radial_only_v6_regression':
            maxerr = details.get('radial_relative_error')
            label, tol = 'B01 radial_only_v6旧力回归', '≤1e-10；局部矩=0'
        elif name == 'B02_zero_state_preload':
            maxerr = max((row.get('relative_error', 0) for row in details.values()), default=None)
            label, tol = 'B02 零扰动预紧', '前/后轴向参考90/70 N；径向力矩近零'
        elif name == 'B03_pure_translation':
            maxerr = max((max(row.get('force_relative_error', 0), row.get('moment_relative_error', 0))
                          for row in details.values()), default=None)
            label, tol = 'B03 纯平移力/矩', '同独立接触律参考比较'
        elif name == 'B04_pure_tilt':
            maxerr = max((max(max(map(abs, row.get('force_N', [0,0]))),
                               max(map(abs, row.get('moment_Nm', [0,0]))))
                          for row in details.values()), default=None)
            label, tol = 'B04 纯倾角响应', '倾角引起的径向力和局部矩应非零'
        elif name == 'B05_relative_tilt_common_rotation':
            maxerr = details.get('relative_wrench_error')
            label, tol = 'B05 内外圈共转抵消', '≤1e-12'
        elif name == 'B06_orientation_sign_reversal':
            maxerr = max((row.get('reversal_relative_error', 0) for row in details.values()), default=None)
            label, tol = 'B06 配对方向反号', '交叉项反号；≤1e-6'
        elif name == 'B07_B08_potential_gradient_tangent':
            maxerr = max((max(row.get('potential_gradient_relative_error', 0), row.get('tangent_relative_error', 0), row.get('tangent_symmetry_relative_error', 0))
                          for row in details.values()), default=None)
            label, tol = 'B07 势能梯度 / B08 切线', '1e-5；对称1e-8'
        elif name == 'B09_pressure_center_equivalence':
            maxerr = details.get('maximum_scaled_generalized_wrench_error')
            label, tol = 'B09 压力中心等价', '≤1e-10'
        elif name == 'B10_DB_DF_pair_stiffness':
            maxerr = max((pair.get('relative_error', 0) for layout in details.values()
                          for pair in layout.values()), default=None)
            label, tol = 'B10 DB/DF线性标尺', '≤1e-5'
        elif name == 'B11_damping_dissipation_and_clipping':
            maxerr = details.get('minimum_dissipation_W')
            label, tol = 'B11 阻尼耗散/裁剪', '耗散≥−1e-12 W；接触力非负'
        elif name == 'B12_open_contact_and_fit_detachment':
            maxerr = max([*details.get('outer_constraint_reaction_relative_error', {}).values()]
                         + [row.get('potential_energy_difference_J', 0)
                            for row in details.get('fully_opened_contacts', {}).values()], default=None)
            label, tol = 'B12 开球/配合脱开', '开球力与能量为0；约束反力反向'
        elif name == 'B13_zero_angle_or_radius_diagnostics':
            maxerr = max((row.get('alpha_zero_force_error', 0) for row in details.values()), default=None)
            label, tol = 'B13 α=0诊断', '倾角贡献消失；仅诊断'
        elif name == 'B14_python_numba_mixed_states':
            maxerr = max(details.get('maximum_normalized_force_error', 0), details.get('maximum_normalized_moment_error', 0))
            label, tol = 'B14 Python/Numba混合状态', '归一化力/矩≤1e-10；N与N·m分开'
        elif name == 'B15_batch_single_and_broadcast':
            maxerr = dict(force_N=details.get('max_force_difference_N'),
                          moment_Nm=details.get('max_moment_difference_Nm'))
            label, tol = 'B15 批处理/单点广播', '36状态逐项差'
        elif name == 'system_virtual_work_assembly':
            maxerr = details.get('maximum_relative_power_difference')
            label, tol = '系统虚功装配', '≤1e-8'
        elif name == 'system_force_moment_reaction_assembly':
            maxerr = f'force={_fmt(details.get("force_balance_error_N"))} N; moment={_fmt(details.get("moment_balance_error_Nm"))} N·m'
            label, tol = '系统反力/约束符号', '逐单位验收，见证据JSON'
        elif name == 'system_gyro_static_equilibrium':
            maxerr = dict(force_N=details.get('equilibrium_max_force_residual_N'),
                          moment_Nm=details.get('equilibrium_max_moment_residual_Nm'),
                          scaled=details.get('equilibrium_scaled_residual'))
            label, tol = '陀螺一次装配/静平衡', '1e-6 N；1e-7 N·m；混合尺度另报'
        elif name == 'radial_only_v6_snapshot_regression':
            trajectories = details.get('short_trajectory_by_clearance', {})
            maxerr = dict(random_states=details.get('random_state_count'),
                normalized_force=details.get('maximum_normalized_force_error'),
                normalized_moment=details.get('maximum_normalized_moment_error'),
                short_trajectory_q_v_acc={gap: row.get('max_difference')
                    for gap, row in trajectories.items()})
            label, tol = 'V6源码快照力/矩与短轨迹回归', '64随机态；c2/c10各约3 ms；快照与现代码均Python'
        else:
            maxerr = None
            label = name.replace('_', ' ')
            tol = ''
        p1_rows.append([label, item.get('status'), _fmt(maxerr), tol, 'validation/production_mechanics.json'])
    lines += _table(['检查','状态','最大误差/诊断','门槛','证据'], p1_rows)
    system = next((row.get('details', {}) for row in p1.get('checks', [])
                   if row.get('name') == 'system_gyro_static_equilibrium'), {})
    lines += ['', f'系统静平衡最大平动力残差{_fmt(system.get("equilibrium_max_force_residual_N"))} N、力矩残差{_fmt(system.get("equilibrium_max_moment_residual_Nm"))} N·m；陀螺反对称误差{_fmt(system.get("gyro_skew_max"))}、陀螺功率{_fmt(system.get("gyro_power_W"))} W。',
        '保存态完整方程力/矩审计要求A-new三个工况和B五个正式工况全部通过；A证据为`validation/A_new/dynamic_balance.json`，B证据为`validation/B/dynamic_balance.json`。B总审计虽集中保存，但JSON内逐项标明原工况目录。B01–B15各项通过状态、样本数、单位尺度及细项误差见永久JSON；系统12/24维、虚功、内反力、外圈约束反力及静平衡均独立记录。', '']
    snapshot_regression = next((row.get('details', {}) for row in p1.get('checks', [])
        if row.get('name') == 'radial_only_v6_snapshot_regression'), {})
    short_trajectories = snapshot_regression.get('short_trajectory_by_clearance', {})
    short_summary = '; '.join(f'c={gap} μm: {row.get("samples")} samples, {1000*row.get("duration_s", 0):.3g} ms, max Δq/v/acc={row.get("max_difference")}'
        for gap, row in short_trajectories.items()) or '未执行'
    lines += [f'源码快照短轨迹独立列为P1回归：{short_summary}。这两条Python轨迹只作约3 ms代码回归，不是稳态响应工况，也不计入动态工况/结果目录总数。', '']

    lines += ['## 5. A组：固定旧U的新旧机制对照', '',
        '新旧A组使用同一U、相位、间隙、绝对测量窗和求解设置；c10的径向旧机制使用`old_matched`。A组保留为固定旧U机制对照；A-new c2健康工况被复用为新U候选标定起点，随后B组c2作为独立冷启动重复并完成h/N/采样确认。A组其它对照不替代B组最终同U间隙扫描的认证。当前径向开关结果另与保存的V6实际结果按V6既有指标门槛比较。本节未另注的Hx/Hy速度RMS及非1×残差为10–900 Hz；轴心E由未滤波原生位移计算。', '']
    a_rows = []
    for gap in (2.0, 5.0, 10.0):
        for label, mapping in (('radial_only_v6', aold), ('relative_tilt_v8', anew)):
            case = mapping.get(f'gap_{gap:g}um')
            m = (case or {}).get('metrics', {})
            a_rows.append([label, _fmt(gap), _fmt((case or {}).get('U_kg_m')),
                _fmt(m.get('E_x_um')), _fmt(m.get('E_y_um')), _fmt(m.get('E_dir_um')),
                f'{_fmt(m.get("E_x_min_um"))}–{_fmt(m.get("E_x_max_um"))} / {_fmt(m.get("E_y_min_um"))}–{_fmt(m.get("E_y_max_um"))}',
                _pair(m.get('mean_x_um'),m.get('mean_y_um')),
                _pair(m.get('OH_mean_x_um'),m.get('OH_mean_y_um')),
                _pair(m.get('OH2_mean_x_um'),m.get('OH2_mean_y_um')),
                _fmt((case or {}).get('theta_peak_rad')),
                _pair(m.get('H_v_rms_x_mm_s'), m.get('H_v_rms_y_mm_s')),
                _pair(m.get('H_v_residual_x_mm_s'), m.get('H_v_residual_y_mm_s')),
                _fmt(m.get('steady_pass')),
                _fmt((groups.get('A_measurement_windows', {}).get(f'gap_{gap:g}um') or {}).get('same_window'))])
    lines += _table(['机制','c μm','U kg·m','Ex μm','Ey μm','Edir μm','单转X/Y范围 μm','M均值 x/y μm','O1-H均值 x/y μm','O2-H均值 x/y μm','θ峰 rad','Hx/Hy RMS mm/s','非1×残差 mm/s','稳态','配对同窗'], a_rows)
    old_unmatched = groups.get('A_old_unmatched', {}).get('gap_10um', {})
    old_unmatched_metrics = old_unmatched.get('metrics', {})
    if old_unmatched:
        lines += ['', '旧c10原始未匹配运行（只作来源保留，不参与同窗百分比对照）：', '']
        lines += _table(['来源','E_dir μm','测量窗 s','U kg·m'], [[
            old_unmatched.get('path'), _fmt(old_unmatched_metrics.get('E_dir_um')),
            f'{_fmt(old_unmatched_metrics.get("measurement_start_s"))}–{_fmt(old_unmatched_metrics.get("measurement_end_s"))}',
            _fmt(old_unmatched.get('U_kg_m'))]])
    relative_rows = []
    for gap in (2.0, 5.0, 10.0):
        old, new = aold.get(f'gap_{gap:g}um'), anew.get(f'gap_{gap:g}um')
        matched_window = (groups.get('A_measurement_windows', {}).get(f'gap_{gap:g}um') or {}).get('same_window')
        if not matched_window:
            relative_rows.append([_fmt(gap), '未配对']+['未执行/未知']*5)
            continue
        relative_rows.append([_fmt(gap), _fmt(_a_relative(old,new,'E_dir_um')),
            _fmt(_a_relative(old,new,'H_v_rms_x_mm_s')),
            _fmt(_a_relative(old,new,'H_v_rms_y_mm_s')),
            _fmt(_a_relative(old,new,'H_v_residual_x_mm_s')),
            _fmt(_a_relative(old,new,'H_v_residual_y_mm_s')),
            _fmt(100*(new['theta_peak_rad']-old['theta_peak_rad'])/abs(old['theta_peak_rad'])
                 if old and new and old.get('theta_peak_rad') not in (None,0) and new.get('theta_peak_rad') is not None else None)])
    lines += ['', 'A-new相对A-old的变化：', '']
    lines += _table(['c μm','Edir %','Hx RMS %','Hy RMS %','Hx非1× %','Hy非1× %','θ峰 %'], relative_rows)
    lines += ['', _figure('A_same_U_mechanism.png', 'A组同U新旧机制对照'), '']
    lines += ['', '轴承球载、接触率、局部矩、力臂矩、逐只外圈角/轴向约束反力与倾角占比逐只列出；约束矩是局部恢复矩取负。H节点整体约束矩另列，二者不相加。局部矩也不与力臂矩相加后再称为单一接触矩。eta_tilt>0.5仅作约化模型风险提示。', '']
    bearing_rows = []
    for gap in (2.0, 5.0, 10.0):
        for label, mapping in (('old', aold), ('new', anew)):
            case = mapping.get(f'gap_{gap:g}um')
            if not case:
                continue
            for name in ('F1', 'F2', 'R1', 'R2'):
                wrench = (case.get('wrench') or {}).get('bearings', {}).get(name, {})
                diag = (case.get('bearing_diagnostics') or {}).get(name, {})
                bearing_rows.append([label, _fmt(gap), name,
                    _bearing_pair(case,name,'radial_N'), _fmt(_metric(case,f'bearing_{name}_ball_peak_N')),
                    _fmt(_metric(case,f'bearing_{name}_contact_fraction')),
                    _bearing_pair(case,name,'local_Nm'), _bearing_pair(case,name,'lever_Nm'),
                    _bearing_pair(case,name,'local_Nm'),
                    _fmt(wrench.get('axial_constraint_reaction_N')),
                    _fmt(diag.get('eta_tilt_max')), _fmt(diag.get('eta_tilt_above_half')),
                    _fmt(diag.get('delta_over_diameter_max')),
                    _fmt(diag.get('maximum_actual_tilt_compression_over_overlap')),
                    case.get('path')])
    lines += _table(['机制','c μm','轴承','Fr RMS x/y N','单球峰值载荷 N','球接触率','局部矩M RMS x/y N·m','力臂矩RMS x/y N·m','外圈角约束矩(−M) RMS x/y N·m','Fa约束反力 N','eta_tilt最大','>0.5提示','δ/d最大','倾角压缩/δ0最大','原始目录'], bearing_rows)
    a_balance_rows = []
    for gap in (2.0, 5.0, 10.0):
        case = anew.get(f'gap_{gap:g}um')
        balance = (case or {}).get('dynamic_balance') or {}
        if case:
            a_balance_rows.append([_fmt(gap), _fmt(balance.get('passed')),
                _fmt(balance.get('maximum_translational_residual_N')),
                _fmt(balance.get('maximum_moment_residual_Nm'))])
    lines += ['', 'A-new保存态力/力矩方程审计：', '']
    lines += _table(['c μm','通过','最大平动力残差 N','最大力矩残差 N·m'], a_balance_rows)
    lines += ['', '逐字段A组配置/测量窗差异、原始未匹配旧c10来源及V6存档指标比较见`acceptance_result.json`的`run_groups.A_config_differences`、`A_measurement_windows`、`A_old_unmatched`和`A_legacy_v6_regression`。逐工况、逐轴承的静态偏移/力矩/球载/接触率/eta_tilt表在[case_wrench_contact_tilt.csv](case_wrench_contact_tilt.csv)；CSV每行带原生波形、逐转CSV、实际配置、接触、诊断和约束反力文件的相对路径。V6回归仍按原指标容差判断，不要求逐位相同。', '']

    lines += ['## 6. 配对方向敏感性', '',
        f'预先按冻结相位静态转动柔度选择{selected or "未选择"}作为敏感替代布局；选择未使用动态E或振动结果。所有方向均为待证实的安装假设。', '']
    layout_rows = []
    for name, data in preflight_layouts.items():
        spans = data.get('bearing_group_pressure_center_spans', {})
        layout_rows.append([name, _fmt(data.get('relative_frobenius_change_vs_DB_DB')),
            _fmt(spans.get('front', {}).get('pressure_center_span_m')),
            _fmt(spans.get('rear', {}).get('pressure_center_span_m')),
            _fmt(data.get('equilibrium_scaled_residual'))])
    lines += _table(['布局','旋转柔度差','当前布局前压力中心跨距 m','当前布局后压力中心跨距 m','平衡缩放残差'], layout_rows)
    lines += ['', '固定DB/DF压力中心跨距标尺：前组0.046077/0.013923 m，后组0.0269955/0.00100446 m；上表列的是每个布局实际采用的跨距。', '']
    layout_dynamic_rows = []
    for case in dynamic_layouts:
        m = case.get('metrics', {})
        layout_dynamic_rows.append([case.get('layout'), _fmt(case.get('gap_um')),
            _fmt(m.get('E_dir_um')), _pair(m.get('H_v_rms_x_mm_s'), m.get('H_v_rms_y_mm_s')),
            _fmt(case.get('theta_peak_rad')), _fmt(case.get('steady_pass')),
            _fmt(case.get('domain_pass'))])
    lines += ['', *_table(['实际布局','c μm','Edir μm','Hx/Hy RMS mm/s','θ峰 rad','稳态','适用域'], layout_dynamic_rows),
        '', f'预设扩展门槛为E_dir或任一主H RMS变化>5%，或稳态/适用域结论改变。首个DF_DF对照超过门槛后，按策略追加DB_DF与DF_DB，最终完成三个替代布局的c2/c10共6点；布局验收状态：{checks["layout_sensitivity"]["status"]}。c10的替代布局E_dir为17.7949–18.0397 μm，名义DB_DB为12.0933 μm。方向均为假设；这六个布局敏感性点未做独立h/2和N1024认证。', '']
    lines += ['', '机制观察：在当前固定角约化中，每球局部矩与径向力来自同一法向载荷Q；压力中心等价位置改变力/矩耦合及支承负荷分配，因此不应解释为额外独立的Kθ。B组F2外圈配合通道接触率（`fit2_contact_fraction`）从c2的0.747降至c7.5/10的0；同时F2内部球接触率仍为1、局部矩仍非零。后支承R2局部Mx RMS从c2的0.1616升至c10的0.4754 N·m。该共同变化与配合开闭下的约束和载荷重分配相符，但不构成唯一因果归因。', '']
    lines += [_figure('gap_motion_contact.png', '间隙下的运动与接触变化'), '',
        _figure('gap_vibration_and_motion_scatter.png', '间隙下的振动与运动轨迹'), '']

    lines += ['## 7. B组：新健康标定与同U退化', '',
        '完整候选顺序保存在`calibration_history.json`及`calibration_selection.json`。只有可用稳态分支参与标定；候选U不是实测残余不平衡，2 μm健康点属于条件标定而非独立实机验证。', '']
    history_rows = []
    if isinstance(history, dict):
        history = history.get('history', [])
    for item in history:
        history_rows.append([_fmt(item.get('F_N')), _fmt(item.get('U_kg_m')),
            _fmt(item.get('E_um')), _fmt(item.get('steady')), _fmt(item.get('domain')),
            _fmt(item.get('sampling')), _fmt(item.get('usable')), item.get('reason','')])
    lines += _table(['力 N','U kg·m','健康E μm','稳态','适用域','采样','用于标定','选择理由'], history_rows)
    outcome = ('U_new等于U_old；既有A-new健康点已进入目标带，因此没有调整激励。'
        if calibration.get('U_new_matches_U_old') is True else
        'U_new与U_old不同；变化量和候选选择过程见下表及JSON。'
        if calibration.get('U_new_matches_U_old') is False else
        'U_new与U_old关系未知。')
    lines += ['', f'健康标定和独立确认：{checks["new_health_calibration"]["status"]}。{outcome}', '']
    b_rows = []
    for gap in (2.0, 3.0, 5.0, 7.5, 10.0):
        case = b.get(f'gap_{gap:g}um')
        m = (case or {}).get('metrics', {})
        b_rows.append([_fmt(gap), _fmt((case or {}).get('U_kg_m')),
            _fmt(m.get('E_dir_um')),
            _fmt(m.get('E_dir_um')-b.get('gap_2um',{}).get('metrics',{}).get('E_dir_um')) if case and b.get('gap_2um') else '未执行',
            _fmt(max(m.get('E_x_max_um',0),m.get('E_y_max_um',0)) if m else None),
            f'{_fmt(m.get("E_x_min_um"))}–{_fmt(m.get("E_x_max_um"))} / {_fmt(m.get("E_y_min_um"))}–{_fmt(m.get("E_y_max_um"))}',
            _pair(m.get('mean_x_um'),m.get('mean_y_um')),
            _pair(m.get('OH_mean_x_um'),m.get('OH_mean_y_um')),
            _pair(m.get('OH2_mean_x_um'),m.get('OH2_mean_y_um')),
            _fmt((case or {}).get('theta_peak_rad')),
            _pair(m.get('H_v_rms_x_mm_s'),m.get('H_v_rms_y_mm_s')),
            _pair(m.get('H_v_1x_x_mm_s'),m.get('H_v_1x_y_mm_s')),
            _pair(m.get('H_v_residual_x_mm_s'),m.get('H_v_residual_y_mm_s')),
            _fmt((case or {}).get('full_waveform_pass'))])
    lines += _table(['c μm','U_new','Edir μm','Δ健康 μm','最大单转 μm','单转X/Y范围 μm','M均值 x/y μm','O1-H均值 x/y μm','O2-H均值 x/y μm','θ峰 rad','Hx/Hy RMS','1× Hx/Hy','非1× Hx/Hy','完整波形'], b_rows)
    b_bearing_rows = []
    for gap in (2.0, 3.0, 5.0, 7.5, 10.0):
        case = b.get(f'gap_{gap:g}um')
        if not case:
            continue
        for name in ('F1','F2','R1','R2'):
            wrench = (case.get('wrench') or {}).get('bearings', {}).get(name, {})
            diag = (case.get('bearing_diagnostics') or {}).get(name, {})
            b_bearing_rows.append([_fmt(gap), name,
                _bearing_pair(case,name,'radial_N'), _fmt(_metric(case,f'bearing_{name}_ball_peak_N')),
                _fmt(_metric(case,f'bearing_{name}_contact_fraction')),
                _bearing_pair(case,name,'local_Nm'), _bearing_pair(case,name,'lever_Nm'),
                _bearing_pair(case,name,'local_Nm'),
                _fmt(wrench.get('axial_constraint_reaction_N')),
                _fmt(diag.get('eta_tilt_max')), _fmt(diag.get('eta_tilt_above_half')),
                _fmt(diag.get('delta_over_diameter_max')),
                _fmt(diag.get('maximum_actual_tilt_compression_over_overlap')),
                case.get('path')])
    lines += ['', 'B组每只轴承球载/接触率/局部矩/力臂矩/约束反力/eta_tilt：', '']
    lines += _table(['c μm','轴承','Fr RMS x/y N','单球峰值载荷 N','球接触率','局部矩M RMS x/y N·m','力臂矩RMS x/y N·m','外圈角约束矩(−M) RMS x/y N·m','Fa约束反力 N','eta_tilt最大','>0.5提示','δ/d最大','倾角压缩/δ0最大','原始目录'], b_bearing_rows)
    lines += ['', '健康与退化输出不能拼成跨U曲线。静态均值是另一个输出诊断，不代替逐转峰峰值。逐例[case_wrench_contact_tilt.csv](case_wrench_contact_tilt.csv)含球载/接触/倾角、局部与力臂矩、两级约束矩、静态偏移及完整原生路径。H整体约束矩单独用`H_aggregate_constraint`行保存，避免在四个轴承行重复计数。', '']
    lines += [_figure('health_calibration_history.png', '健康点候选标定历史'), '',
        _figure('B_local_moments.png', 'B组轴承局部接触矩'), '',
        _figure('B_local_and_constraint_moments.png', 'B组局部矩与约束矩的分别记账'), '']

    lines += ['## 8. 稳态与加密', '',
        f'全波形认证（包括H特征及每只轴承的径向力、局部矩、轴向与接触特征）：{checks["full_waveform_certification"]["status"]}。', '']
    group_counts = '; '.join(f'{name}={count}' for name, count in inventory.get('completed_dynamic_case_directories_by_group', {}).items()) or '无'
    rejected_warmups = inventory.get('rejected_warmup_attempts', [])
    rejected_text = '；'.join(f'{row.get("path")} #{row.get("attempt")} (warmup={_fmt(row.get("warmup_s"))} s)' for row in rejected_warmups) or '无'
    extended_warmups = inventory.get('extended_warmup_cases', [])
    extended_text = '；'.join(f'{row.get("case_path")} { _fmt(row.get("warmup_s"))}→{_fmt(row.get("extended_to_warmup_s"))} s' for row in extended_warmups) or '无'
    lines += [f'完整动态工况/结果目录{_fmt(inventory.get("completed_dynamic_case_result_directories"), 0)}个（每个目录同时有metrics、native和求解设置；分组：{group_counts}）；已启动但未形成完整结果目录{_fmt(inventory.get("started_but_incomplete_dynamic_case_directory_count"), 0)}个。该数按保存结果目录计，不等于求解轨迹次数：warmup延长会续用前段状态；P1短轨迹单列，不计入本数。',
        f'已保存warmup稳态检查{_fmt(inventory.get("warmup_stationarity_attempts"), 0)}次：通过{_fmt(inventory.get("warmup_stationarity_pass_count"), 0)}、未通过{_fmt(inventory.get("rejected_warmup_stationarity_count"), 0)}、状态未知{_fmt(inventory.get("warmup_stationarity_unknown_count"), 0)}。未通过尝试保留在`warmup_history.json`和编排日志中：{rejected_text}。每个拒绝项另列最终native及求解设置路径，完整目录/日志见`acceptance_result.json`的`run_groups.dynamic_case_inventory`。',
        *([f'未通过后继续延长的工况：{extended_text}。这些被拒warmup窗只保留stationarity/warmup统计和求解设置历史（最终native元数据`actual_solver_history`），不另存该拒绝窗的native波形；最终native按延长后获准的export窗及guard保存。延长按预设稳态规则执行，不是从正式输出窗删去高幅值段；正式工况和独立加密工况所用native波形完整。'] if extended_warmups else []), '']
    convergence_rows = []
    for gap in (2.0, 3.0, 5.0, 7.5, 10.0):
        name = f'gap_{gap:g}um'
        base = b.get(name)
        row = [str(gap), _fmt((base or {}).get('steady_pass')),
            _fmt((base or {}).get('domain_pass')),_fmt((base or {}).get('sampling_pass')),
            _fmt((base or {}).get('solver_pass')),_fmt((base or {}).get('full_waveform_pass'))]
        for label in ('h_half','N_double'):
            comp = convergence.get(name,{}).get(label,{})
            main = comp.get('metrics',{}).get('E_dir_um',{})
            bearing = [v for k,v in comp.get('metrics',{}).items() if k.endswith('_local_moment_rms_x_Nm') or k.endswith('_local_moment_rms_y_Nm')]
            row += [_fmt(comp.get('passed')),_fmt(main.get('difference')),
                    _fmt(max((v.get('difference',0) for v in bearing), default=None))]
        convergence_rows.append(row)
    lines += _table(['c μm','稳态','域','采样','求解设置','整波形','h/2','ΔE','最大局部矩差 Nm','N1024','ΔE','最大局部矩差 Nm'], convergence_rows)
    waveform_rows = []
    for gap in (2.0, 3.0, 5.0, 7.5, 10.0):
        audit = waveform_audit.get(f'gap_{gap:g}um', {})
        failed_h = audit.get('failed_H_features', {})
        failed_b = audit.get('failed_bearing_features', {})
        local_failures = {key: value for key, value in failed_b.items() if 'local_moment' in key}
        other_bearing_failures = {key: value for key, value in failed_b.items() if 'local_moment' not in key}
        waveform_rows.append([_fmt(gap), _fmt(audit.get('main_valid')),
            _fmt(audit.get('full_waveform_pass')), json.dumps(failed_h, ensure_ascii=False, sort_keys=True),
            json.dumps(local_failures, ensure_ascii=False, sort_keys=True),
            json.dumps(other_bearing_failures, ensure_ascii=False, sort_keys=True)])
    lines += ['', '逐点全波形门槛分解（失败字段只列未通过项；空对象表示相关字段均通过）：', '']
    lines += _table(['c μm','主量main_valid','full_waveform_pass','H特征失败','局部矩失败','其它轴承特征失败'], waveform_rows)
    lines += ['', 'V6汇总中的`main_valid`只覆盖逐转E、配合接触及主量门槛；这里的V8认证另外要求整窗稳态、所有H特征和所有轴承特征通过。只有`full_waveform_pass=true`的B工况计入认证范围。', '']
    lines += [_figure('per_turn_PP.png', '各工况逐转峰峰值'), '']
    nominal_case = b.get('gap_2um') or {}
    settings_path = root/nominal_case.get('path','')/'actual_solver_settings.json'
    solver_rows = _read(settings_path, [])
    if solver_rows:
        first = solver_rows[0]
        sim = _read(root/nominal_case.get('path','')/'actual_config.json',{}).get('simulation',{})
        lines += ['', f'实际设置示例（c2）：{first.get("method")}，rtol={_fmt(first.get("rtol"))}，最大步长={_fmt(first.get("max_step_s"))} s，N={first.get("fit_direction_count")}；`native.npz`基准采样为65536 Hz，同一稠密解的2×采样为{first.get("output_fs_hz")} Hz（这是加密采样值，不是native.npz基准采样）；保留{sim.get("retained_turns")}转。', '']
    lines += ['', '480转与分支续接：', '']
    long_cases = groups.get('long_window_cases', [])
    long_rows = [[_fmt(c.get('gap_um')),_fmt(c.get('retained_turns')),
        _fmt(c.get('metrics',{}).get('complete_turns')),_fmt(c.get('steady_pass')),
        c.get('path')] for c in long_cases]
    branch = groups.get('branch_evidence', {})
    lines += _table(['c μm','配置保留转数','完整转数','稳态','原始证据'], long_rows)
    branch_rows = []
    for record in branch.get('records', []):
        if record.get('direction') != 'ascending':
            continue
        direct = record.get('direct_flags', {})
        stats = record.get('statistics_comparison', {})
        lag = record.get('PP_lag_statistics', {})
        branch_rows.append([_fmt(record.get('gap_um')),
            record.get('status', 'not_run'), _fmt(direct.get('all_required_pass')),
            _fmt(stats.get('same_window')),
            _fmt(stats.get('passed_ignoring_window_only')),
            _fmt(lag.get('best_integer_turn_lag')),
            _fmt(lag.get('best_lag_rms_vector_difference_um')),
            _fmt(record.get('reference_numerical_error_budget', {}).get('branch_epsilon'))])
    lines += ['', '上行冷启动对照与逐级初值续接：', '']
    lines += _table(['c μm','审计状态','直接物理/种子标志','同绝对窗','特征差异门槛通过','PP整数圈滞后','最佳滞后PP向量RMS差 μm','续接误差预算'], branch_rows)
    lines += ['', f'分支审核状态：{branch.get("status", "未执行")}；“audited”只表示比较过保存的上行续接与冷启动分支，不证明响应唯一。证据：`validation/branches.json`。', '']

    lines += ['## 9. 合理性和未验证项', '',
        '- 倾角使用固定小角度近似；球惯性、离心与陀螺、接触角和运行轴向预紧平衡未求解。',
        '- 外圈转动和轴向增量固定，程序输出的约束反力只是该边界下的需求值，未证明端面夹紧有足够承载。',
        '- DB/DF安装方向未知；压力中心跨距对方向敏感，不能把名义DB_DB当成实机事实。',
        '- H为单一等效节点；模型未验证壳体传播、测点空间差异、探头支架柔度或真实H/T同步信号。',
        '- 没有滚道曲率、赫兹接触椭圆和真实运行预紧，接触压力、寿命和强度不验收。', '']
    reaction_rows = []
    for gap in (2.0, 3.0, 5.0, 7.5, 10.0):
        case = b.get(f'gap_{gap:g}um')
        if not case:
            continue
        balance = case.get('dynamic_balance') or {}
        wrench = case.get('wrench') or {}
        reaction_rows.append([_fmt(case.get('gap_um')),
            _fmt(balance.get('maximum_translational_residual_N')),
            _fmt(balance.get('maximum_moment_residual_Nm')),
            _pair(wrench.get('H_bearing_load_constraint_rms_Nm',[None,None])[0],
                  wrench.get('H_bearing_load_constraint_rms_Nm',[None,None])[1]),
            _pair(wrench.get('H_bearing_load_constraint_peak_Nm',[None,None])[0],
                  wrench.get('H_bearing_load_constraint_peak_Nm',[None,None])[1]),
            _fmt(wrench.get('rotor_axial_constraint_peak_N')),
            _fmt(case.get('theta_peak_rad'))])
    lines += _table(['B工况c μm','ODE平动力残差 N','ODE力矩残差 N·m','H整体轴承载荷约束矩RMS x/y N·m','H整体轴承载荷约束矩峰值 x/y N·m','轴向约束峰值 N','θ峰 rad'], reaction_rows)
    lines += ['', '上表H约束矩是H参考点V8-16的整体承载账目，仅由轴承传入载荷构成；不是任一轴承的外圈角约束矩(−M)，也不与四只轴承的逐只外圈约束矩相加。逐只约束矩和完整约束反力时间序列见前表与`constraint_reactions.npz`。', '']
    coverage = result['range_coverage']
    lines += ['', f'20 μm独立结论：全波形认证最大平均E_dir={_fmt(coverage.get("mean_Edir_max_um"))} μm，认证最大单转={_fmt(coverage.get("maximum_single_turn_um"))} μm，认证健康增量={_fmt(coverage.get("delta_from_health_um"))} μm；认证平均值达到20 μm={_fmt(coverage.get("mean_20um_reached"))}。主量候选最大E_dir={_fmt(coverage.get("main_valid_candidate_Edir_max_um"))} μm；稳态且适用域内候选最大E_dir={_fmt(coverage.get("maximum_steady_in_domain_candidate_Edir_um"))} μm。候选不计作全波形认证。最大单转不能替代平均E_dir，范围覆盖也不是机械方程正确性的门槛。',
        f'实机验证状态：{result["real_machine_validation"]["status"]}。剩余问题见`acceptance_result.json`。', '']

    generated_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines += ['## 10. 打包与复现', '',
        '从仓库根目录执行（不需要执行`pip install`；代码从`src`导入）：', '', '```powershell',
        '$env:PYTHONPATH = "$PWD\\src"',
        'python -m utils.clearance_v8_acceptance --root experiments/clearance_v8_tilt_contact --recompute',
        'python -m scripts.report_clearance_v8 --root experiments/clearance_v8_tilt_contact', '```', '',
        '验收JSON为`experiments/clearance_v8_tilt_contact/acceptance_result.json`；报告写到`experiments/clearance_v8_tilt_contact/reports/V8_回传报告.md`；逐工况诊断汇总为[case_wrench_contact_tilt.csv](case_wrench_contact_tilt.csv)，每行列出对应`native.npz`、`turns.csv`、`actual_config.json`、`bearing_contacts.npz`、`bearing_diagnostics.json`、`wrench_diagnostics.json`及`constraint_reactions.npz`路径。本报告、A/B原生运行、独立收敛、配置、生产单测、动态平衡与分支证据均保留在独立试验目录。汇总命令只重算已保存结果，不会重新积分。', '',
        '补充图表证据：'+ '、'.join(_figure_link(name) for name in (
            'same_U_axis_trajectories.png', 'same_U_housing_velocity.png',
            'same_U_housing_acceleration.png', 'bearing_load_and_outer_orbits.png')) + '。主要图已嵌入第5–8节；所有图由`experiments/clearance_v8_tilt_contact/figures/`提供。', '',
        f'本报告生成时间：{generated_utc} UTC。数值误差预算来自独立h/2、N1024和采样比较，不是实机概率置信区间。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    root = args.root
    acceptance_path = args.acceptance or root/'acceptance_result.json'
    if args.acceptance and not args.recompute:
        result = _read(args.acceptance)
        if result is None:
            raise FileNotFoundError(f'Acceptance JSON not found: {args.acceptance}')
    else:
        result = build_acceptance(root, acceptance_path,
            args.template or Path('GPTOrder/spindle_v8_tilt_contact_plan/templates/acceptance_result.template.json'),
            args.recompute)
    output = args.output or root/'reports'/'V8_回传报告.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_report(root, result), encoding='utf-8')
    print(json.dumps(dict(acceptance=str(acceptance_path), report=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
