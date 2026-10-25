"""Build the V6 Markdown report and filled acceptance template from saved evidence.

This command does not simulate, calibrate, or promote missing evidence to pass.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from utils.clearance_v5_analysis import _read, _write


ASSUMPTIONS = [
    '4 kg是用户转述厂方的内部旋转总成约数，不是4.000 kg称重；内圈等附属件边界未完全确认。',
    '刀柄0.604 kg单列，外圈各0.10 kg为假设分配；CAD整机44.613 kg不是实物称重。',
    'Jt/Jp按旧回转半径和质量比4/2.5缩放，不是CAD惯量；支承和测点坐标未由实机内部图纸确认。',
    '前6009/后6007厂家数据约束等效Keq和初压缩，缺少滚道曲率；实际显式使用Keq，不把兼容用ki=ko解释为厂家真实内外接触系数。',
    '两前同一非线性Hertz规律；后组两个6007作用点外圈约化固定到H，14 mm跨度和DB/DF配对未实证。',
    '正常前配合c=2 μm、Kf/Df、滚动阻尼、壳体安装及接口柔度为条件假设。',
    '健康2 μm测量方向与几何偏心/表面误差贡献未分离，暂全部解释为指定M截面相对固定探头的动力响应。',
    '未解球高速惯性、接触角变化和热预紧；单球离心量级约前17%、后11%名义预紧球载，不等于输出误差率。',
    '没有独立同工况H/轴心/支架、第二转速或FRF实测验证；健康拟合不是实测残余不平衡识别。',
]


def _csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fmt(value, precision=7):
    if value is None or value == '':
        return '未知/未测'
    if isinstance(value, bool):
        return '通过' if value else '未通过'
    try:
        return f'{float(value):.{precision}g}'
    except (TypeError, ValueError):
        return str(value)


def _all_status(values, missing='not_tested'):
    values = list(values)
    if not values or any(value is None for value in values):
        return missing
    return 'passed' if all(values) else 'failed'


def _table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + [
        '| '+' | '.join(str(cell).replace('|', '/') for cell in row)+' |' for row in rows]


def report(root, template=None):
    root = Path(root)
    evidence = _read(root/'recomputed/acceptance.json')
    if evidence is None:
        raise ValueError('Run the V6 evidence assembler before generating the report.')
    records = sorted(evidence['cases'], key=lambda r:r['metrics']['gap_um'])
    health = next((r for r in records if r['metrics']['gap_um'] == 2), None)
    cfg = _read(root/'parameters_v6_calibrated.json') or _read(root/'parameters_v6_candidate.json')
    if cfg is None and health:
        cfg = _read(root/'data'/health['case']/'actual_config.json')
    if cfg is None:
        cfg = _read(root/'parameters_v6_before_calibration.json')
    p, sim, drive = cfg['parameters'], cfg['simulation'], cfg['excitation']
    mechanic = _read(root/'validation/mechanics.json', {})
    mech = mechanic.get('checks', {})
    balance = _read(root/'validation/dynamic_balance.json', {})
    calibration = _read(root/'calibration_selection.json', {})
    cold = _read(root/'validation/health_confirmation.json', {})
    initial = _read(root/'validation/initial_condition.json', {})
    response = _read(root/'validation/response_classification.json', {})
    history = _csv(root/'calibration_history.csv')
    sensitivity = _csv(root/'sensitivity_summary.csv')
    comparisons = _read(root/'recomputed/convergence.json', {})
    differences = _csv(root/'recomputed/same_U_vibration_differences.csv')
    environment = _read(root/'environment.json', {})
    U = evidence.get('frozen_U_kg_m', drive.get('U_kg_m'))
    F = None if U is None else U*(p['rpm']*2*3.141592653589793/60)**2
    eh = evidence.get('health_E_um')
    eps = evidence.get('health_epsilon_um')
    valid = [r for r in records if r['main_valid']]
    maximum = max(valid, key=lambda r:r['metrics']['E_dir_um'], default=None)
    candidate = max(records, key=lambda r:r['metrics']['E_dir_um'], default=None)
    completed = [file for file in root.rglob('metrics.json') if (file.parent/'native.npz').exists()
                 and 'source_before' not in file.parts]
    counts = Counter(file.relative_to(root).parts[0] for file in completed)
    failed = list(root.rglob('failed.json'))
    template_path = Path(template) if template else Path('GPTOrder/spindle_v6_plan/templates/acceptance_result.template.json')
    result = _read(template_path)
    if result is None:
        raise FileNotFoundError(f'Acceptance template missing: {template_path}')
    result.update(execution_status='actual_saved_results', codesnapshot='source_before/; current source in review package',
                  completed_full_dynamic_runs=len(completed), completed_by_group=dict(counts),
                  failed_case_records=len(failed), short_mechanical_runs_included_in_full_count=False)

    def check(name, status, files, reason):
        result['checks'][name] = dict(status=status,
            evidence_files=[file for file in files if (root/file).exists()], reason=reason)

    parameter_keys = ('v6_explicit_12dof', 'no_legacy_linear_bypass',
        'rotor_mass_and_inertia_updated', 'fronts_identical_constitutive_law', 'rear_only_rotor_rank_four')
    check('parameter_update', _all_status(mech.get(k) for k in parameter_keys),
          ['validation/mechanics.json', 'parameters_v6_before_calibration.json'],
          '参数进入新12自由度方程；参数证据强弱另列，不以数值实现证明实机真值。')
    check('mass_budget', _all_status(mech.get(k) for k in ('mass_budget', 'outer_mass_bounds', 'mass_positive')),
          ['validation/mechanics.json'], 'S+H+O1+O2=44.613 kg；T=0.604 kg单列；同一XY质量仅计一次。')
    check('preload_tangent', _all_status(mech.get(k) for k in ('front_preload_and_tangent', 'rear_preload_and_tangent', 'fronts_identical_constitutive_law')),
          ['validation/mechanics.json'], '单只径向初始化标尺144/117 N/μm；实际工作切线由非线性接触给出。')
    force_keys = ('virtual_work_and_moment', 'gyro_skew_zero_power', 'gyro_once',
                  'translational_action_reaction', 'all_support_moments', 'static_gap_2', 'static_gap_10')
    check('force_moment_balance', _all_status([mech.get(k) for k in force_keys]+[balance.get('passed')]),
          ['validation/mechanics.json', 'validation/dynamic_balance.json'],
          '静平衡与动态保存状态分别核验；平均载荷按完整转窗时间积分，不用简单样点均值代替平衡证据。')
    check('python_numba_equivalence', _all_status(mech.get(k) for k in ('python_numba_forces', 'python_numba_short_trajectory', 'legacy_v5_force_regression')),
          ['validation/mechanics.json'], '包含接触切换状态、全部力/矩、健康短轨迹及旧V5回归。')
    check('full_window_stationarity', _all_status(r['metrics'].get('steady_pass') for r in records),
          ['recomputed/acceptance.json'], '按每点完整导出窗和连续四子窗判稳；未稳结果保留，不计稳态范围。')
    for name, label in (('time_step_convergence', 'h_half'), ('angular_integration_convergence', 'N_double')):
        groups = [(case, item[label]) for case, item in comparisons.items() if label in item]
        status = _all_status(row.get('passed') and row.get('independent_refinement_applied') for _,row in groups)
        check(name, status, ['recomputed/convergence.json', 'recomputed/convergence_summary.csv'],
              f'只认证已有独立比较的{len(groups)}点：'+', '.join(case for case,_ in groups)+'；其它点仍为candidate。')
    check('sampling_convergence', _all_status(r.get('sampling_identity_pass') and r['metrics'].get('sampling_pass') for r in records),
          ['recomputed/acceptance.json'], '同一稠密解65536→131072 Hz；逐转PP由原生位移求得，实际高率M数组单独保存。')
    check('initial_condition', initial.get('status', 'not_tested'),
          ['validation/initial_condition.json'], initial.get('reason', '冷启动与反向续接须有实际证据；未执行不推定唯一响应分支。'))
    check('fixed_U_control', _all_status([evidence.get('same_U_control_pass')]),
          ['recomputed/acceptance.json', 'parameters_v6_calibrated.json'], '逐字段比较共同间隙外的物理配置、U、相位、转速及处理；敏感性另列。')
    domain_pass = bool(records and all(r['domain'].get('passed') for r in records))
    check('constitutive_applicability', 'conditional' if domain_pass else ('failed' if records else 'not_tested'),
          ['recomputed/acceptance.json', 'validation/mechanics.json'],
          '倾角≤1 mrad和无拉力筛查；δ/d、δ/δ0、Qmax仅作诊断。未验证接触压力/运行预紧；没有沿用旧1 μm线性域或2δ0强度代理。')
    pair_certified = any(row.get('certified') == 'True' for row in differences)
    check('vibration_numerical_separability', ('passed' if evidence.get('vibration_numerically_resolvable') else
          ('failed' if pair_certified else 'not_tested')), ['recomputed/same_U_vibration_differences.csv'],
          '逐方向逐特征以绝对差>5(εh+εc)判定；部分特征通过不代表所有弱谱线或实机可检测性通过。')
    for name, key, note in (('range_near20', 'near20_pass', '18–22 μm且工程误差边界落入区间'),
                             ('range_strict20', 'strict20_pass', 'E−ε≥20 μm')):
        check(name, ('passed' if evidence.get(key) else 'failed') if valid else 'not_tested',
              ['recomputed/acceptance.json'], note+'；范围结论仅限已测试且已认证的工况，不宣称连续区间上界。')
    check('real_machine_validation', 'not_validated', [], '健康2 μm用于条件标定；没有独立同条件实机验证数据。')
    healthy_numeric = evidence.get('health_target_numerically_certified')
    calibration_checks = [healthy_numeric, cold.get('passed'), calibration.get('all_neighbors_valid')]
    calibration_status = _all_status(calibration_checks)
    if eh is not None and not 1.9 <= eh <= 2.1:
        calibration_status = 'failed'
    result['health'].update(c_h_um=2, U_star_kg_m=U, F_7000_N=F, E_um=eh, epsilon_E_um=eps,
        calibration_pass=True if calibration_status == 'passed' else False if calibration_status == 'failed' else None,
        independent_validation_pass=False, independent_cold_restart_pass=cold.get('passed'),
        neighbor_U_checks_pass=calibration.get('all_neighbors_valid'),
        equivalent_eccentricity_internal_4kg_um=None if U is None else U/4*1e6,
        equivalent_eccentricity_rotating_4p604kg_um=None if U is None else U/4.604*1e6,
        U_g_mm=None if U is None else U*1e6,
        note='等效偏心是U除以指定质量，不是动态轴心幅值；独立冷启动属于数值复核，不是独立实机验证。')
    result['range'].update(valid_gaps_um=evidence['valid_gaps_um'], min_E_um=evidence['minimum_valid_E_um'],
        max_E_um=evidence['maximum_valid_E_um'], gap_at_max_um=evidence['maximum_valid_gap_um'],
        increase_from_health_um=evidence['increase_from_health_um'],
        invalid_cases=[dict(case=r['case'], gap_um=r['metrics']['gap_um'],
            E_um=r['metrics']['E_dir_um'], steady=r['metrics'].get('steady_pass'),
            domain=r['domain'].get('passed'), main_status=r['main_metric_status'],
            contact_status=r['contact_status'], epsilon_E_um=r['epsilon']['E_dir_um']['epsilon'])
            for r in records if not r['main_valid']])
    primary = [r for r in records if r['metrics']['gap_um'] in (2, 3, 5, 7.5, 10)]
    numeric = 'passed' if len(primary) == 5 and all(r['full_waveform_pass'] for r in primary) else 'conditional' if valid else 'not_tested'
    result['numerical_acceptance_scope'] = dict(primary_gaps_um=[2,3,5,7.5,10],
        ideal_zero_gap='Diagnostic control; independent h/N certification not required or inferred')
    result['summary_states'] = dict(
        mechanical_implementation=_all_status([mechanic.get('passed'), balance.get('passed')]),
        parameter_evidence='conditional', healthy_calibration=calibration_status,
        numerical_reliability=numeric, gap_causality=result['checks']['fixed_U_control']['status'],
        rotation_range=result['checks']['range_strict20']['status'],
        vibration_separability=result['checks']['vibration_numerical_separability']['status'],
        real_machine_validation='not_validated')
    result['remaining_assumptions'] = ASSUMPTIONS
    result['files_omitted_from_review_package'] = [
        '全部完整native.npz、逐球bearing_contacts.npz、sampling_131072.npz、actual_input.npz和细化NPZ放入native ZIP；原文件未删除。',
        'review包仅保留选定代表点全窗M/处理后H和末5转全状态；不是全部原生证据的替代。']
    result['cold_restart_evidence'] = cold
    result['initial_condition_evidence'] = initial
    result['per_turn_response_evidence'] = dict(file='validation/response_classification.json',
        formal_metric_reproduced=response.get('all_formal_metrics_reproduced'),
        full_system_periodicity_claimed=False,
        note='单转最大PP和全窗包络不替代max(meanX,meanY)的正式回转指标。')
    result['failed_refinement_features'] = [dict(case=name, refinement=label, feature=key,
        difference=row['difference'], tolerance=row['tolerance'])
        for name, groups in comparisons.items() for label, item in groups.items()
        for key, row in item.get('metrics', {}).items() if not row['passed']]
    _write(root/'acceptance_result.json', result)

    states = '；'.join(f'{k}={v}' for k,v in result['summary_states'].items())
    lines = ['# V6实际回传报告', '', '## 1. 结论', '',
        '- 入口：`scripts.run_clearance_v6`；12自由度，内部旋转总成约4 kg，两前6009同一非线性接触、后6007两作用点健康约化。',
        f'- 健康c=2 μm：U*={fmt(U)} kg·m（{fmt(None if U is None else U*1e6)} g·mm），F@7000={fmt(F)} N；E={fmt(eh)} μm，εE={fmt(eps)} μm。标定状态：{calibration_status}。',
        f'- 认证有效间隙：{evidence["valid_gaps_um"]} μm；有效E范围{fmt(evidence["minimum_valid_E_um"])}–{fmt(evidence["maximum_valid_E_um"])} μm；最大点c={fmt(evidence["maximum_valid_gap_um"])} μm，较健康增加{fmt(evidence["increase_from_health_um"])} μm。',
        f'- 近20：{result["checks"]["range_near20"]["status"]}；严格20：{result["checks"]["range_strict20"]["status"]}。7000 rpm主线均值指标的最大计算值为{fmt(None if candidate is None else candidate["metrics"]["E_dir_um"])} μm，认证状态另列，不能替代最大有效值。',
        f'- 最大有效值距20 μm的名义差为{fmt(None if maximum is None else 20-maximum["metrics"]["E_dir_um"])} μm；该最大值εE={fmt(evidence.get("maximum_valid_epsilon_um"))} μm。',
        f'- 同U壳体振动至少一个指定特征数值可分辨：{result["checks"]["vibration_numerical_separability"]["status"]}；各方向1×与非1×分别列证据。',
        f'- 完成{len(completed)}次有完整原生数据的动态运行，另有{len(failed)}份失败记录；机械短验证不计入此数。',
        '- 参数仍含明确假设。健康拟合是条件标定，没有独立实机验证。', '',
        '八项状态：'+states+'。', '', '## 2. 参数更新与模型', '']
    parameter_rows = [
        ['ms/Jt/Jp', '2.5 / 0.0200687033 / 0.00263740667', f'{fmt(p["ms"])} kg / {fmt(p["Jt"])} / {fmt(p["Jp"])} kg·m²', 'A/D', 'mass_and_gravity、gyro_matrix', '质量约数；惯量按回转半径缩放'],
        ['mO1/mO2/mh', '0.42113 / 0.42113 / 41.27074 kg', f'{fmt(p["mO"])} / {fmt(p["mO2"])} / {fmt(p["mh"])} kg', 'D/A约束', '平动质量和重力', '外圈拆分不是厂家质量'],
        ['F1/F2', 'F1 Hertz；F2旧线性', '6009，18球、8.731 mm球径、60 mm节圆、15°；同Keq和非线性函数', 'A/B/C', 'bearing_force、front2_forces、fast_rhs', '固定接触角、等效Keq'],
        ['后6007', '单R1线性', '两只16球6007；z=-0.127/-0.113 m，外圈约化到H', 'B/C/D', 'rear_bearing_parameters、A(z)ᵀ装配', '跨度/方向/外圈配合未实证'],
        ['健康/观测', '零隙健康、旧输入', 'c=2 μm；z_m='+fmt(p['z_m'])+' m，相对固定探头；保留1×', 'A/D', 'rotor_section、complete_turn_runout', '2 μm方向和几何误差未分离'],
        ['配合/阻尼', '旧Kf及阻尼分配', f'Kf={fmt(p["Kf"])} N/m，Df={fmt(p["Df"])} N·s/m；主曲线冻结', 'D', 'fit_contact_summary', '座孔与安装参数未辨识'],
    ]
    lines += _table(['参数/模块','旧值/实现','实际新值/实现','来源','进入方程','限制'], parameter_rows)
    mass = mechanic.get('details', {}).get('mass_budget_kg', {})
    lines += ['', f'质量账目：S+H+O1+O2={fmt(mass.get("spindle_sum"))} kg；T={fmt(p["mt"])} kg单列，合计{fmt(mass.get("total_with_holder"))} kg。平台405 kg等资产未加入H；XY重复对角项不重复计质量。',
        f'后组单独的转子约束秩为{fmt(mechanic.get("details", {}).get("rear_only_rank"))}；前接触脱开时仍保留有限跨度的最小约束。冻结模态/FRF只作局部筛查，不证明非线性全局稳定。',
        '厂家轴向成对48/39 N/μm经15°经验比例得到单只径向144/117 N/μm。固定角等效模型隐含轴向成对约41.35/33.60 N/μm，约低14%；未调整接触角强行同时拟合，亦未求解轴向精度。',
        '详见 `reports/parameter_change_log.md`、`validation/mechanics.json` 和参数合同。来源A=本机口述/CAD，B=对应型号厂家参考，C=条件推算，D=假设。', '',
        '## 3. 健康标定', '',
        f'主指标E_dir=max(mean(逐转X峰峰),mean(逐转Y峰峰))，保留1×；rpm={fmt(p["rpm"])}、z_m={fmt(p["z_m"])} m、z_u={fmt(p["z_u"])} m、相位={fmt(drive["phase_rad"])} rad。静中心不加入PP。c_h=2 μm及结构/预紧/阻尼固定，仅U进入标定；实际F=UΩ²，限制≤30 N。',
        f'U*/4 kg对应等效偏心{fmt(None if U is None else U/4*1e6)} μm；U*/4.604 kg对应{fmt(None if U is None else U/4.604*1e6)} μm。两者均不是动态回转幅值，也不是实测质心偏心。', '',
    ]
    lines += _table(['类别','F/N','U/kg·m','E/μm','稳态','域筛查','采样','拒绝原因'], [[
        r.get('kind',''), fmt(r.get('F_N')), fmt(r.get('U_kg_m')), fmt(r.get('E_um')),
        r.get('steady',''), r.get('domain',''), r.get('sampling',''), r.get('rejection','')] for r in history])
    lines += ['', f'独立健康冷启动：{fmt(cold.get("passed"))}；U±5%邻域：{fmt(calibration.get("all_neighbors_valid"))}；数值认证健康目标：{fmt(healthy_numeric)}。εE由独立步长、周向、采样和子窗差组成，缺项保持null；不是概率置信区间。',
        '独立复跑证据见 `validation/health_confirmation.json`；反向续接见 `validation/initial_condition.json`。健康数值认证额外要求εE<0.01 μm，即小于目标半宽的十分之一；未检验全部初值，不据单次续接排除所有多稳态。', '',
        '## 4. 固定U间隙扫描', '',
        '数值可信状态针对要求的2、3、5、7.5、10 μm五个主工况。0 μm为额外理想对照，仅进行稳态、域筛查和同解采样检查，未独立h/N认证；其误差预算保持未知，不计入认证范围。', '']
    scan_rows = []
    for r in records:
        m=r['metrics']; er=r['epsilon']['E_dir_um']['epsilon']
        scan_rows.append([fmt(m['gap_um']), '/'.join(fmt(m[k]) for k in ('E_x_um','E_y_um','E_dir_um')),
            fmt(er), fmt(m['E_dir_um']-eh if eh is not None else None),
            '/'.join(fmt(m[k]) for k in ('mean_x_um','mean_y_um')),
            '/'.join(fmt(m[f'H_v_rms_{a}_mm_s']) for a in 'xy'),
            '/'.join(fmt(m[f'H_v_residual_{a}_mm_s']) for a in 'xy'),
            '/'.join(fmt(m[k]) for k in ('fit_contact_fraction','fit2_contact_fraction')),
            f'{m.get("steady_pass")}/{r["domain"].get("passed")}/'+('certified' if r['main_valid'] else 'candidate')])
    lines += _table(['c/μm','Ex/Ey/Edir μm','εE μm','ΔE μm','均值X/Y μm','Hx/Hy RMS mm/s','非1× X/Y mm/s','配合1/2接触率','稳态/域/认证'],scan_rows)
    response_rows = [r for r in response.get('cases', []) if r['group']=='data']
    if response_rows:
        lines += ['', '逐转峰峰值不一定逐转相同。下表的“近p转重复”只描述PP序列：比较完整窗内相隔p转的X/Y峰峰值，最大差≤0.005 μm作描述性标记。保持架仍持续旋转，不能据此断言完整状态严格p转周期。该诊断不改变既定验收门槛。', '']
        lines += _table(['c/μm','正式E_dir μm','最大单转方向PP μm','近重复转数','lag1最大差 μm','选定lag最大差 μm','全窗X/Y包络 μm'], [[
            fmt(r['gap_um']),fmt(r['E_dir_um']),fmt(r['maximum_single_turn_directional_PP_um']),
            fmt(r['approximate_PP_repeat_turns']),fmt(r['lag_errors'][0]['maximum_XY_um']),
            fmt(r['lag_errors'][r['approximate_PP_repeat_turns']-1]['maximum_XY_um'] if r['approximate_PP_repeat_turns'] else None),
            fmt(r['full_window_coordinate_envelope_X_um'])+'/'+fmt(r['full_window_coordinate_envelope_Y_um'])] for r in response_rows])
        lines += ['', '例如c=7.5 μm呈近2转交替，c=10 μm呈近3转重复。单转X峰峰值可超过20 μm，但正式E_dir仍为max(mean(X逐转PP),mean(Y逐转PP))；不能把单转最大、逐圈先取较大再平均或全窗包络换成正式均值指标宣称20 μm达标。',
            '全部主点及对应h/2、N1024的lag1–6、逐转分布保存在 `validation/response_classification.json`、`validation/per_turn_PP.csv`、`validation/per_turn_PP_distribution.csv`；图为 `figures/per_turn_PP.png`。', '']
    lines += ['', '0 μm仅是理想对照；2 μm为健康候选；3/5/7.5/10 μm为绝对单侧径向间隙。c=10对应直径间隙20 μm、相对健康增量8 μm。缺少独立细化的点保留candidate和未知ε，不能混入有效覆盖。',
        '控制变量逐字段差异及每点认证状态见 `recomputed/acceptance.json`；完整指标见 `recomputed/gap_summary.csv`。实际点不强求单调，也不把不同U、转速或敏感性情景拼成主曲线。', '',
        '## 5. 振动差异与机制', '',
        '主对比采用共同10–900 Hz FIR带通，辅助10–2048及900–2048 Hz从同一原生状态单独处理。H速度直接来自状态；加速度来自ODE。保留未带通状态、真实保护和物理幅度，截止频率为FIR半幅边界。1×采用转角谐波拟合，非1×残差去除均值和1×；原生位移PP不由带通速度积分反推。', '']
    feature_keys = ('H_v_rms_x_mm_s','H_v_rms_y_mm_s','H_v_rms_xy_mm_s','H_v_1x_x_mm_s','H_v_1x_y_mm_s','H_v_residual_x_mm_s','H_v_residual_y_mm_s')
    pair_rows = [r for r in differences if r.get('feature') in feature_keys]
    lines += _table(['c/μm','特征','健康','间隙','差值','变化%','ε合计','认证/数值可分辨'], [[
        r['gap_um'], r['feature'], fmt(r['healthy']), fmt(r['degraded']), fmt(r['difference']),
        fmt(r['percent_change']), fmt(r['epsilon_sum']),r['certified']+'/'+r['numerically_resolvable']] for r in pair_rows])
    lines += ['', '数值可分辨判据为|差|>5(εh+εc)，是保守工程设置；接近数值底的健康分母不报告巨大百分比。全部方向、频带能量、1×/2×/3×及加速度见完整CSV；并非所有特征都自动可辨。', '']
    load_rows = []
    for r in records:
        bearings=r['domain'].get('bearings', {})
        load_rows.append([fmt(r['metrics']['gap_um'])]+[
            fmt(bearings.get(k, {}).get('mean_reaction_N',[None,None])[1]) for k in ('F1','F2','R1','R2')]+[
            fmt(r['domain'].get('values',{}).get('maximum_tilt_rad'))])
    lines += _table(['c/μm','F1均值Y/N','F2均值Y/N','R1均值Y/N','R2均值Y/N','最大倾角/rad'],load_rows)
    if maximum and health:
        m, h = maximum['metrics'], health['metrics']
        lines += ['', f'本模型中的量化原因：健康输入已由E≈2 μm约束，最大认证点的运动放大倍数为{fmt(m["E_dir_um"]/h["E_dir_um"])}，达到20 μm则需约{fmt(20/h["E_dir_um"])}倍。当前距离20 μm为{fmt(20-m["E_dir_um"])} μm，远大于数值误差{fmt(maximum["epsilon"]["E_dir_um"]["epsilon"])} μm，不能用数值误差解释缺口。',
            f'第二前外圈配合接触率从{fmt(h["fit2_contact_fraction"])}降至{fmt(m["fit2_contact_fraction"])}，前支承路径间歇化；后两作用点仍通过相反方向载荷形成约束力矩，约束秩保持4。间隙不是可以无约束地全部转化为测量截面运动的位移输入。上述是本模型内的载荷/运动解释，实际幅值仍受后组跨度和预紧假设影响，见下一节敏感性结果。']
    lines += ['', '接触率与载荷转移、两外圈相对H轨迹共同解释响应变化；平均重力下沉只改变中心。精确时间加权反力/力矩与惯性余额见 `validation/dynamic_balance.json`，上表均值是诊断输出；两者用途分开。',
        '未达到20 μm时，量化证据是已冻结健康输入下的最大认证E、其误差、健康增量及接触/载荷结果；并未证明整个连续间隙区间的数学上界，也未通过额外加力、任意调软或缩放补齐差距。', '',
        '## 6. 验收证据', '']
    lines += _table(['项目','状态','证据','限制'], [[name,item['status'], ', '.join('`'+f+'`' for f in item['evidence_files']) or '无独立证据',item['reason']] for name,item in result['checks'].items()])
    lines += ['', '已执行的独立数值细化：', '']
    lines += _table(['工况','细化','主PP','接触','H特征','全部','未过特征（最多列4项）'], [[
        name,label,fmt(item.get('main_metric_pass')),fmt(item.get('contact_pass')),
        fmt(item.get('feature_pass')),fmt(item.get('passed')),
        ', '.join([k for k,row in item.get('metrics',{}).items() if not row['passed']][:4]) or '无']
        for name,groups in comparisons.items() for label,item in groups.items()])
    lines += ['', f'数值证据范围：{len(valid)}个主点通过主量与接触认证；{sum(r["full_waveform_pass"] for r in records)}个主点通过本轮全部指定波形特征。未细化点的误差未知，绝不填零。',
        '积分采用DOP853；实际rtol/分单位atol/max_step、N和采样率逐例落盘并核对。E门槛max(0.005 μm,1%)；速度RMS max(1e−4 mm/s,1%)；弱特征5%、峰力5%、冲量2%、接触率绝对0.02。采样比较与h/2、N512→1024分别进行。', '',
        '## 7. 敏感性与留出', '',
        '局部筛查见 `sensitivity/local_sensitivity.csv`；冻结t=0切线/模态只用于挑选参数情景，不代替时域结论。下表来自实际时域记录，保持U*，改变转速时F按Ω²变化；未经独立h/N细化的敏感性点不能升格为主曲线认证点。', '']
    baseline_by_gap={r['metrics']['gap_um']:r['metrics'] for r in records}
    scenario_changes=[]
    for row in sensitivity:
        if row.get('E_dir_um') and row.get('gap_um') and float(row['gap_um']) in baseline_by_gap:
            change=float(row['E_dir_um'])-baseline_by_gap[float(row['gap_um'])]['E_dir_um']
            row['delta_from_nominal_um']=change
            if (row.get('status') == 'completed' and row.get('steady') == 'True'
                    and row.get('domain_screen') == 'True' and not row.get('scenario','').startswith('rpm_')):
                scenario_changes.append(row)
    lines += _table(['情景','c/μm','rpm','U/kg·m','F/N','E/μm','相对名义ΔE μm','Hx/Hy RMS mm/s','完成/稳态/域/采样'], [[
        r.get('scenario',''),r.get('gap_um',''),r.get('rpm',''),fmt(r.get('U_kg_m')),fmt(r.get('force_N')),
        fmt(r.get('E_dir_um')),fmt(r.get('delta_from_nominal_um')),
        fmt(r.get('H_v_rms_x_mm_s'))+'/'+fmt(r.get('H_v_rms_y_mm_s')),
        '/'.join(r.get(k,'') for k in ('status','steady','domain_screen','sampling'))] for r in sensitivity])
    if scenario_changes:
        strongest=max(scenario_changes,key=lambda row:abs(row['delta_from_nominal_um']))
        lines += ['', f'已完成且稳态/域筛查通过的结构参数情景中，最大已计算偏移为{strongest["scenario"]}、c={strongest["gap_um"]} μm，较同间隙名义分支ΔE={fmt(strongest["delta_from_nominal_um"])} μm。此量与主线最大健康增量{fmt(evidence["increase_from_health_um"])} μm分开比较；敏感性未经h/N全认证时只作条件证据。']
    if not sensitivity:
        lines += ['', '尚无实际时域敏感性汇总；不以局部筛查宣称敏感性验证通过。']
    lines += ['', '敏感性边界是研究假设，不是实测置信区间；后组弹性×0.5不能当成厂家修正刚度。预紧变化保持Keq固定，δ0按Pa^(2/3)更新。转速留出没有对应实测时仅属计算检查。', '',
        '## 8. 复现与打包', '',
        f'环境：{environment.get("platform", "未记录")}；Python {environment.get("python", "未记录")}；NumPy {environment.get("numpy", "未记录")}、SciPy {environment.get("scipy", "未记录")}、Numba {environment.get("numba", "未记录")}。',
        f'完整动态运行数：{len(completed)}；分组：'+', '.join(f'{k}={v}' for k,v in sorted(counts.items()))+f'。失败记录{len(failed)}。判断标准为同目录同时存在metrics.json和native.npz；短后端轨迹和停转衰减另列在机械JSON，不重复计数。',
        '实际命令见 `运行命令.md`；复算、绘图与生成本报告：', '',
        '```powershell',
        'python -m utils.clearance_v6_analysis --root experiments/clearance_v6_GMN --replay',
        'python -m utils.clearance_v6_force_audit --root experiments/clearance_v6_GMN',
        'python -m utils.clearance_v6_response --root experiments/clearance_v6_GMN',
        'python -m scripts.plot_clearance_v6 --root experiments/clearance_v6_GMN',
        'python -m scripts.report_clearance_v6 --root experiments/clearance_v6_GMN',
        'python -m scripts.package_clearance_v6 --root experiments/clearance_v6_GMN',
        '```', '',
        '审阅ZIP集中代码、全部配置/标定历史/报告/图表/JSON/CSV及代表数据；完整原生与数值细化NPZ另包，保持原相对路径。两包合并后把命令中的试验根目录改为`.`复算。主包单独可以用 `scripts.review_v5_representatives` 复算代表全窗PP与已处理H；入口名沿用旧版本，实现读取的是本包V6配置。',
        '主包未附全部整窗12状态、逐球接触、双倍采样和独立细化NPZ；它们完整保留在native ZIP，不以精简代表数据代替全部验收证据。包大小和清单以交付时manifest与package_verification为准。', '',
        '主要剩余问题：', '',
        '1. 结构和预紧工作点证据不足：见参数来源及敏感性变化。应补支承内部坐标、配对方向、热态配合、转子惯量和运行预紧。',
        '2. 健康2 μm观测不能唯一确定真实不平衡：应补同速同工况H/轴心/支架同步测量，以及至少一组留出转速或FRF。',
        '3. 范围只覆盖实际计算且数值有效的点；未稳、未细化及可能的多分支按表保留。若要求更细覆盖，先补对应分支数值证据，不改冻结U追幅值。', '',
        '全部明确假设：', '']
    lines += ['- '+text for text in ASSUMPTIONS]
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/clearance_v6_GMN'))
    parser.add_argument('--template', type=Path)
    args=parser.parse_args()
    result=report(args.root,args.template)
    print(json.dumps(result['summary_states'], ensure_ascii=False, indent=2))
