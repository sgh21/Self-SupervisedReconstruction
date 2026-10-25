"""Read-only V5 force/moment, fixed-physics and reusable V4-data audit."""
import argparse
import ast
import copy
import json
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from spindle import dynamics_model as model
from utils.clearance_v4_validation import validate_config


def config_differences(current, reference, prefix=''):
    result = {}
    for key in current.keys() | reference.keys():
        name = prefix+key
        left, right = current.get(key), reference.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(config_differences(left, right, name+'.'))
        elif left != right:
            result[name] = dict(current=left, reference=right)
    return result


def audit(config_path, output, previous_root):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    p = config['parameters']
    omega = p['rpm'] * 2*np.pi/60
    phase = config['excitation']['phase_rad']
    times = np.array([0., .00123, .125, 1.2345, 2.5, 3.505214763715502])
    levels = [config['excitation']['peak_force_N'], 2., 5., 10., 20., 30.]
    rows = []
    for force in levels:
        u = force/omega**2
        angle = omega*times+phase
        actual = u*omega**2*np.column_stack((np.cos(angle), np.sin(angle)))
        generalized = model.generalized_point_force(actual, p['z_u'], 12)
        expected = np.zeros_like(generalized)
        expected[:, [0, 1, 8, 9]] = np.column_stack((actual, -p['z_u']*actual[:, 1], p['z_u']*actual[:, 0]))
        norm_error = float(np.max(abs(np.linalg.norm(actual, axis=1)/force-1)))
        moment_error = float(np.max(abs(generalized-expected))/force)
        local = copy.deepcopy(config)
        local['excitation'].update(U_kg_m=u, peak_force_N=force)
        validate_config(local)
        rows.append(dict(force_N=force, U_kg_m=u, U_g_mm=u*1e6,
                         moment_peak_N_m=abs(p['z_u'])*force,
                         vector_modulus_relative_error=norm_error,
                         generalized_force_relative_error=moment_error,
                         sample_time_s=times.tolist(), force_xy_N=actual.tolist(),
                         generalized_force=generalized.tolist(),
                         passed=norm_error<=1e-10 and moment_error<=1e-10))
    mapping = []
    for gap in [0., 1., 2., 5., 10.]:
        local = model.front_group_parameters(p, gap)
        differences = [key for key in p if p[key] != local[key]]
        mapping.append(dict(common_gap_um=gap, fit1_um=local['fit_clearance_um'],
                            fit2_um=local['fit2_clearance_um'], changed_parameter_keys=differences,
                            passed=local['fit_clearance_um']==gap and local['fit2_clearance_um']==gap
                            and set(differences)<= {'front_group_clearance_um', 'fit_clearance_um', 'fit2_clearance_um'}))
    wrong = copy.deepcopy(config)
    wrong['excitation']['peak_force_N'] *= 2
    try:
        validate_config(wrong)
        rejects_mismatch = False
    except ValueError:
        rejects_mismatch = True
    m, g = model.mass_and_gravity(p)
    previous = []
    for group in ['pilots', 'validation/n256', 'validation/n512']:
        for gap in [0, 10]:
            folder = Path(previous_root)/group/f'gap_{gap}um'
            cfg = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
            metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            arrays = np.load(folder/'native.npz', allow_pickle=False)
            indices = np.linspace(0, len(arrays['t_raw'])-1, 17, dtype=int)
            q, v, acc = (arrays[key][indices] for key in ['q_raw_m', 'v_raw_m_s', 'a_raw_m_s2'])
            t = arrays['t_raw'][indices]
            masses, gravity = model.mass_and_gravity(cfg['parameters'])
            recovered = np.array([masses*ai + model.internal_force(ti, qi, vi, cfg['parameters'], 0)-gravity
                                  for ti, qi, vi, ai in zip(t, q, v, acc)])
            ex = cfg['excitation']
            f = ex['U_kg_m']*omega**2*np.column_stack((np.cos(omega*t+ex['phase_rad']), np.sin(omega*t+ex['phase_rad'])))
            load = model.generalized_point_force(f, cfg['parameters']['z_u'], 12)
            expected_p = model.front_group_parameters(p, gap)
            expected_p['N'] = cfg['parameters']['N']
            differences = {key: dict(current=expected_p.get(key), saved=cfg['parameters'].get(key))
                           for key in expected_p.keys() | cfg['parameters'].keys()
                           if expected_p.get(key) != cfg['parameters'].get(key)}
            # V4's explicit false flag has no physical effect.
            differences.pop('diagnostic_single_front', None)
            # The backend and N affect evaluation, not mechanical parameters.
            differences.pop('rhs_backend', None)
            row = dict(source=str(folder), force_N=ex['peak_force_N'], U_kg_m=ex['U_kg_m'],
                       gap_um=gap, N=cfg['parameters']['N'], E_dir_um=metrics['E_dir_um'],
                       source_time_integrations='previous V4, not new V5 integrations',
                       parameter_differences=differences, native_keys=arrays.files,
                       native_force_reconstruction_max_abs_error=float(np.max(abs(recovered-load))),
                       native_force_reconstruction_relative_error=float(np.max(abs(recovered-load))/ex['peak_force_N']),
                       full_native_available=True, saved_stationarity=metrics['steady_pass'],
                       saved_domain=metrics['domain_pass'], saved_sampling=metrics['sampling_pass'])
            previous.append(row)
            arrays.close()
    task_ref = Path('GPTOrder/spindle_v5_excitation_plan/reference_inputs/clearance_v4_config.json')
    ref = json.loads(task_ref.read_text(encoding='utf-8'))
    code_zip = Path('deliverables/spindle_v4_code_reports_20261023.zip')
    same_core = None
    unchanged_functions = {}
    if code_zip.exists():
        with zipfile.ZipFile(code_zip) as packed:
            old_source = packed.read('src/spindle/dynamics_model.py')
            new_source = Path('src/spindle/dynamics_model.py').read_bytes()
            same_core = old_source == new_source
            old_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(old_source).body if isinstance(n, ast.FunctionDef)}
            new_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(new_source).body if isinstance(n, ast.FunctionDef)}
            unchanged_functions = {key: value == new_functions.get(key)
                                   for key, value in old_functions.items() if key != 'simulate'}
    all_differences = config_differences(config, ref)
    physical_differences = config_differences(p, ref['parameters'])
    for numerical in ('N', 'rhs_backend', 'diagnostic_single_front'):
        physical_differences.pop(numerical, None)
    backend_file = Path(output).parent/'validation/backend_equivalence.json'
    backend_evidence = json.loads(backend_file.read_text(encoding='utf-8')) if backend_file.exists() else None
    dependencies = {name: version(name) for name in ('numba', 'llvmlite')}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='input_audit_completed; not a V5 dynamic acceptance',
                  omega_rad_s=omega, rotation_frequency_Hz=abs(omega)/(2*np.pi),
                  period_s=2*np.pi/abs(omega), phase_rad=phase,
                  force_checks=rows, common_gap_mapping=mapping,
                  force_metadata_mismatch_rejected=rejects_mismatch,
                  baseline_matches_task_reference=config==ref,
                  physical_parameters_match_v4_except_numerical_options=not physical_differences,
                  physical_parameter_differences=physical_differences,
                  full_configuration_differences=all_differences,
                  numerical_parameter_keys_excluded_from_physical_identity=['N', 'rhs_backend', 'diagnostic_single_front'],
                  core_bytes_match_delivered_v4=same_core,
                  mechanical_functions_ast_equal_to_delivered_v4=unchanged_functions,
                  backend_evidence_path=str(backend_file), backend_equivalence=backend_evidence,
                  optional_backend_dependencies=dependencies,
                  mass_diagonal_kg_or_kg_m2=m.tolist(), gravity_generalized_N_or_N_m=g.tolist(),
                  total_physical_mass_kg=p['ms']+p['mt']+p['mh']+p['mO']+p['mO2'],
                  frozen_structural_parameters=p,
                  reused_v4_evidence=previous,
                  provenance_policy='No hash/SHA256: full actual configuration, explicit differences, source snapshots and byte comparison per AGENTS.md',
                  caveat='V4 native-force substitution is an algebraic consistency check; it is not a new independent time integration or real-machine validation.')
    (output/'input_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    force_table = '\n'.join(f"|{r['force_N']:.12g}|{r['U_kg_m']:.15g}|{r['U_g_mm']:.12g}|{r['moment_peak_N_m']:.12g}|" for r in rows)
    reuse_table = '\n'.join(f"|{r['N']}|{r['gap_um']}|{r['E_dir_um']:.12g}|{r['native_force_reconstruction_max_abs_error']:.3g}|{r['source']}|" for r in previous)
    text = f'''# V5 控制变量与实际输入核查

本文件是公式、配置与旧原生数据的核查，不是新 V5 动态结果。全部参数仍为 assumed_demo。

## 实际施力

Ω={omega:.15g} rad/s，转频={abs(omega)/(2*np.pi):.12g} Hz，周期={2*np.pi/abs(omega):.12g} s。负号保留旋向。
实际求解读取 U：`simulate(..., unbalance=(U, phase))`，内部构造 `U*Ω²*[cos(Ωt+phase),sin(Ωt+phase)]`。
F 是力向量恒定模长和单分量峰值。新 F 替换原输入，不叠加旧力。
`generalized_point_force` 只写入索引 `[0,1,8,9]`，值为 `[Fx,Fy,-z_u*Fy,z_u*Fx]`；z_u=0.1 m。
六个绝对时刻的模长及广义力误差均低于1e−10相对门限，具体时序在 input_checks.json。

|F / N|U / kg·m|U / g·mm|力矩峰值 / N·m|
|---:|---:|---:|---:|
{force_table}

现有 V4 `validate_config` 已拒绝只改 F 元数据而未改 U 的输入。V5 薄入口须同时派生并保存 F、U，继续调用原求解函数。

## 冻结结构及共同间隙

12自由度；质量 S/T/H/O1/O2 为2.5/0.604/41.27074/0.42113/0.42113 kg，总质量45.217 kg；Jt=0.02006870333454608、Jp=0.0026374066690921603 kg·m²。
重力[0,−9.80665] m/s²，转速−7000 rpm、相位−0.3661687497264663 rad、z_u=0.1 m、M=0.165 m、固定探头参考均不变。
前支承 z_F1=0.12、z_F2=0.09 m；后R1 z=−0.12 m，原固定K/C及线性原点不变。原Kf/Df、Kb2/Cb2、预紧重叠、全部矩阵和适用域见 input_checks.json 完整参数及 baseline_config.json。
共同参数经 `front_group_parameters` 同时写入 c1/c2。0/1/2/5/10 μm检查只改变三个共同间隙字段；未移动后支承、未改变质量/惯量/刚度/阻尼/重力或探头。
当前基线与任务包参考配置全文相同：{config==ref}；排除N/rhs_backend等数值选项后，机械参数全文相同：{not physical_differences}。
全部配置差异及排除字段在 input_checks.json 保留。V5提高先导N至256、设置可选rhs_backend=numba，并允许最大热身延长至10 s；这些数值/运行设置不改变物理参数。

## 可选计算后端审查

机械主文件增加了Numba可选挂接，因此全文件与V4字节相同：{same_core}。不能再宣称源码完全未改。
默认 `simulate` 仍读取缺省值 `rhs_backend=python` 并调用原Python力学函数；V5实际配置明确选择numba。
除simulate挂接外，V4已有函数AST逐个比对结果均相同：{all(unchanged_functions.values())}，逐项结果在 input_checks.json。
`fast_rhs.py`只替换同一右端的数值求值，不改变DOP853积分器或容差；没有fastmath、接触插值或输出缩放。

逐式检查：线性联接为DᵀKD/DᵀCD；质量/重力直接调用原函数；陀螺G只加一次；F1逐球Hertz正部、阻尼投影和非负裁剪与原式相同。
两配合仍逐方向采用4K/N、4D/N，先检查投影接近量>0，再裁剪总法向力非负；外圈与壳体力成对装配。F2内部矩阵及R1健康线性支承不变。
有符号Ω继续控制旋转相位及保持架相位，力峰UΩ²；点力及力矩只加入[0,1,8,9]，没有重复载荷。
依赖：numba {dependencies['numba']}、llvmlite {dependencies['llvmlite']}；只选择此后端时导入。
已保存的 `validation/backend_equivalence.json` 记录900个随机状态（N128/256/512，c0/5.5/10）与3条短轨迹对照；最大逐坐标力差约1.10e−11 N或N·m，低于1e−9门限，短轨迹回归通过。
本次只审读并引用该已完成证据，没有重跑长模拟。后端等价不能替代新力值的稳态、适用域及h/N收敛验收。

## 旧结果复用与边界

下表均为已存在的V4积分，完整保护段和12维原生q/v/a均存在。由17个原生样点重装 `M*a+R−g`，与原解析点力/力矩核对。
原力回归可以复用相同N的这两端结果；无需为改薄入口重复长积分。N128半步结果不能用于宣称N256或N512的时间积分已通过。

|N|c / μm|E / μm|广义力最大误差（N或N·m）|来源|
|---:|---:|---:|---:|---|
{reuse_table}

这些代入误差只验证保存状态与当前方程一致。V4正间隙弱谐波仍有角向未收敛项，不能继承为新力值通过；V5代表点必须重新做同一最终N下h/2、N加密及输出采样检查。

## 顺序、域和范围口径

每个2/5/10/20 N均配同力c=0与10 μm。主试验未到20 μm，且20 N代表点有界且原域通过，才可补30 N同力配对；30 N不是自动一级。
健康E超过2 μm后，该力只能检验上端，不能证明完整范围。端点不足时检查适用的10/20 N及近目标力下c=5 μm，不能只据端点否定整个区间。
选共同力后仅改变共同c；完整范围需同一F下健康误差上端≤2 μm、中间误差区间落入[4,6]/[9,11]/[14,16] μm及E−ε≥20 μm。
F2内部及R1相对零隙静态基线的位移变化≤1 μm、倾角≤1 mrad、F1接近量0至2倍原重叠保持冻结。域外结果可探索记录，不计正式覆盖。
不得把静态下沉计入逐转PP，不把不同F拼成间隙曲线，不按健康值强制归零。

## 可追溯方式

任务包要求参数哈希，但本仓库AGENTS.md明确不使用sha256。本轮不计算哈希，以冻结配置全文、逐例actual_config.json、参数差异表、源代码快照和直接字节比对代替；不修改上游任务包清单。
'''
    (output/'control_audit.md').write_text(text, encoding='utf-8')
    (output/'baseline_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v5_config.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments/clearance_v5_excitation/reports'))
    parser.add_argument('--previous-root', type=Path, default=Path('experiments/clearance_v4_20261023'))
    args = parser.parse_args()
    result = audit(args.config, args.output, args.previous_root)
    print(json.dumps(dict(force_checks_passed=all(x['passed'] for x in result['force_checks']),
                          reusable_native_cases=len(result['reused_v4_evidence'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
