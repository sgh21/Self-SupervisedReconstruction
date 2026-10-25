"""Package V5 code/reports/representatives and a separate complete native ZIP.

No original files are deleted or downcast. JSON manifests use paths and byte
sizes, without hashes. Call only after the experiment/report writers finish.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from utils.pitting_analysis import resample_to


CODE_FILES = (
    'src/spindle/__init__.py', 'src/spindle/dynamics_model.py', 'src/spindle/fast_rhs.py',
    'scripts/__init__.py', 'scripts/run_clearance_v3.py', 'scripts/run_clearance_v4.py',
    'scripts/run_clearance_v5.py', 'scripts/plot_clearance_v5.py',
    'scripts/package_clearance_v5.py', 'scripts/review_v5_representatives.py',
    'utils/__init__.py', 'utils/clearance_v3_analysis.py', 'utils/clearance_v3_validation.py',
    'utils/clearance_v3_acceptance.py', 'utils/clearance_v4_analysis.py',
    'utils/clearance_v4_validation.py', 'utils/clearance_v5_analysis.py',
    'utils/clearance_v5_input_audit.py', 'utils/runout_labels.py',
    'utils/data_io.py', 'utils/pitting_analysis.py', 'utils/clearance_validation.py',
    'config/clearance_v3_config.json', 'config/clearance_v4_config.json',
    'config/clearance_v5_config.json', 'requirements.txt', 'pyproject.toml', 'AGENTS.md',
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_npz_lossless(path, arrays):
    """Standard NumPy-readable ZIP-LZMA; compression changes no array values."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_LZMA) as archive:
        for name, values in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(values), allow_pickle=False)
            archive.writestr(name+'.npy', buffer.getvalue())


def representatives(root):
    """Keep every native M sample, boundary interpolation, and final-five-turn states."""
    folder = root/'review_representatives'
    folder.mkdir(exist_ok=True)
    cases = []
    for gap in (0, 8.5, 10):
        case = f'data/F_30N/gap_{gap:g}um'
        source = root/case/'native.npz'
        config_file = root/case/'actual_config.json'
        metrics = json.loads((root/case/'metrics.json').read_text(encoding='utf-8'))
        with np.load(source) as d:
            meta = json.loads(str(d['metadata_json']))
            config = meta['config']; p, s = config['parameters'], config['simulation']
            start, end = meta['export_start_s'], meta['export_end_s']
            t = d['t_raw']; relative = d['r_m_raw_m']
            interior = (t>start)&(t<end)
            boundary = np.array([[np.interp(edge, t, relative[:,axis]) for axis in (0,1)]
                                 for edge in (start,end)])
            tm = np.r_[start, t[interior], end]
            motion = np.vstack((boundary[:1], relative[interior], boundary[1:]))
            tail_start = end-5*60/abs(p['rpm'])
            left = max(0, np.searchsorted(t, tail_start)-1)
            right = min(len(t), np.searchsorted(t, end)+1)
            tail = slice(left, right)
            raw_h = resample_to(d['v_raw_m_s'][:,4:6].T, s['fs_raw_hz'], s['fs_out_hz']).T*1000
            raw_h_t = t[0]+np.arange(len(raw_h))/s['fs_out_hz']
            arrays = dict(t_M_s=tm, M_relative_probe_m=motion,
                angle_M_rad=p['rpm']*2*np.pi/60*tm+config['excitation']['phase_rad'],
                t_H_s=d['t_out'], H_velocity_mm_s=d['H_velocity_mm_s'],
                H_acceleration_m_s2=d['H_acceleration_m_s2'],
                t_H_unbandpassed_s=raw_h_t, H_unbandpassed_guarded_velocity_mm_s=raw_h,
                t_tail_s=t[tail], q_tail=d['q_raw_m'][tail],
                v_tail=d['v_raw_m_s'][tail], a_tail=d['a_raw_m_s2'][tail])
            for key in ('fit_force_N', 'fit2_force_N', 'fit_active_count', 'fit2_active_count',
                        'bearing1_force_N', 'bearing2_force_N', 'rear_force_N'):
                arrays['tail_'+key] = d[key][tail]
            arrays['metadata_json'] = json.dumps(dict(config=config, source_native=case+'/native.npz',
                export_start_s=start, export_end_s=end, state_order=meta['state_order'],
                q_units=['m']*8+['rad']*2+['m']*2,
                v_units=['m/s']*8+['rad/s']*2+['m/s']*2,
                a_units=['m/s2']*8+['rad/s2']*2+['m/s2']*2,
                note='All original native M interior samples kept; two exact phase-window endpoints linearly interpolated like the label routine. Tail states are original native samples bracketing the final five turns. No normalization or downcast.'), ensure_ascii=False)
            target = folder/f'F_30N_gap_{gap:g}um.npz'
            save_npz_lossless(target, arrays)
            cases.append(dict(case=case, file=target.relative_to(root).as_posix(),
                actual_config=config_file.relative_to(root).as_posix(), source_native=case+'/native.npz',
                start_s=start, end_s=end, retained_turns=120, native_rate_hz=s['fs_raw_hz'],
                native_M_interior_samples=int(interior.sum()), interpolated_boundary_samples=2,
                native_M_samples_with_boundaries=len(tm), H_processed_samples=len(d['t_out']),
                H_unbandpassed_guarded_samples=len(raw_h_t), tail_native_samples=len(t[tail]),
                raw_guard_left_s=start-float(t[0]), raw_guard_right_s=float(t[-1])-end,
                bytes=target.stat().st_size, reference_metrics=metrics))
    index = dict(cases=cases, precision='float64; native time/coordinate samples are not decimated or scaled',
        compression='Standard NPZ with ZIP-LZMA members; NumPy reads directly. Lossless bytes, no numeric transformation.',
        scope='Three selected 30 N cases; M and processed H cover the whole 120-turn window. Full 12-DOF states cover the final five turns only in this ZIP.',
        complete_evidence='All full native/actual-input NPZ files are in spindle_v5_excitation_native.zip at unchanged relative paths.')
    write_json(folder/'index.json', index)
    (folder/'README.md').write_text('''# 代表数据的保留范围

保留30 N下c=0、8.5、10 μm三个代表点。M相对固定探头位移保存完整120转的全部原生65536 Hz样点，加与原指标程序一致的两个线性插值周期边界；没有抽样、缩放或转成float32。
代表NPZ采用标准ZIP-LZMA无损压缩，NumPy可直接读取；只改变存储压缩方式，数组数值不变。
处理后H速度及加速度保存原8192 Hz完整主窗；另存带真实保护段的未带通H速度8192 Hz，后者已作抗混叠降采样。实际解析力可由每例配置中的U、带符号rpm、相位及z_u准确重建；全窗实际力数组另在原生包actual_input.npz。
12维q/v/a、两配合力、两轴承力及后支承反力保存原生末5转及边界包围样点。角坐标只有索引8、9，单位分别为rad、rad/s、rad/s²；其余为m、m/s、m/s²。

本包可独立复算三个代表点的全120转方向PP及已处理H特征。完整12维全窗验稳、接触冲量及h/N独立复算需要第二个原生数据包；末5转状态不代表完整窗收敛证据。
''', encoding='utf-8')
    return index


def package(root, output):
    root = Path(root).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    representative_index = representatives(root)
    review_files = {name:repo/name for name in CODE_FILES}
    raw_files = {}
    omissions = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        name = relative.as_posix()
        if any(part in ('__pycache__', 'logs', '.cache') for part in relative.parts) or path.suffix in ('.log', '.pyc', '.nbc', '.nbi'):
            omissions.append(dict(path=name, reason='Runtime log/cache; not simulation data.'))
            continue
        if path.suffix == '.npz' and relative.parts[0] != 'review_representatives':
            raw_files[name] = path
            continue
        if relative.parts[0] == 'source_before':
            canonical = Path(*relative.parts[1:]).as_posix()
            if canonical in review_files and path.read_bytes() == review_files[canonical].read_bytes():
                omissions.append(dict(path=name, reason='Identical source bytes retained once.', identical_to=canonical))
                continue
        if name == 'README.md':
            name = 'reports/original_results_README.md'
        if name not in ('manifest.json', 'package_verification.json'):
            review_files[name] = path
    readme = '''# V5 激励试验验收包

代码、冻结配置、每例实际配置、报告、全部图表、验收JSON、汇总表和三个代表数据集中在本包。主报告为 `reports/V5_回传报告.md`；裁剪和来源见manifest.json及review_representatives/index.json。

实际完成30次动态运行（20个新组合、2个原力回归、6个细化、统计延长与初值检查各1）。最大有效E=19.515319603±0.000709402 μm；严格20 μm上端未通过，同力近0–20 μm完整范围未通过。固定30N健康E=3.975245923 μm；同力间隙10 μm的Hx速度RMS约+55.19%、Hy约−2.11%，均超过五倍合并数值误差。全部是assumed_demo，实机未验证。

安装依赖：`python -m pip install -e .`。Numba后端实测版本为numba 0.68.0、llvmlite 0.50.0；Python原力学后端仍保留。

主包单独复算三个代表点（无需ODE积分）：

```powershell
python -m scripts.review_v5_representatives --root .
```

完整原生包 `spindle_v5_excitation_native.zip` 与本包解压到同一目录，然后独立重做正式和细化组的指标、带通、分窗和收敛证据：

```powershell
python -m utils.clearance_v5_analysis --root . --selected-force-N 30 --replay
```

主包单独保留完整120转M与已处理H，12维状态只保留末5转；主包单独的代表复算不等于全窗接触及收敛重新验收。完整原生包保留所有NPZ及原相对路径，包括完整12维q/v/a、保护段、实际力、半步长和角向细化数据。
没有缩放、float32转换或删除原结果；不以不同力值拼接一条间隙曲线。manifest用路径和大小追溯，不使用哈希。
'''
    entries = []
    for part, files in (('review', review_files), ('native', raw_files)):
        entries.extend(dict(path=name, bytes=path.stat().st_size, part=part,
            purpose='Code/config/report/representative evidence' if part=='review' else 'Unmodified complete native NPZ')
            for name,path in sorted(files.items()))
    manifest = dict(schema='v5_split_review_package', entries=entries, omitted_from_review=omissions,
        raw_npz_moved_to_native=sorted(raw_files), representative_index=representative_index,
        note='Source and original output files remain untouched. No hashes. Extract both ZIPs into the same directory for complete replay.')
    primary = output/'spindle_v5_excitation_review.zip'
    native = output/'spindle_v5_excitation_native.zip'
    with zipfile.ZipFile(primary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name,path in sorted(review_files.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        archive.writestr('README.md', readme)
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(native, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name,path in sorted(raw_files.items()):
            archive.write(path, name)
        archive.writestr('native_manifest.json', json.dumps(dict(entries=[e for e in entries if e['part']=='native'],
            note='Unmodified NPZ files at original relative paths; unpack together with the review ZIP.'), ensure_ascii=False, indent=2))
    verification = {}
    for label,path in (('review',primary), ('native',native)):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            verification[label] = dict(path=str(path), bytes=path.stat().st_size,
                decimal_MB=path.stat().st_size/1e6, entries=len(archive.infolist()),
                crc_pass=bad is None, first_bad_entry=bad)
        if bad:
            raise RuntimeError(f'ZIP CRC failed: {bad}')
    verification['review_below_10_MB'] = primary.stat().st_size <= 10_000_000
    verification['no_evidence_dropped_to_force_size_limit'] = True
    write_json(output/'spindle_v5_package_verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/clearance_v5_excitation'))
    parser.add_argument('--output', type=Path, default=Path('deliverables'))
    args = parser.parse_args()
    package(args.root, args.output)
