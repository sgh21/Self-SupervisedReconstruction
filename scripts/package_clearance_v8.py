"""Create split V8 review/native packages without changing source evidence."""
from __future__ import annotations

import argparse
import io
import json
import posixpath
import re
import zlib
import zipfile
from pathlib import Path

import numpy as np
from utils.runout_labels import complete_turn_runout

REVIEW_LIMIT_BYTES = 10_000_000
NATIVE_SUFFIXES = {'.npz', '.npy', '.pdf'}
BANDS = ((10, 50), (50, 100), (100, 900), (900, 2048))
REPORT_CANDIDATES = ('report.md', 'reports/report.md', 'reports/V8_回传报告.md')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_npz_lossless(path, arrays):
    """Write NumPy arrays losslessly using the existing V5 review-pack format."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_LZMA) as archive:
        for name, values in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(values), allow_pickle=False)
            archive.writestr(name + '.npy', buffer.getvalue())


def harmonics(time, signal, omega, phase, prefix, unit, energy_unit, fs):
    """Replay the shared clearance_v3_analysis harmonic/statistic definition."""
    from scipy.signal import periodogram

    angle = omega * time + phase
    design = np.column_stack((np.ones(len(time)), np.cos(angle), np.sin(angle)))
    coef = np.linalg.lstsq(design, signal, rcond=None)[0]
    residual = signal - design @ coef
    frequencies, psd = periodogram(residual, fs=fs, window='hann', detrend=False,
                                   scaling='density', axis=0)
    metrics = {}
    for name, values in [('rms', np.sqrt(np.mean(signal ** 2, axis=0))),
                         ('1x', np.linalg.norm(coef[1:], axis=0)),
                         ('residual', np.sqrt(np.mean(residual ** 2, axis=0)))]:
        for axis, value in zip('xy', values):
            metrics[f'{prefix}_{name}_{axis}_{unit}'] = float(value)
    for low, high in BANDS:
        band = (frequencies >= low) & (frequencies < high)
        energy = psd[band].sum(axis=0) * (frequencies[1] - frequencies[0])
        for axis, value in zip('xy', energy):
            metrics[f'{prefix}_residual_energy_{low}_{high}_{axis}_{energy_unit}'] = float(value)
    return metrics


def crc32(path):
    value = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(block, value)
    return f'{value & 0xffffffff:08x}'


def crc32_bytes(value):
    return f'{zlib.crc32(value) & 0xffffffff:08x}'


def verify_report_links(archive_path, report_target):
    """Verify image/CSV links in the canonical report and the top-level entry."""
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        report = archive.read(report_target).decode('utf-8')
        shortcut = archive.read('V8_回传报告.md').decode('utf-8')
        links = [(report_target, target) for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report)]
        links.extend(('V8_回传报告.md', target)
                     for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', shortcut))
        checked, missing = [], []
        for source, raw_target in links:
            target = raw_target.strip().split()[0].strip('<>')
            if target.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            suffix = Path(target.split('#', 1)[0]).suffix.lower()
            if suffix not in {'.png', '.jpg', '.jpeg', '.svg', '.csv'} and not (
                    source == 'V8_回传报告.md' and suffix == '.md'):
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), target.split('#', 1)[0]))
            checked.append(dict(source=source, target=target, resolved=resolved))
            if resolved not in names:
                missing.append(resolved)
    return dict(passed=not missing, checked=checked, missing=missing)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def find_report(root):
    for relative in REPORT_CANDIDATES:
        path = Path(root) / relative
        if path.is_file():
            return path
    return None


def case_rows(root, group):
    rows = []
    for metrics_path in sorted((root / group).rglob('metrics.json')):
        folder = metrics_path.parent
        config_path, native_path = folder / 'actual_config.json', folder / 'native.npz'
        if config_path.exists() and native_path.exists():
            rows.append(dict(folder=folder, metrics=read_json(metrics_path),
                             config=read_json(config_path),
                             case=folder.relative_to(root).as_posix()))
    return rows


def select_representatives(root, group, requested=None):
    root = Path(root).resolve()
    rows = case_rows(root, group)
    group_path = Path(group)
    group_prefix = (group_path.resolve().relative_to(Path(root).resolve()).as_posix()
                    if group_path.is_absolute() else group_path.as_posix())
    acceptance = read_json(root / 'recomputed/acceptance.json', {}) or {}
    acceptance_cases = {row.get('case'): row for row in acceptance.get('cases', [])}
    for row in rows:
        case_in_group = Path(row['case']).relative_to(group_prefix).as_posix()
        result = acceptance_cases.get(case_in_group)
        if result is None:
            row['acceptance_case'] = case_in_group
            row['main_valid'] = None
            row['full_waveform_pass'] = None
            continue
        row['acceptance_case'] = case_in_group
        row['main_valid'] = bool(result['main_valid']) if 'main_valid' in result else None
        row['full_waveform_pass'] = bool(result['full_waveform_pass']) if 'full_waveform_pass' in result else None
    if requested:
        by_path = {row['case']: row for row in rows}
        selected = []
        for value in requested:
            path = Path(value)
            relative = path.resolve().relative_to(root).as_posix() if path.is_absolute() else path.as_posix().lstrip('./')
            if relative not in by_path:
                raise FileNotFoundError(f'No completed case at {value!r} under {root}')
            selected.append(by_path[relative])
        return selected, 'Explicit case folders provided by the caller.'
    if not rows:
        raise FileNotFoundError(f'No completed cases found under {root / group}')

    def gap(row):
        return float(row['metrics'].get('gap_um',
                     row['config']['parameters'].get('front_group_clearance_um', float('nan'))))

    health = next((row for row in rows if abs(gap(row) - 2.0) < 1e-12), None)
    if health is None:
        raise ValueError(f'No c=2 um B health case found under {root / group}')
    degraded = [row for row in rows if gap(row) > 2]
    full_pool = [row for row in degraded if row['full_waveform_pass'] is True]
    main_pool = [row for row in degraded if row['main_valid'] is True]
    if full_pool:
        max_pool, certification_level = full_pool, 'full_waveform_certified'
    elif main_pool:
        max_pool, certification_level = main_pool, 'main_valid_only'
    else:
        max_pool, certification_level = degraded or rows, 'computed_only'
    maximum = max(max_pool, key=lambda row: float(row['metrics']['E_dir_um']))
    transition_pool = [row for row in rows if row not in (health, maximum)]
    if transition_pool:
        contact_keys = [key for key in health['metrics'] if key.endswith('_contact_fraction')]
        if contact_keys:
            transition = min(transition_pool, key=lambda row: sum(
                abs(float(row['metrics'].get(key, 0.0)) - .5) for key in contact_keys) / len(contact_keys))
            selection = 'c=2 health; nearest geometric bearing-contact transition; largest degraded E_dir selected from the strongest available level (full_waveform_pass, then main_valid, then computed only).'
        else:
            transition = min(transition_pool, key=lambda row: abs(gap(row) - .5*(gap(health)+gap(maximum))))
            selection = 'c=2 health; middle-gap case; largest degraded E_dir selected from the strongest available level (full_waveform_pass, then main_valid, then computed only).'
        chosen = [health, transition, maximum]
    else:
        chosen = [health, maximum]
        selection = 'c=2 health and largest degraded E_dir selected from the strongest available level (full_waveform_pass, then main_valid, then computed only); fewer than three distinct cases.'
    selection += f' Largest-case level: {certification_level}.'
    def candidate_summary(pool):
        if not pool:
            return 'none'
        candidate = max(pool, key=lambda row: float(row['metrics']['E_dir_um']))
        return (f"{candidate['case']} E_dir={float(candidate['metrics']['E_dir_um']):.9g} um "
                f"(main_valid={candidate['main_valid']}, full_waveform_pass={candidate['full_waveform_pass']})")
    selection += (f" Largest main_valid candidate: {candidate_summary(main_pool)}."
                  f" Largest full_waveform_pass candidate: {candidate_summary(full_pool)}.")
    result, seen = [], set()
    for row in chosen:
        if row['case'] not in seen:
            result.append(row); seen.add(row['case'])
    return result, selection


def representatives(root, group='data_B_recalibrated', requested=None):
    root = Path(root).resolve()
    chosen, selection = select_representatives(root, group, requested)
    output = root / 'review_representatives'
    output.mkdir(exist_ok=True)
    cases = []
    for index, row in enumerate(chosen, 1):
        folder = row['folder']
        with np.load(folder / 'native.npz') as data:
            meta = json.loads(str(data['metadata_json']))
            config = meta['config']
            p, sim = config['parameters'], config['simulation']
            start, end = float(meta['export_start_s']), float(meta['export_end_s'])
            t, relative = data['t_raw'], data['r_m_raw_m']
            interior = (t > start) & (t < end)
            boundary = np.asarray([[np.interp(edge, t, relative[:, axis]) for axis in (0, 1)]
                                   for edge in (start, end)])
            t_m = np.r_[start, t[interior], end]
            motion = np.vstack((boundary[:1], relative[interior], boundary[1:]))
            tail_start = end - 5 * 60 / abs(p['rpm'])
            left = max(0, int(np.searchsorted(t, tail_start)) - 1)
            right = min(len(t), int(np.searchsorted(t, end)) + 1)
            tail = slice(left, right)
            arrays = dict(t_M_s=t_m, M_relative_probe_m=motion,
                t_H_s=data['t_out'], H_velocity_mm_s=data['H_velocity_mm_s'],
                H_acceleration_m_s2=data['H_acceleration_m_s2'], t_tail_s=t[tail],
                q_tail=data['q_raw_m'][tail], v_tail=data['v_raw_m_s'][tail],
                a_tail=data['a_raw_m_s2'][tail])
            constraint_path = folder / 'constraint_reactions.npz'
            if not constraint_path.exists():
                raise FileNotFoundError(f'Missing V8 constraint reactions: {constraint_path}')
            with np.load(constraint_path) as constraints:
                if not np.array_equal(constraints['t_raw'], t):
                    raise ValueError(f'Constraint reaction time grid differs from native grid: {folder}')
                for name in ('F1', 'F2', 'R1', 'R2'):
                    arrays[f'tail_{name}_local_restoring_Nm'] = data[name + '_local_restoring_Nm'][tail]
                    for suffix in ('lever_restoring_Nm', 'outer_rotation_constraint_Nm', 'outer_axial_constraint_N'):
                        key = f'{name}_{suffix}'
                        arrays[f'tail_{key}'] = constraints[key][tail]
                for key in ('H_bearing_load_constraint_Nm', 'rotor_axial_constraint_N'):
                    arrays[f'tail_{key}'] = constraints[key][tail]
            arrays['metadata_json'] = json.dumps(dict(config=config,
                source_native=row['case'] + '/native.npz', export_start_s=start, export_end_s=end,
                state_order=meta['state_order'], q_units=meta['coordinate_units'],
                v_units=meta['velocity_units'], a_units=meta['acceleration_units'], tail_turns=5,
                note='Full-window M relative to the fixed probe and processed H; full state and per-bearing local/constraint torque excerpts cover the final five turns. Complete native evidence is in the companion ZIP.'), ensure_ascii=False)
            safe = row['case'].replace('/', '__')
            target = output / f'{index:02d}_{safe}.npz'
            save_npz_lossless(target, arrays)
            cases.append(dict(case=row['case'], file=target.relative_to(root).as_posix(),
                actual_config=row['case'] + '/actual_config.json',
                source_native=row['case'] + '/native.npz',
                source_constraints=row['case'] + '/constraint_reactions.npz',
                acceptance_case=row['acceptance_case'], main_valid=row['main_valid'],
                full_waveform_pass=row['full_waveform_pass'],
                start_s=start, end_s=end, retained_turns=row['metrics'].get('complete_turns'),
                native_rate_hz=sim['fs_raw_hz'], native_M_samples=int(interior.sum() + 2),
                processed_H_samples=int(len(data['t_out'])), tail_native_samples=int(len(t[tail])),
                bytes=target.stat().st_size, crc32=crc32(target),
                reference_metrics=row['metrics'],
                inclusion_scope='Full M and processed H; final-five-turn full state and per-bearing local/constraint reactions.'))
    index_data = dict(schema='clearance_v8_representatives_v1', cases=cases,
        selection=selection, precision='float64; no rescaling, downcast, or native-time decimation for M/H',
        compression='Lossless NumPy-readable ZIP-LZMA',
        scope='Representative B cases support exact main E/H replay and torque inspection; complete runs and all refinements remain in the native ZIP.')
    write_json(output / 'index.json', index_data)
    (output / 'README.md').write_text('''# V8 B 组代表数据范围

该目录保留 B 组 c=2 健康、接触转变附近及最大已计算回转误差代表点（若完成点少于三个则保留现有代表点）。M 保留完整验收窗及两个插值边界；H 保留原处理速度和加速度。末五转另存全状态、逐轴承局部恢复力矩、力臂矩及外圈/壳体约束反力矩。没有幅值缩放、float32 转换或 M/H 抽样。其余完整原生轨迹、布局和 h/N 收敛数据在同路径原生包中。

用 `python -m scripts.package_clearance_v8 --replay-only --root experiments/clearance_v8_tilt_contact` 复算逐转 E 与已处理 H 特征。复算按各代表点完整验收窗转数进行，支持长窗运行。
''', encoding='utf-8')
    return index_data


def replay_representatives(root):
    """Replay E and processed H features across each representative's full window."""
    root = Path(root)
    index = read_json(root / 'review_representatives/index.json')
    rows = []
    for item in index['cases']:
        config = read_json(root / item['actual_config'])
        p, sim, excitation = config['parameters'], config['simulation'], config['excitation']
        with np.load(root / item['file']) as data:
            turns = complete_turn_runout(data['t_M_s'], data['M_relative_probe_m'] * 1e6,
                p['rpm'], item['start_s'], item['end_s'], phase_at_zero_rad=excitation['phase_rad'])
            mean = turns['peak_to_peak_um'].mean(axis=0)
            metrics = dict(E_x_um=float(mean[0]), E_y_um=float(mean[1]), E_dir_um=float(max(mean)))
            t = data['t_H_s']
            use = (t >= item['start_s'] - 1e-12) & (t < item['end_s'] - 1e-12)
            for key, prefix, unit, energy in (
                    ('H_velocity_mm_s', 'H_v', 'mm_s', 'mm2_s2'),
                    ('H_acceleration_m_s2', 'H_a', 'm_s2', 'm2_s4')):
                metrics.update(harmonics(t[use], data[key][use], p['rpm'] * 2 * np.pi / 60,
                    excitation['phase_rad'], prefix, unit, energy, sim['fs_out_hz']))
        reference = item['reference_metrics']
        differences = {key: abs(value - reference[key]) for key, value in metrics.items() if key in reference}
        expected_turns = int(reference.get('complete_turns', item.get('retained_turns', 0)))
        rows.append(dict(case=item['case'], complete_turns=len(turns['turn_id']),
            recomputed=metrics,
            max_absolute_metric_difference=max(differences.values()) if differences else None,
            passed=bool(expected_turns > 0 and len(turns['turn_id']) == expected_turns
                and differences and max(differences.values()) < 1e-9)))
    result = dict(passed=bool(rows and all(row['passed'] for row in rows)), cases=rows,
        scope='Representative full-window E and saved filtered H only; not independent full-window contact/convergence acceptance.')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)
    return result


def review_files(repo, root, representative_paths):
    review = {}
    for directory in ('src/spindle', 'scripts', 'utils'):
        for path in (repo / directory).rglob('*.py'):
            if '__pycache__' not in path.parts:
                review[path.relative_to(repo).as_posix()] = path
    for name in ('AGENTS.md', 'README.md', 'pyproject.toml', 'requirements.txt',
                 'config/clearance_v8_config.json',
                 'docs/reports/clearance_v8_parameter_contract.md',
                 'experiments/clearance_v6_GMN/data/gap_2um/actual_config.json',
                 'experiments/clearance_v6_GMN/acceptance_result.json'):
        path = repo / name
        if path.exists():
            review['PROJECT_README.md' if name == 'README.md' else name] = path
    plan = repo / 'GPTOrder/spindle_v8_tilt_contact_plan'
    for path in plan.rglob('*'):
        if path.is_file() and path.name != 'manifest_sha256.json' and path.suffix.lower() not in NATIVE_SUFFIXES:
            review[path.relative_to(repo).as_posix()] = path
    for path in root.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.nbc', '.nbi'):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith('review_representatives/'):
            if relative in representative_paths or relative in ('review_representatives/index.json', 'review_representatives/README.md'):
                review['experiments/clearance_v8_tilt_contact/' + relative] = path
            continue
        if path.suffix.lower() in NATIVE_SUFFIXES:
            continue
        if relative == 'README.md':
            relative = 'reports/original_experiment_README.md'
        review['experiments/clearance_v8_tilt_contact/' + relative] = path
    return review


def package(root, output, representative_group='data_B_recalibrated', representative_cases=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    report_path = find_report(root)
    if not (root / 'acceptance_result.json').exists() or report_path is None:
        raise FileNotFoundError('Wait for acceptance_result.json and a final V8 report before packaging.')
    output.mkdir(parents=True, exist_ok=True)
    rep_index = representatives(root, representative_group, representative_cases)
    rep_paths = {item['file'] for item in rep_index['cases']}
    review = review_files(repo, root, rep_paths)
    report_target = 'experiments/clearance_v8_tilt_contact/' + report_path.relative_to(root).as_posix()
    report_shortcut = ('# V8 回传报告\n\n'
        f'完整报告位于规范路径：[{report_path.name}]({report_target})。\n').encode('utf-8')
    native = {}
    omitted = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if '__pycache__' in relative.parts or path.suffix in ('.pyc', '.nbc', '.nbi'):
            omitted.append(dict(path=relative.as_posix(), reason='Regenerable Python/Numba cache'))
            continue
        if relative.parts[0] == 'review_representatives':
            continue
        native['experiments/clearance_v8_tilt_contact/' + relative.as_posix()] = path
    for path in (repo / 'GPTOrder/spindle_v8_tilt_contact_plan').rglob('*.pdf'):
        native[path.relative_to(repo).as_posix()] = path
    entries = []
    for part, files in (('review', review), ('native', native)):
        for name, path in sorted(files.items()):
            entries.append(dict(path=name, bytes=path.stat().st_size, crc32=crc32(path), part=part,
                content_kind='reference_original' if path.suffix.lower() == '.pdf' else
                    'simulation_native' if part == 'native' and path.suffix.lower() in {'.npz', '.npy'} else
                    'experiment_record' if part == 'native' else 'code_config_report_figure_or_representative'))
    entries.append(dict(path='V8_回传报告.md', bytes=len(report_shortcut), crc32=crc32_bytes(report_shortcut),
                        part='review', content_kind='report_shortcut'))
    manifest = dict(schema='clearance_v8_split_review_package_v1',
        hashing='CRC32 only; no SHA-256 calculated or used',
        review_entries=[entry for entry in entries if entry['part'] == 'review'],
        native_entries=[entry for entry in entries if entry['part'] == 'native'],
        omitted=omitted, representatives=rep_index,
        representative_replay_dependencies=['numpy', 'scipy'],
        full_project_dependencies='requirements.txt (numpy, scipy, h5py, matplotlib, numba>=0.68)',
        report_shortcut=dict(path='V8_回传报告.md', target=report_target),
        extraction='The review ZIP can be extracted and replayed on its own in an empty directory. Extract the native ZIP separately into a repository root to restore complete run evidence; its paths match the experiment tree.',
        source_boundary='PDFs are original references; NPZ/NPY arrays are native model evidence. Original input data are not modified.')
    readme = '''# V8 倾转接触审阅与原生数据包

审阅包包含实际运行源码及依赖、V6 会前源码快照、补丁、运行配置、逐例求解设置、验证输出、完整报告、表格、图、PDF 摘页的渲染/文本核对记录，以及 B 组代表数据。完整仿真原生 NPZ/NPY、全部失败/延长/续接记录和原始 PDF 摘页在同名 `_native.zip` 中。审阅包可独立解压到空目录并直接执行代表回放；原生包可另行解压到仓库根目录恢复完整运行证据。

根目录的 `V8_回传报告.md` 是入口，链接到规范位置 `experiments/clearance_v8_tilt_contact/reports/`，图片和 CSV 的相对链接以原报告目录解析。

CRC32 清单位于 `manifest.json` 和 `native_manifest.json`；遵循仓库约定，没有使用 SHA-256。代表数据未改幅值或精度：M 完整保留验收窗，H 保留处理后的速度/加速度，状态和轴承局部/约束力矩保留末五转。完整轨迹、失败记录、独立步长/方向收敛及延长/续接运行证据在原生包中，代表包不替代完整认证。代表点同时标出 `main_valid` 与 `full_waveform_pass`，二者不混作一个验收状态。

```powershell
$env:PYTHONPATH = "src"
python -m scripts.package_clearance_v8 --replay-only --root experiments/clearance_v8_tilt_contact
```

在解压后的审阅包根目录运行以上命令，无需联网安装依赖；代表回放需要本机已有 NumPy 与 SciPy。若要安装完整项目依赖，可在网络可用时运行 `python -m pip install -e .`；完整依赖见 `requirements.txt`（NumPy、SciPy、h5py、Matplotlib、Numba）。

以上两条回放命令只需审阅包；若要运行 `utils.clearance_v6_force_audit`，须另外解压原生包。

V8 的精确重跑命令、U_old/U_new、接受/未接受项和误差预算以报告、验收文件、每例 `actual_config.json` 和 `actual_solver_settings.json` 为准。本文不把计划中的命令写成已完成运行。
'''
    review_path = output / 'spindle_v8_tilt_contact_review.zip'
    native_path = output / 'spindle_v8_tilt_contact_native.zip'
    with zipfile.ZipFile(review_path, 'w', compression=zipfile.ZIP_LZMA, allowZip64=True) as archive:
        for name, path in sorted(review.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == '.npz' else zipfile.ZIP_LZMA)
        archive.writestr('README.md', readme, compress_type=zipfile.ZIP_LZMA)
        archive.writestr('V8_回传报告.md', report_shortcut, compress_type=zipfile.ZIP_LZMA)
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_LZMA)
    with zipfile.ZipFile(native_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(native.items()):
            archive.write(path, name)
        archive.writestr('native_manifest.json', json.dumps(dict(schema='clearance_v8_native_manifest_v1',
            hashing='CRC32 only', entries=[entry for entry in entries if entry['part'] == 'native']),
            ensure_ascii=False, indent=2))
    replay = replay_representatives(root)
    report_links = verify_report_links(review_path, report_target)
    def zip_info(path):
        with zipfile.ZipFile(path) as archive:
            return len(archive.infolist()), archive.testzip()
    review_count, review_bad = zip_info(review_path)
    native_count, native_bad = zip_info(native_path)
    status = dict(review=dict(path=str(review_path), bytes=review_path.stat().st_size,
                              decimal_MB=review_path.stat().st_size / 1e6, entries=review_count,
                              zip_crc_pass=review_bad is None, first_bad_entry=review_bad),
        native=dict(path=str(native_path), bytes=native_path.stat().st_size,
                    decimal_MB=native_path.stat().st_size / 1e6, entries=native_count,
                    zip_crc_pass=native_bad is None, first_bad_entry=native_bad),
        review_below_10_MB=review_path.stat().st_size < REVIEW_LIMIT_BYTES,
        review_size_preference='Historical <10 MB preference; full-precision representative waveforms and required review evidence are retained.',
        report_embedded_links_passed=report_links['passed'],
        report_links_checked=len(report_links['checked']),
        report_links_missing=report_links['missing'],
        representative_replay_passed=replay['passed'], representative_cases=len(rep_index['cases']),
        native_files_retained=len(native), crc32_manifest_entries=len(entries))
    if review_bad or native_bad:
        raise RuntimeError(f'ZIP validation failed: {review_bad or native_bad}')
    if not replay['passed']:
        raise RuntimeError('Representative metric replay failed')
    if not report_links['passed']:
        raise RuntimeError(f'Report contains unresolved image/CSV links: {report_links["missing"]}')
    write_json(output / 'spindle_v8_package_verification.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/clearance_v8_tilt_contact'))
    parser.add_argument('--output', type=Path, default=Path('deliverables'))
    parser.add_argument('--representative-group', default='data_B_recalibrated')
    parser.add_argument('--representative-case', action='append', dest='representatives',
                        help='Experiment-relative B case path; repeat for two or three explicit representatives.')
    parser.add_argument('--replay-only', action='store_true',
                        help='Recompute selected representative E/H metrics without creating ZIPs.')
    args = parser.parse_args()
    if args.replay_only:
        replay_representatives(args.root)
    else:
        package(args.root, args.output, args.representative_group, args.representatives)


if __name__ == '__main__':
    main()
