"""V6 code/review ZIP plus complete native evidence; no original data is changed."""
import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from scripts.package_clearance_v5 import save_npz_lossless, write_json
from utils.pitting_analysis import resample_to


CODE_FILES = (
    'src/spindle/__init__.py', 'src/spindle/dynamics_model.py', 'src/spindle/fast_rhs.py',
    'scripts/__init__.py', 'scripts/run_clearance_v3.py', 'scripts/run_clearance_v6.py',
    'scripts/plot_clearance_v6.py', 'scripts/package_clearance_v5.py',
    'scripts/package_clearance_v6.py', 'scripts/review_v5_representatives.py',
    'utils/__init__.py', 'utils/clearance_v3_analysis.py', 'utils/clearance_v4_analysis.py',
    'utils/clearance_v5_analysis.py', 'utils/clearance_v6_analysis.py',
    'utils/runout_labels.py', 'utils/pitting_analysis.py', 'utils/data_io.py',
    'config/clearance_v5_config.json', 'config/clearance_v6_config.json',
    'GPTOrder/spindle_v6_plan/config/parameter_contract_v6.json',
    'GPTOrder/spindle_v6_plan/00_先读这个.md',
    'GPTOrder/spindle_v6_plan/01_完整实验计划.md',
    'GPTOrder/spindle_v6_plan/03_回传验收报告模板.md',
    'GPTOrder/spindle_v6_plan/04_来源与更新记录.md',
    'GPTOrder/spindle_v6_plan/templates/acceptance_result.template.json',
    'GPTOrder/spindle_v6_plan/references/verified_manufacturer_values.json',
    'requirements.txt', 'pyproject.toml', 'AGENTS.md',
)


def redundant_single_history(history, stationarity, metrics, config):
    return (len(history)==1 and set(history[0])=={'warmup_s','start_s','end_s','stationarity'}
        and history[0]['stationarity']==stationarity
        and history[0]['warmup_s']==config['simulation']['warmup_s']
        and history[0]['start_s']==metrics['measurement_start_s']
        and history[0]['end_s']==metrics['measurement_end_s'])


def history_mapping(name):
    parent=name.rsplit('/',1)[0]
    return dict(path=name, reason='Single-entry warmup history exactly reconstructable; original local file retained',
        reconstructed_list_length=1, fields={
            'stationarity':parent+'/stationarity.json',
            'warmup_s':parent+'/actual_config.json#/simulation/warmup_s',
            'start_s':parent+'/metrics.json#/measurement_start_s',
            'end_s':parent+'/metrics.json#/measurement_end_s'})


def representatives(root, gaps=None):
    acceptance = json.loads((root/'recomputed/acceptance.json').read_text(encoding='utf-8'))
    records = acceptance['cases']
    health = next(r for r in records if r['metrics']['gap_um'] == 2)
    valid = [r for r in records if r['main_valid'] and r['metrics']['gap_um'] > 2]
    candidates = valid or [r for r in records if r['metrics']['gap_um'] > 2]
    maximum = max(candidates, key=lambda r: r['metrics']['E_dir_um'], default=health)
    transition = min((r for r in records if r not in (health, maximum) and r['metrics']['gap_um'] > 2),
        key=lambda r: min(abs(r['metrics'][prefix+'contact_fraction']-.5) for prefix in ('fit_', 'fit2_')),
        default=None)
    chosen = [r for r in (health, transition, maximum) if r is not None]
    chosen = list({r['case']:r for r in chosen}.values())
    if gaps is not None:
        chosen = [r for r in records if r['metrics']['gap_um'] in gaps]
    folder = root/'review_representatives'; folder.mkdir(exist_ok=True)
    entries = []
    for row in chosen:
        case = 'data/'+row['case']
        with np.load(root/case/'native.npz') as d:
            meta = json.loads(str(d['metadata_json']))
            cfg = meta['config']; p, s = cfg['parameters'], cfg['simulation']
            start, end = meta['export_start_s'], meta['export_end_s']
            t, relative = d['t_raw'], d['r_m_raw_m']
            interior = (t > start) & (t < end)
            boundary = np.array([[np.interp(edge, t, relative[:, axis]) for axis in (0, 1)] for edge in (start, end)])
            tm = np.r_[start, t[interior], end]
            motion = np.vstack((boundary[:1], relative[interior], boundary[1:]))
            tail = slice(max(0, np.searchsorted(t, end-5*60/abs(p['rpm']))-1),
                         min(len(t), np.searchsorted(t, end)+1))
            raw_h = resample_to(d['v_raw_m_s'][:, 4:6].T, s['fs_raw_hz'], s['fs_out_hz']).T*1000
            arrays = dict(t_M_s=tm, M_relative_probe_m=motion,
                t_H_s=d['t_out'], H_velocity_mm_s=d['H_velocity_mm_s'],
                H_acceleration_m_s2=d['H_acceleration_m_s2'],
                t_H_unbandpassed_s=t[0]+np.arange(len(raw_h))/s['fs_out_hz'],
                H_unbandpassed_guarded_velocity_mm_s=raw_h,
                t_tail_s=t[tail], q_tail=d['q_raw_m'][tail],
                v_tail=d['v_raw_m_s'][tail], a_tail=d['a_raw_m_s2'][tail])
            for key in ('fit_force_N', 'fit2_force_N', 'fit_active_count', 'fit2_active_count',
                        'bearing1_force_N', 'bearing2_force_N', 'rear_force_N'):
                arrays['tail_'+key] = d[key][tail]
            arrays['metadata_json'] = json.dumps(dict(config=cfg, source_native=case+'/native.npz',
                export_start_s=start, export_end_s=end, state_order=meta['state_order'],
                q_units=['m']*8+['rad']*2+['m']*2,
                v_units=['m/s']*8+['rad/s']*2+['m/s']*2,
                a_units=['m/s2']*8+['rad/s2']*2+['m/s2']*2,
                note='Full native measurement-section samples over the export window; two interpolated phase boundaries. Full-state tail is five turns only; complete evidence is in native ZIP.'), ensure_ascii=False)
            target = folder/f'gap_{row["metrics"]["gap_um"]:g}um.npz'
            save_npz_lossless(target, arrays)
            entries.append(dict(case=case, file=target.relative_to(root).as_posix(),
                actual_config=case+'/actual_config.json', source_native=case+'/native.npz',
                start_s=start, end_s=end, retained_turns=row['metrics']['complete_turns'],
                native_rate_hz=s['fs_raw_hz'], native_M_interior_samples=int(interior.sum()),
                native_M_samples_with_boundaries=len(tm), H_processed_samples=len(d['t_out']),
                tail_native_samples=len(t[tail]), bytes=target.stat().st_size,
                reference_metrics=row['metrics'], main_valid=row['main_valid']))
    index = dict(cases=entries, precision='float64; no rescaling, downcast, or M decimation',
        selection='Healthy c=2, contact transition, maximum computed/certified degraded case; certification states are explicit',
        compression='Lossless NumPy-readable ZIP-LZMA',
        scope='Full-window native M and saved primary 10-900 Hz H. Full-state tail only. Complete contact and all refinements remain in native ZIP.')
    write_json(folder/'index.json', index)
    (folder/'README.md').write_text('''# V6代表数据范围

本目录保留健康点、接触转变点及最大退化代表点。完整导出窗内M相对固定探头的65536 Hz原生位移全部保留，另加与原逐转指标一致的两个插值边界。没有缩放或转为float32。
主10–900 Hz H速度/加速度保留完整8192 Hz输出；另存带保护的未带通H速度（已抗混叠至8192 Hz）。12维状态和四支承力只保留末5转。
完整12维时序、逐球接触、131072 Hz同解漏峰及独立h/2、N1024证据在原生ZIP，保持原路径。代表数据单独只能复算全窗PP和已处理H特征，不等于完整数值认证。

复用通用代表数据复算入口：`python -m scripts.review_v5_representatives --root .`。入口文件名沿用旧版本，读取的是本包V6实际配置和代表数据。
''', encoding='utf-8')
    return index


def package(root, output, gaps=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    index = representatives(root, gaps)
    review = {name:repo/name for name in CODE_FILES}
    # Include additional V6 formal validation/calibration entry points written by
    # other contributors, without bundling unrelated historic experiments.
    for pattern in ('scripts/*v6*.py', 'utils/*v6*.py', 'config/*v6*.json'):
        review.update({p.relative_to(repo).as_posix():p for p in repo.glob(pattern)})
    for path in repo.glob('docs/reports/*v6*.md'):
        review[path.relative_to(repo).as_posix()] = path
    raw, omitted = {}, []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root); name = relative.as_posix()
        if '__pycache__' in relative.parts or path.suffix in ('.pyc', '.nbc', '.nbi'):
            omitted.append(dict(path=name, reason='Regenerable Python/Numba cache'))
            continue
        if ((path.suffix == '.npz' and relative.parts[0] != 'review_representatives')
                or (relative.parts[0] == 'references' and path.suffix.lower() in ('.pdf', '.html', '.htm'))):
            raw[name] = path; continue
        if path.name == 'progress.json' and path.with_name('warmup_history.json').exists() and path.with_name('metrics.json').exists():
            omitted.append(dict(path=name, reason='Intermediate progress duplicated by final saved history and metrics; original local file retained',
                covered_by=[str(relative.with_name('warmup_history.json')).replace('\\','/'),
                            str(relative.with_name('metrics.json')).replace('\\','/')]))
            continue
        if path.name=='warmup_history.json':
            peers=[path.with_name(name) for name in ('stationarity.json','metrics.json','actual_config.json')]
            if all(peer.exists() for peer in peers) and redundant_single_history(
                    json.loads(path.read_text(encoding='utf-8')),
                    *(json.loads(peer.read_text(encoding='utf-8')) for peer in peers)):
                omitted.append(history_mapping(name))
                continue
        if relative.parts[0] == 'source_before':
            canonical = Path(*relative.parts[1:]).as_posix()
            if canonical in review and path.read_bytes() == review[canonical].read_bytes():
                omitted.append(dict(path=name, reason='Identical source retained once', identical_to=canonical))
                continue
        if name == 'README.md': name = 'reports/original_results_README.md'
        if name not in ('manifest.json', 'package_verification.json'):
            review[name] = path
    entries = [dict(path=name, bytes=path.stat().st_size, part=part,
        content_kind='reference_original' if part == 'native' and path.suffix.lower() in ('.pdf', '.html', '.htm')
            else 'simulation_native' if part == 'native' else 'code_config_report_or_representative')
        for part, files in (('review', review), ('native', raw)) for name,path in sorted(files.items())]
    manifest = dict(schema='v6_split_review_package', entries=entries, omitted=omitted,
        native_files=sorted(raw), representatives=index,
        note='Original files untouched. Extract both ZIPs together; no hashes or numerical scaling.')
    readme = '''# V6 GMN健康2 μm标定验收包

报告为 `report.md`（或 `reports/` 下实际回传报告）；实际结论以 `acceptance_result.json`、`recomputed/acceptance.json` 和原始证据为准。健康c=2 μm用于条件标定，随后固定U，仅改变共同前间隙；标定不等于独立实机验证。

本包包含源码、实际配置、标定历史、全部报告/图表/验收表及代表数据。完整原生与数值细化数据另在 `spindle_v6_GMN_health2um_native.zip`，保持相同相对路径；两包解压至同一目录。厂家原始PDF/HTML也放在该包，manifest标为reference_original，和仿真数据simulation_native分开标注；审阅包保留来源摘录与核验记录。

```powershell
python -m pip install -e .
# 主包独立复算代表数据的逐转PP与已处理H
python -m scripts.review_v5_representatives --root .
# 使用包内实际冻结配置重新运行；输出到新目录
python -m scripts.run_clearance_v6 --config parameters_v6_calibrated.json --output rerun --gaps 2 3 5 7.5 10 --workers 3
# 两包合并后，从原生状态重做全窗分析和认证
python -m utils.clearance_v6_analysis --root . --replay
python -m scripts.plot_clearance_v6 --root .
```

代表包保留完整窗的M原生位移及主10–900 Hz H，12维状态仅末5转。逐球接触、整窗全状态、131072 Hz漏峰证据和所有独立h/2、N1024数据均在原生包；没有把裁剪代表数据冒充全部收敛证据。运行模型的命令和依赖版本见报告与运行命令文档。
'''
    review_path = output/'spindle_v6_GMN_health2um_review.zip'
    native_path = output/'spindle_v6_GMN_health2um_native.zip'
    with zipfile.ZipFile(review_path, 'w', compression=zipfile.ZIP_LZMA, allowZip64=True) as archive:
        for name,path in sorted(review.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == '.npz' else zipfile.ZIP_LZMA)
        archive.writestr('README.md', readme)
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(native_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name,path in sorted(raw.items()): archive.write(path, name)
        archive.writestr('native_manifest.json', json.dumps(dict(entries=[e for e in entries if e['part'] == 'native']), indent=2))
    status = {}
    for part,path in (('review', review_path), ('native', native_path)):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            status[part] = dict(path=str(path), bytes=path.stat().st_size,
                decimal_MB=path.stat().st_size/1e6, entries=len(archive.infolist()),
                crc_pass=bad is None, first_bad_entry=bad)
        if bad: raise RuntimeError(f'ZIP CRC failed: {bad}')
    status['review_below_10_MB'] = review_path.stat().st_size <= 10_000_000
    status['all_native_evidence_retained'] = True
    write_json(output/'spindle_v6_package_verification.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def slim_existing_review(output):
    """Atomically recompress only the review ZIP; never read/rewrite the native ZIP."""
    output=Path(output).resolve()
    path=output/'spindle_v6_GMN_health2um_review.zip'
    verification=output/'spindle_v6_package_verification.json'
    status=json.loads(verification.read_text(encoding='utf-8'))
    handle, temp_name=tempfile.mkstemp(prefix='v6_review_',suffix='.zip',dir=output)
    os.close(handle)
    temp=Path(temp_name)
    repo=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(path) as source:
        names=set(source.namelist())
        omitted={name for name in names if name.endswith('/progress.json')
            and name.rsplit('/',1)[0]+'/warmup_history.json' in names
            and name.rsplit('/',1)[0]+'/metrics.json' in names}
        histories=set()
        for name in names:
            if not name.endswith('/warmup_history.json'):continue
            parent=name.rsplit('/',1)[0]
            peers=[parent+'/'+part for part in ('stationarity.json','metrics.json','actual_config.json')]
            if all(peer in names for peer in peers) and redundant_single_history(
                    json.loads(source.read(name)),*(json.loads(source.read(peer)) for peer in peers)):
                histories.add(name)
        manifest=json.loads(source.read('manifest.json'))
        manifest['entries']=[entry for entry in manifest['entries'] if entry['path'] not in omitted|histories]
        for name in sorted(omitted):
            parent=name.rsplit('/',1)[0]
            manifest['omitted'].append(dict(path=name,
                reason='Intermediate progress duplicated by final history and metrics; original local file retained',
                covered_by=[parent+'/warmup_history.json',parent+'/metrics.json']))
        manifest['omitted'].extend(history_mapping(name) for name in sorted(histories))
        script='scripts/package_clearance_v6.py'
        current=(repo/script).read_bytes()
        for entry in manifest['entries']:
            if entry['path']==script: entry['bytes']=len(current)
        manifest['review_storage']=dict(text_compression='ZIP_LZMA_lossless',
            representative_npz='Original bytes stored unchanged',
            progress_deduplicated_count=sum(row['path'].endswith('/progress.json') for row in manifest['omitted']),
            single_history_deduplicated_count=sum(row['path'].endswith('/warmup_history.json') for row in manifest['omitted']),
            native_zip_untouched=True)
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_LZMA,allowZip64=True) as target:
            for name in sorted(names-omitted-histories-{'manifest.json'}):
                data=current if name==script else source.read(name)
                target.writestr(name,data,compress_type=zipfile.ZIP_STORED if name.endswith('.npz') else zipfile.ZIP_LZMA)
            target.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as archive:
        bad=archive.testzip()
        count=len(archive.infolist())
    if bad: raise RuntimeError(f'Review ZIP CRC failed: {bad}')
    os.replace(temp,path)
    status['review']=dict(path=str(path),bytes=path.stat().st_size,
        decimal_MB=path.stat().st_size/1e6,entries=count,crc_pass=True,first_bad_entry=None)
    status['review_below_10_MB']=path.stat().st_size<=10_000_000
    status['review_lossless_slimming']=dict(progress_duplicates_removed=manifest['review_storage']['progress_deduplicated_count'],
        single_history_duplicates_removed=manifest['review_storage']['single_history_deduplicated_count'],
        text_compression='ZIP_LZMA',representative_arrays_unchanged=True,all_figures_retained=True)
    status['native_reused_without_rewrite_or_crc']=True
    write_json(verification,status)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/clearance_v6_GMN'))
    parser.add_argument('--output', type=Path, default=Path('deliverables'))
    parser.add_argument('--representative-gaps', type=float, nargs='+')
    parser.add_argument('--slim-existing-review',action='store_true')
    args = parser.parse_args()
    if args.slim_existing_review:
        slim_existing_review(args.output)
    else:
        package(args.root, args.output, args.representative_gaps)
