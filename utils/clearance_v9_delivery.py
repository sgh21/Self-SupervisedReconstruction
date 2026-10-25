"""Inventory and package the actual V9 evidence; absent results stay absent."""
import argparse
import csv
import json
import platform
import subprocess
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('case','variant','gap_um','U_kg_m','force_N','E_dir_um','E_x_um','E_y_um',
          'E_x_max_um','E_y_max_um','mean_x_um','mean_y_um','H_v_rms_x_mm_s',
          'H_v_rms_y_mm_s','H_v_1x_x_mm_s','H_v_1x_y_mm_s',
          'H_v_residual_x_mm_s','H_v_residual_y_mm_s','fit_contact_fraction',
          'fit2_contact_fraction','measurement_start_s','measurement_end_s',
          'complete_turns','steady_pass','sampling_pass','domain_pass','nfev','wall_seconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def crc_record(path,root):
    value = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            value = zlib.crc32(chunk,value)
    return dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                crc32=f'{value & 0xffffffff:08x}')


def inventory(root):
    rows,failures = [],[]
    for path in sorted(root.rglob('metrics.json')):
        if any(part in ('source_before','source_after') for part in path.parts):
            continue
        cfg = read_json(path.parent/'actual_config.json')
        values = read_json(path)
        values.update(case=path.parent.relative_to(root).as_posix(),
                      variant=cfg.get('mechanism_variant','B0'))
        rows.append({key:values.get(key) for key in FIELDS})
    for path in sorted(root.rglob('failed.json')):
        if any(part in ('source_before','source_after') for part in path.parts):
            continue
        failures.append(dict(case=path.parent.relative_to(root).as_posix(),**read_json(path)))
    with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    write_json(root/'run_inventory.json',dict(completed_cases=len(rows),
        failed_cases=len(failures),runs=rows,failures=failures,
        note='Completed integration does not imply numerical or physical acceptance.'))
    return rows,failures


def environment(root):
    import numpy, scipy, numba
    values = dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        executable=sys.executable,platform=platform.platform(),processor=platform.processor(),
        numpy=numpy.__version__,scipy=scipy.__version__,numba=numba.__version__)
    try:
        values['git_base_commit'] = subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        values['git_status'] = subprocess.check_output(
            ['git','status','--short'],cwd=ROOT,text=True)
    except subprocess.CalledProcessError as error:
        values['git_error'] = str(error)
    write_json(root/'environment/actual_environment.json',values)


def source_files():
    for folder in ('src','scripts','utils'):
        yield from sorted((ROOT/folder).rglob('*.py'))
    yield from sorted((ROOT/'config').glob('*.json'))
    for name in ('requirements.txt','pyproject.toml','README.md'):
        if (ROOT/name).exists():
            yield ROOT/name


def package(root,destination,review_only=False):
    """Review zip plus one raw zip per completed case, all same extraction root."""
    destination.mkdir(parents=True,exist_ok=True)
    source = list(source_files())
    frozen_source = [p for p in sorted((root/'source_before').rglob('*'))
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix.lower() in ('.py','.json','.md','.txt','.toml')]
    evidence = [p for p in sorted(root.rglob('*')) if p.is_file()
                and p != root/'manifest_crc32.json'
                and not any(x in p.relative_to(root).parts for x in ('source_before','source_after','__pycache__'))]
    if review_only:
        evidence = [p for p in evidence if p.suffix.lower() not in
                    ('.npz','.npy','.xz','.pdf','.zip')]
    manifest = [crc_record(p,root) for p in evidence]
    write_json(root/'manifest_crc32.json',dict(algorithm='CRC32',
        reason='User explicitly prohibits SHA256; CRC32 is a transfer-integrity check, not a security signature.',
        source=[crc_record(p,ROOT) for p in source],evidence=manifest))
    # The actual frozen V8 code is needed to audit the independently run baseline.
    manifest_record = read_json(root/'manifest_crc32.json')
    manifest_record['frozen_v8_source'] = [crc_record(p,root) for p in frozen_source]
    manifest_record['delivery_scope'] = ('Code, configurations, figures and reported evidence only; '
        'raw trajectories, independent refinement arrays and PDFs excluded at user request; '
        'no package-native replay performed.' if review_only else 'Full evidence split across archives.')
    write_json(root/'manifest_crc32.json',manifest_record)
    review = destination/'spindle_v9_review.zip'
    prefix = 'spindle_v9_review/'
    with zipfile.ZipFile(review,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in source:
            archive.write(path,prefix+'source/'+path.relative_to(ROOT).as_posix())
        for path in frozen_source:
            archive.write(path,prefix+path.relative_to(root).as_posix())
        contract = ROOT/'GPTOrder/spindle_v9_tai_bearing_plan/templates/v9_contract.json'
        archive.write(contract,prefix+'source/'+contract.relative_to(ROOT).as_posix())
        frozen = root/'config/parameters_v8_frozen.json'
        archive.write(frozen,prefix+'source/experiments/clearance_v9_tai_bearing/config/parameters_v8_frozen.json')
        for path in evidence+[root/'manifest_crc32.json']:
            relative = path.relative_to(root)
            if path.suffix.lower() in ('.npz','.xz','.pdf','.zip','.npy'):
                continue
            archive.write(path,prefix+relative.as_posix())
    if review_only:
        listing = dict(review=dict(file=review.name,bytes=review.stat().st_size),native=[],
            delivery_scope=manifest_record['delivery_scope'],
            extraction='Extract spindle_v9_review.zip; source/ contains runnable code.',
            original_evidence_root=str(root),raw_data_deleted=False,
            package_native_replay='not_run_at_user_request')
        write_json(destination/'spindle_v9_packages.json',listing)
        return listing
    native_archives=[]
    # Keeping already-compressed NPZ members stored avoids expensive recompression.
    case_folders = sorted({p.parent for p in root.rglob('native.npz')
                           if not any(x in p.parts for x in ('source_before','source_after'))})
    packed=set()
    for folder in case_folders:
        relative=folder.relative_to(root)
        filename='spindle_v9_native_'+'_'.join(relative.parts)+'.zip'
        target=destination/filename
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix.lower() in ('.npz','.npy','.xz'):
                    archive.write(path,prefix+path.relative_to(root).as_posix());packed.add(path)
        native_archives.append(dict(file=filename,bytes=target.stat().st_size,
                                    case=relative.as_posix()))
    supplemental=[p for p in evidence if p.suffix.lower() in ('.npz','.npy','.pdf','.xz') and p not in packed]
    if supplemental:
        target=destination/'spindle_v9_supplemental.zip'
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for path in supplemental:
                archive.write(path,prefix+path.relative_to(root).as_posix())
        native_archives.append(dict(file=target.name,bytes=target.stat().st_size,case='local evidence and sources'))
    listing=dict(review=dict(file=review.name,bytes=review.stat().st_size),native=native_archives,
        extraction='Extract review and selected/all raw zips into the same directory.',
        original_evidence_root=str(root))
    write_json(destination/'spindle_v9_packages.json',listing)
    return listing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'experiments/clearance_v9_tai_bearing')
    parser.add_argument('--package',action='store_true')
    parser.add_argument('--review-only',action='store_true',
                        help='Exclude raw arrays and PDFs; skip native/supplemental archives.')
    parser.add_argument('--destination',type=Path,default=ROOT/'deliverables/spindle_v9')
    args=parser.parse_args()
    rows,failures=inventory(args.root)
    environment(args.root)
    print(json.dumps(dict(completed=len(rows),failed=len(failures))))
    if args.package:
        print(json.dumps(package(args.root,args.destination,args.review_only),ensure_ascii=False))


if __name__=='__main__':
    main()
