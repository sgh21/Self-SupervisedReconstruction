"""Independent V6 baseline, half-step and doubled-direction runs."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scripts.run_clearance_v6 import case_config, execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/clearance_v6_GMN/parameters_v6_candidate.json'))
    parser.add_argument('--output',type=Path,default=Path('experiments/clearance_v6_GMN'))
    parser.add_argument('--gaps',type=float,nargs='+',required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    baseline=json.loads(args.config.read_text(encoding='utf-8'))
    jobs=[]
    for gap in args.gaps:
        for group,step,count in (('data',1,512),('certification/h_half',.5,512),
                                 ('certification/n1024',1,1024)):
            jobs.append((case_config(baseline,gap,directions=count,step_factor=step),
                         args.output/group/f'gap_{gap:g}um'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,*job):job[1] for job in jobs}
        for future in as_completed(futures):
            result=future.result()
            print(json.dumps(dict(folder=str(futures[future]),E_um=result['E_dir_um'],
                steady=result['steady_pass'],domain=result['domain_pass'])),flush=True)


if __name__=='__main__':main()
