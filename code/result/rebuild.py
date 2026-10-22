"""Reproduce P1-P4 into an isolated output directory."""
import argparse, os, subprocess, sys, shutil, json, time
from pathlib import Path
from ..common.paths import SUPPORT_ROOT, protected_output_path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--smoke-days',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    output=protected_output_path(args.output_dir)
    if args.smoke_days is not None and not 32<=args.smoke_days<=365:
        raise SystemExit('--smoke-days must be 32..365 to include output-period rows')
    if not args.dry_run and output.exists() and any(output.iterdir()):
        raise SystemExit('Use a new or empty output directory to avoid stale results')
    env=os.environ.copy(); env['MCM_OUTPUT_ROOT']=str(output)
    days=['--max-days',str(args.smoke_days)] if args.smoke_days else []
    steps=[['mcm_code.result.problem1.solve_problem1','--time-mapping','A'],['mcm_code.result.problem2.solve_problem2_A'],['mcm_code.result.problem2.solve_problem2_yesterday_dynamic',*days],['mcm_code.result.export_dynamic',*days],['mcm_code.result.problem3.solve_problem3_baseline',*days],['mcm_code.result.problem3.solve_problem3_rolling',*days]]
    if args.smoke_days: steps.pop(1) # the perfect-information P2 annual LP has no bounded-day interface
    for method in ['att1','yest','wd2']:
        steps.append(['mcm_code.result.problem4.solve_p4_2','--price-method',method,*days,*(['--export'] if not days and method=='wd2' else [])])
    for mode,price in [('a','wd2'),('b0','yest'),('bstar','wd2'),('bfreeze','wd2')]:
        steps.append(['mcm_code.result.problem4.solve_p4_3','--mode',mode,'--price-method',price,*days,*(['--export'] if not days and mode=='bfreeze' else [])])
    completed=[]
    for step in steps:
        command=[sys.executable,str(SUPPORT_ROOT/'code/run.py'),*step]; print(' '.join(command),flush=True)
        if not args.dry_run:
            output.mkdir(parents=True,exist_ok=True)
            started=time.monotonic()
            with (output/'run.log').open('a',encoding='utf-8') as log:
                log.write('\nCOMMAND '+ ' '.join(command)+'\n'); log.flush()
                result=subprocess.run(command,cwd=SUPPORT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            completed.append({'module':step[0],'arguments':step[1:],'returncode':result.returncode,'seconds':round(time.monotonic()-started,3)})
            (output/'steps.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2),encoding='utf-8')
            if result.returncode:
                raise SystemExit(f'{step[0]} failed ({result.returncode}); see {output / "run.log"}')
            print(f'Completed {step[0]} in {completed[-1]["seconds"]:.1f}s',flush=True)
    if args.dry_run or days: return
    mapping={'result1.xlsx':'problem1/formal','result2.xlsx':'problem2/formal/dynamic','result3.xlsx':'problem3/formal','result4-2.xlsx':'problem4/formal','result4-3.xlsx':'problem4/formal'}
    for name,directory in mapping.items(): shutil.copy2(output/'result/来源证据'/directory/name,output/'result'/name)
    subprocess.run([sys.executable,str(SUPPORT_ROOT/'code/run.py'),'table.build_tables','--output-dir',str(output/'table'),'--result-dir',str(output/'result')],cwd=SUPPORT_ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(SUPPORT_ROOT/'code/run.py'),'figure.generate_paper_figures','--output-dir',str(output/'figure')],cwd=SUPPORT_ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(SUPPORT_ROOT/'code/run.py'),'check.verify_paper_results','--root',str(output),'--report',str(output/'verification.json')],cwd=SUPPORT_ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(SUPPORT_ROOT/'code/run.py'),'check.verify_tables','--root',str(output/'table'),'--report',str(output/'table-verification.json')],cwd=SUPPORT_ROOT,env=env,check=True)
if __name__=='__main__': main()
