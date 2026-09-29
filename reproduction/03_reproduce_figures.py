#!/usr/bin/env python3
"""Render scientific figures from the restored, verified numerical sources."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DRIVERS=['01_checkm2_domain_analysis.py','make_figures_1_3.py','make_figure_4.py','make_figure_5.py','make_supp_figures.py']
TARGETS={'01_checkm2_domain_analysis.py':[],
 'make_figures_1_3.py':['Figure_1','Figure_2','Figure_3'],
 'make_figure_4.py':['Figure_4'],'make_figure_5.py':['Figure_5'],
 'make_supp_figures.py':[f'Figure_S{i}' for i in range(1,9)]}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace',type=Path,required=True)
 ap.add_argument('--drivers',nargs='+',choices=DRIVERS,default=DRIVERS);args=ap.parse_args()
 root=args.workspace.resolve();base=root/'nature_communications/resubmission7';audit=root/'reproduction_audit/figures';audit.mkdir(parents=True,exist_ok=True)
 assert (root/'RESTORE_MANIFEST.json').is_file(),'Run 01_restore.py first'
 env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(audit/'matplotlib'),MAGICC_ARTIST_LEDGER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 records=[];declared=[];absent=[]
 for name in args.drivers:
  for target in TARGETS[name]:
   output=base/('supp_figures' if target.startswith('Figure_S') else 'figures')
   output.mkdir(parents=True,exist_ok=True)
   for suffix in ['.png','.pdf','.svg']:
    (output/(target+suffix)).unlink(missing_ok=True)
   ledger=output/'_values'/(target+'_artists.tsv');ledger.unlink(missing_ok=True)
   declared.extend([output/(target+'.png'),output/(target+'.pdf'),ledger])
   if target=='Figure_2':declared.append(output/(target+'.svg'))
   absent.extend([output/(target+suffix) for suffix in ['.png','.pdf','.svg']]+[ledger])
 assert all(not p.exists() for p in absent),'Declared output survived pre-run removal'
 started_ns=time.time_ns()
 for name in args.drivers:
  script=base/'scripts'/name;command=[sys.executable,str(script)]
  with (audit/(Path(name).stem+'.log')).open('w') as f:
   process=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  records.append(dict(script=name,sha256=hashlib.sha256(script.read_bytes()).hexdigest(),command=command,return_code=process.returncode))
  if process.returncode:raise RuntimeError(f'{name} failed; see {audit}')
 for p in declared:
  assert p.is_file() and p.stat().st_size>0,f'Figure output missing: {p}'
  assert p.stat().st_mtime_ns>=started_ns,f'Figure output was not newly created: {p}'
 outputs=[]
 for directory in ['figures','supp_figures']:
  for p in sorted((base/directory).rglob('*')):
   if p.is_file():outputs.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 repo=Path(__file__).resolve().parents[1];selected={target for name in args.drivers for target in TARGETS[name]};comparisons=[]
 with (repo/'reproduction/FIGURE_MANIFEST.tsv').open() as f:expected=list(csv.DictReader(f,delimiter='\t'))
 for record in expected:
  name=Path(record['output_path']).name.removesuffix('_artists.tsv')
  if name not in selected:continue
  actual=base/record['output_path'];reference=repo/record['reference_path']
  assert hashlib.sha256(reference.read_bytes()).hexdigest()==record['sha256'],'Reference artist ledger changed'
  assert actual.read_bytes()==reference.read_bytes(),f'Numerical artist values differ: {actual}'
  comparisons.append(dict(figure=name,records=int(record['records']),sha256=record['sha256'],status='EXACT'))
 assert len(comparisons)==len(selected),'Missing numerical artist reference for selected figure'
 report=dict(status='PASS',declared_outputs_absent_before_run=True,started_ns=started_ns,newly_created_declared_outputs=[str(p.relative_to(root)) for p in declared],commands=records,outputs=outputs,artist_comparisons=comparisons,scope='Scientific figures and numerical artist ledgers; canonical journal captions and document rendering are separate author materials.')
 (audit/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f'Rendered {len(records)} drivers; recorded {len(outputs)} files under {base/"figures"}')

if __name__=='__main__':main()
