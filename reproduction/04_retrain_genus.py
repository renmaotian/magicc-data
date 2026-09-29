#!/usr/bin/env python3
"""Run the full genus-exclusion experiment using its documented source inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PRODUCTION_SHA='b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'

def archive_deposited_metadata(root,step):
 """Preserve reference-only run records before fresh raw reproduction.

 A deposited completion certificate is evidence about the published run, not
 evidence that its large raw files or research weights exist in this workspace.
 Never move records once raw/checkpoint state or changed metadata is present.
 """
 experiment=root/'results/revision/holdout_resubmission7'
 restore=json.loads((root/'RESTORE_MANIFEST.json').read_text())
 identities={r['path']:r['restored_sha256'] for r in restore['files']}
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 groups=[]
 if step=='train':
  for arm in ['holdout','matched_full']:
   base=experiment/arm;data=base/'data';models=base/'models/full'
   names=[data/n for n in ['build_manifest.json','normalization_params.json','donor_count_audit.json']]
   names += [models/n for n in ['completion.json','training_config.json','training_history.json']]
   raw=data/'raw_batches'
   extra=[p for p in models.rglob('*') if p.is_file() and p not in names]
   unknown=any(p.suffix not in {'.pt','.onnx'} for p in extra)
   active=(raw.exists() and any(raw.iterdir())) or any(data.glob('*.h5')) or bool(extra)
   state='EXTRA_STATE_PRESERVED' if unknown else 'GENERATED_STATE_PRESERVED' if active else None
   groups.append((arm,names,state))
 evaluation=experiment/'evaluation'
 names=[evaluation/n for n in ['preparation_status.json','verification_manifest.tsv']]
 extra=[p for p in evaluation.iterdir() if p not in names] if evaluation.exists() else []
 state=None
 if extra:
  certificate=evaluation/'preparation_status.json'
  prior=json.loads(certificate.read_text()) if certificate.is_file() else {}
  artifacts=prior.get('artifact_sha256',{})
  if prior.get('status')=='EVALUATION_INPUTS_COMPLETE' and artifacts and all((evaluation/p).is_file() for p in artifacts):
   state='GENERATED_STATE_PRESERVED'
  elif prior.get('status')=='PREPARING':
   state='GENERATED_STATE_PRESERVED'
  else:state='INCOMPLETE_OR_EXTRA_EVALUATION_STATE_PRESERVED'
 groups.append(('evaluation',names,state))
 records=[]
 for group,names,state in groups:
  present=[p for p in names if p.is_file()]
  if state:
   records.append(dict(group=group,status=state));continue
  if not present:
   records.append(dict(group=group,status='NO_DEPOSITED_RECORDS'));continue
  matched=all(identities.get(str(p.relative_to(root)))==sha(p) for p in present)
  if not matched:
   records.append(dict(group=group,status='CHANGED_METADATA_PRESERVED'));continue
  moves=[]
  for source in present:
   target=experiment/'recorded_original_provenance'/source.relative_to(experiment)
   if target.exists() and sha(target)!=sha(source):raise ValueError('Archived provenance conflict: '+str(target))
  for source in present:
   digest=sha(source);target=experiment/'recorded_original_provenance'/source.relative_to(experiment)
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():source.unlink()  # Same bytes already archived after an interrupted move.
   else:source.rename(target)
   assert sha(target)==digest
   moves.append(dict(original_path=str(source.relative_to(root)),archive_path=str(target.relative_to(root)),sha256=digest))
  records.append(dict(group=group,status='DEPOSITED_REFERENCE_RECORDS_ARCHIVED',files=moves))
 audit=root/'reproduction_audit/genus';audit.mkdir(parents=True,exist_ok=True)
 path=audit/'deposited_metadata_archive.json'
 history=json.loads(path.read_text()) if path.exists() else []
 history.append(dict(step=step,groups=records,scope='Reference metadata only; no generated raw data, model weights or checkpoints moved.'))
 path.write_text(json.dumps(history,indent=2)+'\n')
 return records

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace',required=True,type=Path)
 ap.add_argument('--step',required=True,choices=['fetch','prepare','train','evaluate'])
 ap.add_argument('--workers',type=int,default=8);ap.add_argument('--production-model',type=Path)
 ap.add_argument('--fetch-limit',type=int,help='Optional bounded real retrieval test; omit for all 99,957 genomes')
 args=ap.parse_args();root=args.workspace.resolve()
 if not (root/'RESTORE_MANIFEST.json').is_file():ap.error('Run 01_restore.py first')
 audit=root/'reproduction_audit/genus';audit.mkdir(parents=True,exist_ok=True)
 if args.production_model:
  source=args.production_model.resolve()
  if hashlib.sha256(source.read_bytes()).hexdigest()!=PRODUCTION_SHA:ap.error('Production model hash mismatch')
  target=root/'models/magicc_v5.onnx';target.parent.mkdir(parents=True,exist_ok=True)
  if source!=target:shutil.copy2(source,target)
 scripts=root/'scripts';commands=[]
 if args.step=='fetch':
  command=[sys.executable,str(scripts/'250_reconstruct_resubmission5_inputs.py'),'fetch']
  if args.fetch_limit:command.extend(['--limit',str(args.fetch_limit)])
  commands=[command]
 elif args.step=='prepare':
  commands=[[sys.executable,str(scripts/'282_select_resubmission7_holdout.py')],
            [sys.executable,str(scripts/'283_reselect_resubmission7_features.py'),'--workers',str(args.workers)]]
 elif args.step=='train':
  model=root/'models/magicc_v5.onnx'
  if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest()!=PRODUCTION_SHA:ap.error('Supply the verified released model using --production-model')
  commands=[['bash',str(scripts/'288_run_resubmission7_holdout.sh')]]
 else:
  commands=[[sys.executable,str(scripts/'286_generate_resubmission7_holdout_eval.py'),'--workers',str(args.workers)],
            [sys.executable,str(scripts/'287_evaluate_resubmission7_holdout.py'),'--threads',str(args.workers)]]
 metadata_archive=archive_deposited_metadata(root,args.step) if args.step in {'train','evaluate'} else []
 if any(r['status'] in {'CHANGED_METADATA_PRESERVED','EXTRA_STATE_PRESERVED','INCOMPLETE_OR_EXTRA_EVALUATION_STATE_PRESERVED'} for r in metadata_archive):
  ap.error('Changed metadata or incompatible partial state preserved; use a separate fresh restoration for reproduction. No training/evaluation command was launched.')
 records=[];env=dict(os.environ,MAGICC_R7_PYTHON=sys.executable,PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1')
 for i,command in enumerate(commands):
  log=audit/f'{args.step}_{i+1:02d}.log'
  with log.open('w') as f:process=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  records.append(dict(command=command,return_code=process.returncode,log=str(log.relative_to(root))))
  (audit/(args.step+'.json')).write_text(json.dumps(dict(status='PASS' if process.returncode==0 else 'FAILED',commands=records,metadata_archive=metadata_archive),indent=2)+'\n')
  if process.returncode:raise RuntimeError(f'Genus {args.step} failed; see {log}')
 print(f'Genus {args.step}: completed {len(commands)} commands; see {audit}')

if __name__=='__main__':main()
