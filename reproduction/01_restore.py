#!/usr/bin/env python3
"""Verify committed inputs and restore a separate, portable analysis workspace."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--destination',required=True,type=Path)
 args=ap.parse_args();dst=args.destination.resolve()
 if dst==REPO or REPO in dst.parents:ap.error('Choose a workspace outside this repository')
 if dst.exists() and any(dst.iterdir()):ap.error('Destination must be empty or new')
 with (HERE/'INPUT_MANIFEST.tsv').open() as f:inputs=list(csv.DictReader(f,delimiter='\t'))
 with (HERE/'DEPENDENCIES.tsv').open() as f:code=list(csv.DictReader(f,delimiter='\t'))
 # Verify every source before materializing anything; duplicate sources hashed once.
 checked={}
 for row in inputs+code:
  source=row.get('source') or row['path'];p=REPO/source
  if not p.is_file():raise FileNotFoundError(p)
  actual=checked.setdefault(source,sha(p))
  if actual!=row['sha256']:raise ValueError('Source SHA256 mismatch: '+source)
  if row.get('bytes') and p.stat().st_size!=int(row['bytes']):raise ValueError('Source size mismatch: '+source)
 archives=[]
 archive_manifest=HERE/'ARCHIVES.json'
 if archive_manifest.is_file():
  for archive in json.loads(archive_manifest.read_text()):
   blocks=[]
   for part in archive['parts']:
    p=REPO/part['path']
    if p.stat().st_size!=part['bytes'] or sha(p)!=part['sha256']:raise ValueError('Archive part mismatch: '+part['path'])
    blocks.append(p.read_bytes())
   payload=b''.join(blocks)
   if len(payload)!=archive['archive_bytes'] or hashlib.sha256(payload).hexdigest()!=archive['archive_sha256']:raise ValueError('Combined archive mismatch')
   mp=REPO/archive['member_manifest']
   if sha(mp)!=archive['member_manifest_sha256']:raise ValueError('Member manifest mismatch')
   members=json.loads(mp.read_text());names=[m['path'] for m in members]
   if len(names)!=len(set(names)):raise ValueError('Duplicate manifest members')
   with zipfile.ZipFile(io.BytesIO(payload)) as z:
    znames=z.namelist()
    if len(znames)!=len(set(znames)) or set(znames)!=set(names) or len(names)!=archive['members']:raise ValueError('Archive member identity mismatch')
    targets=set()
    for member in members:
     target=Path(member['restored_path'])
     if target.is_absolute() or '..' in target.parts or (dst/target).resolve().is_relative_to(dst) is False:raise ValueError('Archive member escapes destination')
     if str(target) in targets:raise ValueError('Duplicate restored archive path')
     targets.add(str(target));content=z.read(member['path'])
     if len(content)!=member['bytes'] or hashlib.sha256(content).hexdigest()!=member['sha256']:raise ValueError('Archive member mismatch: '+member['path'])
   archives.append((archive,payload,members))
 dst.mkdir(parents=True,exist_ok=True);records=[]
 for row in inputs+code:
  source=row.get('source') or row['path'];target=row.get('restored_path') or row['path']
  p=REPO/source;q=dst/target;q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.gz' and q.suffix!='.gz':
   with gzip.open(p,'rb') as a,q.open('wb') as b:shutil.copyfileobj(a,b)
  else:shutil.copy2(p,q)
  if q.suffix in {'.py','.sh','.yaml','.yml'}:
   s=q.read_text().replace('__WORKSPACE__',str(dst))
   # These ROOT-relative modules retain both historical input prefixes as
   # remapping keys. Preserve their bytes and the locked selection-script hash.
   if q.name not in {'resubmission7_holdout_config.py','282_select_resubmission7_holdout.py'}:
    s=s.replace('/media/Data_1/tianrm/projects/magicc2',str(dst))
   q.write_text(s)
  records.append(dict(path=target,source=source,source_sha256=row['sha256'],restored_sha256=sha(q)))
 for archive,payload,members in archives:
  with zipfile.ZipFile(io.BytesIO(payload)) as z:
   for member in members:
    q=dst/member['restored_path'];q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists() and sha(q)!=member['sha256']:raise ValueError('Archive/input identity conflict: '+member['restored_path'])
    q.write_bytes(z.read(member['path']))
    records.append(dict(path=member['restored_path'],source=archive['name']+':'+member['path'],source_sha256=member['sha256'],restored_sha256=sha(q)))
 manifest=dict(status='RESTORED',input_manifest_sha256=sha(HERE/'INPUT_MANIFEST.tsv'),dependency_manifest_sha256=sha(HERE/'DEPENDENCIES.tsv'),files=records)
 (dst/'RESTORE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(f'Verified {len(checked)} source files; restored {len(records)} paths to {dst}')

if __name__=='__main__':main()
