#!/usr/bin/env python3
"""Manifest versioned genomic inputs and restore missing FASTAs from NCBI FTP.

The manifest command reads metadata and stats genomic files; it does not read
all genomic sequences. Core-gene input FASTAs are small and are fully hashed.
The fetch command downloads only missing paths, verifies NCBI's compressed MD5,
then records the restored uncompressed SHA256. Never substitutes an accession
version, GenBank/RefSeq representation, or another genome after a failed URL.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/holdout_resubmission5/reconstruction'

def sha(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def write(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def relative(path):
    text=str(path)
    marker='/data/'
    if marker not in text:raise ValueError(f'Unexpected archived input path: {text}')
    return 'data/'+text.split(marker,1)[1]

def manifests():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];seen=set()
    for split in ['train','val','test']:
        for row in read(ROOT/f'data/splits/{split}_genomes.tsv'):
            path=relative(row['fasta_path']);p=ROOT/path
            name=p.name
            if not name.endswith('_genomic.fna'):raise ValueError(name)
            folder=name[:-len('_genomic.fna')]
            match=re.match(r'^(GC[AF])_(\d{9})\.(\d+)_',folder)
            if not match:raise ValueError(folder)
            prefix,digits,version=match.groups();accession=f'{prefix}_{digits}.{version}'
            if path in seen:raise ValueError(f'Duplicate split FASTA: {path}')
            seen.add(path)
            url='https://ftp.ncbi.nlm.nih.gov/genomes/all/'+'/'.join([prefix,digits[:3],digits[3:6],digits[6:],folder,name+'.gz'])
            rows.append({'split':split,'gtdb_accession':row['gtdb_accession'],'ncbi_accession':row['ncbi_accession'],
                         'download_accession':accession,'fasta_relative_path':path,'genomic_fasta_url':url,
                         'metadata_genome_size':row['genome_size'],'original_file_bytes':p.stat().st_size if p.exists() else '',
                         'original_genomic_sha256':'','hash_note':'Genomic byte hashes not calculated by metadata-only manifest.'})
    write(OUT/'genomic_source_manifest.tsv',rows)
    cores=[]
    for domain in ['bacterial','archaeal']:
        representatives={r['ncbi_accession'] for r in read(ROOT/f'data/kmer_selection/selected_{domain}_1000.tsv')}
        for row in read(ROOT/f'data/kmer_selection/{domain}_core_genes/core_gene_results.tsv'):
            if row['accession'] not in representatives:continue
            path=relative(row['core_gene_path']);p=ROOT/path
            cores.append({'domain':domain,'accession':row['accession'],'relative_path':path,
                          'n_core_genes':row['n_core_genes'],'file_bytes':p.stat().st_size,'sha256':sha(p)})
    assert len(cores)==2000
    write(OUT/'core_gene_input_manifest.tsv',cores)
    inputs=[ROOT/f'data/splits/{s}_genomes.tsv' for s in ['train','val','test']]
    inputs.extend(ROOT/f'data/kmer_selection/{name}' for name in ['selected_kmers.txt','selected_bacterial_1000.tsv',
        'selected_archaeal_1000.tsv','bacterial_kmer_prevalence.tsv','archaeal_kmer_prevalence.tsv',
        'bacterial_core_genes/core_gene_results.tsv','archaeal_core_genes/core_gene_results.tsv'])
    write(OUT/'metadata_input_manifest.tsv',[{'relative_path':str(p.relative_to(ROOT)),'file_bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs])
    summary={'genomic_files':len(rows),'genomic_file_bytes':sum(int(r['original_file_bytes'] or 0) for r in rows),
             'core_gene_files':len(cores),'core_gene_file_bytes':sum(r['file_bytes'] for r in cores),
             'genomic_byte_hashes_calculated':False,'script_dependencies':'Python 3 standard library only.',
             'download_policy':'Exact archived FASTA filename and version; compressed MD5 from the same NCBI assembly directory; no substitution.',
             'NCBI_reference':'https://www.ncbi.nlm.nih.gov/datasets/docs/v2/data-processing/policies-annotation/genomeftp/'}
    (OUT/'reconstruction_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

def fetch(args):
    records=read(args.manifest)
    if args.split:records=[r for r in records if r['split']==args.split]
    if args.limit:records=records[:args.limit]
    base=Path(args.destination_root).resolve();log=base/'results/revision/holdout_resubmission5/reconstruction/download_receipts.jsonl'
    log.parent.mkdir(parents=True,exist_ok=True)
    errors=0
    for i,row in enumerate(records):
        p=(base/row['fasta_relative_path']).resolve()
        if not p.is_relative_to(base):raise ValueError('Manifest path escapes destination.')
        if p.exists():
            if row['original_file_bytes'] and p.stat().st_size!=int(row['original_file_bytes']):
                raise ValueError(f'Existing file size mismatch, inspect manually: {p}')
            if row['original_genomic_sha256'] and sha(p)!=row['original_genomic_sha256']:
                raise ValueError(f'Existing file SHA256 mismatch, inspect manually: {p}')
            continue
        url=row['genomic_fasta_url'];p.parent.mkdir(parents=True,exist_ok=True)
        compressed=Path(str(p)+'.download.gz');temporary=Path(str(p)+'.download.tmp')
        receipt={'accession':row['download_accession'],'url':url,'relative_path':row['fasta_relative_path']}
        try:
            checksum_url=url.rsplit('/',1)[0]+'/md5checksums.txt'
            checks=urllib.request.urlopen(checksum_url,timeout=90).read().decode()
            expected=[line.split()[0] for line in checks.splitlines() if line.split()[-1].removeprefix('./')==url.rsplit('/',1)[1]]
            if len(expected)!=1:raise ValueError('No unique NCBI MD5 entry for exact genomic FASTA.')
            with urllib.request.urlopen(url,timeout=120) as response,open(compressed,'wb') as output:shutil.copyfileobj(response,output)
            observed=sha(compressed,'md5')
            if observed!=expected[0]:raise ValueError('NCBI compressed MD5 mismatch.')
            with gzip.open(compressed,'rb') as source,open(temporary,'wb') as output:shutil.copyfileobj(source,output)
            with open(temporary,'rb') as f:
                if f.read(1)!=b'>':raise ValueError('Downloaded content is not FASTA.')
            if row['original_file_bytes'] and temporary.stat().st_size!=int(row['original_file_bytes']):
                raise ValueError('Restored byte count differs from original; inspect before reproducing.')
            restored_sha=sha(temporary)
            if row['original_genomic_sha256'] and restored_sha!=row['original_genomic_sha256']:
                raise ValueError('Restored SHA256 differs from original; inspect before reproducing.')
            receipt.update(status='RESTORED',compressed_md5=observed,restored_sha256=restored_sha,restored_bytes=temporary.stat().st_size)
            os.replace(temporary,p)
        except Exception as e:
            errors+=1;receipt.update(status='FAILED',error=str(e))
        finally:
            for q in [compressed,temporary]:
                if q.exists():q.unlink()
        with open(log,'a') as f:f.write(json.dumps(receipt)+'\n')
        print(f'{i+1}/{len(records)} {receipt["accession"]}: {receipt["status"]}',flush=True)
        time.sleep(.35)
    if errors:raise SystemExit(f'{errors} exact-version retrievals failed; see {log}. No substitutes were used.')

def hash_genomes():
    manifest=OUT/'genomic_source_manifest.tsv';records=read(manifest)
    journal=OUT/'genomic_hashes.jsonl';known={}
    if journal.exists():
        with open(journal) as f:
            for line in f:
                if line.strip():
                    r=json.loads(line);known[r['relative_path']]=r
    started=time.time();read_bytes=0
    with open(journal,'a') as log:
        for i,row in enumerate(records):
            path=row['fasta_relative_path'];p=ROOT/path
            if path not in known:
                actual=p.stat().st_size
                if actual!=int(row['original_file_bytes']):raise ValueError(f'Input size changed: {p}')
                r={'relative_path':path,'bytes':actual,'sha256':sha(p)}
                known[path]=r;log.write(json.dumps(r)+'\n');read_bytes+=actual
                if (i+1)%100==0:log.flush()
            row['original_genomic_sha256']=known[path]['sha256'];row['hash_note']='Original FASTA byte SHA256 verified by full sequence-file read.'
            if (i+1)%2000==0:print(json.dumps({'files':i+1,'total':len(records),'new_bytes_read':read_bytes,'elapsed_seconds':time.time()-started}),flush=True)
    temporary=Path(str(manifest)+'.tmp');write(temporary,records);os.replace(temporary,manifest)
    summary=json.loads((OUT/'reconstruction_summary.json').read_text());summary['genomic_byte_hashes_calculated']=True
    summary['genomic_sha256_manifest_sha256']=sha(manifest)
    (OUT/'reconstruction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'COMPLETE: hashed {len(records)} original genomic FASTAs.',flush=True)

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
    sp.add_parser('manifest')
    sp.add_parser('hash')
    p=sp.add_parser('fetch');p.add_argument('--manifest',type=Path,default=OUT/'genomic_source_manifest.tsv')
    p.add_argument('--destination-root',type=Path,default=ROOT);p.add_argument('--split',choices=['train','val','test'])
    p.add_argument('--limit',type=int,help='Optional bounded verification download.')
    args=ap.parse_args()
    if args.command=='manifest':manifests()
    elif args.command=='hash':hash_genomes()
    else:fetch(args)

if __name__=='__main__':main()
