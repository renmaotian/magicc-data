#!/usr/bin/env python3
"""Generate full V5-recipe matched arms with auditable, atomic batch artifacts.

Reuses script 121's frozen generators, planners, validity filters and top-ups.
Differences: actual reselected features, per-sample contaminant identities, seeded
normalization fitted only on training data, raw immutable batch checkpoints and
atomic finalization (never normalizes already-normalized data after interruption).
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import h5py
import numpy as np
from resubmission5_holdout_config import OUT,configure,load_script

M=None
ORIGINAL={}
TRACE=[]

class OrderedPool:
    """Preserve top-up acceptance order even when underlying workers finish out of order."""
    def __init__(self,pool):self.pool=pool
    def imap_unordered(self,fn,plans,chunksize):return self.pool.imap(fn,plans,chunksize)

def traced_work(args,kind):
    r=ORIGINAL[kind](args)
    if r is None:return None
    if kind=='v4':
        sid,stype,di,cis,tcomp,tcont,qt,seed=args
    elif kind=='A':
        sid,di,seed=args;cis=[];tcomp=1.;tcont=0.;qt='high'
    else:
        sid,di,cis,tcont,ctype,seed=args;tcomp=1.;qt='high'
    trace={'sample_plan_id':int(sid),'seed':int(seed),
           'dominant_accession':M._gi.all_genomes[di]['accession'],
           'contaminant_accessions':[M._gi.all_genomes[i]['accession'] for i in (cis or [])],
           'contaminant_phyla':[M._gi.all_genomes[i]['phylum'] for i in (cis or [])],
           'target_completeness':float(tcomp)*100,'target_contamination':float(tcont),
           'quality_tier':str(qt),'observed_completeness':float(r[3]),'observed_contamination':float(r[4])}
    return r+(trace,)
def work_v4(args):return traced_work(args,'v4')
def work_A(args):return traced_work(args,'A')
def work_B(args):return traced_work(args,'B')

def traced_assemble(results,batch_tag,logger):
    global TRACE
    # imap_unordered completion order must not influence persisted data order.
    results=sorted(results,key=lambda r:(r[0],r[-1]['seed']))
    TRACE=[r[-1] for r in results]
    return ORIGINAL['assemble']([r[:-1] for r in results],batch_tag,logger)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)

def main():
    global M,ORIGINAL
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['holdout','matched_full'],required=True)
    ap.add_argument('--workers',type=int,default=12);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();c=configure(args.variant)
    if args.smoke:
        c.HOLDOUT_H5=c.HOLDOUT_DIR/'smoke_features.h5'
        c.HOLDOUT_NORM_PARAMS=c.HOLDOUT_DIR/'smoke_normalization_params.json'
    M=load_script('121','r5_build_base')
    ORIGINAL={'v4':M.work_v4,'A':M.work_A,'B':M.work_B,'assemble':M.assemble}
    M.work_v4=work_v4;M.work_A=work_A;M.work_B=work_B;M.assemble=traced_assemble
    logger=M.setup_logging(args.smoke)
    dest=c.HOLDOUT_DIR/('smoke_batches' if args.smoke else 'raw_batches');dest.mkdir(parents=True,exist_ok=True)
    M.init_kmer_globals();np.random.seed(42)
    gis={s:M.GenomeIndex(c.HOLDOUT_DIR/f'{s}_genomes_holdout.tsv',logger) for s in ('train','val','test')}
    for s,gi in gis.items():
        expected=__import__('pandas').read_csv(c.HOLDOUT_DIR/f'{s}_genomes_holdout.tsv',sep='\t')
        assert len(gi.all_genomes)==len(expected), f'Missing FASTAs in {s}: expected {len(expected)}, found {len(gi.all_genomes)}'
        assert set(x['accession'] for x in gi.all_genomes)==set(expected.ncbi_accession)
    # Shared split safety applies to contaminants as well as dominants.
    accession_sets={s:set(x['accession'] for x in gi.all_genomes) for s,gi in gis.items()}
    assert not(accession_sets['train']&accession_sets['val'] or accession_sets['train']&accession_sets['test'] or accession_sets['val']&accession_sets['test'])
    if args.smoke:
        types={k:max(1,v//100) for k,v in c.SAMPLE_TYPES.items()};n=sum(types.values())
        queue=[('v4',0,'train',0),('A',0,'train',n),('B',0,'train',2*n),('v4',80,'val',0),('v4',90,'test',0)]
        sizes={'train':3*n,'val':n,'test':n}
    else:
        n=10000;types=None
        queue=([('v4',b,s,o) for b,s,o in M.V4_ASSIGN]+[('A',b,s,o) for b,s,o in M.A_ASSIGN]+[('B',b,s,o) for b,s,o in M.B_ASSIGN])
        sizes={'train':1000000,'val':100000,'test':100000}
    pool=None;cur=None;t0=time.time();completed=[]
    try:
        for kind,bid,split,offset in queue:
            path=dest/f'{kind}_{bid:03d}_{split}.npz';tracepath=path.with_suffix('.jsonl.gz')
            if path.exists() and tracepath.exists():
                with np.load(path) as z:
                    assert z['labels'].shape==(n,2) and z['metadata'].shape==(n,)
                    expected_acc=[x.decode() for x in z['metadata']['dominant_accession']]
                with gzip.open(tracepath,'rt') as f:traces=[json.loads(line) for line in f]
                assert len(traces)==n and [r['dominant_accession'] for r in traces]==expected_acc
                assert all(r['row']==offset+i and r['split']==split for i,r in enumerate(traces))
                assert all(r['dominant_accession'] in accession_sets[split] and set(r['contaminant_accessions'])<=accession_sets[split] for r in traces)
                completed.append(str(path));continue
            if cur!=split:
                if pool is not None:pool.terminate();pool.join()
                M._gi=gis[split];pool=mp.Pool(args.workers);cur=split
            kc,asm,lab,md,elapsed=M.run_batch(kind,bid,split,offset,gis[split],OrderedPool(pool),args.workers,logger,n_target=n,types=types)
            assert len(lab)==n
            allowed=accession_sets[split]
            assert all(r['dominant_accession'] in allowed and set(r['contaminant_accessions'])<=allowed for r in TRACE)
            temp=Path(str(path)+'.tmp')
            with open(temp,'wb') as f:np.savez(f,kmer=kc.astype(np.int32),assembly=asm.astype(np.float64),labels=lab,metadata=md)
            os.replace(temp,path)
            temp=Path(str(tracepath)+'.tmp')
            with gzip.open(temp,'wt') as f:
                for i,r in enumerate(TRACE):f.write(json.dumps({'split':split,'row':offset+i,'batch':f'{kind}:{bid}',**r})+'\n')
            os.replace(temp,tracepath);completed.append(str(path))
            atomic_json(dest/'progress.json',{'variant':args.variant,'done_batches':len(completed),'total_batches':len(queue),
                                              'elapsed_seconds':time.time()-t0,'last_batch_seconds':elapsed,
                                              'remaining_hours_estimate':elapsed*(len(queue)-len(completed))/3600,
                                              'last_batch':str(path),'normalization':'not_started'})
    finally:
        if pool is not None:pool.terminate();pool.join()
    # Recompute training-only normalization from immutable raw batches on resume.
    # Exact Welford moments, exact robust medians/IQR across all training summaries.
    # This avoids the legacy reservoir's unseeded/biased quantile sampling.
    from holdout_lib.normalization import FeatureNormalizer,LOG10_INDICES
    nrm=FeatureNormalizer(n_kmer_features=c.N_KMER_FEATURES,reservoir_size=1)
    count=0;mean=np.zeros(c.N_KMER_FEATURES);m2=np.zeros_like(mean);summaries=[]
    for kind,bid,split,offset in queue:
        if split!='train':continue
        with np.load(dest/f'{kind}_{bid:03d}_{split}.npz') as z:
            x=np.log1p(z['kmer'].astype(np.float64));b=len(x);bm=x.mean(axis=0);bd=x-bm;bm2=(bd*bd).sum(axis=0)
            delta=bm-mean;new=count+b;m2+=bm2+delta*delta*(count*b/max(new,1));mean+=delta*(b/max(new,1));count=new
            summaries.append(z['assembly'])
    assert count==sizes['train'];summary=np.concatenate(summaries)
    transformed=summary.copy()
    for idx in LOG10_INDICES:transformed[:,idx]=np.log10(transformed[:,idx]+1.)
    nrm.kmer_stats.count=count;nrm.kmer_stats.mean=mean;nrm.kmer_stats.m2=m2
    nrm.assembly_stats.count=count;nrm.assembly_stats.mean=transformed.mean(axis=0)
    nrm.assembly_stats.m2=((transformed-nrm.assembly_stats.mean)**2).sum(axis=0)
    nrm.assembly_stats.min_vals=transformed.min(axis=0);nrm.assembly_stats.max_vals=transformed.max(axis=0)
    nrm.assembly_stats.reservoir=transformed;nrm.assembly_stats.reservoir_count=count;nrm.assembly_stats.reservoir_size=count
    nrm.finalize()
    # Save only meaningful fitted parameters; k-mer quantiles are not estimated
    # or used, so do not emit placeholders or infinite min/max values.
    params={'n_kmer_features':c.N_KMER_FEATURES,'n_assembly_features':7,'finalized':True,
            'kmer_stats':{'count':count,'mean':mean.tolist()},
            'assembly_stats':{'count':count,'mean':transformed.mean(axis=0).tolist()},
            'fit_split':'train_only','summary_quantiles':'exact_all_training_samples'}
    for name in ['kmer_mean','kmer_std','assembly_minmax_min','assembly_minmax_range','assembly_robust_median','assembly_robust_iqr']:
        params[name]=getattr(nrm,name).tolist()
    params['assembly_log10_offset']=1.
    atomic_json(c.HOLDOUT_NORM_PARAMS,params)
    temp=Path(str(c.HOLDOUT_H5)+'.tmp')
    with h5py.File(temp,'w') as f:
        for split,total in sizes.items():
            g=f.create_group(split)
            g.create_dataset('kmer_features',(total,c.N_KMER_FEATURES),dtype='f4')
            g.create_dataset('assembly_features',(total,7),dtype='f4')
            g.create_dataset('labels',(total,2),dtype='f4')
            g.create_dataset('metadata',(total,),dtype=M.METADATA_DTYPE)
        for kind,bid,split,offset in queue:
            with np.load(dest/f'{kind}_{bid:03d}_{split}.npz') as z:
                end=offset+len(z['labels']);g=f[split]
                g['kmer_features'][offset:end]=nrm.normalize_kmer(z['kmer']).astype('f4')
                g['assembly_features'][offset:end]=nrm.normalize_assembly(z['assembly']).astype('f4')
                g['labels'][offset:end]=z['labels'];g['metadata'][offset:end]=z['metadata']
        f.attrs['normalized']=True;f.attrs['normalization_fit_split']='train_only_all_1000000'
        f.attrs['panel_taxa']=json.dumps(c.PANEL_TAXA);f.attrs['panel_level']='family'
        f.attrs['features_sha256']=sha(c.SELECTED_KMERS);f.attrs['variant']=args.variant
    os.replace(temp,c.HOLDOUT_H5)
    manifest={'variant':args.variant,'smoke':args.smoke,'samples':sizes,'n_kmer_features':c.N_KMER_FEATURES,
              'feature_sha256':sha(c.SELECTED_KMERS),'normalization_sha256':sha(c.HOLDOUT_NORM_PARAMS),
              'features_h5_sha256':sha(c.HOLDOUT_H5),'wall_hours':(time.time()-t0)/3600,
              'normalization':'training-only exact moments/summary quantiles; no validation or test rows',
              'status':'DATA_COMPLETE','role_leakage':'All actual dominant/contaminant accessions audited against the arm-specific split pool; holdout excludes the panel, matched_full includes it.',
              'raw_batches':[{'path':str(dest/f'{k}_{b:03d}_{s}.npz'),'sha256':sha(dest/f'{k}_{b:03d}_{s}.npz')} for k,b,s,o in queue]}
    atomic_json(c.HOLDOUT_DIR/('smoke_build_manifest.json' if args.smoke else 'build_manifest.json'),manifest)
    logger.info('DATA COMPLETE %s',json.dumps({k:manifest[k] for k in ('samples','n_kmer_features','wall_hours')}))

if __name__=='__main__':
    mp.set_start_method('fork',force=True);main()
