#!/usr/bin/env python3
"""Inference-only sensitivity to excluding held-out data from legacy normalization.

Recover 800k V4-training k-mer moments by subtracting 200k val/test moments
from the known full-data standardized moments (N=1M; ddof=1). All transforms
are positive affine; summaries use exact training min/max and quantiles. This
is NOT a retrained-model leakage correction. Frozen V5 weights never change.
"""
import argparse,hashlib,json,os,sys,time,zlib
from pathlib import Path
from multiprocessing import Pool
import h5py,numpy as np,pandas as pd,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from magicc.normalization import FeatureNormalizer,MINMAX_INDICES,ROBUST_INDICES
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
OUT=ROOT/'results/revision/normalization_sensitivity_resubmission5'
H5=ROOT/'data/features/magicc_features.h5'
NORM=ROOT/'data/features/normalization_params.json'
SETS=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E']
MODEL=ROOT/'models/magicc_v5.onnx'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def savenpz(path,compressed=False,**arrays):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        (np.savez_compressed if compressed else np.savez)(f,**arrays)
    os.replace(temp,path)
def moments():
    norm=json.loads(NORM.read_text());assert norm['kmer_stats']['count']==1000000
    oldstd=np.asarray(norm['kmer_std']);oldmean=np.asarray(norm['kmer_mean'])
    assert np.min(oldstd)>1e-10 and np.array_equal(oldstd,np.asarray(norm['kmer_stats']['std']))
    assert np.array_equal(oldmean,np.asarray(norm['kmer_stats']['mean']))
    cache=OUT/'moment_chunks';cache.mkdir(exist_ok=True)
    identity={'normalizer_sha256':sha(NORM),'source_hdf_size':H5.stat().st_size,'source_hdf_mtime_ns':H5.stat().st_mtime_ns}
    if (cache/'source_identity.json').exists():
        assert json.loads((cache/'source_identity.json').read_text())==identity
    else:savejson(cache/'source_identity.json',identity)
    with h5py.File(H5,'r') as f:
        ntrain=f['train/kmer_features'].shape[0];nval=f['val/kmer_features'].shape[0];ntest=f['test/kmer_features'].shape[0]
        assert (ntrain,nval,ntest)==(800000,100000,100000)
        sh=np.zeros(9249);qh=np.zeros(9249)
        for split in ['val','test']:
            for start in range(0,100000,10000):
                path=cache/f'{split}_{start:06d}.npz'
                if not path.exists():
                    t=time.monotonic();x=f[f'{split}/kmer_features'][start:start+10000].astype('f8')
                    savenpz(path,sum=x.sum(axis=0),sumsq=np.einsum('ij,ij->j',x,x),n=len(x))
                    print(f'{split} {start+len(x)}/100000: {time.monotonic()-t:.1f}s',flush=True)
                c=np.load(path);assert int(c['n'])==10000;sh+=c['sum'];qh+=c['sumsq']
        # Exact before storage rounding: all1M normalized rows sum0, square-sumN-1.
        trainmean=-sh/ntrain;trainsumsq=999999.-qh
        trainvar=(trainsumsq-ntrain*trainmean**2)/(ntrain-1)
        assert np.min(trainvar)>0
        trainstd=np.sqrt(trainvar)
        # Verify the old-normalizer fingerprint against integer count inversion.
        z=f['train/kmer_features'][:32].astype('f8')
        raw=np.expm1(z*oldstd+oldmean);inverse_error=float(np.max(np.abs(raw-np.rint(raw))))
        assert inverse_error<.02,inverse_error
        a=f['train/assembly_features'][:].astype('f8')
    amin=a.min(axis=0);amax=a.max(axis=0);amed=np.median(a,axis=0);aq=np.quantile(a,[.25,.75],axis=0);aiqr=aq[1]-aq[0]
    oldrange=np.asarray(norm['assembly_minmax_range']);oldiqr=np.asarray(norm['assembly_robust_iqr'])
    new_raw_range=oldrange*(amax-amin);new_raw_range[new_raw_range<=1e-10]=1.
    new_raw_iqr=oldiqr*aiqr;new_raw_iqr[new_raw_iqr<1e-10]=1.
    rawkstd=oldstd*trainstd;rawkstd[rawkstd<1e-10]=1.
    params=dict(n_train=ntrain,n_held_out=nval+ntest,old_normalizer_sha256=sha(NORM),
        kmer_mean_z=trainmean.tolist(),kmer_scale_z=(rawkstd/oldstd).tolist(),
        summary_min_z=amin.tolist(),summary_minmax_scale_z=(new_raw_range/oldrange).tolist(),
        summary_median_z=amed.tolist(),summary_robust_scale_z=(new_raw_iqr/oldiqr).tolist(),
        old_normalizer_integer_count_inversion_max_error=inverse_error,
        kmer_mean_z_abs_max=float(np.abs(trainmean).max()),kmer_scale_z_min=float(trainstd.min()),kmer_scale_z_max=float(trainstd.max()),
        source_hdf=str(H5.relative_to(ROOT)),source_hdf_size=H5.stat().st_size,
        source_hdf_mtime_ns=H5.stat().st_mtime_ns,
        source_shapes={'train':[800000,9249],'val':[100000,9249],'test':[100000,9249]},
        method='Subtract held-out sum and square-sum from known full-data standardized moments; remaining error is float32 storage rounding.',
        robust_summary_caveat='Historical median/IQR used an unseeded reservoir. Exact training quantiles also change quantile estimation; they do not isolate data exclusion alone.')
    savejson(OUT/'affine_parameters.json',params);return params

COUNTER=None
def init_counter():
    global COUNTER;COUNTER=KmerCounter(str(ROOT/'magicc/data/selected_kmers.txt'))
def extract(item):
    gid,path=item;contigs=[];part=[]
    for line in open(path):
        if line.startswith('>'):
            if part:contigs.append(''.join(part).upper());part=[]
        else:part.append(line.strip())
    if part:contigs.append(''.join(part).upper())
    k=COUNTER.count_contigs(contigs);a=compute_assembly_stats(COUNTER.total_kmer_count(k),k)
    return gid,k.astype('f4'),a.astype('f4')
def get_features(dataset):
    folder=ROOT/'data/benchmarks'/dataset;m=pd.read_csv(folder/'metadata.tsv',sep='\t')
    cache=OUT/f'{dataset}_raw_features.npz'
    if not cache.exists():
        jobs=[(g,str(folder/'fasta'/f'{g}.fasta')) for g in m.genome_id]
        with Pool(4,initializer=init_counter) as pool:result=list(pool.imap(extract,jobs,chunksize=4))
        ids,k,a=zip(*result);savenpz(cache,compressed=True,genome_id=np.array(ids,dtype='S80'),kmer=np.stack(k),summary=np.stack(a))
    f=np.load(cache);assert np.array_equal(f['genome_id'],m.genome_id.to_numpy(dtype='S80'))
    return m,f['kmer'],f['summary']
def bootstrap_delta(delta,clusters,key):
    d=pd.DataFrame({'delta':delta,'cluster':clusters}).groupby('cluster').delta.agg(['sum','count'])
    rng=np.random.default_rng(zlib.crc32(key.encode())&0xffffffff);picks=rng.integers(0,len(d),(2000,len(d)))
    v=d['sum'].to_numpy()[picks].sum(axis=1)/d['count'].to_numpy()[picks].sum(axis=1)
    return np.quantile(v,[.025,.975])
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-recorded-parameters',action='store_true',help='Replay with supplied affine_parameters.json without requiring the original V4 HDF5; retain model, normalizer and fresh-inference verification.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'affine_parameters.json'
    if args.from_recorded_parameters and not p.exists():raise FileNotFoundError('Recorded-parameter replay requires '+str(p))
    params=json.loads(p.read_text()) if p.exists() else moments()
    assert params['old_normalizer_sha256']==sha(NORM)
    if not args.from_recorded_parameters:
        assert params['source_hdf_size']==H5.stat().st_size and params['source_hdf_mtime_ns']==H5.stat().st_mtime_ns
    norm=FeatureNormalizer.load(str(NORM));so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
    session=ort.InferenceSession(str(MODEL),sess_options=so,providers=['CPUExecutionProvider'])
    assert sha(MODEL)=='b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'
    allrows=[];stats=[];baseline_diffs=[]
    for dataset in SETS:
        print('Benchmark',dataset,flush=True);m,k,a=get_features(dataset)
        zk=norm.normalize_kmer(k);za=norm.normalize_assembly(a)
        models={}
        for arm in ['released_baseline','kmer_minmax_training_only','all_summary_exact_training']:
            x=zk.copy();b=za.copy()
            if arm!='released_baseline':
                x=(x-np.array(params['kmer_mean_z']))/np.array(params['kmer_scale_z'])
                for j in MINMAX_INDICES:b[:,j]=(b[:,j]-params['summary_min_z'][j])/params['summary_minmax_scale_z'][j]
                if arm=='all_summary_exact_training':
                    for j in ROBUST_INDICES:b[:,j]=(b[:,j]-params['summary_median_z'][j])/params['summary_robust_scale_z'][j]
            pred=[]
            for start in range(0,len(m),64):
                pred.append(session.run(None,{'kmer_features':x[start:start+64].astype('f4'),'assembly_features':b[start:start+64].astype('f4')})[0])
            models[arm]=np.concatenate(pred)
        saved=pd.read_csv(ROOT/'data/benchmarks'/dataset/'magicc_v5_predictions.tsv',sep='\t').set_index('genome_id').loc[m.genome_id]
        baseline_diff=float(np.max(np.abs(models['released_baseline']-saved[['pred_completeness','pred_contamination']].to_numpy())))
        assert baseline_diff<.006,(dataset,baseline_diff)
        baseline_diffs.append(baseline_diff)
        for arm,pred in models.items():
            row=m[['genome_id','dominant_accession','true_completeness','true_contamination']].copy()
            row['dataset']=dataset;row['arm']=arm;row['pred_completeness']=pred[:,0];row['pred_contamination']=pred[:,1];allrows.append(row)
            for j,metric in enumerate(['completeness','contamination']):
                truth=m[f'true_{metric}'].to_numpy();err=pred[:,j]-truth
                base=models['released_baseline'][:,j];shift=pred[:,j]-base;diff=np.abs(err)-np.abs(base-truth)
                lo,hi=bootstrap_delta(diff,m.dominant_accession,f'{dataset}:{metric}:{arm}')
                stats.append(dict(dataset=dataset,arm=arm,metric=metric,n=len(m),n_refs=m.dominant_accession.nunique(),
                    mae=float(np.mean(np.abs(err))),signed_bias=float(err.mean()),paired_mae_change=float(diff.mean()),
                    paired_mae_change_ci_lo=float(lo),paired_mae_change_ci_hi=float(hi),
                    prediction_shift_mean=float(shift.mean()),prediction_abs_shift_median=float(np.median(np.abs(shift))),prediction_abs_shift_max=float(np.abs(shift).max()),
                    baseline_max_difference_saved_predictions=baseline_diff))
    pd.concat(allrows).to_csv(OUT/'per_sample_sensitivity.tsv.gz',sep='\t',index=False)
    pd.DataFrame(stats).to_csv(OUT/'normalization_sensitivity.tsv',sep='\t',index=False)
    savejson(OUT/'summary.json',dict(status='COMPLETE',n_samples=5000,arms=['released_baseline','kmer_minmax_training_only','all_summary_exact_training'],
        model_sha256=sha(MODEL),normalizer_sha256=sha(NORM),max_baseline_difference_saved_pp=max(baseline_diffs),
        inference_only=True,changes_training_weights=False,affine_parameters_sha256=sha(p),
        parameter_provenance_mode='recorded_parameters_without_original_hdf_check' if args.from_recorded_parameters else 'original_hdf_identity_verified',
        limitation='Inference-only scaler perturbation cannot remove historical preprocessing dependence during training. Exact robust quantiles also replace historical unseeded reservoir estimation.',
        methods_script=str(Path(__file__).relative_to(ROOT))))
    print(pd.DataFrame(stats).to_string(index=False),flush=True)
if __name__=='__main__':main()
