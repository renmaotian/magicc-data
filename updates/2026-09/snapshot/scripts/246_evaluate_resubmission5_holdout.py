#!/usr/bin/env python3
"""Paired three-model evaluation with reference-cluster uncertainty and provenance.

Primary contrast: jointly excluding the ten panel families, with a shared
panel-free feature vocabulary, versus the seeded full-family training arm.
DiD adjusts the observed MAE difference by its common-control counterpart; it
cannot identify taxonomy alone independently of changes to the training pool.
Frozen production V5 is included as a descriptive deployment reference.
"""
import argparse
import hashlib
import json
import zlib
from pathlib import Path
import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
from resubmission5_holdout_config import OUT,ROOT,configure
from holdout_lib.normalization import FeatureNormalizer

N_BOOT=2000
SEED=20260908
TOOLS={'holdout':'MAGICC_holdout','matched_full':'MAGICC_matched_full','production':'MAGICC_V5'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rng_for(label):return np.random.default_rng((SEED+zlib.crc32(label.encode()))%(2**32))
def resample_means(values,clusters,label):
    a=pd.DataFrame({'cluster':clusters,'value':values}).groupby('cluster',sort=True).value.agg(['sum','count'])
    picks=rng_for(label).integers(0,len(a),(N_BOOT,len(a)))
    return a['sum'].to_numpy()[picks].sum(axis=1)/a['count'].to_numpy()[picks].sum(axis=1)
def ci(x):return [float(v) for v in np.quantile(x,[.025,.975])]
def r2(y,p):
    ss=float(((y-y.mean())**2).sum())
    return float(1-((p-y)**2).sum()/ss) if ss>1e-10 else np.nan
def bh(p):
    p=np.asarray(p);ix=np.argsort(p);q=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p));result[ix]=np.minimum(q,1.);return result
def quality(comp,cont):return np.where((comp>=90)&(cont<5),'high',np.where((comp>=50)&(cont<10),'medium','low'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=8);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();c=configure('holdout')
    target=OUT/'smoke_evaluation' if args.smoke else OUT;target.mkdir(parents=True,exist_ok=True)
    base=c.HOLDOUT_DIR/'eval_sets_smoke' if args.smoke else c.EVAL_DIR
    so=ort.SessionOptions();so.intra_op_num_threads=args.threads;so.inter_op_num_threads=1
    suffix='smoke' if args.smoke else 'full';models={};provenance=[]
    for model in TOOLS:
        if model=='production':
            path=ROOT/'models/magicc_v5.onnx';norm=ROOT/'data/features/normalization_params.json'
            assert sha(path)=='b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'
            assert sha(norm)=='b1e3f211a43560d8bed1dd8264921c7ec8ce0529c9bdbdbdc000894c9ce1d7d9'
        else:
            path=OUT/model/f'models/{suffix}/model.onnx'
            norm=OUT/model/'data'/('smoke_normalization_params.json' if args.smoke else 'normalization_params.json')
            complete=path.parent/'completion.json';assert complete.exists(),f'Incomplete training: {model}'
            report=json.loads(complete.read_text());assert report['status']=='TRAINING_COMPLETE'
            assert report['pytorch_onnx_max_difference']<1e-4
            assert report['onnx_sha256']==sha(path)
            assert report['training_config']['normalization_sha256']==sha(norm)
            assert report['training_config']['feature_sha256']==sha(c.SELECTED_KMERS)
            if not args.smoke:assert report['training_config']['training_samples']==1000000
        models[model]=(ort.InferenceSession(str(path),sess_options=so,providers=['CPUExecutionProvider']),FeatureNormalizer.load(str(norm)))
        provenance.extend([{'artifact':str(path),'sha256':sha(path)},{'artifact':str(norm),'sha256':sha(norm)}])
    predictions=[]
    test_taxonomy=pd.read_csv(c.TEST_TSV,sep='\t',usecols=['ncbi_accession','gtdb_taxonomy'])
    test_taxonomy['family']=test_taxonomy.gtdb_taxonomy.str.extract(r'f__([^;]+)')
    family_by_accession=test_taxonomy.set_index('ncbi_accession').family
    for group in c.EVAL_DESIGN:
        directory=base/group;d=pd.read_csv(directory/'metadata.tsv',sep='\t')
        assert (d.group==group).all() and d.genome_id.is_unique
        assert d.genome_id.str.len().max()<=40, 'Genome ID would truncate in feature identity check'
        families=d.dominant_accession.map(family_by_accession)
        assert not families.isna().any()
        if group=='in_distribution':assert not families.isin(c.PANEL_TAXA).any()
        else:assert (families==group).all()
        for artifact in [directory/'metadata.tsv',directory/'features.h5',directory/'production_features.h5']:
            provenance.append({'artifact':str(artifact),'sha256':sha(artifact)})
        d['analysis_group']=group
        for model,(session,nrm) in models.items():
            h5=directory/('production_features.h5' if model=='production' else 'features.h5')
            with h5py.File(h5,'r') as f:
                raw=f['kmer_counts_raw'][:];summary=f['summary_features_raw'][:]
                assert np.array_equal(f['genome_ids'][:],d.genome_id.to_numpy(dtype='S40'))
            result=[]
            for s in range(0,len(d),512):
                x=nrm.normalize_kmer(raw[s:s+512]).astype('f4');a=nrm.normalize_assembly(summary[s:s+512]).astype('f4')
                result.append(session.run(None,{'kmer_features':x,'assembly_features':a})[0])
            p=np.concatenate(result);assert p.shape==(len(d),2) and np.isfinite(p).all()
            d[f'{model}_comp']=p[:,0];d[f'{model}_cont']=p[:,1]
        d['primary_in_domain']=(d.true_completeness>=50)&(d.true_completeness<=100)&(d.true_contamination>=0)&(d.true_contamination<=d.true_completeness+1e-6)
        predictions.append(d)
    allrows=pd.concat(predictions,ignore_index=True)
    assert not allrows.duplicated(['analysis_group','genome_id']).any()
    # Original simulator IDs are local to each group (genome_0, genome_1, ...).
    # Preserve their feature/FASTA identity and provide an unambiguous pooled key.
    allrows['evaluation_id']=allrows.analysis_group.astype(str)+'::'+allrows.genome_id.astype(str)
    assert allrows.evaluation_id.is_unique, 'Pooled evaluation sample IDs must be globally unique'
    assert allrows.groupby('dominant_accession').analysis_group.nunique().max()==1
    allrows.to_csv(target/'per_sample_predictions.tsv.gz',sep='\t',index=False,compression='gzip')
    d=allrows[allrows.primary_in_domain].copy();rows=[];reference_rows=[];confusion=[]
    assert set(d.analysis_group)==set(c.EVAL_DESIGN), 'A primary group has no in-domain samples'
    assert (d.groupby('analysis_group').dominant_accession.nunique() >= (2 if args.smoke else 20)).all()
    for group,sub in d.groupby('analysis_group',sort=False):
        for model,tool in TOOLS.items():
            row={'group':group,'tool':tool,'n_samples':len(sub),'n_refs':sub.dominant_accession.nunique()}
            for metric,truth in [('comp','true_completeness'),('cont','true_contamination')]:
                error=sub[f'{model}_{metric}'].to_numpy()-sub[truth].to_numpy();cl=sub.dominant_accession.to_numpy()
                bs=resample_means(np.abs(error),cl,f'{group}:{metric}:mae');lo,hi=ci(bs)
                row.update({f'{metric}_mae':float(np.abs(error).mean()),f'{metric}_mae_ci_low':lo,f'{metric}_mae_ci_high':hi,
                            f'{metric}_bias':float(error.mean()),f'{metric}_rmse':float(np.sqrt((error**2).mean())),
                            f'{metric}_r2':r2(sub[truth].to_numpy(),sub[f'{model}_{metric}'].to_numpy())})
                elo,ehi=ci(resample_means(error,cl,f'{group}:{metric}:bias'))
                row.update({f'{metric}_bias_ci_low':elo,f'{metric}_bias_ci_high':ehi})
                frame=pd.DataFrame({'reference':cl,'absolute_error':np.abs(error),'signed_error':error})
                for ref,rf in frame.groupby('reference'):
                    reference_rows.append({'group':group,'tool':tool,'metric':metric,'reference':ref,'n_samples':len(rf),
                                           'mae':rf.absolute_error.mean(),'bias':rf.signed_error.mean()})
            rows.append(row)
            truth=quality(sub.true_completeness.to_numpy(),sub.true_contamination.to_numpy())
            pred=quality(sub[f'{model}_comp'].to_numpy(),sub[f'{model}_cont'].to_numpy())
            for t in ['high','medium','low']:
                for p in ['high','medium','low']:
                    confusion.append({'group':group,'tool':tool,'true_class':t,'predicted_class':p,'n':int(((truth==t)&(pred==p)).sum())})
    pd.DataFrame(rows).to_csv(target/'head_to_head_by_group.tsv',sep='\t',index=False)
    pd.DataFrame(reference_rows).to_csv(target/'per_reference_errors.tsv',sep='\t',index=False)
    pd.DataFrame(confusion).to_csv(target/'mimag_confusion_by_group.tsv',sep='\t',index=False)
    control=d[d.analysis_group=='in_distribution'];dids=[];controls=[]
    for metric,truth in [('comp','true_completeness'),('cont','true_contamination')]:
        ec=np.abs(control[f'holdout_{metric}']-control[truth])-np.abs(control[f'matched_full_{metric}']-control[truth])
        bc=resample_means(ec.to_numpy(),control.dominant_accession.to_numpy(),f'control:{metric}:did')
        lo,hi=ci(bc);controls.append({'metric':metric,'delta_mae':float(ec.mean()),'ci_low':lo,'ci_high':hi,
                                     'n_refs':control.dominant_accession.nunique(),'n_samples':len(control)})
        for group in c.EVAL_GROUPS:
            sub=d[d.analysis_group==group]
            eg=np.abs(sub[f'holdout_{metric}']-sub[truth])-np.abs(sub[f'matched_full_{metric}']-sub[truth])
            bg=resample_means(eg.to_numpy(),sub.dominant_accession.to_numpy(),f'{group}:{metric}:did')
            obs=float(eg.mean()-ec.mean());boot=bg-bc;lo,hi=ci(boot)
            # Two-sided centred bootstrap tail probability, with plus-one correction.
            p=float((1+np.sum(np.abs(boot-obs)>=abs(obs)))/(N_BOOT+1))
            dids.append({'group':group,'metric':metric,'did':obs,'ci_low':lo,'ci_high':hi,'p':p,
                         'delta_mae':float(eg.mean()),'control_delta_mae':float(ec.mean()),
                         'n_samples':len(sub),'n_refs':sub.dominant_accession.nunique(),
                         'control_n_samples':len(control),'control_n_refs':control.dominant_accession.nunique()})
    did=pd.DataFrame(dids);did['q']=bh(did.p.to_numpy());did.to_csv(target/'did.tsv',sep='\t',index=False)
    pd.DataFrame(controls).to_csv(target/'control_model_difference.tsv',sep='\t',index=False)
    pd.DataFrame(provenance).to_csv(target/'model_provenance.tsv',sep='\t',index=False)
    convergence=[]
    for arm in ['holdout','matched_full']:
        completion=json.loads((OUT/arm/f'models/{suffix}/completion.json').read_text())
        hist=pd.read_json(OUT/arm/f'models/{suffix}/training_history.json')
        convergence.append({'arm':arm,'epochs_completed':completion['epochs_completed'],'best_epoch':completion['best_epoch'],
                            'best_validation_loss':completion['best_validation_loss'],
                            'early_stopped':int(hist.iloc[-1].epochs_without_improvement)>=20,
                            'hit_maximum_epochs':completion['epochs_completed']>=completion['training_config']['max_epochs'],
                            'best_in_last_20_epochs':completion['epochs_completed']-completion['best_epoch']<20})
    pd.DataFrame(convergence).to_csv(target/'convergence_check.tsv',sep='\t',index=False)
    summary={'status':'SMOKE_ONLY' if args.smoke else 'EVALUATION_COMPLETE','total_samples':len(allrows),'primary_samples':len(d),
             'outside_domain_samples':len(allrows)-len(d),'primary_references':d.dominant_accession.nunique(),
             'n_panel_groups':10,'bootstrap_replicates':N_BOOT,'bootstrap_unit':'dominant reference genome',
             'multiplicity':'BH over all 20 family×metric DiD contrasts','seed':SEED,
             'interpretation':'Joint family-panel exclusion effect relative to matched full-family training, adjusted for common-control model difference; training-pool interactions and one-seed model variability remain.',
             'feature_selection':'Both newly retrained arms use the same panel-free reselected vocabulary of 9,243 canonical 9-mers. No target family enters feature selection.',
             'frozen_production_role':'Descriptive deployed-model comparator; excluded from primary matched-arm DiD.',
             'convergence':convergence}
    (target/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(did.to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
