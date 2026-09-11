#!/usr/bin/env python3
"""Audit affine-transform equivalence and threshold consequences of script 249.

Uses independently reconstructed raw-unit normalizers and the existing paired
predictions. Does not recount FASTAs or retrain V5. Classification uses all three
MIMAG-inspired labels, with zero F1 for a class with no truth/predicted instances.
"""
import copy,hashlib,json,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from magicc.normalization import FeatureNormalizer,MINMAX_INDICES,ROBUST_INDICES
from magicc.assembly_stats import FEATURE_NAMES
OUT=ROOT/'results/revision/normalization_sensitivity_resubmission5'
ARMS=['released_baseline','kmer_minmax_training_only','all_summary_exact_training']
SETS=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def labels(comp,cont):return np.where((comp>=90)&(cont<5),0,np.where((comp>=50)&(cont<10),1,2))
def f1(y,p):
    c=np.zeros((3,3),dtype=int);np.add.at(c,(y,p),1)
    den=c.sum(0)+c.sum(1);val=np.divide(2*c.diagonal(),den,out=np.zeros(3),where=den!=0)
    return float(val.mean()),c

def main():
    params=json.loads((OUT/'affine_parameters.json').read_text());norm=FeatureNormalizer.load(str(ROOT/'data/features/normalization_params.json'))
    equivalence=[]
    for dataset in SETS:
        raw=np.load(OUT/f'{dataset}_raw_features.npz');k=raw['kmer'].astype('f8');a=raw['summary'].astype('f8')
        oldk=norm.normalize_kmer(k);olda=norm.normalize_assembly(a)
        for arm in ARMS[1:]:
            new=copy.deepcopy(norm)
            new.kmer_mean=norm.kmer_mean+norm.kmer_std*np.asarray(params['kmer_mean_z'])
            new.kmer_std=norm.kmer_std*np.asarray(params['kmer_scale_z'])
            for j in MINMAX_INDICES:
                new.assembly_minmax_min[j]=norm.assembly_minmax_min[j]+norm.assembly_minmax_range[j]*params['summary_min_z'][j]
                new.assembly_minmax_range[j]=norm.assembly_minmax_range[j]*params['summary_minmax_scale_z'][j]
            if arm=='all_summary_exact_training':
                for j in ROBUST_INDICES:
                    new.assembly_robust_median[j]=norm.assembly_robust_median[j]+norm.assembly_robust_iqr[j]*params['summary_median_z'][j]
                    new.assembly_robust_iqr[j]=norm.assembly_robust_iqr[j]*params['summary_robust_scale_z'][j]
            affine_k=(oldk-np.asarray(params['kmer_mean_z']))/np.asarray(params['kmer_scale_z']);affine_a=olda.copy()
            for j in MINMAX_INDICES:affine_a[:,j]=(olda[:,j]-params['summary_min_z'][j])/params['summary_minmax_scale_z'][j]
            if arm=='all_summary_exact_training':
                for j in ROBUST_INDICES:affine_a[:,j]=(olda[:,j]-params['summary_median_z'][j])/params['summary_robust_scale_z'][j]
            kd=float(np.max(np.abs(new.normalize_kmer(k)-affine_k)));ad=float(np.max(np.abs(new.normalize_assembly(a)-affine_a)))
            assert kd<1e-12 and ad<1e-12,(dataset,arm,kd,ad)
            equivalence.append(dict(dataset=dataset,arm=arm,n=len(k),n_kmer_features=k.shape[1],kmer_feature_max_abs_difference=kd,summary_feature_max_abs_difference=ad,numeric_precision='float64 for algebra audit'))
    pd.DataFrame(equivalence).to_csv(OUT/'raw_transform_equivalence.tsv',sep='\t',index=False)
    d=pd.read_csv(OUT/'per_sample_sensitivity.tsv.gz',sep='\t');assert not d.duplicated(['dataset','arm','genome_id']).any()
    published=pd.read_csv(ROOT/'results/revision/metrics/definitive_mimag.tsv',sep='\t')
    thresholds=pd.read_csv(ROOT/'results/revision/metrics/definitive_thresholds.tsv',sep='\t')
    rows=[];confusions=[];changes=[]
    for dataset in SETS:
        base=d[(d.dataset==dataset)&(d.arm==ARMS[0])].set_index('genome_id').sort_index();assert len(base)==1000
        truecls=labels(base.true_completeness.to_numpy(),base.true_contamination.to_numpy())
        basecls=labels(base.pred_completeness.to_numpy(),base.pred_contamination.to_numpy())
        basef1,_=f1(truecls,basecls);truth_pass=base.true_contamination.to_numpy()<5;basepass=base.pred_contamination.to_numpy()<5
        nr_clean=int(truth_pass.sum());nr_cont=int((~truth_pass).sum())
        baseff=int((truth_pass&~basepass).sum());basefp=int((~truth_pass&basepass).sum())
        baseba=(1-(baseff/nr_clean+basefp/nr_cont)/2) if nr_clean and nr_cont else np.nan
        orig=published[(published['set']==dataset)&(published.tool=='magicc_v5')].macro_f1.unique();assert len(orig)==1 and abs(basef1-orig[0])<1e-12
        ot=thresholds[(thresholds['set']==dataset)&(thresholds.tool=='magicc_v5')&(thresholds.criterion=='contamination')&(thresholds.threshold==5)].iloc[0]
        assert baseff==ot.n_false_fail and basefp==ot.n_false_pass
        for arm in ARMS:
            p=d[(d.dataset==dataset)&(d.arm==arm)].set_index('genome_id').loc[base.index]
            assert np.array_equal(p[['true_completeness','true_contamination','dominant_accession']],base[['true_completeness','true_contamination','dominant_accession']])
            predcls=labels(p.pred_completeness.to_numpy(),p.pred_contamination.to_numpy());macro,c=f1(truecls,predcls)
            predpass=p.pred_contamination.to_numpy()<5;ff=int((truth_pass&~predpass).sum());fp=int((~truth_pass&predpass).sum())
            ffr=ff/nr_clean if nr_clean else np.nan;fpr=fp/nr_cont if nr_cont else np.nan;ba=1-(ffr+fpr)/2
            rows.append(dict(dataset=dataset,arm=arm,n=len(p),n_refs=p.dominant_accession.nunique(),macro_f1=macro,macro_f1_change=macro-basef1,
                n_changed_quality_classes=int((predcls!=basecls).sum()),n_changed_contamination_threshold_calls=int((predpass!=basepass).sum()),
                n_true_clean=nr_clean,n_true_contaminated=nr_cont,false_fail_n=ff,false_pass_n=fp,false_fail_rate=ffr,false_pass_rate=fpr,balanced_accuracy_5pct=ba,balanced_accuracy_change=ba-baseba))
            for i,lab in enumerate(['high','medium','low']):
                for j,plab in enumerate(['high','medium','low']):confusions.append(dict(dataset=dataset,arm=arm,true_class=lab,predicted_class=plab,n=int(c[i,j])))
            for n,gid in enumerate(base.index):
                if predcls[n]!=basecls[n] or predpass[n]!=basepass[n]:
                    changes.append(dict(dataset=dataset,arm=arm,genome_id=gid,true_completeness=p.loc[gid,'true_completeness'],true_contamination=p.loc[gid,'true_contamination'],baseline_completeness=base.loc[gid,'pred_completeness'],baseline_contamination=base.loc[gid,'pred_contamination'],perturbed_completeness=p.loc[gid,'pred_completeness'],perturbed_contamination=p.loc[gid,'pred_contamination'],baseline_class=['high','medium','low'][basecls[n]],perturbed_class=['high','medium','low'][predcls[n]],baseline_contamination_pass=bool(basepass[n]),perturbed_contamination_pass=bool(predpass[n])))
    cls=pd.DataFrame(rows);cls.to_csv(OUT/'classification_sensitivity.tsv',sep='\t',index=False)
    pd.DataFrame(confusions).to_csv(OUT/'classification_confusion_matrices.tsv',sep='\t',index=False)
    pd.DataFrame(changes).to_csv(OUT/'classification_changed_samples.tsv',sep='\t',index=False)
    summary={'status':'PASS','n_samples':5000,'n_prediction_rows':len(d),'baseline_classification_reproduces_published':True,'all_raw_unit_transform_equivalence_checks_pass':True,'max_raw_equivalence_difference':max(max(x['kmer_feature_max_abs_difference'],x['summary_feature_max_abs_difference']) for x in equivalence),'classification_rules':'High: completeness >=90 and contamination <5; medium: completeness >=50 and contamination <10, excluding high; low: otherwise. Fixed three-label macro F1 with zero_division=0. 5% contamination balanced accuracy is undefined without both truth classes.','summary_feature_names':FEATURE_NAMES,'minmax_feature_indices':MINMAX_INDICES,'robust_feature_indices':ROBUST_INDICES,'software_versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths=[Path(__file__),ROOT/'scripts/249_normalization_sensitivity.py',ROOT/'models/magicc_v5.onnx',ROOT/'data/features/normalization_params.json',ROOT/'magicc/data/selected_kmers.txt',ROOT/'magicc/normalization.py',ROOT/'magicc/kmer_counter.py',ROOT/'magicc/assembly_stats.py',OUT/'INTERPRETATION_REPORT.md']
    paths += list(OUT.glob('*.json'))+list(OUT.glob('*.npz'))+list(OUT.glob('*.tsv.gz'))+[OUT/'normalization_sensitivity.tsv',OUT/'classification_sensitivity.tsv',OUT/'classification_confusion_matrices.tsv',OUT/'classification_changed_samples.tsv',OUT/'raw_transform_equivalence.tsv']
    for dataset in SETS:paths += [ROOT/'data/benchmarks'/dataset/'metadata.tsv',ROOT/'data/benchmarks'/dataset/'magicc_v5_predictions.tsv']
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    pd.DataFrame(manifest).to_csv(OUT/'input_output_manifest.tsv',sep='\t',index=False)
    print(cls.to_string(index=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
