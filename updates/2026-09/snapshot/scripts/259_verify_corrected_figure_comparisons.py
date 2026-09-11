#!/usr/bin/env python3
"""Independent stage-corrected duplication and binary-threshold recalculation.

Reuses the completed FASTA and fresh released-MAGICC inference audit only after
hashing its records. Reconstructs all statistical values independently from
prediction rows, without importing the published statistical generators.
"""
import hashlib,json,zlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'independent_figures'
TOOLS=['magicc_v5','checkm2','cocopye','deepcheck']
def read(p):return pd.read_csv(p,sep='\t')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boot(values,refs,key):
    order=np.argsort(np.asarray(refs).astype(str),kind='stable');values=np.asarray(values)[order];assert len(set(refs))==len(refs)
    picks=np.random.default_rng(zlib.crc32(key.encode())&0xffffffff).integers(0,len(values),size=(2000,len(values)))
    return map(float,np.percentile(values[picks].mean(axis=1),[2.5,97.5]))
def main():
    DEST.mkdir(exist_ok=True);g=ROOT/'data/benchmarks/set_G';gm=read(g/'generation_metadata.tsv');selected=gm[gm.error_type.isin(['none','uneven_coverage'])].copy();assert len(selected)==400
    oldaudit=ROOT/'results/revision/fig5c_verification_resubmission5';fa=read(oldaudit/'fasta_duplication_audit.tsv');assert len(fa)==320 and fa.original_contigs_retained.all() and fa.added_segments_exact_substrings.all()
    fresh=read(oldaudit/'fresh_inference_comparison.tsv');assert len(fresh)==400
    for m in ['completeness','contamination']:assert (fresh[f'pred_{m}_fresh']-fresh[f'pred_{m}_saved']).abs().max()<.00007
    inputs=[g/'generation_metadata.tsv',oldaudit/'fasta_duplication_audit.tsv',oldaudit/'fresh_inference_comparison.tsv',oldaudit/'independent_duplication_statistics.tsv',WORK/'results/revision/set_G/set_G_curves.tsv',WORK/'results/revision/set_G/set_G_paired_degradation.tsv',WORK/'results/revision/metrics/definitive_thresholds.tsv',Path(__file__).resolve()];records=[];raw=[];maxdiff=0.
    curves=read(WORK/'results/revision/set_G/set_G_curves.tsv');paired=read(WORK/'results/revision/set_G/set_G_paired_degradation.tsv')
    for tool in TOOLS:
        p=WORK/'data/benchmarks/set_G'/f'{tool}_predictions.tsv';inputs.append(p);pred=read(p);assert pred.genome_id.is_unique
        d=selected.merge(pred[['genome_id','pred_completeness','pred_contamination']],on='genome_id',validate='one_to_one');assert len(d)==400 and d.pred_completeness.notna().all();d['tool']=tool;raw.append(d)
        control=d[d.error_type.eq('none')].set_index('ref_index')
        for rate in [0.,.05,.1,.2,.4]:
            z=d[d.error_rate.eq(rate)].set_index('ref_index');assert len(z)==80
            for metric,prefix in [('completeness','comp'),('contamination','cont')]:
                e=z[f'pred_{metric}']-z[f'true_{metric}'];ce=control[f'pred_{metric}']-control[f'true_{metric}'];delta=e.abs()-ce.abs()
                lo,hi=boot(e.abs().values,z.index,f'setG|curve|{tool}|uneven_coverage|{rate}')
                dl,dh=boot(delta.values,delta.index,f'setG|paired|{tool}|uneven_coverage|{rate}|{metric}') if rate else (0.,0.)
                rec=dict(tool=tool,metric=metric,added_percent_original=rate*100,duplicate_percent_final=100*rate/(1+rate),n=80,mae=float(e.abs().mean()),mae_ci_lo=lo,mae_ci_hi=hi,signed_bias=float(e.mean()),delta_mae_vs_control=float(delta.mean()),delta_ci_lo=dl,delta_ci_hi=dh);records.append(rec)
                r=curves[curves.tool.eq(tool)&curves.error_type.eq('uneven_coverage')&curves.error_rate.eq(rate)].iloc[0]
                for value,column in [(rec['mae'],f'{prefix}_mae'),(lo,f'{prefix}_mae_ci_lo'),(hi,f'{prefix}_mae_ci_hi'),(rec['signed_bias'],f'{prefix}_bias')]:maxdiff=max(maxdiff,abs(value-r[column]));assert abs(value-r[column])<=.000051
                if rate:
                    r=paired[paired.tool.eq(tool)&paired.error_type.eq('uneven_coverage')&paired.error_rate.eq(rate)&paired.metric.eq(metric)].iloc[0]
                    for value,column in [(rec['delta_mae_vs_control'],'delta_mae'),(dl,'delta_mae_ci_lo'),(dh,'delta_mae_ci_hi')]:maxdiff=max(maxdiff,abs(value-r[column]));assert abs(value-r[column])<=.000051
    stats=pd.DataFrame(records);stats.to_csv(DEST/'independent_duplication_statistics.tsv',sep='\t',index=False);pd.concat(raw).to_csv(DEST/'duplication_raw_predictions_and_truth.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    old=read(oldaudit/'independent_duplication_statistics.tsv');q=stats[stats.tool.ne('cocopye')].merge(old,on=['tool','metric','added_percent_original'],suffixes=('_new','_old'))
    for column in ['mae','signed_bias','delta_mae_vs_control','delta_ci_lo','delta_ci_hi']:assert np.allclose(q[column+'_new'],q[column+'_old'],atol=1e-12)
    definitive=read(WORK/'results/revision/metrics/definitive_thresholds.tsv');thr=[]
    for dataset in ['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E']:
        folder=WORK/'data/benchmarks'/dataset;meta=read(folder/'metadata.tsv');inputs.append(folder/'metadata.tsv')
        for tool in TOOLS:
            p=folder/f'{tool}_predictions.tsv';inputs.append(p);pred=read(p);d=meta[['genome_id','true_contamination']].merge(pred[['genome_id','pred_contamination']],on='genome_id',validate='one_to_one');assert len(d)==1000
            clean=d.true_contamination.lt(5);passed=d.pred_contamination.lt(5);ff=int((clean&~passed).sum());fp=int((~clean&passed).sum());ffr=ff/clean.sum() if clean.any() else np.nan;fpr=fp/(~clean).sum() if (~clean).any() else np.nan;ba=1-(ffr+fpr)/2
            r=definitive[definitive['set'].eq(dataset)&definitive.tool.eq(tool)&definitive.criterion.eq('contamination')&definitive.threshold.eq(5)].iloc[0]
            for value,column in [(ffr,'false_fail_rate'),(fpr,'false_pass_rate'),(ba,'balanced_accuracy')]:assert (np.isnan(value) and np.isnan(r[column])) or np.isclose(value,r[column],atol=1e-12)
            thr.append(dict(dataset=dataset,tool=tool,n=len(d),n_true_clean=int(clean.sum()),n_true_contaminated=int((~clean).sum()),false_fail_n=ff,false_pass_n=fp,false_fail_rate=ffr,false_pass_rate=fpr,balanced_accuracy=ba))
    pd.DataFrame(thr).to_csv(DEST/'independent_figure3f_thresholds.tsv',sep='\t',index=False)
    manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(set(inputs))];pd.DataFrame(manifest).to_csv(DEST/'input_manifest.tsv',sep='\t',index=False)
    summary=dict(status='INDEPENDENT_CORRECTED_FIGURE_AUDIT_PASS',duplication_rows=40,threshold_rows=20,reference_clusters_per_dose=80,max_statistic_difference_from_rounded_replay=maxdiff,original_fasta_and_fresh_magicc_inference_audit_preserved=True,non_cocopye_duplication_statistics_unchanged=True,interpretation='Localized exact sequence duplication. Dose is added bp/original assembly bp, not empirical coverage frequency. Completeness-specific MAGICC degradation persists; at40% added/original bp MAGICC has the lowest contamination-MAE point estimate among the four tools, while all four contamination errors increase.',input_manifest=manifest)
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(stats[stats.added_percent_original.eq(40)].to_string(index=False))
if __name__=='__main__':main()
