"""Final shared-reference pooled intervals and reference-level Set F tests."""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]

def pooled():
 spec=importlib.util.spec_from_file_location('pooled_framework',ROOT/'scripts/101_metrics_framework.py')
 fw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fw;spec.loader.exec_module(fw)
 cfg=fw.load_config(ROOT/'scripts/config_revision_metrics.yaml')
 objects={b.name:b for b in fw.discover_sets(cfg,include_missing=False)}
 names=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E']
 frames=[fw.load_set(cfg,objects[name])[0] for name in names]
 refs=sorted(set().union(*(set(d.cluster_id) for d in frames)))
 assert len(refs)==1648 and all(len(d)==1000 for d in frames)
 index={a:i for i,a in enumerate(refs)}
 rng=np.random.default_rng(cfg.seed+777)
 counts=rng.multinomial(len(refs),np.full(len(refs),1/len(refs)),size=cfg.n_boot).astype(float)
 counts=np.concatenate([np.ones((1,len(refs))),counts]);rows=[]
 for tool in ['magicc_v5','checkm2','cocopye','deepcheck']:
  for metric in ['completeness','contamination']:
   draws=[]
   for d in frames:
    t=d[f'true_{metric}'].to_numpy(float);p=d[f'pred_{metric}__{tool}'].to_numpy(float);e=p-t
    values=np.column_stack([np.ones(len(d)),t,t*t,p,p*p,t*p,np.abs(e),e*e,e])
    idx=np.array([index[a] for a in d.cluster_id]);sums=np.zeros((len(refs),9));np.add.at(sums,idx,values)
    weighted=counts@sums;assert (weighted[:,0]>0).all();draws.append(weighted[:,1:]/weighted[:,0,None])
    assert np.max(np.abs(np.average(values[:,1:],axis=0,weights=counts[1,idx])-draws[-1][1]))<1e-9
   t,t2,p,p2,tp,ae,e2,e=np.mean(draws,axis=0).T
   quantities=dict(mae=ae,rmse=np.sqrt(e2),bias=e,r2=1-e2/(t2-t*t),r2_pearson_sq=(tp-t*p)**2/((t2-t*t)*(p2-p*p)))
   row=dict(set='POOLED_leakage_free_5_sets',tool=tool,metric=metric,n=5000,n_clusters=len(refs))
   for key,value in quantities.items():
    row[key]=float(value[0]);row[key+'_ci_lo'],row[key+'_ci_hi']=map(float,np.quantile(value[1:],[.025,.975]))
   rows.append(row)
 path=ROOT/'results/revision/metrics/ws5.5_table_S2_rebuilt.tsv';table=pd.read_csv(path,sep='\t')
 for row in rows:
  mask=table.set.eq(row['set'])&table.tool.eq(row['tool'])&table.metric.eq(row['metric']);assert mask.sum()==1
  for key,value in row.items():table.loc[mask,key]=value
  table.loc[mask,'set_label']='Pooled five-set benchmark (equal set weights; shared dominant-accession clusters)'
 table.to_csv(path,sep='\t',index=False)

def set_f():
 base=ROOT/'results/revision/set_F';path=base/'set_F_paired_comparisons.tsv'
 old=pd.read_csv(path,sep='\t');long=pd.read_csv(base/'set_F_long_predictions.tsv',sep='\t')
 long=long[long.contamination_type.ne('none')];rows=[]
 for r in old.to_dict('records'):
  z=long
  if r['stratum']=='type':z=z[z.contamination_type.eq(r['level'])]
  elif r['stratum']=='distance':z=z[z.distance.eq(r['level'])]
  elif r['stratum']=='cell':
   ty,di=r['level'].split('|');z=z[z.contamination_type.eq(ty)&z.distance.eq(di)]
  elif r['stratum']=='pooled' and r['level']=='NON_REDUNDANT':z=z[z.contamination_type.isin(['replaced','single'])]
  else:assert r['stratum']=='pooled' and r['level']=='ALL'
  m=z[z.tool.eq(r['tool_a'])].merge(z[z.tool.eq(r['tool_b'])],on='genome_id',validate='one_to_one',suffixes=('_a','_b'))
  assert len(m)==r['n_pairs'] and (m.ref_index_a==m.ref_index_b).all()
  d=m.abs_err_contamination_a.to_numpy()-m.abs_err_contamination_b.to_numpy()
  rd=pd.DataFrame(dict(reference=m.ref_index_a,difference=d)).groupby('reference').difference.mean();assert len(rd)==100
  p=float(wilcoxon(rd,alternative='two-sided',zero_method='wilcox').pvalue) if not np.allclose(rd,0) else 1.
  r.update(p_wilcoxon_sample_level_historical=r['p_wilcoxon'],q_bh_sample_level_historical=r['q_bh'],winner_sample_level_historical=r['winner'],p_wilcoxon=p,p_wilcoxon_reference_mean=p,n_paired_references=len(rd),reference_mean_difference=float(rd.mean()));rows.append(r)
 result=pd.DataFrame(rows);p=result.p_wilcoxon_reference_mean.to_numpy();order=np.argsort(p)
 q=np.empty(len(p));q[order]=np.minimum(np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1],1)
 result['q_bh']=q;result['q_bh_reference_mean']=q
 result['supported_reference_test_and_hl_interval']=result.q_bh.lt(.05)&((result.hl_ci_lo.gt(0)&result.reference_mean_difference.gt(0))|(result.hl_ci_hi.lt(0)&result.reference_mean_difference.lt(0)))
 result['winner']=np.where(result.supported_reference_test_and_hl_interval,np.where(result.hl.lt(0),'magicc_v5',result.tool_b),'unresolved')
 result['inference_definition']='Two-sided Wilcoxon on100 per-reference mean paired absolute-error differences; BH across87 comparisons. Support also requires concordant reference-cluster HL95% interval excluding0; unresolved is not equivalence.'
 result['support_decision_changed']=result.winner.ne(result.winner_sample_level_historical.replace('tie','unresolved'))
 result.to_csv(path,sep='\t',index=False)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('analysis',choices=['pooled','set_f']);args=ap.parse_args()
 {'pooled':pooled,'set_f':set_f}[args.analysis]()
