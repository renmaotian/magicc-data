#!/usr/bin/env python3
"""Canonical CoCoPyE 0.5.0 full-CSV parser and complete correction inventory.

CSV/API stage1 is unscored prefilter rejection, stage2 selects the marker
estimate and stage3 selects the neural estimate. The paper numbers its two
estimation stages I/II; these are not the three CSV/API status values.
No estimate is clipped, and a rejected row is retained with NaN quantitative
predictions plus the exact official sentinel fractions (-1).
"""
import argparse,csv,hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse_full(path):
    d=pd.read_csv(path);required=['bin','stage','2_completeness','2_contamination','3_completeness','3_contamination','method']
    assert set(required)<=set(d),(path,d.columns.tolist())
    assert d.bin.is_unique and not d.bin.isna().any()
    stage=d.stage.to_numpy();assert np.isin(stage,[1,2,3]).all()
    comp=np.select([stage==1,stage==2],[-1.,d['2_completeness']],default=d['3_completeness']).astype(float)
    cont=np.select([stage==1,stage==2],[-1.,d['2_contamination']],default=d['3_contamination']).astype(float)
    scored=stage!=1;assert np.isfinite(comp[scored]).all() and np.isfinite(cont[scored]).all()
    out=pd.DataFrame({'genome_id':d.bin.astype(str),'selected_stage':stage.astype(int),'tool_scored':scored,
        'pred_completeness':np.where(scored,100*comp,np.nan),'pred_contamination':np.where(scored,100*cont,np.nan),
        'official_completeness_fraction':comp,'official_contamination_fraction':cont,'selection_method':d.method})
    return out,d

def main():
    from cocopye.core import Result
    import cocopye.core as core
    assert importlib.metadata.version('cocopye')=='0.5.0', 'Recorded analysis requires CoCoPyE 0.5.0 API'
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inventory-only',action='store_true');args=parser.parse_args()
    roots=[ROOT/'data/benchmarks',ROOT/'results/revision'];paths=[]
    for base in roots:
        for p in base.rglob('*.csv'):
            if OUT in p.parents:continue
            with p.open() as f:head=next(csv.reader(f),[])
            if 'cocopye' in str(p).lower() or {'stage','3_completeness','3_contamination'}<=set(head):paths.append(p)
    inventory=[];artifacts=[]
    for path in sorted(paths):
        header=pd.read_csv(path,nrows=0).columns.tolist();full='3_completeness' in header and 'stage' in header
        if not full:
            d=pd.read_csv(path);inventory.append({'raw_path':str(path.relative_to(ROOT)),'rows':len(d),'schema':'standard_selected_output','role':'timing_only' if any(x in path.parts for x in ['speed','speed_v3','speed_v033']) else 'selected_output_requires_consumer_mapping','n_stage1':0,'n_stage2':0,'n_stage3':0,'quantitative_rows_changed':0,'raw_sha256':sha(path)});continue
        pred,d=parse_full(path)
        for row,rec in zip(d.itertuples(index=False,name=None),pred.itertuples(index=False)):
            r=dict(zip(d.columns,row));official=Result();official.stage=int(r['stage']);official.comp_2=r['2_completeness'];official.cont_2=r['2_contamination'];official.comp_3=r['3_completeness'];official.cont_3=r['3_contamination']
            assert rec.official_completeness_fraction==official.completeness()
            assert rec.official_contamination_fraction==official.contamination()
        oldc=d['3_completeness'].fillna(d['2_completeness']).to_numpy()*100
        oldx=d['3_contamination'].fillna(d['2_contamination']).to_numpy()*100
        changed=pred.tool_scored.to_numpy() & ((pred.pred_completeness.to_numpy()!=oldc)|(pred.pred_contamination.to_numpy()!=oldx))
        rel=path.relative_to(ROOT);dest=OUT/'corrected_predictions'/rel.parent/(path.stem.replace('raw_output','predictions')+'.tsv')
        if not args.inventory_only:
            dest.parent.mkdir(parents=True,exist_ok=True);pred.to_csv(dest,sep='\t',index=False)
            artifacts.append({'raw_path':str(rel),'corrected_path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'n_rows':len(pred)})
        inventory.append({'raw_path':str(rel),'rows':len(d),'schema':'full_all_stage_estimates','role':'timing_only' if any(x in path.parts for x in ['speed','speed_v3','speed_v033']) else 'accuracy','n_stage1':int((d.stage==1).sum()),'n_stage2':int((d.stage==2).sum()),'n_stage3':int((d.stage==3).sum()),'quantitative_rows_changed':int(changed.sum()),'rejected_rows_previously_assigned_stage3':int(((d.stage==1)&np.isfinite(oldc)&np.isfinite(oldx)).sum()),'max_selected_vs_historical_completeness_difference_pp':float(np.nanmax(np.abs(pred.pred_completeness.to_numpy()-oldc))),'max_selected_vs_historical_contamination_difference_pp':float(np.nanmax(np.abs(pred.pred_contamination.to_numpy()-oldx))),'raw_sha256':sha(path)})
    inv=pd.DataFrame(inventory);inv.to_csv(OUT/'affected_cohort_inventory.tsv',sep='\t',index=False)
    pd.DataFrame(artifacts).to_csv(OUT/'corrected_prediction_manifest.tsv',sep='\t',index=False)
    verification={'status':'PARSER_VERIFIED','distribution_version':importlib.metadata.version('cocopye'),'official_source_path':str(Path(core.__file__)),'official_source_sha256':sha(core.__file__),'parser_script':str(Path(__file__).relative_to(ROOT)),'parser_sha256':sha(__file__),'official_api_checked_on_every_full_csv_row':True,'official_paper':'https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae079/7841111','numbering_note':'Paper estimation stages I/II correspond to API/CSV marker/neural stages2/3; CSV stage1 is prefilter rejection.','stage_rules':{'1':'unscored; API sentinel−1 retained separately; quantitative columns NaN','2':'2_completeness/2_contamination ×100','3':'3_completeness/3_contamination ×100'},'clipping':'None','original_outputs_modified':False,'inventory_only':args.inventory_only}
    (OUT/'parser_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(inv[inv.role=='accuracy'][['raw_path','rows','n_stage1','n_stage2','n_stage3','quantitative_rows_changed']].to_string(index=False));print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
