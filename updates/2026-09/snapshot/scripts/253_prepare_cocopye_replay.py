#!/usr/bin/env python3
"""Prepare isolated re-analysis inputs without modifying historical artifacts."""
import hashlib,json,shutil,os
from pathlib import Path
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';TEXT={'.py','.yaml','.yml','.tsv','.csv','.json','.md','.txt','.sh'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def link(src,dst):
    if dst.exists() or dst.is_symlink():return
    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(os.path.relpath(src,dst.parent),target_is_directory=src.is_dir())
def copy_tables(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
        q=dst/p.name
        if q.exists() or q.is_symlink():continue
        if p.is_dir():
            if p.name in {'figures','plots','__pycache__','.stage_done'}:q.mkdir(exist_ok=True)
            elif p.name in {'fasta','genomes','checkm2_output','checkpoints_pilot','checkpoints_full','alignments','raw','competitors','proteins'}:link(p,q)
            else:copy_tables(p,q)
        elif p.suffix in TEXT or p.name.endswith('.tsv.gz'):shutil.copy2(p,q)
        else:link(p,q)

def main():
    assert json.loads((OUT/'parser_verification.json').read_text())['status']=='PARSER_VERIFIED'
    WORK.mkdir(parents=True,exist_ok=True)
    for name in ['magicc','models','tools','nature_communications','configs','tests']: 
        if (ROOT/name).exists():link(ROOT/name,WORK/name)
    (WORK/'data').mkdir(exist_ok=True)
    for p in (ROOT/'data').iterdir():
        if p.name=='benchmarks':copy_tables(p,WORK/'data/benchmarks')
        else:link(p,WORK/'data'/p.name)
    affected={'metrics','set_F','set_G','real_data','cami2','ws11'}
    (WORK/'results/revision').mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'results/revision').iterdir():
        if p==OUT:continue
        q=WORK/'results/revision'/p.name
        if p.name in affected and p.is_dir():copy_tables(p,q)
        else:link(p,q)
    scripts=WORK/'scripts';scripts.mkdir(exist_ok=True)
    copied=[]
    for p in (ROOT/'scripts').iterdir():
        if not p.is_file() or p.suffix not in TEXT:continue
        q=scripts/p.name;q.write_text(p.read_text().replace(str(ROOT),str(WORK)))
        copied.append({'original':str(p.relative_to(ROOT)),'original_sha256':sha(p),'replay_path':str(q.relative_to(ROOT)),'replay_sha256':sha(q),'modification':'Redirect literal project root to isolated replay root only'})
    cfgpath=scripts/'config_revision_metrics.yaml';cfg=yaml.safe_load(cfgpath.read_text());cfg['project_root']=str(WORK)
    # Keep the original explicitly named benchmark families, excluding datasets
    # added after the historical definitive analysis via auto-discovery.
    cfg['autodiscover']['enabled']=False;cfg['n_jobs']=4
    cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
    artifacts=pd.read_csv(OUT/'corrected_prediction_manifest.tsv',sep='\t');updates=[]
    for row in artifacts.itertuples():
        raw=Path(row.raw_path);corrected=pd.read_csv(ROOT/row.corrected_path,sep='\t');canonical=corrected.set_index('genome_id')
        if str(raw).startswith('data/benchmarks/'):
            target=WORK/raw.parent/('cocopye_predictions.tsv' if raw.name=='cocopye_raw_output.csv' else raw.stem+'_selected.tsv')
            corrected.to_csv(target,sep='\t',index=False);updates.append({'consumer':str(target.relative_to(ROOT)),'source':row.corrected_path,'n_rows':len(corrected)})
        elif 'real_data' in raw.parts:
            target=WORK/raw.parent/'predictions.tsv';d=pd.read_csv(target,sep='\t')
            assert d.genome_id.is_unique and not d.genome_id.isna().any()
            attempted = d.genome_id.isin(canonical.index)
            if raw.parent.name=='meslier':expected_attempted=d.bin_fasta.notna() & d.bin_fasta.ne('')
            elif raw.parent.name=='ncbi_pairs':expected_attempted=d.status.eq('ok')
            else:expected_attempted=pd.Series(True,index=d.index)
            assert np.array_equal(attempted,expected_attempted), f'Unmatched eligible real-data IDs: {target}'
            for metric in ['completeness','contamination']:
                d[f'cocopye_{metric}']=d.genome_id.map(canonical[f'pred_{metric}'])
            d['cocopye_selected_stage']=d.genome_id.map(canonical.selected_stage)
            d['cocopye_tool_attempted']=attempted
            d['cocopye_tool_scored']=d.genome_id.map(canonical.tool_scored).eq(True)
            assert d.cocopye_tool_scored.notna().all()
            assert np.array_equal(d.cocopye_completeness.isna(),d.cocopye_selected_stage.eq(1) | ~attempted)
            assert np.array_equal(d.cocopye_contamination.isna(),d.cocopye_selected_stage.eq(1) | ~attempted)
            if raw.parent.name=='ncbi_pairs':
                cm=pd.read_csv(ROOT/raw.parent/'checkm2_output/quality_report.tsv',sep='\t').set_index('Name')
                d['checkm2_completeness']=d.genome_id.map(cm.Completeness);d['checkm2_contamination']=d.genome_id.map(cm.Contamination)
            d.to_csv(target,sep='\t',index=False);updates.append({'consumer':str(target.relative_to(ROOT)),'source':row.corrected_path,'n_rows':len(d)})
        elif 'cami2' in raw.parts and 'competitors' in raw.parts:
            tag=raw.parent.name;target=WORK/'results/revision/cami2/predictions'/f'{tag}_all_tools.tsv';d=pd.read_csv(target,sep='\t')
            canonical.index=canonical.index.str.replace(r'\.fasta$','',regex=True);mask=d.tool.eq('CoCoPyE')
            assert canonical.index.is_unique and d.loc[mask,'bin_id'].is_unique
            assert set(d.loc[mask,'bin_id'])<=set(canonical.index), f'Unmatched CAMI IDs: {target}'
            for metric in ['completeness','contamination']:d.loc[mask,f'pred_{metric}']=d.loc[mask,'bin_id'].map(canonical[f'pred_{metric}'])
            d['selected_stage']=np.nan;d.loc[mask,'selected_stage']=d.loc[mask,'bin_id'].map(canonical.selected_stage)
            assert np.array_equal(d.loc[mask,'pred_completeness'].isna(),d.loc[mask,'selected_stage'].eq(1))
            assert np.array_equal(d.loc[mask,'pred_contamination'].isna(),d.loc[mask,'selected_stage'].eq(1))
            d.to_csv(target,sep='\t',index=False);updates.append({'consumer':str(target.relative_to(ROOT)),'source':row.corrected_path,'n_rows':len(d)})
    pd.DataFrame(copied).to_csv(OUT/'replay_script_manifest.tsv',sep='\t',index=False)
    pd.DataFrame(updates).to_csv(OUT/'prediction_consumer_map.tsv',sep='\t',index=False)
    (OUT/'replay_setup.json').write_text(json.dumps({'status':'REPLAY_PREPARED','workspace':str(WORK.relative_to(ROOT)),'originals_modified':False,'autodiscovery':'Disabled; all original explicitly configured benchmark sets retained; F/G and external analyses run separately.','script_paths_redirected':len(copied),'prediction_tables_updated':len(updates)},indent=2)+'\n')
    print(f'Prepared {WORK}; {len(updates)} corrected prediction consumers')
if __name__=='__main__':main()
