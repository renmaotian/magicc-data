#!/usr/bin/env python3
"""Read-only CAMI marine source-scope audit; no prediction/truth mutation."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/holdout_resubmission5/cami_source_audit'
CAMI=ROOT/'data/real_data/cami2'
RES=ROOT/'results/revision/cami2'
MEMBERSHIP=ROOT/'results/revision/cocopye_stage_resubmission5/cami_pairwise_coverage/membership.tsv'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MAGICC-resubmission-source-audit'})
    with urllib.request.urlopen(req,timeout=30) as r:return r.read()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    upstream=OUT/'upstream';upstream.mkdir(exist_ok=True)
    commit_path=upstream/'github_commit.json'
    if not commit_path.exists():
        commit_path.write_bytes(get('https://api.github.com/repos/CAMI-challenge/second_challenge_evaluation/commits/master'))
    commit=json.loads(commit_path.read_text())['sha'];sources=[]
    for name in ['README.md','add_plasmids.py']:
        p=upstream/name
        url=f'https://raw.githubusercontent.com/CAMI-challenge/second_challenge_evaluation/{commit}/scripts/data_generation/{name}'
        if not p.exists():p.write_bytes(get(url))
        sources.append({'path':str(p.relative_to(ROOT)),'url':url,'sha256':sha(p)})
    meta_path=CAMI/'marine/setup/simulation_short_read/metadata.tsv'
    meta=pd.read_csv(meta_path,sep='\t');assert meta.genome_ID.is_unique
    meta['source_scope']=meta.genome_ID.map(lambda s:'circular_element' if s.startswith('RNODE_') else 'microbial_genome' if s.startswith('Otu') else 'unresolved')
    assert not (meta.source_scope=='unresolved').any()
    elements=meta[meta.source_scope=='circular_element'].copy()
    assert len(elements)==200 and len(meta)==977
    assert elements.novelty_category.value_counts().to_dict()=={'plasmid':108,'unknown':88,'virus':4}
    assert set(elements.loc[elements.novelty_category=='unknown','NCBI_ID'])=={32644}
    elements['legacy_excluded_by_novelty_label']=elements.novelty_category.isin(['plasmid','virus'])
    elements.to_csv(OUT/'marine_circular_element_manifest.tsv',sep='\t',index=False)
    meta.groupby(['source_scope','novelty_category'],dropna=False).size().rename('n_sources').reset_index().to_csv(OUT/'source_category_counts.tsv',sep='\t',index=False)
    original_audit=RES/'provenance/marine_source_genome_audit.tsv'
    audit=pd.read_csv(original_audit,sep='\t')
    retained=audit.merge(meta[['genome_ID','source_scope']],left_on='genome',right_on='genome_ID',validate='one_to_one')
    assert len(retained)==len(audit)
    retained[retained.source_scope=='circular_element'].to_csv(OUT/'legacy_retained_circular_elements.tsv',sep='\t',index=False)
    membership=pd.read_csv(MEMBERSHIP,sep='\t');membership=membership[membership.dataset=='marine'].copy()
    gold_path=RES/'truth/marine_gold_truth.tsv';mixed_path=RES/'truth/marine_mixed_truth.tsv'
    gold=pd.read_csv(gold_path,sep='\t');mixed=pd.read_csv(mixed_path,sep='\t')
    gt=gold[['bin_id','genome','completeness_pct','excluded_non_genome']].rename(columns={'genome':'dominant'})
    gt['binset']='gold';gt['contaminant']='';gt['distance_rank']=''
    mt=mixed[['bin_id','dominant','contaminant','distance_rank','completeness_pct']].copy();mt['binset']='mixed';mt['excluded_non_genome']=False
    truth=pd.concat([gt,mt],ignore_index=True)
    joined=membership.merge(truth,on=['binset','bin_id'],validate='one_to_one',how='left')
    assert joined.dominant.notna().all()
    element_ids=set(elements.genome_ID)
    joined['dominant_circular_element']=joined.dominant.isin(element_ids)
    joined['contaminant_circular_element']=joined.contaminant.isin(element_ids)
    joined['any_circular_element']=joined.dominant_circular_element|joined.contaminant_circular_element
    joined.to_csv(OUT/'pairwise_membership_source_scope.tsv',sep='\t',index=False)
    counts=joined.groupby(['binset','four_tool_common','any_circular_element'],dropna=False).size().rename('n_bins').reset_index()
    counts.to_csv(OUT/'pairwise_source_scope_counts.tsv',sep='\t',index=False)
    assert not joined.loc[joined.four_tool_common,'any_circular_element'].any()
    assert joined.loc[~joined.four_tool_common,'dominant_circular_element'].all()
    tax_records=[]
    for name in ['nodes.dmp','names.dmp']:
        p=ROOT/'tools/kraken2_db_standard'/name
        with open(p) as f:
            for line in f:
                if line.startswith('32644\t'):
                    tax_records.append({'file':name,'line':line.rstrip()})
    (OUT/'local_taxid_32644_records.json').write_text(json.dumps(tax_records,indent=2)+'\n')
    for p in [meta_path,CAMI/'marine/setup/simulation_short_read/genome_to_id.tsv',original_audit,MEMBERSHIP,gold_path,mixed_path,
              ROOT/'scripts/195_ws38_fetch_cami2.py',ROOT/'scripts/196_ws38_cami2_truth_and_bins.py',
              ROOT/'scripts/197_ws38_cami2_leakage_audit.py',ROOT/'scripts/201_ws38_cami2_analysis.py',Path(__file__)]:
        sources.append({'path':str(p.relative_to(ROOT)),'url':'','sha256':sha(p)})
    pd.DataFrame(sources).to_csv(OUT/'provenance.tsv',sep='\t',index=False)
    summary={'status':'SOURCE_SCOPE_AUDIT_COMPLETE','audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'official_generation_commit':commit,'metadata_sources':len(meta),'microbial_genome_sources':777,
        'circular_element_sources':200,'circular_categories':elements.novelty_category.value_counts().to_dict(),
        'legacy_source_audit_rows':len(audit),'legacy_retained_unknown_circular_elements':int((retained.source_scope=='circular_element').sum()),
        'pairwise_counts':counts.to_dict('records'),
        'exclusive_mixed_distance_counts':joined.loc[(~joined.four_tool_common)&(joined.binset=='mixed'),'distance_rank'].value_counts().to_dict(),
        'conclusion':'All pairwise-exclusive marine bins have circular-element dominant references. The former plasmid/virus-only filter missed 88 unknown circular elements; catchall taxid 32644 is not biological evidence of species relatedness.',
        'scope_rule':'CAMI source identity and official generation provenance, independent of sequence length, prediction values or comparator success.',
        'inputs_or_predictions_modified':False}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
