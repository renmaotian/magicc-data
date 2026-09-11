#!/usr/bin/env python3
"""Verify existing FASTAs/raw scores for all354 marine coverage-exclusive bins."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';DEST=OUT/'cami_sequence_input_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True);inputs=[];rows=[]
    mp=OUT/'cami_pairwise_coverage/membership.tsv';lp=OUT/'cami_microbial_scope/before_scope/cami2_long_predictions.tsv'
    m=pd.read_csv(mp,sep='\t');d=pd.read_csv(lp,sep='\t');inputs +=[mp,lp,Path(__file__).resolve()]
    primary=d[d.tool.eq('MAGICC_V5')&d.in_domain&d.scoreable&d.leakage_free&d.all_tools_scored]
    assert not primary.dominant.fillna('').str.startswith('RNODE').any()
    assert not primary.contaminant.fillna('').str.startswith('RNODE').any()
    exclusive=m[m.dataset.eq('marine')&~m.four_tool_common];assert len(exclusive)==354
    for bs in ['gold','mixed']:
        cp=ROOT/f'results/revision/cami2/provenance/marine_{bs}_competitor_cohort.tsv';rp=ROOT/f'results/revision/cami2/competitors/marine_{bs}/checkm2_output/quality_report.tsv';tp=ROOT/f'results/revision/cami2/truth/marine_{bs}_truth.tsv'
        cohort=pd.read_csv(cp,sep='\t').set_index('bin_id');raw=pd.read_csv(rp,sep='\t').set_index('Name');truth=pd.read_csv(tp,sep='\t').set_index('bin_id');inputs +=[cp,rp,tp]
        z=exclusive[exclusive.binset.eq(bs)];assert set(z.bin_id)<=set(cohort.index)&set(raw.index)&set(truth.index)
        for r in z.itertuples():
            p=Path(cohort.loc[r.bin_id,'path']);count=Counter();ncontigs=0
            with p.open() as f:
                for line in f:
                    if line.startswith('>'):ncontigs+=1
                    else:count.update(line.strip().upper())
            n=sum(count.values());assert n>0 and ncontigs>0 and n==int(raw.loc[r.bin_id,'Genome_Size'])
            saved=d[d.dataset.eq('marine')&d.binset.eq(bs)&d.bin_id.eq(r.bin_id)&d.tool.eq('CheckM2')].iloc[0]
            for metric,rawcol in [('completeness','Completeness'),('contamination','Contamination')]:assert saved['pred_'+metric]==raw.loc[r.bin_id,rawcol]
            rows.append(dict(dataset='marine',binset=bs,bin_id=r.bin_id,dominant=r.cluster,contaminant=truth.loc[r.bin_id,'contaminant'] if bs=='mixed' else '',comparator_input_selected=True,fasta_path=str(p.relative_to(ROOT)),fasta_sha256=sha(p),n_contigs=ncontigs,total_bp=n,acgt_bp=sum(count[b] for b in 'ACGT'),ambiguous_bp=sum(v for b,v in count.items() if b not in 'ACGT'),cocopye_prediction_row=r.cocopye_has_row,cocopye_stage=r.cocopye_stage,raw_checkm2_completeness=raw.loc[r.bin_id,'Completeness'],raw_checkm2_contamination=raw.loc[r.bin_id,'Contamination'],raw_checkm2_coding_density=raw.loc[r.bin_id,'Coding_Density'],raw_checkm2_n_cds=raw.loc[r.bin_id,'Total_Coding_Sequences'],scope='RNODE circular-element reference; not a prokaryotic genome-quality validation unit'))
    a=pd.DataFrame(rows);a.to_csv(DEST/'sequence_and_raw_score_checks.tsv',sep='\t',index=False)
    summary=dict(status='CAMI_SEQUENCE_INPUT_AUDIT_COMPLETE',n_exclusive_bins=354,n_gold=148,n_mixed=206,all_selected_as_comparator_inputs=True,all_fasta_bp_match_raw_checkm2_genome_size=True,all_raw_checkm2_predictions_match_merged_rows=True,n_primary_rnode_dominants=0,n_primary_rnode_donors=0,minimum_bp=int(a.total_bp.min()),maximum_bp=int(a.total_bp.max()),total_ambiguous_bp=int(a.ambiguous_bp.sum()),interpretation='Existing FASTAs contain real sequence and raw scores, but their source units are CAMI RNODE circular elements. Real sequence does not make their completeness denominator or catchall species taxid suitable for prokaryotic genome quality or within-species inference. CoCoPyE absent-row bins were input selected; without process-level evidence, the exact cause of absent rows is not assigned.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],output_manifest=[dict(path=str((DEST/'sequence_and_raw_score_checks.tsv').relative_to(ROOT)),sha256=sha(DEST/'sequence_and_raw_score_checks.tsv'))])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if not k.endswith('manifest')},indent=2))
if __name__=='__main__':main()
