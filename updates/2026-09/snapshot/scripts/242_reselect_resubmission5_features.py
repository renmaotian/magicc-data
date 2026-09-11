#!/usr/bin/env python3
"""Reselect training-only k-mers, shared by both new matched model arms.

Reuse the validated canonical 9-mer prevalence implementation of script 125.
Only panel-free training representatives participate. Unlike the former holdout
experiments, these features are the actual inputs to both newly trained models.
"""
import hashlib
import json
import sys
import argparse
import numpy as np
from resubmission5_holdout_config import OUT,ROOT,configure,load_script

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    c=configure(selection=True)
    import pandas as pd
    tr=pd.read_csv(c.TRAIN_TSV,sep='\t')
    all_ids=set(tr.ncbi_accession)
    representatives=[]
    for domain in ('bacterial','archaeal'):
        d=pd.read_csv(c.KMER_DIR/f'selected_{domain}_1000.tsv',sep='\t')
        assert set(d.ncbi_accession)<=all_ids
        c.add_taxon_column(d)
        d=d[~d.family.isin(c.PANEL_TAXA)].copy()
        d['feature_domain']=domain
        representatives.append(d)
    pd.concat(representatives).to_csv(OUT/'features/feature_selection_representatives.tsv',sep='\t',index=False)
    implementation=load_script('125','r5_reselection')
    selections=[];original_selections=[];checks=[]
    for domain,n in [('bacterial',9000),('archaeal',1000)]:
        allreps=pd.read_csv(c.KMER_DIR/f'selected_{domain}_1000.tsv',sep='\t')
        c.add_taxon_column(allreps)
        core=pd.read_csv(c.KMER_DIR/f'{domain}_core_genes/core_gene_results.tsv',sep='\t')
        core.core_gene_path=core.core_gene_path.map(c.remap_fasta_path)
        allreps=allreps.merge(core[['accession','core_gene_path']],left_on='ncbi_accession',right_on='accession',validate='one_to_one')
        assert len(allreps)==1000
        allprev,n_all=implementation.prevalence(allreps.core_gene_path.tolist(),args.workers)
        keep=allreps[~allreps.family.isin(c.PANEL_TAXA)]
        prev,n_kept=implementation.prevalence(keep.core_gene_path.tolist(),args.workers)
        assert n_kept==len(keep) and n_all==1000
        stored=pd.read_csv(c.KMER_DIR/f'{domain}_kmer_prevalence.tsv',sep='\t')
        codes=np.array([implementation.encode_kmer(k) for k in stored.kmer])
        maxdiff=int(np.max(np.abs(allprev[codes]-stored.prevalence.to_numpy())))
        assert maxdiff==0, f'Cannot reproduce {domain} production prevalence: {maxdiff}'
        original_sel,_=implementation.top_n(allprev,n);original_selections.extend(original_sel.tolist())
        sel,_=implementation.top_n(prev,n)
        selections.extend(sel.tolist())
        obs=np.flatnonzero(prev>0)
        pd.DataFrame({'kmer':[implementation.code_to_kmer(int(i)) for i in obs],
                      'prevalence_nonpanel':prev[obs],'prevalence_all':allprev[obs]}).to_csv(
                          OUT/f'features/kmer_prevalence_{domain}.tsv',sep='\t',index=False)
        checks.append({'domain':domain,'original_representatives':n_all,'panel_free_representatives':n_kept,
                       'production_prevalence_max_difference':maxdiff,'number_selected':n})
    reproduced={implementation.code_to_kmer(i) for i in original_selections}
    original=set((c.KMER_DIR/'selected_kmers.txt').read_text().splitlines())
    assert reproduced==original,'The original prevalence/tie-breaking rule did not reproduce the production vocabulary.'
    audit={'reproduced_original_union_features':len(reproduced),'original_features':len(original),
           'intersection':len(reproduced&original),'original_only':len(original-reproduced),'recount_only':len(reproduced-original),
           'original_selection_exactly_reproduced':True,
           'method':'Reapply prevalence-descending/canonical-code-ascending tie break to independently verified original prevalence tables.'}
    (OUT/'features/original_selection_rule_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    codes=sorted(set(selections))
    (OUT/'features/selected_kmers_holdout.txt').write_text('\n'.join(implementation.code_to_kmer(i) for i in codes)+'\n')
    (OUT/'features/reselection_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    p=OUT/'features/selected_kmers_holdout.txt'
    spec={'n_features':len(p.read_text().splitlines()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'selection_scope':'Only nonpanel representatives from original training split; top 9000 bacterial +1000 archaeal canonical k-mers, merged.',
          'model_inputs':'Both holdout and matched_full models use this exact reselected vocabulary.',
          'production_v5':'Frozen production model retained as a separate descriptive comparator with its original vocabulary.',
          'normalization':'Each new model fits its own normalization on all and only its training samples.',
          'split_leakage_checks':'Every representative accession confirmed in original training split; no primary-panel family survives feature selection.'}
    (OUT/'features/model_input_contract.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(spec,indent=2))
if __name__=='__main__':main()
