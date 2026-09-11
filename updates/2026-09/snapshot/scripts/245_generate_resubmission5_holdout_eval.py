#!/usr/bin/env python3
"""Generate paired test-only evaluation genomes and both feature vocabularies."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from resubmission5_holdout_config import OUT,ROOT,configure,load_script

INDEX=None
NK=None

def prod_init():
    global INDEX,NK
    from holdout_lib.kmer_counter import load_selected_kmers,build_kmer_index
    codes=load_selected_kmers(str(ROOT/'data/kmer_selection/selected_kmers.txt'))
    INDEX=build_kmer_index(codes);NK=len(codes)

def count_production(path):
    from holdout_lib.fragmentation import load_original_contigs
    from holdout_lib.kmer_counter import _count_kmers_single,K
    from holdout_lib.assembly_stats import compute_assembly_stats
    total=np.zeros(NK,dtype=np.int64)
    for c in load_original_contigs(path):
        if len(c)>=K:total+=_count_kmers_single(np.frombuffer(c.encode('ascii'),dtype=np.uint8),INDEX,NK,K)
    s=total.sum();summary=compute_assembly_stats(np.log10(float(s)) if s else 0.,total)
    return total,summary

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();c=configure('holdout')
    source_test=pd.read_csv(c.TEST_TSV,sep='\t')
    missing=[p for p in source_test.fasta_path.map(c.remap_fasta_path) if not Path(p).is_file()]
    assert not missing,f'Evaluation source pool is incomplete: {len(missing)} missing FASTAs; first {missing[:3]}'
    original=sys.argv[:];sys.argv=[sys.argv[0],'--workers',str(args.workers)]+(['--smoke'] if args.smoke else [])
    load_script('123','r5_eval_generator').main();sys.argv=original
    base=c.HOLDOUT_DIR/'eval_sets_smoke' if args.smoke else c.EVAL_DIR
    train=pd.read_csv(c.TRAIN_TSV,sep='\t');val=pd.read_csv(c.VAL_TSV,sep='\t');test=pd.read_csv(c.TEST_TSV,sep='\t')
    forbidden=set(train.ncbi_accession)|set(val.ncbi_accession);allowed=set(test.ncbi_accession)
    c.add_taxon_column(test)
    panel=set(test.loc[test.family.isin(c.PANEL_TAXA),'ncbi_accession'])
    manifest=[]
    for group,spec in c.EVAL_DESIGN.items():
        directory=base/group;metadata=pd.read_csv(directory/'metadata.tsv',sep='\t')
        expected=4 if args.smoke else spec['n_refs']*spec['sims']
        assert len(metadata)==expected and metadata.genome_id.is_unique
        assert set(metadata.dominant_accession)<=allowed and not(set(metadata.dominant_accession)&forbidden)
        assert (metadata.dominant_v5_split=='test').all()
        contam={a for text in metadata.contaminant_accessions.fillna('') for a in text.split(';') if a}
        assert contam<=allowed and not(contam&forbidden) and not(contam&panel)
        with h5py.File(directory/'features.h5','r') as f:
            assert f['kmer_counts_raw'].shape==(expected,c.N_KMER_FEATURES)
            assert np.allclose(f['labels'][:],metadata[['true_completeness','true_contamination']],atol=1e-4)
        p=directory/'production_features.h5'
        if not p.exists():
            with mp.Pool(args.workers,initializer=prod_init) as pool:
                data=list(pool.imap(count_production,metadata.fasta.map(c.remap_fasta_path).tolist(),chunksize=8))
            temp=Path(str(p)+'.tmp')
            with h5py.File(temp,'w') as f:
                f.create_dataset('kmer_counts_raw',data=np.stack([x[0] for x in data]),compression='gzip',compression_opts=1)
                f.create_dataset('summary_features_raw',data=np.stack([x[1] for x in data]))
                f.attrs['selected_kmers_sha256']=sha(ROOT/'data/kmer_selection/selected_kmers.txt')
            os.replace(temp,p)
        with h5py.File(p,'r') as f:assert f['kmer_counts_raw'].shape==(expected,9249)
        # Independent FASTA recount under the production vocabulary checks row
        # identity as well as count correctness on every shared k-mer (8575).
        # The counter sorts canonical integer codes; A/C/G/T lexical order agrees.
        new_kmers=sorted(c.SELECTED_KMERS.read_text().splitlines())
        prod_kmers=sorted((ROOT/'data/kmer_selection/selected_kmers.txt').read_text().splitlines())
        new_pos={k:i for i,k in enumerate(new_kmers)};prod_pos={k:i for i,k in enumerate(prod_kmers)}
        common=sorted(set(new_pos)&set(prod_pos))
        with h5py.File(directory/'features.h5','r') as f, h5py.File(p,'r') as g:
            for start in range(0,expected,128):
                left=f['kmer_counts_raw'][start:start+128][:,[new_pos[k] for k in common]]
                right=g['kmer_counts_raw'][start:start+128][:,[prod_pos[k] for k in common]]
                assert np.array_equal(left,right),f'Feature row/count mismatch in {group}'
        for artifact in [directory/'features.h5',p]:
            with h5py.File(artifact,'a') as f:
                ids=metadata.genome_id.to_numpy(dtype='S40')
                if 'genome_ids' in f:assert np.array_equal(f['genome_ids'][:],ids)
                else:f.create_dataset('genome_ids',data=ids)
                f.attrs['row_identity_verification']='Independent FASTA recount: all shared k-mer counts agree row-for-row'
                f.attrs['shared_kmer_columns_verified']=len(common)
        manifest.append({'group':group,'n_samples':len(metadata),'n_refs':metadata.dominant_accession.nunique(),
                         'metadata_sha256':sha(directory/'metadata.tsv'),'new_features_sha256':sha(directory/'features.h5'),
                         'production_features_sha256':sha(p),
                         'shared_kmer_columns_verified':len(common),
                         'domain_violations':int(((metadata.true_completeness<50)|(metadata.true_contamination>metadata.true_completeness+1e-6)).sum())})
    pd.DataFrame(manifest).to_csv(base/'verification_manifest.tsv',sep='\t',index=False)
    print(pd.DataFrame(manifest)[['group','n_samples','n_refs','domain_violations']].to_string(index=False),flush=True)
if __name__=='__main__':
    mp.set_start_method('fork',force=True);main()
