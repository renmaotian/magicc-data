#!/usr/bin/env python3
"""Generate paired test-only evaluation genomes and both feature vocabularies."""
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from resubmission7_holdout_config import OUT,ROOT,configure,load_script

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

def atomic_json(path,obj):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)

def preparation_identity(c,smoke):
    files=[Path(__file__),ROOT/'scripts/123_generate_holdout_eval_sets.py',
        ROOT/'scripts/resubmission7_holdout_config.py',c.SELECTED_KMERS,
        ROOT/'data/kmer_selection/selected_kmers.txt',OUT/'panel_design.json',
        OUT/'lineage_selection_manifest.tsv',c.TRAIN_TSV,c.VAL_TSV,c.TEST_TSV,
        ROOT/'results/revision/holdout_resubmission5/reconstruction/genomic_source_manifest.tsv']
    files.extend(sorted((ROOT/'scripts/holdout_lib').glob('*.py')))
    return {'smoke':smoke,'seed':int(c.EVAL_SEED_BASE),'design':c.EVAL_DESIGN,
        'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'source_scope':'Original source-manifest identity is bound here; actual source FASTA byte validation is performed by audit290.'}

def validate_group(directory,group,spec,c,smoke,complete=False):
    """Reject partial/incompatible data before any reuse or bookkeeping write."""
    metadata=pd.read_csv(directory/'metadata.tsv',sep='\t')
    nrefs,sims=(2,2) if smoke else (int(spec['n_refs']),int(spec['sims']))
    expected=nrefs*sims
    required={'genome_id','group','dominant_accession','dominant_v5_split',
        'contaminant_accessions','true_completeness','true_contamination','fasta'}
    assert required<=set(metadata.columns),f'Incomplete metadata schema: {group}'
    assert len(metadata)==expected and (metadata.group==group).all()
    assert metadata.genome_id.tolist()==[f'genome_{i}' for i in range(expected)]
    assert metadata.dominant_accession.nunique()==nrefs
    assert (metadata.groupby('dominant_accession').size()==sims).all()
    truth=metadata[['true_completeness','true_contamination']].to_numpy()
    assert np.isfinite(truth).all()
    ids=metadata.genome_id.to_numpy(dtype='S40')
    for row in metadata.itertuples():
        path=Path(c.remap_fasta_path(row.fasta))
        assert path.resolve()==(directory/'fasta'/f'{row.genome_id}.fasta').resolve()
        assert path.is_file(),f'Missing generated FASTA: {path}'
    for name,nfeatures in [('features.h5',c.N_KMER_FEATURES),('production_features.h5',9249)]:
        path=directory/name
        if not path.exists():
            assert name=='production_features.h5' and not complete,f'Incomplete evaluation artifact: {path}'
            continue
        with h5py.File(path,'r') as f:
            assert f['kmer_counts_raw'].shape==(expected,nfeatures)
            assert f['kmer_counts_raw'].dtype==np.dtype('int64')
            assert f['summary_features_raw'].shape==(expected,7)
            assert f['summary_features_raw'].dtype==np.dtype('float64')
            assert np.isfinite(f['summary_features_raw'][:]).all()
            if name=='features.h5':
                assert f['labels'].shape==(expected,2) and f['labels'].dtype==np.dtype('float32')
                assert np.allclose(f['labels'][:],truth,atol=1e-4,rtol=0)
            else:assert f.attrs['selected_kmers_sha256']==sha(ROOT/'data/kmer_selection/selected_kmers.txt')
            if complete:assert 'genome_ids' in f
            if 'genome_ids' in f:assert np.array_equal(f['genome_ids'][:],ids)
    return metadata

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();c=configure('holdout')
    base=c.HOLDOUT_DIR/'eval_sets_smoke' if args.smoke else c.EVAL_DIR
    base.mkdir(parents=True,exist_ok=True)
    preparation_lock=(base/'.prepare.lock').open('a+')
    fcntl.flock(preparation_lock,fcntl.LOCK_EX)
    identity=preparation_identity(c,args.smoke);status_path=base/'preparation_status.json'
    prior=json.loads(status_path.read_text()) if status_path.exists() else None
    if prior is not None:
        assert prior['identity']==identity,'Evaluation preparation identity/configuration changed.'
        assert prior['status'] in ['PREPARING','EVALUATION_INPUTS_COMPLETE']
        if prior['status']=='EVALUATION_INPUTS_COMPLETE':
            for name,expected_sha in prior['artifact_sha256'].items():
                assert sha(base/name)==expected_sha,f'Completed evaluation artifact changed: {name}'
    else:
        existing=any((base/group/name).exists() for group in c.EVAL_DESIGN for name in ['metadata.tsv','features.h5'])
        assert args.smoke or not existing,'Existing full inputs have no preparation identity contract.'
        atomic_json(status_path,{'status':'PREPARING','identity':identity,
            'legacy_smoke_inputs_validated_on_first_contract':bool(args.smoke and existing)})
    source_test=pd.read_csv(c.TEST_TSV,sep='\t')
    missing=[p for p in source_test.fasta_path.map(c.remap_fasta_path) if not Path(p).is_file()]
    assert not missing,f'Evaluation source pool is incomplete: {len(missing)} missing FASTAs; first {missing[:3]}'
    for group,spec in c.EVAL_DESIGN.items():
        directory=base/group;exists=[(directory/name).exists() for name in ['metadata.tsv','features.h5']]
        assert not any(exists) or all(exists),f'Partial group inputs require recovery: {group}'
        if all(exists):validate_group(directory,group,spec,c,args.smoke,complete=bool(prior and prior['status']=='EVALUATION_INPUTS_COMPLETE'))
    # A completed preparation is checked read-only. This preserves its original
    # generator manifest and exact HDF5 bytes when launcher288 reaches this step.
    if any(not (base/group/name).exists() for group in c.EVAL_DESIGN
            for name in ['metadata.tsv','features.h5']):
        original=sys.argv[:];sys.argv=[sys.argv[0],'--workers',str(args.workers)]+(['--smoke'] if args.smoke else [])
        try:load_script('123','r7_eval_generator').main()
        finally:sys.argv=original
    train=pd.read_csv(c.TRAIN_TSV,sep='\t');val=pd.read_csv(c.VAL_TSV,sep='\t');test=pd.read_csv(c.TEST_TSV,sep='\t')
    forbidden=set(train.ncbi_accession)|set(val.ncbi_accession);allowed=set(test.ncbi_accession)
    c.add_taxon_column(test)
    panel=set(test.loc[test.genus.isin(c.PANEL_TAXA),'ncbi_accession'])
    manifest=[]
    for group,spec in c.EVAL_DESIGN.items():
        directory=base/group;metadata=validate_group(directory,group,spec,c,args.smoke)
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
        # identity as well as count correctness on every shared k-mer.
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
            ids=metadata.genome_id.to_numpy(dtype='S40')
            attributes={'row_identity_verification':'Independent FASTA recount: all shared k-mer counts agree row-for-row',
                'shared_kmer_columns_verified':len(common)}
            with h5py.File(artifact,'r') as f:
                missing_ids='genome_ids' not in f
                if not missing_ids:assert np.array_equal(f['genome_ids'][:],ids)
                missing_attributes={key:value for key,value in attributes.items() if key not in f.attrs}
                for key,value in attributes.items():
                    if key in f.attrs:assert f.attrs[key]==value
            if missing_ids or missing_attributes:
                with h5py.File(artifact,'a') as f:
                    if missing_ids:f.create_dataset('genome_ids',data=ids)
                    for key,value in missing_attributes.items():f.attrs[key]=value
        manifest.append({'group':group,'n_samples':len(metadata),'n_refs':metadata.dominant_accession.nunique(),
                         'metadata_sha256':sha(directory/'metadata.tsv'),'new_features_sha256':sha(directory/'features.h5'),
                         'production_features_sha256':sha(p),
                         'shared_kmer_columns_verified':len(common),
                         'domain_violations':int(((metadata.true_completeness<50)|(metadata.true_contamination>metadata.true_completeness+1e-6)).sum())})
    pd.DataFrame(manifest).to_csv(base/'verification_manifest.tsv',sep='\t',index=False)
    artifacts=[base/'manifest.json',base/'verification_manifest.tsv']
    for group,spec in c.EVAL_DESIGN.items():
        directory=base/group;metadata=validate_group(directory,group,spec,c,args.smoke,complete=True)
        artifacts.extend(directory/name for name in ['metadata.tsv','features.h5','production_features.h5'])
        artifacts.extend(Path(c.remap_fasta_path(path)) for path in metadata.fasta)
    hashes={str(path.relative_to(base)):sha(path) for path in artifacts}
    if prior and prior['status']=='EVALUATION_INPUTS_COMPLETE':
        assert hashes==prior['artifact_sha256'],'Completed evaluation artifacts changed during read-only reuse.'
        print('Reused exact verified evaluation inputs; no simulation or model inference.',flush=True)
    else:
        atomic_json(status_path,{'status':'EVALUATION_INPUTS_COMPLETE','identity':identity,
            'artifact_sha256':hashes,'n_samples':sum(row['n_samples'] for row in manifest),
            'n_references':sum(row['n_refs'] for row in manifest),
            'scope':'Simulation/features only; no model predictions or accuracy evaluation.'})
    print(pd.DataFrame(manifest)[['group','n_samples','n_refs','domain_violations']].to_string(index=False),flush=True)
if __name__=='__main__':
    mp.set_start_method('fork',force=True);main()
