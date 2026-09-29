#!/usr/bin/env python3
"""Audit genus exclusion, provenance, actual synthesis roles and paired estimates.

Run --preflight before synthesis; run without it after complete evaluation.
--smoke audits the deliberately undersized integration test, never study results.
--data-only checks completed synthesis/source provenance without a training claim.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zlib
import numpy as np
import pandas as pd
from resubmission7_holdout_config import OUT,ROOT

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def aliases(frame,versionless=False):
    result=set()
    for col in ['ncbi_accession','gcf_accession','gtdb_accession']:
        values=frame[col].dropna().str.replace(r'^(RS_|GB_)','',regex=True).str.replace(r'^GC[AF]_','',regex=True)
        if versionless:values=values.str.replace(r'\.\d+$','',regex=True)
        result.update(values)
    return result

def taxonomy(frame):
    for rank,prefix in [('species','s'),('genus','g'),('family','f'),('order','o'),('class','c')]:
        frame[rank]=frame.gtdb_taxonomy.str.extract(prefix+r'__([^;]+)')
    return frame

def preflight():
    spec=json.loads((OUT/'panel_design.json').read_text());panel=set(spec['genera'])
    splits={s:taxonomy(pd.read_csv(ROOT/f'data/splits/{s}_genomes.tsv',sep='\t')) for s in ('train','val','test')}
    overlap={a+'_'+b:len(aliases(splits[a])&aliases(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert sum(overlap.values())==0
    source_overlap={a+'_'+b:len(aliases(splits[a],True)&aliases(splits[b],True)) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert sum(source_overlap.values())==0
    held=pd.concat([d[d.genus.isin(panel)] for d in splits.values()],ignore_index=True)
    assert (held.groupby('genus').family.nunique()==1).all()
    assert (held.groupby('genus').phylum.nunique()==1).all()
    suffix=[]
    for genus in spec['genera']:
        related=splits['train'][splits['train'].genus.str.startswith(genus+'_',na=False)]
        suffix.append({'target_genus':genus,'retained_same_basename_suffixed_gtdb_genera':';'.join(sorted(related.genus.unique())),
            'retained_training_genomes':len(related),
            'interpretation':'Distinct GTDB genera by declared rank. Shared Latin name alone does not establish phylogenetic proximity; exclusion does not collapse historical/NCBI genus synonyms.'})
    pd.DataFrame(suffix).to_csv(OUT/'gtdb_genus_name_sensitivity.tsv',sep='\t',index=False)
    feature=taxonomy(pd.read_csv(OUT/'features/feature_selection_representatives.tsv',sep='\t'))
    records=[]
    for arm in ['holdout','matched_full']:
        for split in splits:
            d=taxonomy(pd.read_csv(OUT/arm/f'data/{split}_genomes_holdout.tsv',sep='\t'))
            expected=splits[split]
            if arm=='holdout':expected=expected[~expected.genus.isin(panel)]
            assert set(d.ncbi_accession)==set(expected.ncbi_accession)
            for rank in ['genus','species']:
                n=len(set(d[rank].dropna())&set(held[rank].dropna()))
                records.append({'pool':arm+':'+split,'rank':rank,'target_overlap_taxa':n,'n_genomes':len(d)})
                if arm=='holdout':assert n==0
            if arm=='holdout' and split=='train':
                for rank in ['family','order','class','phylum']:
                    assert set(d[rank].dropna())==set(splits['train'][rank].dropna())
    assert set(feature.ncbi_accession)<=set(splits['train'].ncbi_accession)
    for rank in ['genus','species']:
        n=len(set(feature[rank].dropna())&set(held[rank].dropna()));assert n==0
        records.append({'pool':'feature_selection','rank':rank,'target_overlap_taxa':n,'n_genomes':len(feature)})
    pd.DataFrame(records).to_csv(OUT/'taxonomic_exclusion_audit.tsv',sep='\t',index=False)
    a={'status':'PREFLIGHT_PASS','canonical_accession_split_overlaps':overlap,
       'version_insensitive_source_split_overlaps':source_overlap,
       'canonicalization':'Strip RS_/GB_ and GCA_/GCF_ prefixes, include linked gcf_accession, retain version.',
       'target_genus_and_species_overlap_in_holdout_training_validation_features':0,
       'all_training_families_orders_classes_phyla_retained':True,
       'scope':'Original genome splits are accession-independent, not globally species-independent; entire target genera/species are excluded only in the holdout arm.',
       'production_model_sha256':sha(ROOT/'models/magicc_v5.onnx')}
    assert a['production_model_sha256']=='b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'
    atomic_json(OUT/'canonical_split_audit.json',a)
    return splits

def audit_traces(splits,smoke,require_training_complete=True):
    records=[];spec=json.loads((OUT/'panel_design.json').read_text())
    for arm in ['holdout','matched_full']:
        base=OUT/arm/'data';batchdir=base/('smoke_batches' if smoke else 'raw_batches')
        manifest=json.loads((base/('smoke_build_manifest.json' if smoke else 'build_manifest.json')).read_text())
        allowed={s:set(pd.read_csv(base/f'{s}_genomes_holdout.tsv',sep='\t').ncbi_accession) for s in splits}
        seen={s:set() for s in splits};roles={s:{'dominant':set(),'donor':set()} for s in splits}
        donor_counts={'v4_samples':0,'v4_donors_planned':0,'v4_donors_loaded':0,'v4_count_mismatch_samples':0,
            'B_samples':0,'B_donors_planned':0,'B_count_fields_are_planned':True}
        for batch in manifest['raw_batches']:
            path=Path(batch['path']);assert sha(path)==batch['sha256']
            with np.load(path) as raw:metadata=raw['metadata']
            trace=path.with_suffix('.jsonl.gz')
            if 'actual_role_trace_sha256' in batch:assert sha(trace)==batch['actual_role_trace_sha256']
            with gzip.open(trace,'rt') as f:
                for row_index,line in enumerate(f):
                    r=json.loads(line);s=r['split'];assert r['row'] not in seen[s];seen[s].add(r['row'])
                    assert r['dominant_accession'] in allowed[s] and set(r['contaminant_accessions'])<=allowed[s]
                    md=metadata[row_index];planned=len(r['contaminant_accessions']);recorded=int(md['n_contaminants'])
                    assert md['dominant_accession'].decode()==r['dominant_accession']
                    assert int(md['genome_full_length'])>=500
                    if path.name.startswith('v4_'):
                        assert 0<=recorded<=planned
                        donor_counts['v4_samples']+=1;donor_counts['v4_donors_planned']+=planned
                        donor_counts['v4_donors_loaded']+=recorded
                        donor_counts['v4_count_mismatch_samples']+=int(recorded!=planned)
                    elif path.name.startswith('B_'):
                        assert recorded==planned
                        donor_counts['B_samples']+=1;donor_counts['B_donors_planned']+=planned
                    roles[s]['dominant'].add(r['dominant_accession']);roles[s]['donor'].update(r['contaminant_accessions'])
                    assert 50<=r['observed_completeness']<=100
                    assert 0<=r['observed_contamination']<=r['observed_completeness']+1e-6
        for split in seen:
            assert seen[split]==set(range(manifest['samples'][split]))
            records.append({'arm':arm,'split':split,'samples':len(seen[split]),
                'distinct_dominants':len(roles[split]['dominant']),'distinct_donors':len(roles[split]['donor']),
                'all_roles_in_arm_specific_split':True,
                'trace_scope':'Accepted simulation-plan donor identities; a superset of possible contributing donors, not per-donor emitted-bp attribution.'})
        atomic_json(base/('smoke_donor_count_audit.json' if smoke else 'donor_count_audit.json'),donor_counts)
        if not smoke and require_training_complete:
            completion=json.loads((OUT/arm/'models/full/completion.json').read_text())
            config=completion['training_config'];hist=pd.read_json(OUT/arm/'models/full/training_history.json')
            assert config['max_epochs']==150 and config['patience']==20
            assert config['training_samples']==1000000 and config['validation_samples']==100000
            assert config['feature_sha256']==sha(OUT/'features/selected_kmers_holdout.txt')
            assert config['normalization_sha256']==sha(base/'normalization_params.json')
            assert config['data_sha256']==manifest['features_h5_sha256']
            assert int(hist.iloc[-1].epoch)==completion['epochs_completed']
            assert int(hist.loc[hist.val_loss.idxmin(),'epoch'])==completion['best_epoch']
            assert int(hist.iloc[-1].epochs_without_improvement)>=20 or completion['epochs_completed']==150
            assert completion['fixed_epoch20_sensitivity']['epoch']==20
    return records

def numerical_audit(target):
    cols={f'{m}_{metric}':'float32' for m in ['holdout','matched_full','production','holdout_epoch20','matched_full_epoch20'] for metric in ['comp','cont']}
    d=pd.read_csv(target/'per_sample_predictions.tsv.gz',sep='\t',dtype=cols,float_precision='round_trip')
    assert d.evaluation_id.is_unique and not d.duplicated(['analysis_group','genome_id']).any()
    d=d[d.primary_in_domain];spec=json.loads((OUT/'panel_design.json').read_text())
    truth={'comp':'true_completeness','cont':'true_contamination'}
    obs=pd.read_csv(target/'did.tsv',sep='\t');maximum=0.;cimax=0.
    def draws(frame,label):
        rng=np.random.default_rng((spec['seed']+zlib.crc32(label.encode()))%(2**32))
        selected=rng.integers(0,len(frame),(2000,len(frame)))
        weights=np.zeros((2000,len(frame)),dtype=np.int64)
        np.add.at(weights,(np.arange(2000)[:,None],selected),1)
        return (weights@frame['sum'].to_numpy())/(weights@frame['count'].to_numpy())
    for metric in truth:
        q=d.assign(delta=np.abs(d[f'holdout_{metric}']-d[truth[metric]])-np.abs(d[f'matched_full_{metric}']-d[truth[metric]]))
        ref=q.groupby(['analysis_group','dominant_accession'],sort=True).delta.agg(['sum','count'])
        vc=ref.loc['in_distribution'];bc=draws(vc,f'control:{metric}:did')
        for group in spec['genera']:
            vg=ref.loc[group];delta=vg['sum'].sum()/vg['count'].sum()-vc['sum'].sum()/vc['count'].sum()
            row=obs[(obs.group==group)&(obs.metric==metric)].iloc[0]
            bounds=np.quantile(draws(vg,f'{group}:{metric}:did')-bc,[.025,.975])
            maximum=max(maximum,abs(delta-row.did));cimax=max(cimax,float(np.max(np.abs(bounds-row[['ci_low','ci_high']].to_numpy(dtype=float)))))
    assert maximum<1e-10 and cimax<1e-10,(maximum,cimax)
    secondary=None
    if target==OUT:
        sensitivity=pd.read_csv(target/'sensitivity_did.tsv',sep='\t');assert len(sensitivity)==60
        smax=0.;scimax=0.
        def cluster_draw(frame,left,right,metric,label):
            y=frame[truth[metric]]
            values=np.abs(frame[f'{left}_{metric}']-y)-np.abs(frame[f'{right}_{metric}']-y)
            agg=pd.DataFrame({'reference':frame.dominant_accession,'difference':values}).groupby('reference',sort=True).difference.agg(['sum','count'])
            rng=np.random.default_rng((spec['seed']+zlib.crc32(label.encode()))%(2**32))
            indices=rng.integers(0,len(agg),(2000,len(agg)))
            result=agg['sum'].to_numpy()[indices].sum(axis=1)/agg['count'].to_numpy()[indices].sum(axis=1)
            return float(values.mean()),result
        for name,limit,left,right in [('contamination_le35',35,'holdout','matched_full'),
                ('contamination_le10',10,'holdout','matched_full'),
                ('fixed_epoch20',100,'holdout_epoch20','matched_full_epoch20')]:
            data=d[d.true_contamination<=limit];control=data[data.analysis_group=='in_distribution']
            for metric in truth:
                cm,cb=cluster_draw(control,left,right,metric,f'sensitivity:{name}:control:{metric}')
                for group in spec['genera']:
                    gm,gb=cluster_draw(data[data.analysis_group==group],left,right,metric,f'sensitivity:{name}:{group}:{metric}')
                    row=sensitivity[(sensitivity.sensitivity==name)&(sensitivity.group==group)&(sensitivity.metric==metric)].iloc[0]
                    smax=max(smax,abs(gm-cm-row.did));bounds=np.quantile(gb-cb,[.025,.975])
                    scimax=max(scimax,float(np.max(np.abs(bounds-row[['ci_low','ci_high']].to_numpy(dtype=float)))))
        assert smax<1e-10 and scimax<1e-10,(smax,scimax)
        secondary={'contrasts':60,'max_did_difference_pp':smax,'max_ci_difference_pp':scimax}
    return {'status':'PASS','n_samples':len(d),'n_references':d.dominant_accession.nunique(),
        'max_independent_did_difference_pp':maximum,'max_independent_ci_difference_pp':cimax,
        'secondary_sensitivity_independent_audit':secondary,
        'method':'Independent reference-resampling multiplicity weights and cluster sufficient statistics, allowing unequal retained samples/reference; native float32 predictions restored from serialized TSV.'}

def compatible_source_certificate(certificate,manifest_hashes,n_genomes,n_cores,total_bytes):
    """Only exact manifest-wide readability facts can be reused for equal bytes."""
    return (certificate.get('status')=='ALL_ORIGINAL_SOURCE_BYTES_VERIFIED'
        and certificate.get('manifest_sha256')==manifest_hashes
        and certificate.get('genomic_fastas')==n_genomes
        and certificate.get('core_gene_fastas')==n_cores
        and certificate.get('bytes_read')==total_bytes
        and certificate.get('mismatches')==0
        and certificate.get('all_fastas_utf8_readable_with_ascii_sequence_and_at_least500bp') is True
        and isinstance(certificate.get('minimum_sequence_bp'),int)
        and certificate['minimum_sequence_bp']>=500)

def verify_source_file(row,parse_fasta=True):
    relative,size,expected=row;path=ROOT/relative
    assert path.stat().st_size==size,relative
    content=path.read_bytes();observed=hashlib.sha256(content).hexdigest();assert observed==expected,relative
    if not parse_fasta:return size,None
    content.decode('utf-8');assert content.startswith(b'>'),relative
    sequence=re.sub(br'(?m)^>[^\n]*(?:\n|$)',b'',content).translate(None,b'\r\n\t ')
    sequence.decode('ascii')
    assert len(sequence)>=500,relative
    return size,len(sequence)

def audit_original_sources(target=OUT):
    """Re-read actual FASTA bytes, binding this run to restoration manifests."""
    source=ROOT/'results/revision/holdout_resubmission5/reconstruction'
    genomes=pd.read_csv(source/'genomic_source_manifest.tsv',sep='\t')
    cores=pd.read_csv(source/'core_gene_input_manifest.tsv',sep='\t')
    inputs=[(r.fasta_relative_path,int(r.original_file_bytes),r.original_genomic_sha256)
        for r in genomes.itertuples()]
    inputs.extend((r.relative_path,int(r.file_bytes),r.sha256) for r in cores.itertuples())
    manifest_hashes={str((source/n).relative_to(ROOT)):sha(source/n)
        for n in ['genomic_source_manifest.tsv','core_gene_input_manifest.tsv']}
    total_bytes=sum(size for _,size,_ in inputs)
    certificate_path=OUT/'data_only_audit/actual_input_hash_verification.json'
    certificate={};certificate_hash=None
    if certificate_path.exists() and certificate_path!=(target/'actual_input_hash_verification.json'):
        try:
            content=certificate_path.read_bytes();certificate=json.loads(content)
            certificate_hash=hashlib.sha256(content).hexdigest()
        except (OSError,UnicodeDecodeError,json.JSONDecodeError):certificate={}
    reuse=isinstance(certificate,dict) and compatible_source_certificate(certificate,manifest_hashes,len(genomes),len(cores),total_bytes)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked=list(pool.map(lambda row:verify_source_file(row,parse_fasta=not reuse),inputs))
    total=sum(size for size,_ in checked)
    result={'status':'ALL_ORIGINAL_SOURCE_BYTES_VERIFIED','genomic_fastas':len(genomes),
        'core_gene_fastas':len(cores),'bytes_read':total,'mismatches':0,
        'all_fastas_utf8_readable_with_ascii_sequence_and_at_least500bp':True,
        'minimum_sequence_bp':certificate['minimum_sequence_bp'] if reuse else min(n for _,n in checked),
        'every_actual_file_read_and_sha256_checked_this_run':True,
        'readability_validation':({'method':'Reused readability/length facts only after freshly rereading and matching every file byte to the same exact manifests.',
            'certificate_path':str(certificate_path.relative_to(ROOT)),'certificate_sha256':certificate_hash}
            if reuse else {'method':'Parsed all actual FASTA bytes; no compatible independent early certificate was available.'}),
        'donor_trace_scope':'Accepted simulation plans bound possible donors; readability at audit time does not prove every planned donor emitted bp or rule out transient earlier read failures.',
        'manifest_sha256':manifest_hashes}
    atomic_json(target/'actual_input_hash_verification.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--data-only',action='store_true',help='Audit completed synthesis and source bytes while models train; no final numerical/training certification.')
    args=ap.parse_args()
    if args.data_only and (args.preflight or args.smoke):ap.error('--data-only is distinct from --preflight/--smoke.')
    splits=preflight()
    if args.preflight:print('Genus holdout preflight PASS');return
    if args.data_only:
        target=OUT/'data_only_audit';target.mkdir(exist_ok=True)
        traces=audit_traces(splits,False,require_training_complete=False)
        pd.DataFrame(traces).to_csv(target/'actual_role_audit.tsv',sep='\t',index=False)
        report={'status':'DATA_ONLY_PASS','scope':'Completed synthesis and source provenance only; training and evaluation completion are not certified.',
            'accepted_simulation_plan_rows':sum(row['samples'] for row in traces),
            'actual_source_inputs':audit_original_sources(target)}
        atomic_json(target/'report.json',report);print(json.dumps(report,indent=2));return
    target=OUT/'smoke_evaluation' if args.smoke else OUT
    traces=audit_traces(splits,args.smoke);pd.DataFrame(traces).to_csv(target/'actual_role_audit.tsv',sep='\t',index=False)
    report=numerical_audit(target);report['scope']='SMOKE_ONLY' if args.smoke else 'FULL_EXPERIMENT'
    if not args.smoke:report['actual_source_inputs']=audit_original_sources()
    atomic_json(target/'independent_numerical_audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
