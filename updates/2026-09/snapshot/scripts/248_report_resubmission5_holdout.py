#!/usr/bin/env python3
"""Write current/final analysis report and preserve broad CPR sensitivity evidence."""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from resubmission5_holdout_config import OUT,ROOT

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def table(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(map(str,row))+' |')
    return '\n'.join(lines)

def main():
    legacy=OUT/'legacy_sensitivity';legacy.mkdir(parents=True,exist_ok=True)
    records=[];cpr=[];dpann=[]
    for level,directory in [('phylum','holdout'),('family','holdout_family'),('genus','holdout_genus')]:
        src=ROOT/'results/revision'/directory
        for name in ['lineage_novelty_effect_did.tsv','head_to_head_by_group.tsv','per_sample_predictions.tsv.gz']:
            p=src/name;q=legacy/f'{level}_{name}'
            if not q.exists():shutil.copy2(p,q)
            if p.exists():assert sha(p)==sha(q)
            records.append({'role':'legacy_sensitivity','path':str(p.relative_to(ROOT)),'sha256':sha(q),'preserved_copy':str(q.relative_to(ROOT))})
        d=pd.read_csv(legacy/f'{level}_lineage_novelty_effect_did.tsv',sep='\t')
        s=d[d.group.str.contains('Patescibacteriota')].copy();s.insert(0,'level',level);cpr.append(s)
        if level=='phylum':
            allp=pd.read_csv(legacy/f'{level}_per_sample_predictions.tsv.gz',sep='\t')
            pp=allp[allp.group=='DPANN']
            dpann=pp.groupby('dominant_v5_split').agg(samples=('genome_id','size'),references=('dominant_accession','nunique')).reset_index().to_dict('records')
    pd.concat(cpr).to_csv(legacy/'CPR_legacy_effects.tsv',sep='\t',index=False)
    (legacy/'README.md').write_text('''# Preserved broad-lineage sensitivity analyses

These are exact copies of the former phylum/family/genus holdout outputs. They
remain scientifically relevant and have not been removed because they are
unfavorable to MAGICC. They are distinct experiments from the new ten-family
primary panel and must not be pooled with it.

The historical phylum Patescibacteriota estimate is a broad-CPR-removal stress
test. GTDB treats this historical radiation as one rank-normalized phylum;
its historical superphylum description does not invalidate its observed loss.
The older models retained production features and mixed-split preprocessing.
The DPANN phylum comparison additionally contains training/validation dominant
references for the production V5 baseline; only its test subset is an independent
genome comparison. All split labels are preserved in the copied predictions.
''')
    spec=json.loads((OUT/'panel_design.json').read_text());panel=pd.read_csv(OUT/'lineage_selection_manifest.tsv',sep='\t')
    # The running build originally used a shared "panel-free" audit label even
    # for the deliberately full-family arm. Clarify wording without changing
    # any samples, arrays, fitted normalization or recorded feature hashes.
    for arm in ['holdout','matched_full']:
        p=OUT/arm/'data/build_manifest.json'
        if p.exists():
            manifest=json.loads(p.read_text())
            changed=False
            old='All actual dominant/contaminant accessions audited against split-specific panel-free pool.'
            if manifest.get('role_leakage')==old:
                manifest['role_leakage']='All actual dominant/contaminant accessions audited against the arm-specific split pool; holdout excludes the panel, matched_full includes it.'
                manifest['audit_wording_clarification']='Only the common descriptive label was clarified; no data or numerical checksum changed.'
                changed=True
            for batch in manifest.get('raw_batches',[]):
                trace=Path(batch['path']).with_suffix('.jsonl.gz')
                if trace.exists() and 'actual_role_trace_sha256' not in batch:
                    batch['actual_role_trace_path']=str(trace)
                    batch['actual_role_trace_sha256']=sha(trace);changed=True
            if changed:p.write_text(json.dumps(manifest,indent=2)+'\n')
    split_frames={s:pd.read_csv(ROOT/f'data/splits/{s}_genomes.tsv',sep='\t') for s in ['train','val','test']}
    aliases={}
    for split,frame in split_frames.items():
        aliases[split]=set()
        for col in ['ncbi_accession','gcf_accession','gtdb_accession']:
            aliases[split]|=set(frame[col].dropna().str.replace(r'^(RS_|GB_)','',regex=True).str.replace(r'^GC[AF]_','',regex=True))
    overlaps={a+'_'+b:len(aliases[a]&aliases[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert sum(overlaps.values())==0
    (OUT/'canonical_split_audit.json').write_text(json.dumps({'overlap_counts':overlaps,'canonicalization':'Remove RS_/GB_ and GCA_/GCF_ prefixes, include linked gcf_accession aliases, retain versions.'},indent=2)+'\n')
    tr=split_frames['train']
    for rank in ['family','order','class']:tr[rank]=tr.gtdb_taxonomy.str.extract(rank[0]+r'__([^;]+)')
    retained=tr[~tr.family.isin(spec['families'])];rank_rows=[]
    for family in spec['families']:
        sub=tr[tr.family==family];orders=sorted(sub.order.unique());classes=sorted(sub['class'].unique())
        n_remaining=int(retained.order.isin(orders).sum())
        rank_rows.append({'family':family,'parent_orders':';'.join(orders),'parent_order_training_remaining':n_remaining,
                          'whole_order_removed':n_remaining==0,'parent_classes':';'.join(classes),
                          'parent_class_training_remaining':int(retained['class'].isin(classes).sum())})
    pd.DataFrame(rank_rows).to_csv(OUT/'higher_rank_exclusion_audit.tsv',sep='\t',index=False)
    feature=json.loads((OUT/'features/model_input_contract.json').read_text())
    fcheck=json.loads((OUT/'features/reselection_verification.json').read_text())
    prodset=set((ROOT/'data/kmer_selection/selected_kmers.txt').read_text().splitlines())
    newset=set((OUT/'features/selected_kmers_holdout.txt').read_text().splitlines())
    for p in [ROOT/f'data/splits/{s}_genomes.tsv' for s in ['train','val','test']]+[
        ROOT/'data/kmer_selection/selected_kmers.txt',OUT/'features/selected_kmers_holdout.txt',
        ROOT/'models/magicc_v5.onnx',ROOT/'data/features/normalization_params.json']:
        records.append({'role':'input','path':str(p.relative_to(ROOT)),'sha256':sha(p),'preserved_copy':''})
    for pattern in ['24[0-9]_*.py','250_*.py','247_*.sh','resubmission5_holdout_config.py']:
        for p in sorted((ROOT/'scripts').glob(pattern)):
            records.append({'role':'implementation','path':str(p.relative_to(ROOT)),'sha256':sha(p),'preserved_copy':''})
    for p in sorted((ROOT/'scripts/holdout_lib').glob('*.py'))+[
        ROOT/'scripts/121_build_holdout_training_data.py',ROOT/'scripts/123_generate_holdout_eval_sets.py',
        ROOT/'scripts/125_kmer_reselection_control.py']:
        records.append({'role':'reused_implementation','path':str(p.relative_to(ROOT)),'sha256':sha(p),'preserved_copy':''})
    for name in ['genomic_source_manifest.tsv','core_gene_input_manifest.tsv','metadata_input_manifest.tsv','software_versions.json',
                 'gpu_access_diagnostic_20260908.json','gpu_access_evidence.md','execution_completion.json','execution_completion.md']:
        p=OUT/'reconstruction'/name
        if p.exists():records.append({'role':'source_reconstruction_manifest','path':str(p.relative_to(ROOT)),'sha256':sha(p),'preserved_copy':''})
    for name in ['independent_numerical_audit.json','model_provenance.tsv','evaluation_summary.json']:
        p=OUT/name
        if p.exists():records.append({'role':'completed_evaluation_audit','path':str(p.relative_to(ROOT)),'sha256':sha(p),'preserved_copy':''})
    pd.DataFrame(records).to_csv(OUT/'provenance_manifest.tsv',sep='\t',index=False)
    summary_path=OUT/'evaluation_summary.json'
    complete=summary_path.exists() and json.loads(summary_path.read_text()).get('status')=='EVALUATION_COMPLETE'
    report=[
        '# Ten-family holdout analysis for resubmission 5',
        '**Status:** '+('Complete; final matched-model predictions available.' if complete else 'Full synthesis/retraining in progress. No new holdout accuracy result is claimed yet.'),
        '## Selection and scope',
        f'Ten established-name bacterial families were selected from taxonomy and split counts before computing new predictions. Eligibility required ≥50 training, ≥5 validation and ≥20 test genomes, exclusion of Patescibacteriota, retention of each parent phylum\'s largest family and ≥50% of parent training genomes. One family per eligible parent phylum was selected first; remaining slots were filled by training abundance with a maximum of two per parent. The rules yield seven parent phyla; no archaeal family meets all requirements. This is an availability-defined bacterial panel, not a claim of universal taxonomic representativeness.',
        table(panel[['family','phylum','train','val','test','eval_references','parent_remaining_fraction']].round(4)),
        f'The panel removes {spec["n_training_removed"]:,}/{spec["n_training_total"]:,} training genomes ({spec["pct_training_removed"]:.2f}%). All 110 original training phyla remain, with 65.7–87.4% of each selected family\'s parent phylum retained. The full design targets 872 panel test references ×10 simulations =8,720 assemblies plus 100 nonpanel test references ×10=1,000 control assemblies. Every dominant and contaminant is drawn from the test split during evaluation.',
        'Leptospiraceae and Cyanobiaceae are the only represented families in their respective orders (Leptospirales and PCC-6307), so their exclusion also removes those orders. The other eight families retain their orders; all classes and phyla remain represented. These two rows must be labeled as family exclusions that also create order-level absence. Canonical/linked accession audits find zero overlaps between any two original genome splits.',
        '## Features, controls and retraining',
        f'The actual model vocabulary was reselected from 878 panel-free bacterial and all 1,000 archaeal training representatives, using the original top-9,000 bacterial plus top-1,000 archaeal prevalence rule and merging canonical 9-mers. The independent recount reproduced all original prevalence counts exactly. The resulting {feature["n_features"]:,} features retain {len(prodset&newset):,}/{len(prodset):,} ({100*len(prodset&newset)/len(prodset):.2f}%) production features; Jaccard={len(prodset&newset)/len(prodset|newset):.6f}. Both new arms use this exact vocabulary; feature reselection is not merely a diagnostic.',
        'The holdout arm excludes the ten families from dominant and contaminant roles in training and validation. The matched full-family arm includes them. Both arms regenerate 1,000,000 training samples (800,000 V4 recipe +100,000 complete/clean +100,000 complete/low contamination), 100,000 validation and 100,000 auxiliary test samples. The original six-category synthesis mixture, fragmentation, contamination definition and training-domain constraint are retained. Planners and seed formulas come from the frozen V5 implementation. Every accepted sample has a saved seed, dominant accession, contaminant accessions, target/observed labels and quality tier.',
        'Each arm fits its own normalization on all and only its training samples, using exact k-mer moments and exact summary-feature quantiles. No validation/test values participate. The frozen production model uses the original normalizer, which was fitted to 800,000 training +100,000 validation +100,000 test V4-recipe samples, then reused for the additional 200,000 V5 training samples. Therefore the frozen deployment comparator must not be described as fully independent at preprocessing level.',
        'There is one seeded retraining per arm (seed 42). Hidden layers/output bounds, AdamW learning rate 0.001, weight decay 0.0005, weighted MSE 2:1, batch size 512, cosine warm restarts 10/2, masking 2%, noise SD 0.01, gradient clipping at 1, maximum 150 epochs and patience 20 match the V5 recipe. GPU device access was unavailable to the execution environment, so training used CPU FP32; gradient checkpointing is disabled because memory is ample. The first layer has 9,243 inputs rather than 9,249, identically in both matched arms. Vectorized random augmentation preserves the original independent masking/noise distributions. Atomic checkpoints contain all RNG, optimizer and scheduler states. Training/validation sample counts are matched, but validation taxonomy differs by arm; validation MAEs are not same-population comparisons.',
        '## Estimand and uncertainty',
        'Every model predicts the same evaluation assemblies. Report raw MAE, signed error, R², both-arm MAE differences and the common-control model difference. The primary adjusted contrast is (MAE_holdout−MAE_matched_full)_family −(MAE_holdout−MAE_matched_full)_control. This is the excess error change following joint panel exclusion, after subtracting a global control change. Exclusion changes training-pool composition and normalizers; the adjustment does not isolate taxonomy alone or remove lineage-specific training interactions.',
        '95% percentile intervals resample dominant reference genomes (2,000 replicates), with model predictions paired within references and the family/control reference sets sampled independently. Two-sided centered-bootstrap p-values use a plus-one correction; BH covers all 20 family×outcome contrasts. These intervals cover evaluation-reference variation, not model-seed uncertainty. Primary analyses enforce the stated completeness 50–100% and contamination ≤ completeness domain; all predictions remain available, including any samples outside that domain.',
        'Feature-row identity is checked against independent FASTA recounts under the production vocabulary: every shared k-mer must agree for every row. Simulator genome IDs are local to each group and are stored alongside features; they are checked within group during model evaluation. The pooled evaluation_id combines analysis_group and genome_id and must be globally unique. PyTorch–ONNX equivalence is required within 1e−4. Convergence records include best epoch, stopping reason and proximity of the best epoch to the final epoch; capped or still-improving runs require cautious interpretation.',
        '## Historical CPR and DPANN evidence',
        'Parks et al. (2018) explicitly consolidated the historical Candidate Phyla Radiation into a single GTDB phylum ([primary paper](https://pubmed.ncbi.nlm.nih.gov/30148503/)); [GTDB R226](https://gtdb.ecogenomic.org/stats/r226) lists Patescibacteriota at phylum rank. Its broad evolutionary scope motivates reporting it separately from a common-family panel. It does not make the earlier unfavorable observation invalid, and diversity alone has not been shown to explain that loss.',
        'Exact legacy outputs and provenance are preserved in legacy_sensitivity/. Their historical CPR adjusted completeness errors were 22.8114 pp (phylum), 5.0338 pp (family) and 4.6431 pp (genus), on different evaluation panels; they must not be interpreted as a strictly paired rank ladder. Their frozen production features and mixed-split normalization limit claims of complete training-process exclusion.',
        'The former DPANN phylum panel contains these production-model split categories: '+json.dumps(dpann)+'. The full DPANN contrast includes seen dominant references and must remain labeled accordingly.',
        '## Reproduction',
        'Use the software versions and input restoration instructions in reconstruction/. Run scripts 240, 242, then 247. Script 247 runs 243/244 for both arms and then 245/246/248. Script 241 records runtime only. Two simultaneous full-width CPU benchmarks averaged 0.418/0.328 seconds per training batch with each 12-thread model restricted to a separate physical socket, versus 0.834/0.975 seconds unbound. Runtime affinity is recorded separately and changes no scientific setting. Full training remains governed by the specified maximum epochs and validation patience. The persistent exec session is necessary in the current sandbox; detached shell jobs are terminated when their execution namespace closes.',
        'Family selection is deterministic without random sampling. Evaluation base seed is 20260908: reference sampling uses base + CRC32(group) % 100000, while each simulated assembly uses base + sum(character codes of group) × 1000003 + sample index. Bootstrap substreams use (base + CRC32(contrast label)) modulo 2^32. Synthetic training retains the original V5 batch planner formulas; both neural-network arms use seed 42. Every evaluation sample stores its exact seed. All paths, scripts, input hashes and exact legacy copies are recorded in provenance_manifest.tsv. Raw batch artifacts and per-sample provenance make regeneration resumable without treating frozen-model predictions as holdout retraining.',
    ]
    if complete:
        result=json.loads(summary_path.read_text())
        contrasts=pd.read_csv(OUT/'did.tsv',sep='\t')
        controls=pd.read_csv(OUT/'control_model_difference.tsv',sep='\t').set_index('metric')
        comp=contrasts[contrasts.metric=='comp'];cont=contrasts[contrasts.metric=='cont']
        report.extend(['## Final results',
            f'Both full training arms completed under the prespecified patience-20 stopping rule: holdout 49 epochs with best epoch 29; matched full-family 90 epochs with best epoch 70. Neither reached the 150-epoch cap. All 9,720 evaluation assemblies from 972 references met the primary domain. The adjusted completeness MAE increases range from {comp.did.min():.2f} to {comp.did.max():.2f} pp and contamination increases from {cont.did.min():.2f} to {cont.did.max():.2f} pp. Every 95% reference-bootstrap interval excludes zero; all BH q-values equal the minimum attainable bootstrap p-value, 1/2,001 (approximately 0.000500). The common-control MAE changes are {controls.loc["comp","delta_mae"]:.3f} pp for completeness and {controls.loc["cont","delta_mae"]:.3f} pp for contamination. These are results of the joint exclusion experiment with one training seed per arm; they must not be presented as an isolated causal effect of novelty or as accuracy of the unchanged deployed model.',
            'The unchanged production V5 comparator has lower completeness MAE than the new matched full-family model in every panel family. Its contamination MAE is slightly higher in nine families and slightly lower in Leptospiraceae. The exclusion results therefore need to be distinguished from the deployment benchmark, whose predictions and model were unchanged. Full raw errors and adjusted contrasts are retained below. The independent numerical audit reconstructs MAEs by reference means and contrasts by a separate group/model pivot; all agree within 7.11e−15 pp after restoring the native float32 prediction dtype from CSV.',
            'The original long-running shell completed synthesis, training, export and input verification, then encountered a truncated command after an earlier edit to its source file. Scripts 246 and 248 were subsequently invoked directly on the completed artifacts. No data, model weights, training settings or statistical settings were changed. Exact continuation commands and hashes are recorded in reconstruction/execution_completion.json; the final launcher passes bash syntax validation.',
            json.dumps(result,indent=2),
            table(pd.read_csv(OUT/'did.tsv',sep='\t').round(5)),
            table(pd.read_csv(OUT/'head_to_head_by_group.tsv',sep='\t')[['group','tool','n_refs','comp_mae','cont_mae','comp_bias','cont_bias']].round(4))])
    else:
        report.append('## Pending outputs\nFinal head_to_head_by_group.tsv, did.tsv, per_sample_predictions.tsv.gz and convergence_check.tsv will be generated only after both full training arms finish. Smoke outputs are labeled SMOKE_ONLY and cannot support manuscript claims.')
    (OUT/'analysis_report.md').write_text('\n\n'.join(report)+'\n')
    print('Wrote',OUT/'analysis_report.md',flush=True)

if __name__=='__main__':main()
