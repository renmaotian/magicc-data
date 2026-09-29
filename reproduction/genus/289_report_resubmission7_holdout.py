#!/usr/bin/env python3
"""Summarize the actual genus experiment; preserve prior family/CPR evidence."""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from resubmission7_holdout_config import ROOT,OUT

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def table(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(map(str,row))+' |')
    return '\n'.join(lines)

def main():
    spec=json.loads((OUT/'panel_design.json').read_text())
    panel=pd.read_csv(OUT/'lineage_selection_manifest.tsv',sep='\t')
    feature=json.loads((OUT/'features/model_input_contract.json').read_text())
    representatives=pd.read_csv(OUT/'features/feature_selection_representatives.tsv',sep='\t')
    legacy=OUT/'legacy_sensitivity';legacy.mkdir(exist_ok=True)
    inputs=[]
    for src in sorted((ROOT/'results/revision/holdout_resubmission5/legacy_sensitivity').glob('*')):
        if not src.is_file():continue
        dst=legacy/('README_prior_broad_lineage.md' if src.name=='README.md' else src.name)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
        inputs.append({'role':'historical_broad_lineage_result','path':str(src.relative_to(ROOT)),
            'sha256':sha(src),'preserved_copy':str(dst.relative_to(ROOT))})
    for name in ['did.tsv','head_to_head_by_group.tsv','per_reference_errors.tsv','per_sample_predictions.tsv.gz',
            'panel_design.json','lineage_selection_manifest.tsv','evaluation_summary.json','convergence_check.tsv',
            'control_model_difference.tsv','independent_numerical_audit.json']:
        src=ROOT/'results/revision/holdout_resubmission5'/name;dst=legacy/('ten_family_'+name)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
        inputs.append({'role':'historical_ten_family_result','path':str(src.relative_to(ROOT)),
            'sha256':sha(src),'preserved_copy':str(dst.relative_to(ROOT))})
    (legacy/'README.md').write_text('''# Historical holdout results retained with the genus analysis

The ten_family_* files are byte-identical records of the complete resubmission5/6
ten-family matched retraining. They remain valid results of that experiment.
The broad phylum/family/genus files predate those matched arms and retain their
documented preprocessing and DPANN split caveats. The genus experiment in the
parent folder uses newly selected genera and new models; its results cannot
establish a paired difference between taxonomic ranks. No historical result is
relabelled as a new genus result.
''')
    paths=[OUT/'panel_design.json',OUT/'analysis_contract.json',OUT/'lineage_selection_manifest.tsv',
        OUT/'genus_candidate_census.tsv',OUT/'features/selected_kmers_holdout.txt',
        OUT/'features/feature_selection_representatives.tsv',OUT/'features/model_input_contract.json',
        OUT/'features/reselection_verification.json',ROOT/'models/magicc_v5.onnx',
        ROOT/'data/features/normalization_params.json']
    paths.extend(ROOT/f'data/splits/{s}_genomes.tsv' for s in ['train','val','test'])
    scripts=[p for n in range(282,291) for p in (ROOT/'scripts').glob(f'{n}_*') if p.suffix in ['.py','.sh']]
    scripts.extend(ROOT/'scripts'/s for s in ['resubmission7_holdout_config.py',
        '121_build_holdout_training_data.py','123_generate_holdout_eval_sets.py','125_kmer_reselection_control.py'])
    scripts.extend(sorted((ROOT/'scripts/holdout_lib').glob('*.py')))
    for path in paths+scripts:
        inputs.append({'role':'implementation' if path in scripts else 'input','path':str(path.relative_to(ROOT)),
            'sha256':sha(path),'preserved_copy':''})
    for name in ['genomic_source_manifest.tsv','core_gene_input_manifest.tsv','metadata_input_manifest.tsv','software_versions.json']:
        src=ROOT/'results/revision/holdout_resubmission5/reconstruction'/name
        inputs.append({'role':'original_input_restoration_manifest','path':str(src.relative_to(ROOT)),
            'sha256':sha(src),'preserved_copy':''})
    summary_path=OUT/'evaluation_summary.json'
    complete=summary_path.exists() and json.loads(summary_path.read_text())['status']=='EVALUATION_COMPLETE'
    newset=set((OUT/'features/selected_kmers_holdout.txt').read_text().splitlines())
    production=set((ROOT/'data/kmer_selection/selected_kmers.txt').read_text().splitlines())
    lines=['# Resubmission7 genus holdout',
        '**Status: '+('Complete.' if complete else 'Full synthesis/retraining in progress. No new study accuracy result is claimed.')+'**',
        '## Design and scope',
        'Ten named bacterial genera were selected from taxonomy and genome counts before any new model outcomes. Eligibility required at least 50 training, 5 validation and 20 test genomes, an unsuffixed established-name genus and exclusion of Patescibacteriota from this primary panel. Each family retained its largest genus and at least 50% of training genomes. Selection first spread across families by training abundance, then filled remaining slots, at most 2 genera/family and 4 genera/phylum. This is an availability-defined panel, with no archaeal genus meeting all rules.',
        table(panel[['genus','family','phylum','train','val','test','eval_references','parent_remaining_fraction']].round(4)),
        f'Exclusion removes {spec["n_training_removed"]:,}/{spec["n_training_total"]:,} training genomes ({spec["pct_training_removed"]:.3f}%). All original training families, orders, classes and 110 phyla remain. The panel spans {len(spec["parent_families"])} parent families/{len(spec["parent_phyla"])} phyla and evaluates {spec["n_panel_eval_references"]} panel test references plus 100 controls, 10 assemblies/reference ({spec["n_panel_eval_samples"]+1000:,} total). Controls are square-root stratified across nonpanel test phyla. Evaluation donors come only from the nonpanel test pool.',
        '## Actual features and matched retraining',
        f'Both new arms use {feature["n_features"]:,} reselected canonical 9-mers, selected from {int((representatives.feature_domain=="bacterial").sum())} bacterial and {int((representatives.feature_domain=="archaeal").sum())} archaeal training representatives after excluding target genera. The independent recount exactly reproduces the original production prevalence/tie-breaking rule. The new vocabulary retains {len(newset&production):,}/{len(production):,} production features. This vocabulary is an actual model input, not a post hoc diagnostic.',
        'The holdout arm excludes every target genus from dominant and donor pools in training, validation and its auxiliary test set; the matched-full arm retains them. Both synthesize 1,000,000 training, 100,000 validation and 100,000 auxiliary test examples using the full V5 recipe: 800,000 mixed-quality plus 100,000 complete/clean plus 100,000 complete/low-contamination training examples. Exact training-only moments and summary quantiles fit each arm normalizer. Every accepted example retains its seed, dominant accession and donor identities from its simulation plan. Donor lists bound possible contributors; they do not attribute emitted base pairs to individual donors.',
        'One retraining per arm uses seed 42, CPU FP32, the V5 hidden architecture and output bounds, AdamW 0.001/weight decay 0.0005, batch 512, weighted MSE 2:1, cosine restarts 10/2, 2% masking, noise SD 0.01, gradient clip 1, maximum 150 epochs and validation patience 20. The current session exposes no CUDA device. CPU benchmarks select two 12-thread models pinned to separate physical sockets. No epoch/data shortening is permitted. Validation populations differ by arm; their validation errors are not a same-population comparison.',
        '## Estimand, uncertainty and secondary checks',
        'Primary DiD is (MAE_holdout−MAE_matched_full)_genus −(MAE_holdout−MAE_matched_full)_control on identical test assemblies. Reference-cluster percentile intervals use 2,000 draws, paired predictions within each reference, and independently resampled target/control references. Two-sided centered-bootstrap p-values use a plus-one correction, with BH over 20 genus×outcome contrasts. These intervals measure evaluation-reference uncertainty and omit model-seed uncertainty. Joint pool exclusion changes training composition and normalization; the contrast is not an isolated causal taxonomic-novelty effect.',
        'The supported simulation domain is completeness 50–100% and contamination 0–100% of the full dominant reference, with contamination≤completeness. Prespecified secondary results restrict true contamination to ≤35% or ≤10%, and separately compare both models at fixed epoch 20. The latter uses checkpoints saved independently of validation/test performance; neither primary training run stops at 20. Secondary tests receive separate BH adjustment within each 20-contrast analysis.',
        'Canonical GCA/GCF/linked accession checks include accession versions and an additional version-insensitive source check. Target genera and their species are absent from holdout pools and feature selection; ordinary original splits are accession-independent, not globally species-independent. Per-row donor traces and independent numerical recomputation are required for final completion.',
        'Rank definitions follow GTDB. Three targets retain same-base-name GTDB genera in training: Rhizobium_E (1 genome), Bordetella_A/B/C (10 total), and Phocaeicola_A (4). These are distinct GTDB genera. A shared Latin name alone does not establish phylogenetic proximity; the experiment excludes exact GTDB genera rather than every historical or NCBI genus synonym.',
        '## Historical evidence and reproduction',
        'The complete earlier ten-family experiment remains in legacy_sensitivity/ten_family_* with its original estimates. Broad CPR and DPANN experiments also remain, with their historical preprocessing and DPANN source-overlap caveats. The new genus panel differs in references and composition: smaller or larger effects do not establish a paired rank trend.',
        'Restore original inputs using the hashed genomic/core-gene/metadata manifests, then run scripts 282, 283, 288. Launcher 288 runs 284/285 for both arms and 286/287/290/289 after actual completion. Module dependencies are 121/123/125, resubmission7_holdout_config.py and active holdout_lib Python modules. Optional hardware profiling remains local; runtime affinity may be adapted to available CPUs. The original model remains a descriptive deployment comparator using original features and mixed-split normalization, separate from the primary matched-arm contrast.',
        'Selection and analysis contracts, immutable raw-batch checksums, normalization hashes, model hashes and final numerical audits make the chain reviewable. Atomic epoch checkpoints include model/optimizer/scheduler/RNG states. Run the same launcher to resume; identity/configuration differences are rejected. If synthesis already completed, script 285 can instead resume each arm directly after verifying the completed build manifest and full HDF5/normalizer/feature hashes, avoiding reconstruction. Reconstruction was byte-identical in the isolated smoke resume test.',
        'For statistical reproduction from deposited predictions, run script 287 with --from-predictions per_sample_predictions.tsv.gz --output-dir a_separate_directory. This calls the same aggregation used after inference, restores native float32 predictions and round-trip float64 truth, and recomputes all six primary/secondary tables. Its hashed replay manifest certifies statistical recomputation only; it does not claim new training, ONNX inference or source-FASTA validation.']
    if complete:
        audit=json.loads((OUT/'independent_numerical_audit.json').read_text());assert audit['status']=='PASS' and audit['scope']=='FULL_EXPERIMENT'
        did=pd.read_csv(OUT/'did.tsv',sep='\t');sensitivity=pd.read_csv(OUT/'sensitivity_did.tsv',sep='\t')
        assert len(did)==20 and len(sensitivity)==60
        for arm in ['holdout','matched_full']:
            c=json.loads((OUT/arm/'models/full/completion.json').read_text());h=pd.read_json(OUT/arm/'models/full/training_history.json')
            assert c['training_config']['max_epochs']==150 and c['training_config']['patience']==20
            assert int(h.iloc[-1].epochs_without_improvement)>=20 or c['epochs_completed']==150
            lines.append(f'{arm}: completed {c["epochs_completed"]} epochs, selected epoch {c["best_epoch"]}; ONNX SHA256 `{c["onnx_sha256"]}`; PyTorch–ONNX maximum difference {c["pytorch_onnx_max_difference"]:.8f} pp.')
        lines.extend(['## Completed primary results',table(did.round(6)),
            table(pd.read_csv(OUT/'head_to_head_by_group.tsv',sep='\t')[['group','tool','n_refs','comp_mae','cont_mae','comp_bias','cont_bias']].round(4)),
            '## Prespecified secondary sensitivities',table(sensitivity.round(6))])
        for name in ['did.tsv','sensitivity_did.tsv','head_to_head_by_group.tsv','per_sample_predictions.tsv.gz',
                'per_reference_errors.tsv','evaluation_summary.json','model_provenance.tsv','convergence_check.tsv',
                'independent_numerical_audit.json','actual_role_audit.tsv','canonical_split_audit.json']:
            path=OUT/name;inputs.append({'role':'completed_result','path':str(path.relative_to(ROOT)),'sha256':sha(path),'preserved_copy':''})
    pd.DataFrame(inputs).to_csv(OUT/'provenance_manifest.tsv',sep='\t',index=False)
    (OUT/'analysis_report.md').write_text('\n\n'.join(lines)+'\n')
    print('Wrote',OUT/'analysis_report.md','complete=',complete)

if __name__=='__main__':main()
