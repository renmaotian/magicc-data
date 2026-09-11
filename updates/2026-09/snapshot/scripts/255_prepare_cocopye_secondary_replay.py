#!/usr/bin/env python3
"""Apply explicit, auditable coverage and cached-input changes to isolated replay.

Original scripts and outputs are retained. The only scientific changes are the
official stage selector and an all-four-scored comparison intersection where
stage-1 rejection makes the historical intersection invalid. Full-cohort MAGICC
results and coverage counts are emitted separately, never silently discarded.
"""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def replace_once(s,old,new):
    assert s.count(old)==1,(old[:80],s.count(old));return s.replace(old,new)
def main():
    prep=module('prep253',ROOT/'scripts/253_prepare_cocopye_replay.py')
    # Circularity is another accuracy consumer; its old directory was read-only.
    p=WORK/'results/revision/circularity'
    if p.is_symlink():p.unlink()
    prep.copy_tables(ROOT/'results/revision/circularity',p)
    records=[]
    def write(name,s):
        dest=WORK/'scripts'/name;dest.write_text(s)
        records.append({'original':str((ROOT/'scripts'/name).relative_to(ROOT)),'original_sha256':sha(ROOT/'scripts'/name),'replay_path':str(dest.relative_to(ROOT)),'replay_sha256':sha(dest)})
    def source(name):return (ROOT/'scripts'/name).read_text().replace(str(ROOT),str(WORK))
    name='201_ws38_cami2_analysis.py';s=source(name)
    s=replace_once(s,"    long['in_domain'] = long['true_contamination'] <= long['true_completeness']", "    long['tool_scored'] = np.isfinite(long['pred_completeness']) & np.isfinite(long['pred_contamination'])\n    long['in_domain'] = long['true_contamination'] <= long['true_completeness']")
    s=replace_once(s,"        n_tools = sub['tool'].nunique()\n        cnt = sub.groupby('bin_id')['tool'].nunique()", "        assert set(sub['tool']) == set(TOOL_ORDER)\n        n_tools = len(TOOL_ORDER)\n        cnt = sub[sub.tool_scored].groupby('bin_id')['tool'].nunique()")
    s=replace_once(s,"    return long\n\n\ndef attach_leakage", "    return long\n\n\ndef attach_leakage")
    s=replace_once(s,"    long = attach_leakage(load_long())", "    long = attach_leakage(load_long())\n    coverage = []\n    for (ds, bset), sub in long.groupby(['dataset', 'binset']):\n        for cohort, mask in [('all_attempted', np.ones(len(sub), dtype=bool)), ('primary_in_domain_scoreable_training_overlap_screened', sub.in_domain & sub.scoreable & sub.leakage_free)]:\n            z = sub.loc[mask]\n            old_counts = z.groupby('bin_id').tool.nunique()\n            old_common = set(old_counts[old_counts == len(TOOL_ORDER)].index)\n            for tool, zz in z.groupby('tool'):\n                coverage.append(dict(dataset=ds, binset=bset, cohort=cohort, tool=tool, n_attempted=len(zz), n_scored=int(zz.tool_scored.sum()), n_unscored=int((~zz.tool_scored).sum()), original_common_n=len(old_common), corrected_common_n=int(z.loc[z.all_tools_scored, 'bin_id'].nunique())))\n    pd.DataFrame(coverage).to_csv(OUT_DIR / 'cami2_tool_scoring_coverage.tsv', sep='\\t', index=False)")
    # Keep all attempted rows in the source table, then exclude only unscored tool
    # rows from quantitative summaries (common-subset flags already computed).
    s=replace_once(s,"    long.to_csv(OUT_DIR / 'cami2_long_predictions.tsv', sep='\\t', index=False)","    long.to_csv(OUT_DIR / 'cami2_long_predictions.tsv', sep='\\t', index=False)\n    long = long[long.tool_scored].copy()")
    write(name,s)
    name='143_realdata_analysis.py';s=source(name)
    anchor='    rows, mrows, prows = [], [], []\n    for name, sub in cohorts.items():'
    # First occurrence is Meslier. Other tracks have no stage-1 rejections.
    assert s.count(anchor)==2
    replacement='''    full_rows, coverage = [], []
    for name, sub in cohorts.items():
        common = np.ones(len(sub), dtype=bool)
        for tool in TOOLS:
            scored = np.isfinite(sub[f'{tool}_completeness']) & np.isfinite(sub[f'{tool}_contamination'])
            common &= scored
            coverage.append(dict(cohort=name, tool=tool, n_attempted=len(sub), n_scored=int(scored.sum()), n_unscored=int((~scored).sum())))
            full_rows.append(metrics_row(sub, tool, cluster='accession', cohort=name, note=notes[name] + ' Tool-specific scoring coverage; not an all-tool comparison.'))
        cohorts[name] = sub.loc[common].copy()
        for row in coverage[-len(TOOLS):]:row['original_common_n'] = len(sub);row['corrected_common_n'] = int(common.sum())
    emit(full_rows, d / 'metrics_full_tool_scoring_coverage.tsv')
    pd.DataFrame(coverage).to_csv(d / 'tool_scoring_coverage.tsv', sep='\\t', index=False)
    rows, mrows, prows = [], [], []
    for name, sub in cohorts.items():'''
    s=s.replace(anchor,replacement,1)
    # A balanced gradient requires each selected organism to be scored by every
    # comparator on every assembly. The source per-bin table still retains all.
    s=replace_once(s,'    cnt = p.groupby("accession")["assembly"].nunique()', '''    all_scored = np.logical_and.reduce([np.isfinite(p[f'{t}_completeness']) & np.isfinite(p[f'{t}_contamination']) for t in TOOLS])
    score_panel = p.loc[all_scored].groupby('accession').assembly.nunique()
    scored_every_assembly = set(score_panel[score_panel == p.assembly.nunique()].index)
    cnt = p.groupby("accession")["assembly"].nunique()''')
    s=replace_once(s,'    panel = set(cnt[cnt == p["assembly"].nunique()].index)','    panel = set(cnt[cnt == p["assembly"].nunique()].index) & scored_every_assembly')
    s=replace_once(s,'    panel50 = set(ok50[ok50 == p["assembly"].nunique()].index)','    panel50 = set(ok50[ok50 == p["assembly"].nunique()].index) & scored_every_assembly')
    write(name,s)
    name='192_ws1_11_reference_level_scores.py';s=source(name)
    start=s.index('    # Stage-3 (marker + neural network)');end=s.index('\n\ndef main()',start)
    s=s[:start]+'''    selected = pd.read_csv(SET_DIR / 'reference_level_cocopye_selected.tsv', sep='\\t')
    assert selected.genome_id.is_unique and selected.tool_scored.all()
    return pd.DataFrame({'primary_accession': selected.genome_id.astype(str).str.replace(r'\\.fasta$', '', regex=True),
                         'cocopye_completeness': selected.pred_completeness,
                         'cocopye_contamination': selected.pred_contamination})
'''+s[end:]
    # Existing MAGICC / CheckM2 cached files are mandatory: no inference allowed.
    s=replace_once(s,"    print(f'  created {build_links(sel)} new .fasta symlinks in {LINK_DIR}')", "    assert (SET_DIR / 'reference_level_magicc_v5.tsv').is_file(), 'Cached MAGICC scores required'\n    assert (SET_DIR / 'reference_level_checkm2_output/quality_report.tsv').is_file(), 'Cached CheckM2 scores required'\n    assert (SET_DIR / 'reference_level_cocopye.csv').is_file(), 'Cached CoCoPyE scores required'\n    print('Using existing reference scores; no FASTA links or inference are regenerated')")
    write(name,s)
    # Validate all overlay joins, including deliberate stage-1 NaNs.
    joins=[]
    for row in pd.read_csv(OUT/'prediction_consumer_map.tsv',sep='\t').itertuples():
        d=pd.read_csv(ROOT/row.consumer,sep='\t');c=pd.read_csv(ROOT/row.source,sep='\t')
        if 'cocopye_selected_stage' in d:
            assert d.genome_id.is_unique
            attempted=d.genome_id.isin(c.genome_id)
            if '/meslier/' in row.consumer:eligible=d.bin_fasta.notna() & d.bin_fasta.ne('')
            elif '/ncbi_pairs/' in row.consumer:eligible=d.status.eq('ok')
            else:eligible=pd.Series(True,index=d.index)
            assert np.array_equal(attempted,eligible), 'Eligible truth row failed prediction join'
            assert np.array_equal(d.cocopye_completeness.isna(),d.cocopye_selected_stage.eq(1) | ~attempted)
            assert np.array_equal(d.cocopye_contamination.isna(),d.cocopye_selected_stage.eq(1) | ~attempted)
            d['cocopye_tool_attempted']=attempted
            d.to_csv(ROOT/row.consumer,sep='\t',index=False)
        elif 'bin_id' in d:
            c.genome_id=c.genome_id.str.replace(r'\.fasta$','',regex=True);z=d[d.tool.eq('CoCoPyE')]
            assert c.genome_id.is_unique and z.bin_id.is_unique and set(z.bin_id)<=set(c.genome_id)
            assert np.array_equal(z.pred_completeness.isna(),z.selected_stage.eq(1))
            assert np.array_equal(z.pred_contamination.isna(),z.selected_stage.eq(1))
        joins.append({'consumer':row.consumer,'sha256':sha(ROOT/row.consumer),'join_and_rejection_validation':'PASS'})
    pd.DataFrame(records).to_csv(OUT/'secondary_replay_script_changes.tsv',sep='\t',index=False)
    pd.DataFrame(joins).to_csv(OUT/'consumer_join_validation.tsv',sep='\t',index=False)
    # Snapshot historical tabular inputs and scripts before downstream execution.
    files=set()
    for base in [ROOT/'data/benchmarks',ROOT/'results/revision']:
        for p in base.rglob('*'):
            if OUT in p.parents or not p.is_file() or p.is_symlink():continue
            if p.suffix in {'.tsv','.csv','.json'} and p.stat().st_size<100_000_000:files.add(p)
    files.update(ROOT/'scripts'/x['original'].split('/')[-1] for x in records)
    pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]).to_csv(OUT/'historical_immutability_before.tsv',sep='\t',index=False)
    print('Secondary replay prepared; joins verified; historical table hashes recorded.')
if __name__=='__main__':main()
