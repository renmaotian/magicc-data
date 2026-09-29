#!/usr/bin/env python3
"""Decompose the five-set contamination MAE by the CheckM2 training dose limit.

Descriptive accounting only: training dose, contamination construction and
taxonomic representation are not experimentally separated by this analysis.
"""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_sources import resolve_source

SETS = ['set_A_v2', 'set_B_v2', 'set_C_clean', 'set_D_clean', 'set_E']
TOOLS = ['magicc_v5', 'checkm2', 'cocopye', 'deepcheck']

def main():
    source = Path(resolve_source(ROOT/'results/revision/metrics/ws5.3_signed_errors_per_genome.tsv.gz'))
    d = pd.read_csv(source, sep='\t')
    d = d[d['set'].isin(SETS) & d.tool.isin(TOOLS) & (d.metric == 'contamination')].copy()
    assert len(d) == 20000
    assert d.groupby(['set','tool']).size().eq(1000).all()
    assert not d.duplicated(['set','tool','genome_id']).any()
    d['signed_error'] = d.predicted - d.true_contamination
    d['absolute_error'] = d.signed_error.abs()
    d['dose_band'] = d.true_contamination.map(lambda x: '0_to_35' if x <= 35 else 'above_35')
    records = []
    for cohort in ['pooled_equal_set'] + SETS:
        panel = d if cohort == 'pooled_equal_set' else d[d['set'] == cohort]
        for tool in TOOLS:
            td = panel[panel.tool == tool]
            for band in ['all', '0_to_35', 'above_35']:
                bd = td if band == 'all' else td[td.dose_band == band]
                ae = float(bd.absolute_error.sum())
                records.append(dict(cohort=cohort,tool=tool,dose_band=band,
                    n_samples=len(bd),n_reference_clusters=bd.cluster_id.nunique(),
                    sample_fraction=len(bd)/len(td),mae_pp=bd.absolute_error.mean(),
                    signed_bias_pp=bd.signed_error.mean(),
                    contribution_to_full_mae_pp=ae/len(td),
                    fraction_of_full_absolute_error=ae/td.absolute_error.sum()))
    out = ROOT/'results/revision/checkm2_domain_resubmission7'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'dose_decomposition.tsv',sep='\t',index=False,float_format='%.12g')
    d.to_csv(out/'per_sample_contamination.tsv.gz',sep='\t',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    checks=[]
    for (cohort,tool),g in pd.DataFrame(records).groupby(['cohort','tool']):
        a=g[g.dose_band=='all'].iloc[0]
        rest=g[g.dose_band!='all']
        difference=abs(rest.contribution_to_full_mae_pp.sum()-a.mae_pp)
        assert difference < 1e-12
        checks.append({'cohort':cohort,'tool':tool,'sum_contribution_error_pp':difference})
    report={'status':'VERIFIED','input':str(source.relative_to(ROOT)),
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'n_samples_per_tool':5000,'full_panel_weighting':'Each set weight 1/5; each has exactly 1000 rows, so equivalent to uniform sample weighting.',
        'within_band_weighting':'Conditional distribution induced by the full panel; sample-weighted within band. Not equal-weight reweighting of retained set subsets.',
        'interpretation':'Descriptive decomposition, not causal attribution to dose or training construction.',
        'checks':checks}
    (out/'decomposition_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(records).query("cohort == 'pooled_equal_set' and tool in ['magicc_v5','checkm2']").to_string(index=False))

if __name__=='__main__': main()
