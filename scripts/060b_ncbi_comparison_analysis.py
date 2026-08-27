#!/usr/bin/env python3
"""Step 6: Compare MAGICC V5 vs CheckM2 on NCBI GenBank genomes."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir('/path/to/magicc-legacy')

NCBI_DIR = Path('data/ncbi')
RESULTS_DIR = Path('results/ncbi_comparison')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load all data
pure_v5 = pd.read_csv(NCBI_DIR / 'magicc_v5_pure_culture.tsv', sep='\t')
mag_v5 = pd.read_csv(NCBI_DIR / 'magicc_v5_mags.tsv', sep='\t')
pure_ckm2 = pd.read_csv(NCBI_DIR / 'checkm2_pure_culture' / 'quality_report.tsv', sep='\t')
mag_ckm2 = pd.read_csv(NCBI_DIR / 'checkm2_mags' / 'quality_report.tsv', sep='\t')

print(f'Data loaded: V5 pure={len(pure_v5)}, V5 MAG={len(mag_v5)}, '
      f'CheckM2 pure={len(pure_ckm2)}, CheckM2 MAG={len(mag_ckm2)}')

# Rename CheckM2 columns
for ckm2_df in [pure_ckm2, mag_ckm2]:
    ckm2_df.rename(columns={
        'Name': 'accession',
        'Completeness': 'checkm2_completeness',
        'Contamination': 'checkm2_contamination',
    }, inplace=True)


def classify_mimag(comp, cont):
    if comp >= 90 and cont <= 5:
        return 'HQ'
    elif comp >= 50 and cont <= 10:
        return 'MQ'
    else:
        return 'LQ'


def analyze(v5_df, ckm2_df, label):
    merged = v5_df.merge(
        ckm2_df[['accession', 'checkm2_completeness', 'checkm2_contamination']],
        on='accession', how='inner'
    )
    n = len(merged)

    print(f"\n{'='*70}")
    print(f"  {label} (n={n})")
    print(f"{'='*70}")

    # Distribution
    print(f"\n  Distribution of Predictions:")
    print(f"  {'Metric':<30} {'MAGICC V5':>12} {'CheckM2':>12}")
    print(f"  {'-'*56}")
    for name, v5_col, ck_col, func in [
        ('Completeness mean', 'v5_completeness', 'checkm2_completeness', 'mean'),
        ('Completeness median', 'v5_completeness', 'checkm2_completeness', 'median'),
        ('Completeness std', 'v5_completeness', 'checkm2_completeness', 'std'),
        ('Completeness min', 'v5_completeness', 'checkm2_completeness', 'min'),
        ('Completeness max', 'v5_completeness', 'checkm2_completeness', 'max'),
        ('Contamination mean', 'v5_contamination', 'checkm2_contamination', 'mean'),
        ('Contamination median', 'v5_contamination', 'checkm2_contamination', 'median'),
        ('Contamination std', 'v5_contamination', 'checkm2_contamination', 'std'),
        ('Contamination min', 'v5_contamination', 'checkm2_contamination', 'min'),
        ('Contamination max', 'v5_contamination', 'checkm2_contamination', 'max'),
    ]:
        v5_val = getattr(merged[v5_col], func)()
        ck_val = getattr(merged[ck_col], func)()
        print(f"  {name:<30} {v5_val:>11.2f}% {ck_val:>11.2f}%")

    # Agreement metrics
    comp_diff = merged['v5_completeness'].values - merged['checkm2_completeness'].values
    cont_diff = merged['v5_contamination'].values - merged['checkm2_contamination'].values
    comp_ae = np.abs(comp_diff)
    cont_ae = np.abs(cont_diff)

    comp_mae = np.mean(comp_ae)
    comp_rmse = np.sqrt(np.mean(comp_ae**2))
    cont_mae = np.mean(cont_ae)
    cont_rmse = np.sqrt(np.mean(cont_ae**2))

    v5c = merged['v5_completeness'].values
    ckc = merged['checkm2_completeness'].values
    v5x = merged['v5_contamination'].values
    ckx = merged['checkm2_contamination'].values

    comp_r = np.corrcoef(v5c, ckc)[0, 1] if np.std(v5c) > 0 and np.std(ckc) > 0 else float('nan')
    cont_r = np.corrcoef(v5x, ckx)[0, 1] if np.std(v5x) > 0 and np.std(ckx) > 0 else float('nan')
    comp_bias = np.mean(comp_diff)
    cont_bias = np.mean(cont_diff)

    print(f"\n  Agreement (V5 vs CheckM2):")
    print(f"  {'Metric':<35} {'Completeness':>14} {'Contamination':>14}")
    print(f"  {'-'*65}")
    print(f"  {'MAE':<35} {comp_mae:>13.2f}% {cont_mae:>13.2f}%")
    print(f"  {'RMSE':<35} {comp_rmse:>13.2f}% {cont_rmse:>13.2f}%")
    print(f"  {'Bias (V5 - CheckM2)':<35} {comp_bias:>+13.2f}% {cont_bias:>+13.2f}%")
    print(f"  {'Pearson r':<35} {comp_r:>14.4f} {cont_r:>14.4f}")

    # MIMAG classification
    merged['v5_mimag'] = [classify_mimag(c, x) for c, x in
                           zip(merged['v5_completeness'], merged['v5_contamination'])]
    merged['checkm2_mimag'] = [classify_mimag(c, x) for c, x in
                                zip(merged['checkm2_completeness'], merged['checkm2_contamination'])]

    v5_counts = merged['v5_mimag'].value_counts()
    ck_counts = merged['checkm2_mimag'].value_counts()

    print(f"\n  MIMAG Tier Distribution:")
    print(f"  {'Tier':<6} {'CheckM2':>10} {'MAGICC V5':>10}")
    print(f"  {'-'*28}")
    for tier in ['HQ', 'MQ', 'LQ']:
        print(f"  {tier:<6} {ck_counts.get(tier, 0):>10} {v5_counts.get(tier, 0):>10}")

    # Confusion matrix
    tiers = ['HQ', 'MQ', 'LQ']
    confusion = pd.crosstab(merged['checkm2_mimag'], merged['v5_mimag'],
                            rownames=['CheckM2'], colnames=['V5'], dropna=False)
    for t in tiers:
        if t not in confusion.columns:
            confusion[t] = 0
        if t not in confusion.index:
            confusion.loc[t] = 0
    confusion = confusion.reindex(index=tiers, columns=tiers, fill_value=0)

    print(f"\n  Confusion Matrix (rows=CheckM2, cols=V5):")
    print(f"  {'':>12} {'V5-HQ':>8} {'V5-MQ':>8} {'V5-LQ':>8} {'Total':>8}")
    print(f"  {'-'*46}")
    for row_tier in tiers:
        vals = [confusion.loc[row_tier, col_tier] for col_tier in tiers]
        total = sum(vals)
        print(f"  CkM2-{row_tier:<5} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {total:>8}")
    col_totals = [confusion[t].sum() for t in tiers]
    print(f"  {'Total':<12} {col_totals[0]:>8} {col_totals[1]:>8} {col_totals[2]:>8} {sum(col_totals):>8}")

    # Agreement
    agree = (merged['checkm2_mimag'] == merged['v5_mimag']).sum()
    agreement_pct = agree / n * 100
    print(f"\n  MIMAG Agreement: {agree}/{n} ({agreement_pct:.1f}%)")

    # HQ analysis
    checkm2_hq = merged['checkm2_mimag'] == 'HQ'
    v5_hq = merged['v5_mimag'] == 'HQ'

    hq_both = int((checkm2_hq & v5_hq).sum())
    hq_checkm2_only = int((checkm2_hq & ~v5_hq).sum())
    hq_v5_only = int((~checkm2_hq & v5_hq).sum())
    hq_neither = int((~checkm2_hq & ~v5_hq).sum())

    print(f"\n  HQ Classification Comparison:")
    print(f"    Both HQ (agree):      {hq_both:>6}")
    print(f"    CheckM2 HQ only:      {hq_checkm2_only:>6}  (CheckM2 says HQ, V5 does NOT)")
    print(f"    V5 HQ only:           {hq_v5_only:>6}  (V5 says HQ, CheckM2 does NOT)")
    print(f"    Neither HQ (agree):   {hq_neither:>6}")

    if hq_checkm2_only > 0:
        downgraded = merged[checkm2_hq & ~v5_hq]
        v5_tier = downgraded['v5_mimag'].value_counts()
        print(f"\n    CheckM2-HQ genomes downgraded by V5:")
        for tier, cnt in v5_tier.items():
            print(f"      -> V5 {tier}: {cnt}")
        comp_low = (downgraded['v5_completeness'] < 90).sum()
        cont_high = (downgraded['v5_contamination'] > 5).sum()
        both_fail = ((downgraded['v5_completeness'] < 90) & (downgraded['v5_contamination'] > 5)).sum()
        print(f"    Reasons: comp<90: {comp_low}, cont>5: {cont_high}, both: {both_fail}")

    if hq_v5_only > 0:
        upgraded = merged[~checkm2_hq & v5_hq]
        ck_tier = upgraded['checkm2_mimag'].value_counts()
        print(f"\n    V5-HQ genomes NOT HQ by CheckM2:")
        for tier, cnt in ck_tier.items():
            print(f"      CheckM2 {tier}: {cnt}")
        comp_low_ck = (upgraded['checkm2_completeness'] < 90).sum()
        cont_high_ck = (upgraded['checkm2_contamination'] > 5).sum()
        print(f"    CheckM2 reasons: comp<90: {comp_low_ck}, cont>5: {cont_high_ck}")

    # Save per-genome comparison
    merged_out = RESULTS_DIR / f'{label.lower().replace(" ", "_")}_comparison.tsv'
    merged.to_csv(merged_out, sep='\t', index=False)
    print(f"\n  Saved: {merged_out}")

    return {
        'n': n,
        'v5_comp_mean': float(merged['v5_completeness'].mean()),
        'v5_comp_median': float(merged['v5_completeness'].median()),
        'v5_cont_mean': float(merged['v5_contamination'].mean()),
        'v5_cont_median': float(merged['v5_contamination'].median()),
        'checkm2_comp_mean': float(merged['checkm2_completeness'].mean()),
        'checkm2_comp_median': float(merged['checkm2_completeness'].median()),
        'checkm2_cont_mean': float(merged['checkm2_contamination'].mean()),
        'checkm2_cont_median': float(merged['checkm2_contamination'].median()),
        'comp_mae': float(comp_mae),
        'comp_rmse': float(comp_rmse),
        'comp_bias': float(comp_bias),
        'comp_r': float(comp_r),
        'cont_mae': float(cont_mae),
        'cont_rmse': float(cont_rmse),
        'cont_bias': float(cont_bias),
        'cont_r': float(cont_r),
        'mimag_agreement_n': int(agree),
        'mimag_agreement_pct': float(agreement_pct),
        'checkm2_hq': int(ck_counts.get('HQ', 0)),
        'checkm2_mq': int(ck_counts.get('MQ', 0)),
        'checkm2_lq': int(ck_counts.get('LQ', 0)),
        'v5_hq': int(v5_counts.get('HQ', 0)),
        'v5_mq': int(v5_counts.get('MQ', 0)),
        'v5_lq': int(v5_counts.get('LQ', 0)),
        'hq_both': hq_both,
        'hq_checkm2_only': hq_checkm2_only,
        'hq_v5_only': hq_v5_only,
        'hq_neither': hq_neither,
    }


# Run analysis
pure_results = analyze(pure_v5, pure_ckm2, 'Pure Culture')
mag_results = analyze(mag_v5, mag_ckm2, 'MAG')

# Combined summary table
rp = pure_results
rm = mag_results
print(f"\n\n{'='*76}")
print("COMBINED COMPARISON TABLE")
print(f"{'='*76}")
print(f"{'Metric':<44} {'Pure Culture':>14} {'MAGs':>14}")
print(f"{'-'*76}")
print(f"{'N genomes':<44} {rp['n']:>14} {rm['n']:>14}")
print(f"{'CheckM2 completeness mean':<44} {rp['checkm2_comp_mean']:>13.2f}% {rm['checkm2_comp_mean']:>13.2f}%")
print(f"{'CheckM2 completeness median':<44} {rp['checkm2_comp_median']:>13.2f}% {rm['checkm2_comp_median']:>13.2f}%")
print(f"{'V5 completeness mean':<44} {rp['v5_comp_mean']:>13.2f}% {rm['v5_comp_mean']:>13.2f}%")
print(f"{'V5 completeness median':<44} {rp['v5_comp_median']:>13.2f}% {rm['v5_comp_median']:>13.2f}%")
print(f"{'CheckM2 contamination mean':<44} {rp['checkm2_cont_mean']:>13.2f}% {rm['checkm2_cont_mean']:>13.2f}%")
print(f"{'CheckM2 contamination median':<44} {rp['checkm2_cont_median']:>13.2f}% {rm['checkm2_cont_median']:>13.2f}%")
print(f"{'V5 contamination mean':<44} {rp['v5_cont_mean']:>13.2f}% {rm['v5_cont_mean']:>13.2f}%")
print(f"{'V5 contamination median':<44} {rp['v5_cont_median']:>13.2f}% {rm['v5_cont_median']:>13.2f}%")
print(f"{'-'*76}")
print(f"{'Completeness MAE (V5 vs CheckM2)':<44} {rp['comp_mae']:>13.2f}% {rm['comp_mae']:>13.2f}%")
print(f"{'Completeness RMSE':<44} {rp['comp_rmse']:>13.2f}% {rm['comp_rmse']:>13.2f}%")
print(f"{'Completeness Bias (V5 - CheckM2)':<44} {rp['comp_bias']:>+13.2f}% {rm['comp_bias']:>+13.2f}%")
print(f"{'Completeness Pearson r':<44} {rp['comp_r']:>14.4f} {rm['comp_r']:>14.4f}")
print(f"{'Contamination MAE (V5 vs CheckM2)':<44} {rp['cont_mae']:>13.2f}% {rm['cont_mae']:>13.2f}%")
print(f"{'Contamination RMSE':<44} {rp['cont_rmse']:>13.2f}% {rm['cont_rmse']:>13.2f}%")
print(f"{'Contamination Bias (V5 - CheckM2)':<44} {rp['cont_bias']:>+13.2f}% {rm['cont_bias']:>+13.2f}%")
print(f"{'Contamination Pearson r':<44} {rp['cont_r']:>14.4f} {rm['cont_r']:>14.4f}")
print(f"{'-'*76}")
print(f"{'MIMAG Agreement':<44} {rp['mimag_agreement_pct']:>13.1f}% {rm['mimag_agreement_pct']:>13.1f}%")
print(f"{'CheckM2 HQ count':<44} {rp['checkm2_hq']:>14} {rm['checkm2_hq']:>14}")
print(f"{'V5 HQ count':<44} {rp['v5_hq']:>14} {rm['v5_hq']:>14}")
print(f"{'Both HQ':<44} {rp['hq_both']:>14} {rm['hq_both']:>14}")
print(f"{'CheckM2 HQ only (V5 disagrees)':<44} {rp['hq_checkm2_only']:>14} {rm['hq_checkm2_only']:>14}")
print(f"{'V5 HQ only (CheckM2 disagrees)':<44} {rp['hq_v5_only']:>14} {rm['hq_v5_only']:>14}")
print(f"{'Neither HQ':<44} {rp['hq_neither']:>14} {rm['hq_neither']:>14}")
print(f"{'CheckM2 MQ count':<44} {rp['checkm2_mq']:>14} {rm['checkm2_mq']:>14}")
print(f"{'V5 MQ count':<44} {rp['v5_mq']:>14} {rm['v5_mq']:>14}")
print(f"{'CheckM2 LQ count':<44} {rp['checkm2_lq']:>14} {rm['checkm2_lq']:>14}")
print(f"{'V5 LQ count':<44} {rp['v5_lq']:>14} {rm['v5_lq']:>14}")

# Save summary JSON
summary = {
    'description': 'NCBI GenBank comparison: 1000 pure culture + 1000 MAG bacterial genomes',
    'pure_culture_pool': 73556,
    'mag_pool': 634485,
    'seed': 42,
    'pure_culture': rp,
    'mag': rm,
}
with open(RESULTS_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {RESULTS_DIR / 'summary.json'}")

# Save combined comparison table as TSV
table_rows = []
for label, r in [('Pure Culture', rp), ('MAG', rm)]:
    table_rows.append({
        'Dataset': label,
        'N': r['n'],
        'CheckM2_Comp_Mean': round(r['checkm2_comp_mean'], 2),
        'CheckM2_Comp_Median': round(r['checkm2_comp_median'], 2),
        'V5_Comp_Mean': round(r['v5_comp_mean'], 2),
        'V5_Comp_Median': round(r['v5_comp_median'], 2),
        'CheckM2_Cont_Mean': round(r['checkm2_cont_mean'], 2),
        'CheckM2_Cont_Median': round(r['checkm2_cont_median'], 2),
        'V5_Cont_Mean': round(r['v5_cont_mean'], 2),
        'V5_Cont_Median': round(r['v5_cont_median'], 2),
        'Comp_MAE': round(r['comp_mae'], 2),
        'Cont_MAE': round(r['cont_mae'], 2),
        'Comp_r': round(r['comp_r'], 4),
        'Cont_r': round(r['cont_r'], 4),
        'MIMAG_Agreement_Pct': round(r['mimag_agreement_pct'], 1),
        'CheckM2_HQ': r['checkm2_hq'],
        'V5_HQ': r['v5_hq'],
        'HQ_Both': r['hq_both'],
        'HQ_CheckM2_Only': r['hq_checkm2_only'],
        'HQ_V5_Only': r['hq_v5_only'],
    })
table_df = pd.DataFrame(table_rows)
table_df.to_csv(RESULTS_DIR / 'comparison_table.tsv', sep='\t', index=False)
print(f"Saved: {RESULTS_DIR / 'comparison_table.tsv'}")
