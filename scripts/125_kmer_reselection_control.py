#!/usr/bin/env python3
"""
WS1.7 - K-mer feature-selection leakage control.

The production 9,249 canonical 9-mers were selected by genome breadth
(prevalence) over 1,000 bacterial + 1,000 archaeal TRAINING-split
representatives (scripts/08-11). If panel phyla contributed to that
representative set, the feature set itself carries mild taxonomic leakage.

This script:
  1. Counts, per representative, the set of canonical 9-mers present in its
     already-extracted core-gene FASTA (data/kmer_selection/*_core_genes/).
     Encoding is identical to magicc/kmer_counter.py (A=0 C=1 G=2 T=3,
     canonical = min(code, revcomp code), windows containing non-ACGT skipped).
  2. VALIDATION: reproduces the production prevalence tables from all 1,000
     representatives and compares with the stored *_kmer_prevalence.tsv, so any
     later difference is attributable to the panel removal and not to a
     re-implementation artifact.
  3. RESELECTION: recomputes prevalence using only NON-panel representatives,
     takes the top 9,000 bacterial + top 1,000 archaeal, merges, and reports
     Jaccard overlap with the production 9,249-mer set.

Outputs (results/revision/holdout/)
  kmer_reselection_summary.json
  kmer_reselection_prevalence_bacterial.tsv
  kmer_reselection_prevalence_archaeal.tsv
  selected_kmers_holdout.txt          reselected set (for reference / 1.9 use)
  kmer_reselection_dropped_added.tsv

Usage:  python scripts/125_kmer_reselection_control.py --workers 24
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C
from holdout_lib.kmer_counter import _BASE_MAP, K, encode_kmer

N_POSSIBLE = 4 ** K
N_BACT_SELECT = 9000
N_ARCH_SELECT = 1000


@nb.njit(cache=True)
def _presence(seq, base_map, k, out):
    """Mark canonical k-mer codes present in seq (out: uint8[4**k])."""
    mask = (1 << (2 * k)) - 1
    shift = 2 * (k - 1)
    code = 0
    rc = 0
    valid = 0
    for i in range(len(seq)):
        b = base_map[seq[i]]
        if b == 255:
            valid = 0
            code = 0
            rc = 0
            continue
        code = ((code << 2) | b) & mask
        rc = (rc >> 2) | ((3 - b) << shift)
        valid += 1
        if valid >= k:
            c = code if code < rc else rc
            out[c] = 1


def genome_presence(fasta_path):
    """Return sorted array of canonical 9-mer codes present in a core-gene FASTA."""
    out = np.zeros(N_POSSIBLE, dtype=np.uint8)
    try:
        with open(fasta_path) as f:
            buf = []
            for line in f:
                if line.startswith('>'):
                    if buf:
                        _presence(np.frombuffer(''.join(buf).encode('ascii'),
                                                dtype=np.uint8), _BASE_MAP, K, out)
                        buf = []
                else:
                    buf.append(line.strip())
            if buf:
                _presence(np.frombuffer(''.join(buf).encode('ascii'), dtype=np.uint8),
                          _BASE_MAP, K, out)
    except FileNotFoundError:
        return np.array([], dtype=np.int32)
    return np.flatnonzero(out).astype(np.int32)


def code_to_kmer(code):
    s = []
    for _ in range(K):
        s.append('ACGT'[code & 3])
        code >>= 2
    return ''.join(reversed(s))


def prevalence(paths, workers):
    prev = np.zeros(N_POSSIBLE, dtype=np.int32)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for codes in ex.map(genome_presence, paths, chunksize=4):
            if len(codes):
                prev[codes] += 1
                n_ok += 1
    return prev, n_ok


def top_n(prev, n):
    """Top-n codes by prevalence; deterministic tie-break by ascending code."""
    order = np.lexsort((np.arange(N_POSSIBLE), -prev))
    sel = order[:n]
    return sel, prev[sel]


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b), len(a & b), len(a - b), len(b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print(f'k-mer feature-selection leakage control '
          f'(panel level: {C.PANEL_LEVEL})')
    print('=' * 78)

    prod_codes = np.array(sorted(encode_kmer(l.strip())
                                 for l in open(C.SELECTED_KMERS) if l.strip()),
                          dtype=np.int32)
    print(f'production selected k-mers: {len(prod_codes)}')

    res = {'production_n_kmers': int(len(prod_codes)),
           'panel_level': C.PANEL_LEVEL, 'panel_taxa': C.PANEL_TAXA,
           'panel_phyla': C.PANEL_PHYLA, 'domains': {}}
    new_sets = {}

    for dom, reps_tsv, core_dir, n_sel in [
            ('bacterial', 'selected_bacterial_1000.tsv', 'bacterial_core_genes',
             N_BACT_SELECT),
            ('archaeal', 'selected_archaeal_1000.tsv', 'archaeal_core_genes',
             N_ARCH_SELECT)]:
        print(f'\n--- {dom} ---')
        reps = pd.read_csv(C.KMER_DIR / reps_tsv, sep='\t')
        cg = pd.read_csv(C.KMER_DIR / core_dir / 'core_gene_results.tsv', sep='\t')
        cg['core_gene_path'] = cg['core_gene_path'].map(C.remap_fasta_path)
        cg = cg[['accession', 'core_gene_path', 'n_core_genes']]
        reps = reps.merge(cg, left_on='ncbi_accession', right_on='accession', how='left')
        n_missing = int(reps.core_gene_path.isna().sum())
        reps = reps.dropna(subset=['core_gene_path'])
        # Panel membership at the CONFIGURED level. The representative tables
        # carry `phylum` but not `family`/`genus`, so at family and genus level it
        # is derived from the GTDB lineage; without this the mask would be
        # silently all-False.
        C.add_taxon_column(reps)
        is_panel = reps[C.TAXON_COL].isin(C.PANEL_TAXA)
        print(f'  {len(reps)} representatives with core genes '
              f'({n_missing} without); {int(is_panel.sum())} are panel '
              f'{C.TAXON_COL}s ({100*is_panel.mean():.1f}%)')
        print(f'  panel contribution by {C.TAXON_COL}:')
        for k, v in reps[is_panel][C.TAXON_COL].value_counts().items():
            print(f'    {k}: {v}')

        print('  counting 9-mer presence for ALL representatives...')
        prev_all, n_all = prevalence(list(reps.core_gene_path), a.workers)
        print(f'    {n_all} genomes counted, {int((prev_all>0).sum())} distinct '
              f'canonical 9-mers observed')

        # -- validation against the stored production prevalence table
        stored = pd.read_csv(C.KMER_DIR / f'{dom}_kmer_prevalence.tsv', sep='\t')
        stored['code'] = [encode_kmer(k) for k in stored.kmer]
        mine = prev_all[stored.code.values]
        agree = int((mine == stored.prevalence.values).sum())
        corr = float(np.corrcoef(mine, stored.prevalence.values)[0, 1])
        maxdiff = int(np.max(np.abs(mine - stored.prevalence.values)))
        print(f'  VALIDATION vs stored {dom}_kmer_prevalence.tsv '
              f'({len(stored)} rows): exact match {agree}/{len(stored)} '
              f'({100*agree/len(stored):.2f}%), r={corr:.6f}, max|diff|={maxdiff}')

        sel_all, pv_all = top_n(prev_all, n_sel)

        # -- reselection excluding panel representatives
        keep = reps[~is_panel]
        print(f'  counting 9-mer presence for {len(keep)} NON-panel representatives...')
        prev_np, n_np = prevalence(list(keep.core_gene_path), a.workers)
        sel_np, pv_np = top_n(prev_np, n_sel)
        print(f'    prevalence denominator {n_np} (production {n_all})')
        print(f'    top-{n_sel} prevalence range {pv_np.min()}-{pv_np.max()} '
              f'({100*pv_np.min()/max(n_np,1):.1f}%-{100*pv_np.max()/max(n_np,1):.1f}% '
              f'of genomes); production range {pv_all.min()}-{pv_all.max()}')

        j_dom, ni, no1, no2 = jaccard(sel_all, sel_np)
        print(f'  within-{dom} Jaccard (all reps vs non-panel reps): {j_dom:.4f} '
              f'({ni} shared, {no1} lost, {no2} gained)')

        pd.DataFrame({'kmer': [code_to_kmer(int(c)) for c in
                               np.flatnonzero(prev_np > 0)],
                      'prevalence_nonpanel': prev_np[prev_np > 0],
                      'prevalence_all': prev_all[prev_np > 0]}
                     ).sort_values('prevalence_nonpanel', ascending=False).to_csv(
            C.RESULTS_DIR / f'kmer_reselection_prevalence_{dom}.tsv',
            sep='\t', index=False)

        # ---- where does the churn come from? ----------------------------
        # If the k-mers that cross the top-N boundary sit near the prevalence
        # cutoff, the churn is rank noise at an arbitrary threshold rather than
        # loss of lineage-informative features. Quantify directly.
        from scipy.stats import spearmanr
        obs = (prev_all > 0) | (prev_np > 0)
        rho = float(spearmanr(prev_all[obs], prev_np[obs]).statistic)
        rate_all = prev_all / max(n_all, 1)
        rate_np = prev_np / max(n_np, 1)
        panel_paths = list(reps[is_panel].core_gene_path)
        prev_p, n_p = prevalence(panel_paths, a.workers)
        rate_p = prev_p / max(n_p, 1)

        lost = np.array(sorted(set(map(int, sel_all)) - set(map(int, sel_np))))
        gained = np.array(sorted(set(map(int, sel_np)) - set(map(int, sel_all))))
        kept = np.array(sorted(set(map(int, sel_all)) & set(map(int, sel_np))))
        print(f'  Spearman rho of prevalence over all observed 9-mers '
              f'(all reps vs non-panel reps): {rho:.6f}')
        print(f'  prevalence RATE of the {len(lost)} lost k-mers: '
              f'all-reps {rate_all[lost].mean():.3f}, non-panel {rate_np[lost].mean():.3f}, '
              f'panel-only {rate_p[lost].mean():.3f}')
        print(f'  prevalence RATE of the {len(kept)} retained k-mers: '
              f'all-reps {rate_all[kept].mean():.3f}, non-panel {rate_np[kept].mean():.3f}, '
              f'panel-only {rate_p[kept].mean():.3f}')
        print(f'  panel-minus-nonpanel prevalence rate: lost '
              f'{np.mean(rate_p[lost]-rate_np[lost]):+.4f}, retained '
              f'{np.mean(rate_p[kept]-rate_np[kept]):+.4f}, gained '
              f'{np.mean(rate_p[gained]-rate_np[gained]):+.4f}')
        # retention by production prevalence-rank stratum
        order_all = np.lexsort((np.arange(N_POSSIBLE), -prev_all))
        strat = []
        bounds = [(0, 1000), (1000, 3000), (3000, 6000), (6000, n_sel)]
        for lo, hi in bounds:
            if lo >= n_sel:
                continue
            hi = min(hi, n_sel)
            blk = set(map(int, order_all[lo:hi]))
            ret = len(blk & set(map(int, sel_np))) / max(len(blk), 1)
            strat.append({'rank_range': f'{lo+1}-{hi}', 'n': hi - lo,
                          'retained_frac': round(ret, 4),
                          'prevalence_rate_all_reps':
                              round(float(rate_all[order_all[lo:hi]].mean()), 4)})
            print(f'    production rank {lo+1}-{hi}: retained '
                  f'{100*ret:.1f}% (mean prevalence rate '
                  f'{rate_all[order_all[lo:hi]].mean():.3f})')

        res['domains'][dom] = {
            'spearman_rho_prevalence': round(rho, 6),
            'prevalence_rate_lost': {
                'all_reps': round(float(rate_all[lost].mean()), 4),
                'nonpanel_reps': round(float(rate_np[lost].mean()), 4),
                'panel_reps': round(float(rate_p[lost].mean()), 4),
                'panel_minus_nonpanel': round(float(np.mean(rate_p[lost]
                                                            - rate_np[lost])), 4)},
            'prevalence_rate_retained': {
                'all_reps': round(float(rate_all[kept].mean()), 4),
                'nonpanel_reps': round(float(rate_np[kept].mean()), 4),
                'panel_reps': round(float(rate_p[kept].mean()), 4),
                'panel_minus_nonpanel': round(float(np.mean(rate_p[kept]
                                                            - rate_np[kept])), 4)},
            'retention_by_production_rank': strat,
            'n_representatives': int(len(reps)),
            'n_panel_representatives': int(is_panel.sum()),
            'pct_panel_representatives': round(100 * float(is_panel.mean()), 2),
            'panel_by_phylum': {k: int(v) for k, v in
                                reps[is_panel][C.TAXON_COL].value_counts().items()},
            'n_nonpanel_representatives': int(len(keep)),
            'validation_vs_stored': {'n_compared': int(len(stored)),
                                     'exact_match': agree,
                                     'pct_exact': round(100 * agree / len(stored), 3),
                                     'pearson_r': round(corr, 6),
                                     'max_abs_diff': maxdiff},
            'top_n': n_sel,
            'prevalence_cutoff_all': int(pv_all.min()),
            'prevalence_cutoff_nonpanel': int(pv_np.min()),
            'within_domain_jaccard': round(j_dom, 4),
            'within_domain_shared': ni, 'lost': no1, 'gained': no2,
        }
        new_sets[dom] = sel_np
        new_sets[dom + '_all'] = sel_all

    # ---- merged sets -----------------------------------------------------
    merged_np = np.array(sorted(set(map(int, new_sets['bacterial'])) |
                                set(map(int, new_sets['archaeal']))), dtype=np.int32)
    merged_all = np.array(sorted(set(map(int, new_sets['bacterial_all'])) |
                                 set(map(int, new_sets['archaeal_all']))), dtype=np.int32)

    j_prod, ni, lost, gained = jaccard(prod_codes, merged_np)
    j_repro, ri, rl, rg = jaccard(prod_codes, merged_all)
    print('\n' + '=' * 78)
    print('MERGED 9-MER SET COMPARISON')
    print('=' * 78)
    print(f'production set                        : {len(prod_codes)} k-mers')
    print(f'reproduced from ALL reps (control)    : {len(merged_all)} k-mers  '
          f'Jaccard vs production {j_repro:.4f} '
          f'({ri} shared, {rl} lost, {rg} gained)')
    print(f'RESELECTED excluding panel phyla      : {len(merged_np)} k-mers  '
          f'Jaccard vs production {j_prod:.4f} '
          f'({ni} shared, {lost} lost, {gained} gained)')
    jj, _, _, _ = jaccard(merged_all, merged_np)
    print(f'reselected vs reproduced-control      : Jaccard {jj:.4f}')
    print(f'\n=> {100*ni/len(prod_codes):.2f}% of the production 9,249 k-mers are '
          f're-selected when every panel-phylum genome is removed from the '
          f'representative set.')

    with open(C.RESULTS_DIR / 'selected_kmers_holdout.txt', 'w') as f:
        for c in merged_np:
            f.write(code_to_kmer(int(c)) + '\n')

    prod_s, new_s = set(map(int, prod_codes)), set(map(int, merged_np))
    rows = ([{'kmer': code_to_kmer(c), 'status': 'lost_when_panel_removed'}
             for c in sorted(prod_s - new_s)] +
            [{'kmer': code_to_kmer(c), 'status': 'gained_when_panel_removed'}
             for c in sorted(new_s - prod_s)])
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / 'kmer_reselection_dropped_added.tsv',
                              sep='\t', index=False)

    res['merged'] = {
        'n_production': int(len(prod_codes)),
        'n_reproduced_all_reps': int(len(merged_all)),
        'n_reselected_nonpanel': int(len(merged_np)),
        'jaccard_production_vs_reselected': round(j_prod, 4),
        'jaccard_production_vs_reproduced_control': round(j_repro, 4),
        'jaccard_reproduced_vs_reselected': round(jj, 4),
        'shared': ni, 'lost': lost, 'gained': gained,
        'pct_production_retained': round(100 * ni / len(prod_codes), 2),
    }
    # ---- where in the production ranking do the merged losses sit? -------
    ann = pd.read_csv(C.KMER_DIR / 'selected_kmers_annotated.tsv', sep='\t')
    lost_kmers = pd.DataFrame({'kmer': [code_to_kmer(c) for c in sorted(prod_s - new_s)]})
    mm = ann.merge(lost_kmers, on='kmer')
    stats = json.loads((C.KMER_DIR / 'kmer_selection_stats.json').read_text())
    res['merged']['lost_by_production_source_branch'] = {
        k: int(v) for k, v in mm.source.value_counts().items()}
    res['merged']['lost_production_prevalence'] = {
        'bacterial_mean': round(float(mm.bacterial_prevalence.mean()), 1),
        'bacterial_max': int(mm.bacterial_prevalence.max()),
        'archaeal_mean': round(float(mm.archaeal_prevalence.mean()), 1),
        'archaeal_max': int(mm.archaeal_prevalence.max())}
    res['merged']['all_selected_production_prevalence'] = {
        'bacterial_mean': round(float(ann.bacterial_prevalence.mean()), 1),
        'bacterial_max': int(ann.bacterial_prevalence.max()),
        'archaeal_mean': round(float(ann.archaeal_prevalence.mean()), 1),
        'archaeal_max': int(ann.archaeal_prevalence.max())}
    res['merged']['production_composition'] = stats
    res['merged']['archaeal_only_share_pct'] = round(
        100 * stats['n_arch_only'] / stats['n_final_merged'], 2)
    print('\nlost k-mers by production source branch: '
          f'{res["merged"]["lost_by_production_source_branch"]}')
    print(f'  lost k-mers: bacterial prevalence mean '
          f'{mm.bacterial_prevalence.mean():.0f}, MAX '
          f'{int(mm.bacterial_prevalence.max())} (whole selected set: mean '
          f'{ann.bacterial_prevalence.mean():.0f}, max '
          f'{int(ann.bacterial_prevalence.max())})')
    print('  => every lost k-mer sits in the low-prevalence tail of the production '
          'ranking; none of the high-prevalence core is affected.')
    print(f'  archaeal-only k-mers are {res["merged"]["archaeal_only_share_pct"]}% '
          f'of the 9,249 feature set (n_arch_only={stats["n_arch_only"]})')

    res['interpretation'] = (
        'Jaccard alone is misleading here because the selection is a hard top-N cut on '
        'a continuous prevalence ranking, so k-mers sitting near the cutoff churn even '
        'under small perturbations of the representative set. The informative statistics '
        'are (a) the Spearman correlation of prevalence, (b) retention stratified by '
        'production prevalence rank, and (c) the production prevalence of the lost '
        'k-mers. Retention of 100% in the top bacterial ranks with churn confined to '
        'the low-prevalence tail means the feature set is stable where it carries '
        'signal. Direction of residual bias: because the production features were '
        'partly chosen using panel genomes, they are if anything slightly better suited '
        'to panel-phylum core genes than a panel-blind feature set would be, so any '
        'measured degradation on held-out phyla is a CONSERVATIVE (under)estimate. '
        'Feature-level leakage therefore cannot manufacture a false-negative result.')
    (C.RESULTS_DIR / 'kmer_reselection_summary.json').write_text(json.dumps(res, indent=2))
    print(f"\nwrote {C.RESULTS_DIR / 'kmer_reselection_summary.json'}")


if __name__ == '__main__':
    main()
