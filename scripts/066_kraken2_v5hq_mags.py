#!/usr/bin/env python3
"""Step 8: Kraken2 taxonomic contamination check for 93 MAGs.

These 93 MAGs are classified as HQ by MAGICC V5 but NOT by CheckM2.
This is the opposite case from script 61 (which analyzed 141 CheckM2-HQ-but-V5-not-HQ MAGs).

For these 93 MAGs:
- V5 says HQ (completeness >= 90%, contamination <= 5%)
- CheckM2 says NOT HQ (completeness < 90% OR contamination > 5%)

Key questions:
- Are these MAGs taxonomically clean? (supporting V5's low contamination estimate)
- Or does Kraken2 find contamination that V5 missed? (supporting CheckM2)
- How many have phylum contamination > 5%?
- Does CheckM2's lower completeness reflect genuine incompleteness, or underestimate?

Strategy: same as script 61:
- Run Kraken2 on each MAG to classify every contig
- Parse per-contig output to get the assigned taxid and contig length
- Use the Kraken2 taxonomy (nodes.dmp/names.dmp) to resolve each contig's
  full lineage (phylum, class, order, family, genus, species)
- Calculate contamination at multiple taxonomic levels
- Compare with V5 and CheckM2 predictions
"""

import sys
import os
import json
import subprocess
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

os.chdir('/path/to/magicc-legacy')

# ============================================================================
# Configuration
# ============================================================================
KRAKEN2_DB = Path('tools/kraken2_db')
MAG_DIR = Path('data/ncbi/mags')
KRAKEN2_OUT_DIR = Path('data/ncbi/kraken2_results_v5hq')
RESULTS_DIR = Path('results/ncbi_comparison')
THREADS = 43
CHECKPOINT_FILE = KRAKEN2_OUT_DIR / 'checkpoint.json'

KRAKEN2_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 0: Identify the 93 MAGs (V5 HQ but NOT CheckM2 HQ)
# ============================================================================
print("=" * 70)
print("Step 0: Identifying 93 V5-HQ but CheckM2-not-HQ MAGs")
print("=" * 70)

mag_comp = pd.read_csv(RESULTS_DIR / 'mag_comparison.tsv', sep='\t')
v5_hq = (mag_comp['v5_completeness'] >= 90) & (mag_comp['v5_contamination'] <= 5)
checkm2_hq = (mag_comp['checkm2_completeness'] >= 90) & (mag_comp['checkm2_contamination'] <= 5)
target_df = mag_comp[v5_hq & ~checkm2_hq].copy().reset_index(drop=True)

print(f"  Total MAGs: {len(mag_comp)}")
print(f"  V5 HQ: {v5_hq.sum()}")
print(f"  CheckM2 HQ: {checkm2_hq.sum()}")
print(f"  V5 HQ but NOT CheckM2 HQ: {len(target_df)}")

# Breakdown of why CheckM2 downgrades
low_comp = target_df['checkm2_completeness'] < 90
high_cont = target_df['checkm2_contamination'] > 5
both = low_comp & high_cont
print(f"\n  Reasons CheckM2 downgrades:")
print(f"    CheckM2 completeness < 90%: {low_comp.sum()}")
print(f"    CheckM2 contamination > 5%: {high_cont.sum()}")
print(f"    Both: {both.sum()}")
print(f"    Only low completeness: {(low_comp & ~high_cont).sum()}")
print(f"    Only high contamination: {(~low_comp & high_cont).sum()}")

accessions = target_df['accession'].tolist()

# Verify FASTA files exist
missing = []
for acc in accessions:
    fasta = MAG_DIR / f'{acc}.fna'
    if not fasta.exists():
        missing.append(acc)
if missing:
    print(f"  ERROR: Missing FASTA files for {len(missing)} MAGs:")
    for m in missing[:5]:
        print(f"    {m}")
    sys.exit(1)
print(f"  All {len(accessions)} FASTA files verified.")

# ============================================================================
# Step 1: Run Kraken2 on all 93 MAGs (with checkpointing)
# ============================================================================
print()
print("=" * 70)
print(f"Step 1: Running Kraken2 on {len(accessions)} MAGs")
print("=" * 70)

# Load checkpoint
completed = set()
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE) as f:
        completed = set(json.load(f).get('completed', []))
    print(f"  Checkpoint loaded: {len(completed)} already completed")


def save_checkpoint():
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': sorted(completed)}, f)


remaining = [a for a in accessions if a not in completed]
if remaining:
    t0 = time.time()
    for i, acc in enumerate(remaining):
        fasta = MAG_DIR / f'{acc}.fna'
        output = KRAKEN2_OUT_DIR / f'{acc}.kraken2.txt'
        report = KRAKEN2_OUT_DIR / f'{acc}.kraken2.report'

        cmd = [
            'kraken2',
            '--db', str(KRAKEN2_DB),
            '--threads', str(THREADS),
            '--output', str(output),
            '--report', str(report),
            '--use-names',
            str(fasta),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR on {acc}: {result.stderr.strip()}")
            continue

        completed.add(acc)

        if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(remaining)} MAGs processed ({elapsed:.1f}s elapsed)")
            save_checkpoint()

    save_checkpoint()
    elapsed = time.time() - t0
    print(f"  Kraken2 completed: {len(completed)}/{len(accessions)} MAGs in {elapsed:.1f}s")
else:
    print(f"  All {len(accessions)} MAGs already processed (from checkpoint)")


# ============================================================================
# Step 2: Load NCBI taxonomy from Kraken2 database
# ============================================================================
print()
print("=" * 70)
print("Step 2: Loading NCBI taxonomy")
print("=" * 70)

# Parse nodes.dmp: taxid -> (parent_taxid, rank)
nodes_file = KRAKEN2_DB / 'nodes.dmp'
names_file = KRAKEN2_DB / 'names.dmp'

print("  Loading nodes.dmp...")
taxid_parent = {}   # taxid -> parent_taxid
taxid_rank = {}     # taxid -> rank
with open(nodes_file) as f:
    for line in f:
        parts = line.strip().split('\t|\t')
        if len(parts) >= 3:
            tid = int(parts[0].strip())
            pid = int(parts[1].strip())
            rank = parts[2].strip().rstrip('\t|')
            taxid_parent[tid] = pid
            taxid_rank[tid] = rank
print(f"  Loaded {len(taxid_parent)} taxa from nodes.dmp")

print("  Loading names.dmp...")
taxid_name = {}  # taxid -> scientific name
with open(names_file) as f:
    for line in f:
        parts = line.strip().split('\t|\t')
        if len(parts) >= 4:
            tid = int(parts[0].strip())
            name = parts[1].strip()
            name_class = parts[3].strip().rstrip('\t|')
            if name_class == 'scientific name':
                taxid_name[tid] = name
print(f"  Loaded {len(taxid_name)} scientific names from names.dmp")


def get_lineage(taxid):
    """Get the full lineage for a taxid as dict: {rank: (taxid, name)}."""
    lineage = {}
    current = taxid
    visited = set()
    while current and current != 1 and current not in visited:
        visited.add(current)
        rank = taxid_rank.get(current, 'no rank')
        name = taxid_name.get(current, f'taxid_{current}')
        if rank != 'no rank' and rank != 'clade':
            lineage[rank] = (current, name)
        current = taxid_parent.get(current)
    return lineage


# ============================================================================
# Step 3: Parse Kraken2 output and compute per-contig lineages
# ============================================================================
print()
print("=" * 70)
print(f"Step 3: Analyzing per-contig taxonomy for all {len(accessions)} MAGs")
print("=" * 70)


def parse_taxid_from_output(taxon_field):
    """Extract taxid from Kraken2 output field like 'Escherichia coli (taxid 562)'."""
    m = re.search(r'\(taxid\s+(\d+)\)', taxon_field)
    if m:
        return int(m.group(1))
    return 0


def analyze_mag(acc):
    """Analyze a single MAG's Kraken2 results using full taxonomy."""
    output_file = KRAKEN2_OUT_DIR / f'{acc}.kraken2.txt'
    if not output_file.exists():
        return None

    # Parse per-contig output
    contigs = []
    total_bp = 0
    classified_bp = 0
    unclassified_bp = 0

    with open(output_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            status = parts[0]  # C or U
            contig_name = parts[1]
            taxon_field = parts[2]
            contig_len = int(parts[3])
            taxid = parse_taxid_from_output(taxon_field)

            total_bp += contig_len
            if status == 'C':
                classified_bp += contig_len
            else:
                unclassified_bp += contig_len

            contigs.append({
                'contig': contig_name,
                'status': status,
                'taxid': taxid,
                'length': contig_len,
            })

    if total_bp == 0:
        return None

    # Get lineage for each classified contig
    # Aggregate bp at each taxonomic level
    levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    level_bp = {lev: defaultdict(int) for lev in levels}  # level -> {name: bp}
    level_contigs = {lev: defaultdict(int) for lev in levels}

    for contig in contigs:
        if contig['status'] != 'C':
            continue
        bp = contig['length']
        taxid = contig['taxid']

        lineage = get_lineage(taxid)

        for lev in levels:
            if lev in lineage:
                _, name = lineage[lev]
                level_bp[lev][name] += bp
                level_contigs[lev][name] += 1

    # For each level, find dominant taxon and calculate contamination
    result = {
        'accession': acc,
        'total_bp': total_bp,
        'classified_bp': classified_bp,
        'unclassified_bp': unclassified_bp,
        'classified_pct': round(classified_bp / total_bp * 100, 2),
        'unclassified_pct': round(unclassified_bp / total_bp * 100, 2),
        'n_contigs': len(contigs),
        'n_classified': sum(1 for c in contigs if c['status'] == 'C'),
    }

    for lev in levels:
        bp_dict = level_bp[lev]
        if not bp_dict:
            result[f'{lev}_dominant'] = 'unresolved'
            result[f'{lev}_dominant_bp'] = 0
            result[f'{lev}_dominant_pct'] = 0
            result[f'{lev}_contam_bp'] = 0
            result[f'{lev}_contam_pct'] = 0
            result[f'{lev}_n_taxa'] = 0
            continue

        sorted_taxa = sorted(bp_dict.items(), key=lambda x: -x[1])
        dominant_name = sorted_taxa[0][0]
        dominant_bp = sorted_taxa[0][1]
        total_resolved = sum(bp_dict.values())
        non_dominant_bp = total_resolved - dominant_bp

        result[f'{lev}_dominant'] = dominant_name
        result[f'{lev}_dominant_bp'] = dominant_bp
        result[f'{lev}_dominant_pct'] = round(dominant_bp / total_bp * 100, 2)
        result[f'{lev}_contam_bp'] = non_dominant_bp
        # Contamination: non-dominant bp as % of total bp
        result[f'{lev}_contam_pct'] = round(non_dominant_bp / total_bp * 100, 2)
        result[f'{lev}_n_taxa'] = len(bp_dict)

        # Also store breakdown for the top level
        if lev in ['phylum', 'genus', 'species']:
            breakdown = []
            for name, bp in sorted_taxa[:5]:
                breakdown.append(f"{name}({bp / total_bp * 100:.1f}%)")
            result[f'{lev}_breakdown'] = '; '.join(breakdown)

    return result


results = []
for i, row in target_df.iterrows():
    acc = row['accession']
    r = analyze_mag(acc)
    if r is None:
        print(f"  WARNING: No results for {acc}")
        continue
    # Add V5 and CheckM2 predictions
    r['v5_completeness'] = row['v5_completeness']
    r['v5_contamination'] = row['v5_contamination']
    r['v5_mimag'] = row['v5_mimag']
    r['checkm2_completeness'] = row['checkm2_completeness']
    r['checkm2_contamination'] = row['checkm2_contamination']
    r['checkm2_mimag'] = row['checkm2_mimag']
    results.append(r)

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(target_df)} processed")

df = pd.DataFrame(results)
print(f"  Processed {len(df)} MAGs")

# ============================================================================
# Step 4: Determine best contamination metric
# ============================================================================
print()
print("=" * 70)
print("Step 4: Contamination at multiple taxonomic levels")
print("=" * 70)

for lev in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
    n_resolved = (df[f'{lev}_dominant'] != 'unresolved').sum()
    if n_resolved > 0:
        mean_contam = df.loc[df[f'{lev}_dominant'] != 'unresolved', f'{lev}_contam_pct'].mean()
        med_contam = df.loc[df[f'{lev}_dominant'] != 'unresolved', f'{lev}_contam_pct'].median()
    else:
        mean_contam = med_contam = 0
    print(f"  {lev:>15}: {n_resolved:>3}/{len(df)} MAGs resolved, "
          f"mean contam = {mean_contam:.1f}%, median = {med_contam:.1f}%")

print()
print("  Using PHYLUM-LEVEL contamination as primary metric")
print("  (cross-phylum = definitive contamination)")
primary_contam_col = 'phylum_contam_pct'

print("  Using GENUS-LEVEL contamination as secondary metric")
print("  (cross-genus within same phylum = possible contamination)")

# ============================================================================
# Step 5: Summary statistics
# ============================================================================
print()
print("=" * 70)
print("Step 5: Summary Statistics")
print("=" * 70)

resolved = df[df['phylum_dominant'] != 'unresolved']
print(f"\n  MAGs with phylum-level resolution: {len(resolved)}/{len(df)}")

for threshold in [1, 2, 5, 10, 15, 20]:
    n = (resolved['phylum_contam_pct'] > threshold).sum()
    print(f"  Phylum-level contamination > {threshold:>2}%: {n:>3} MAGs "
          f"({n/len(resolved)*100:.1f}%)")

print()
genus_resolved = df[df['genus_dominant'] != 'unresolved']
print(f"  MAGs with genus-level resolution: {len(genus_resolved)}/{len(df)}")
for threshold in [1, 2, 5, 10, 15, 20]:
    n = (genus_resolved['genus_contam_pct'] > threshold).sum()
    print(f"  Genus-level contamination > {threshold:>2}%: {n:>3} MAGs "
          f"({n/len(genus_resolved)*100:.1f}%)")

# Distribution of number of phyla per MAG
print(f"\n  Number of phyla per MAG:")
phylum_counts = resolved['phylum_n_taxa']
for n_phyla in sorted(phylum_counts.unique()):
    count = (phylum_counts == n_phyla).sum()
    print(f"    {n_phyla} phylum/phyla: {count} MAGs")

# ============================================================================
# Step 6: Compare with V5 and CheckM2
# ============================================================================
print()
print("=" * 70)
print("Step 6: Comparing Kraken2 vs V5 vs CheckM2")
print("=" * 70)

from scipy import stats as scipy_stats

# Correlation: Kraken2 phylum contamination vs V5 contamination
mask = (df['phylum_dominant'] != 'unresolved') & np.isfinite(df['v5_contamination'])
r_v5_phylum = sr_v5_phylum = p_v5_phylum = sp_v5_phylum = float('nan')
r_ck_phylum = sr_ck_phylum = p_ck_phylum = sp_ck_phylum = float('nan')
r_v5_genus = sr_v5_genus = p_v5_genus = sp_v5_genus = float('nan')
r_ck_genus = sr_ck_genus = p_ck_genus = sp_ck_genus = float('nan')

if mask.sum() > 2:
    r_v5_phylum, p_v5_phylum = scipy_stats.pearsonr(
        df.loc[mask, 'phylum_contam_pct'], df.loc[mask, 'v5_contamination'])
    sr_v5_phylum, sp_v5_phylum = scipy_stats.spearmanr(
        df.loc[mask, 'phylum_contam_pct'], df.loc[mask, 'v5_contamination'])
    print(f"\n  Correlation: Kraken2 phylum contamination vs V5 contamination:")
    print(f"    Pearson r  = {r_v5_phylum:.4f} (p = {p_v5_phylum:.2e})")
    print(f"    Spearman r = {sr_v5_phylum:.4f} (p = {sp_v5_phylum:.2e})")

# Correlation: Kraken2 phylum contamination vs CheckM2 contamination
if mask.sum() > 2:
    r_ck_phylum, p_ck_phylum = scipy_stats.pearsonr(
        df.loc[mask, 'phylum_contam_pct'], df.loc[mask, 'checkm2_contamination'])
    sr_ck_phylum, sp_ck_phylum = scipy_stats.spearmanr(
        df.loc[mask, 'phylum_contam_pct'], df.loc[mask, 'checkm2_contamination'])
    print(f"\n  Correlation: Kraken2 phylum contamination vs CheckM2 contamination:")
    print(f"    Pearson r  = {r_ck_phylum:.4f} (p = {p_ck_phylum:.2e})")
    print(f"    Spearman r = {sr_ck_phylum:.4f} (p = {sp_ck_phylum:.2e})")

# Genus-level correlations
mask_genus = (df['genus_dominant'] != 'unresolved') & np.isfinite(df['v5_contamination'])
if mask_genus.sum() > 2:
    r_v5_genus, p_v5_genus = scipy_stats.pearsonr(
        df.loc[mask_genus, 'genus_contam_pct'], df.loc[mask_genus, 'v5_contamination'])
    sr_v5_genus, sp_v5_genus = scipy_stats.spearmanr(
        df.loc[mask_genus, 'genus_contam_pct'], df.loc[mask_genus, 'v5_contamination'])
    print(f"\n  Correlation: Kraken2 genus contamination vs V5 contamination:")
    print(f"    Pearson r  = {r_v5_genus:.4f} (p = {p_v5_genus:.2e})")
    print(f"    Spearman r = {sr_v5_genus:.4f} (p = {sp_v5_genus:.2e})")

if mask_genus.sum() > 2:
    r_ck_genus, p_ck_genus = scipy_stats.pearsonr(
        df.loc[mask_genus, 'genus_contam_pct'], df.loc[mask_genus, 'checkm2_contamination'])
    sr_ck_genus, sp_ck_genus = scipy_stats.spearmanr(
        df.loc[mask_genus, 'genus_contam_pct'], df.loc[mask_genus, 'checkm2_contamination'])
    print(f"\n  Correlation: Kraken2 genus contamination vs CheckM2 contamination:")
    print(f"    Pearson r  = {r_ck_genus:.4f} (p = {p_ck_genus:.2e})")
    print(f"    Spearman r = {sr_ck_genus:.4f} (p = {sp_ck_genus:.2e})")

# ============================================================================
# Step 7: Agreement analysis - who is right, V5 or CheckM2?
# ============================================================================
print()
print("=" * 70)
print("Step 7: Agreement analysis - V5 vs CheckM2 vs Kraken2")
print("=" * 70)

# For these 93 MAGs, V5 says ALL are HQ (contamination <= 5%).
# CheckM2 downgrades them. Key question: is Kraken2 supporting V5's low
# contamination estimate, or finding contamination like CheckM2 suspects?

# Split by reason for CheckM2 downgrade
print(f"\n  --- Breakdown by CheckM2 downgrade reason ---")

# MAGs where CheckM2 downgrades due to HIGH CONTAMINATION (> 5%)
high_cont_mags = df[df['checkm2_contamination'] > 5]
print(f"\n  CheckM2 cont > 5% group: {len(high_cont_mags)} MAGs")
if len(high_cont_mags) > 0:
    hc_resolved = high_cont_mags[high_cont_mags['phylum_dominant'] != 'unresolved']
    if len(hc_resolved) > 0:
        print(f"    Phylum-resolved: {len(hc_resolved)}")
        print(f"    Kraken2 phylum contam > 5%: {(hc_resolved['phylum_contam_pct'] > 5).sum()}")
        print(f"    Kraken2 phylum contam <= 5%: {(hc_resolved['phylum_contam_pct'] <= 5).sum()}")
        print(f"    Mean Kraken2 phylum contam: {hc_resolved['phylum_contam_pct'].mean():.1f}%")
        print(f"    Mean V5 contamination: {high_cont_mags['v5_contamination'].mean():.2f}%")
        print(f"    Mean CheckM2 contamination: {high_cont_mags['checkm2_contamination'].mean():.2f}%")

# MAGs where CheckM2 downgrades due to LOW COMPLETENESS (< 90%)
low_comp_mags = df[df['checkm2_completeness'] < 90]
print(f"\n  CheckM2 comp < 90% group: {len(low_comp_mags)} MAGs")
if len(low_comp_mags) > 0:
    lc_resolved = low_comp_mags[low_comp_mags['phylum_dominant'] != 'unresolved']
    if len(lc_resolved) > 0:
        print(f"    Phylum-resolved: {len(lc_resolved)}")
        print(f"    Kraken2 phylum contam > 5%: {(lc_resolved['phylum_contam_pct'] > 5).sum()}")
        print(f"    Kraken2 phylum contam <= 5%: {(lc_resolved['phylum_contam_pct'] <= 5).sum()}")
        print(f"    Mean Kraken2 phylum contam: {lc_resolved['phylum_contam_pct'].mean():.1f}%")
        print(f"    Mean V5 completeness: {low_comp_mags['v5_completeness'].mean():.1f}%")
        print(f"    Mean CheckM2 completeness: {low_comp_mags['checkm2_completeness'].mean():.1f}%")

# For ALL 93 MAGs - what does Kraken2 say about contamination?
print(f"\n  --- All {len(df)} MAGs (V5 says ALL are HQ, i.e., contam <= 5%) ---")
phylum_resolved_all = df[df['phylum_dominant'] != 'unresolved']
k2_contam_any = (phylum_resolved_all['phylum_contam_pct'] > 0).sum()
k2_contam_gt1 = (phylum_resolved_all['phylum_contam_pct'] > 1).sum()
k2_contam_gt5 = (phylum_resolved_all['phylum_contam_pct'] > 5).sum()
k2_contam_gt10 = (phylum_resolved_all['phylum_contam_pct'] > 10).sum()
k2_clean_le5 = (phylum_resolved_all['phylum_contam_pct'] <= 5).sum()
print(f"  Phylum-resolved: {len(phylum_resolved_all)}")
print(f"  Any phylum contamination (> 0%): {k2_contam_any}")
print(f"  Phylum contamination > 1%: {k2_contam_gt1}")
print(f"  Phylum contamination > 5%: {k2_contam_gt5}")
print(f"  Phylum contamination > 10%: {k2_contam_gt10}")
print(f"  Taxonomically clean (<= 5%): {k2_clean_le5}")

# Multi-phylum MAGs (definitive contamination signal)
n_multi_phylum = (phylum_resolved_all['phylum_n_taxa'] >= 2).sum()
n_three_plus_phylum = (phylum_resolved_all['phylum_n_taxa'] >= 3).sum()
print(f"\n  Multi-phylum MAGs (>= 2 phyla): {n_multi_phylum}")
print(f"  3+ phylum MAGs: {n_three_plus_phylum}")

# ============================================================================
# Step 8: Detailed categorization
# ============================================================================
print()
print("=" * 70)
print("Step 8: Detailed categorization of 93 MAGs")
print("=" * 70)

# Since V5 says ALL have contamination <= 5%, the categorization is different
# from script 61. Focus on whether Kraken2 agrees with V5 (clean) or CheckM2.

# Category A: Kraken2 agrees with V5 - MAG is clean (phylum contam <= 5%)
cat_a = df[(df['phylum_dominant'] != 'unresolved') & (df['phylum_contam_pct'] <= 5)]
print(f"\n  Category A: Kraken2 agrees with V5 (clean, phylum contam <= 5%): {len(cat_a)} MAGs")
if len(cat_a) > 0:
    print(f"    Mean V5 contamination: {cat_a['v5_contamination'].mean():.2f}%")
    print(f"    Mean CheckM2 contamination: {cat_a['checkm2_contamination'].mean():.2f}%")
    print(f"    Mean Kraken2 phylum contamination: {cat_a['phylum_contam_pct'].mean():.2f}%")
    print(f"    Mean V5 completeness: {cat_a['v5_completeness'].mean():.1f}%")
    print(f"    Mean CheckM2 completeness: {cat_a['checkm2_completeness'].mean():.1f}%")

# Category B: Kraken2 disagrees with V5 - MAG has contamination (phylum contam > 5%)
cat_b = df[(df['phylum_dominant'] != 'unresolved') & (df['phylum_contam_pct'] > 5)]
print(f"\n  Category B: Kraken2 finds contamination V5 missed (phylum contam > 5%): {len(cat_b)} MAGs")
if len(cat_b) > 0:
    print(f"    Mean V5 contamination: {cat_b['v5_contamination'].mean():.2f}%")
    print(f"    Mean CheckM2 contamination: {cat_b['checkm2_contamination'].mean():.2f}%")
    print(f"    Mean Kraken2 phylum contamination: {cat_b['phylum_contam_pct'].mean():.1f}%")
    print(f"    Mean V5 completeness: {cat_b['v5_completeness'].mean():.1f}%")
    print(f"    Mean CheckM2 completeness: {cat_b['checkm2_completeness'].mean():.1f}%")
    # Among these, how many does CheckM2 flag for contamination?
    ck2_also_flags = (cat_b['checkm2_contamination'] > 5).sum()
    print(f"    CheckM2 also flags contamination > 5%: {ck2_also_flags}")

# Category C: Not resolved at phylum level
cat_c = df[df['phylum_dominant'] == 'unresolved']
print(f"\n  Category C: Not resolved at phylum level: {len(cat_c)} MAGs")

# ============================================================================
# Step 9: Detailed examples
# ============================================================================
print()
print("=" * 70)
print("Step 9: Detailed Examples")
print("=" * 70)

# Top contaminated MAGs by Kraken2 phylum contamination
phylum_resolved = df[df['phylum_dominant'] != 'unresolved'].copy()
print(f"\n  --- Top 20 MAGs by Kraken2 phylum-level contamination ---")
top_contam = phylum_resolved.nlargest(20, 'phylum_contam_pct')
print(f"  {'Accession':<20} {'V5_cont':>7} {'CkM2_cont':>9} {'K2_phy_cont':>11} {'#Phy':>4} {'Cls%':>5} {'Dominant Phylum':<25} {'Phylum Breakdown'}")
for _, r in top_contam.iterrows():
    phyl_breakdown = r.get('phylum_breakdown', '')
    if len(str(phyl_breakdown)) > 60:
        phyl_breakdown = str(phyl_breakdown)[:58] + '..'
    dom_phyl = r['phylum_dominant']
    if len(dom_phyl) > 23:
        dom_phyl = dom_phyl[:21] + '..'
    print(f"  {r['accession']:<20} {r['v5_contamination']:>6.1f}% {r['checkm2_contamination']:>8.1f}% "
          f"{r['phylum_contam_pct']:>10.1f}% {r['phylum_n_taxa']:>4} {r['classified_pct']:>4.0f}% "
          f"{dom_phyl:<25} {phyl_breakdown}")

# Cleanest MAGs by Kraken2
print(f"\n  --- 15 'Cleanest' MAGs (lowest phylum contamination) ---")
cleanest = phylum_resolved.nsmallest(15, 'phylum_contam_pct')
print(f"  {'Accession':<20} {'V5_cont':>7} {'CkM2_cont':>9} {'V5_comp':>7} {'CkM2_comp':>9} {'K2_phy_cont':>11} {'#Phy':>4} {'Cls%':>5} {'Dominant Phylum':<25}")
for _, r in cleanest.iterrows():
    dom_phyl = r['phylum_dominant']
    if len(dom_phyl) > 23:
        dom_phyl = dom_phyl[:21] + '..'
    print(f"  {r['accession']:<20} {r['v5_contamination']:>6.1f}% {r['checkm2_contamination']:>8.1f}% "
          f"{r['v5_completeness']:>6.1f}% {r['checkm2_completeness']:>8.1f}% "
          f"{r['phylum_contam_pct']:>10.1f}% {r['phylum_n_taxa']:>4} {r['classified_pct']:>4.0f}% "
          f"{dom_phyl:<25}")

# ============================================================================
# Step 10: Comparison with the 141-MAG analysis
# ============================================================================
print()
print("=" * 70)
print("Step 10: Comparison with 141-MAG analysis (script 61)")
print("=" * 70)

prev_summary_file = RESULTS_DIR / 'kraken2_summary.json'
if prev_summary_file.exists():
    with open(prev_summary_file) as f:
        prev_summary = json.load(f)
    print(f"\n  {'Metric':<45} {'141 MAGs (CkM2-HQ)':>18} {'93 MAGs (V5-HQ)':>18}")
    print(f"  {'':45} {'(V5 downgrades)':>18} {'(CkM2 downgrades)':>18}")
    print(f"  {'-'*83}")

    prev_phy = prev_summary['phylum_level_contamination']
    curr_phy_resolved = df[df['phylum_dominant'] != 'unresolved']

    print(f"  {'MAGs analyzed':<45} {prev_summary['n_mags_analyzed']:>18} {len(df):>18}")
    print(f"  {'Mean classified %':<45} {prev_summary['classification_stats']['mean_classified_pct']:>17.1f}% {df['classified_pct'].mean():>17.1f}%")
    print(f"  {'Phylum-resolved':<45} {prev_phy['n_resolved']:>18} {len(curr_phy_resolved):>18}")
    print(f"  {'Phylum contam > 5%':<45} {prev_phy['n_contaminated_gt5']:>18} {(curr_phy_resolved['phylum_contam_pct'] > 5).sum():>18}")
    print(f"  {'Phylum contam <= 5% (clean)':<45} {prev_phy['n_clean_le5']:>18} {(curr_phy_resolved['phylum_contam_pct'] <= 5).sum():>18}")
    print(f"  {'Mean phylum contam %':<45} {prev_phy['mean_contamination_pct']:>17.1f}% {curr_phy_resolved['phylum_contam_pct'].mean():>17.1f}%")
    print(f"  {'Median phylum contam %':<45} {prev_phy['median_contamination_pct']:>17.1f}% {curr_phy_resolved['phylum_contam_pct'].median():>17.1f}%")
    print(f"  {'Multi-phylum MAGs':<45} {prev_phy['n_multi_phylum']:>18} {(curr_phy_resolved['phylum_n_taxa'] >= 2).sum():>18}")

    prev_genus = prev_summary['genus_level_contamination']
    curr_genus_resolved = df[df['genus_dominant'] != 'unresolved']
    print(f"  {'Genus-resolved':<45} {prev_genus['n_resolved']:>18} {len(curr_genus_resolved):>18}")
    print(f"  {'Genus contam > 5%':<45} {prev_genus['n_contaminated_gt5']:>18} {(curr_genus_resolved['genus_contam_pct'] > 5).sum():>18}")
    print(f"  {'Mean genus contam %':<45} {prev_genus['mean_contamination_pct']:>17.1f}% {curr_genus_resolved['genus_contam_pct'].mean():>17.1f}%")
else:
    print("  Previous 141-MAG summary not found, skipping comparison.")

# ============================================================================
# Step 11: Save results
# ============================================================================
print()
print("=" * 70)
print("Step 11: Saving results")
print("=" * 70)

# Save per-MAG detailed results
output_cols = [
    'accession',
    'v5_completeness', 'v5_contamination', 'v5_mimag',
    'checkm2_completeness', 'checkm2_contamination', 'checkm2_mimag',
    'total_bp', 'classified_bp', 'unclassified_bp',
    'classified_pct', 'unclassified_pct',
    'n_contigs', 'n_classified',
    'phylum_dominant', 'phylum_dominant_pct', 'phylum_contam_pct', 'phylum_n_taxa', 'phylum_breakdown',
    'class_dominant', 'class_dominant_pct', 'class_contam_pct', 'class_n_taxa',
    'order_dominant', 'order_dominant_pct', 'order_contam_pct', 'order_n_taxa',
    'family_dominant', 'family_dominant_pct', 'family_contam_pct', 'family_n_taxa',
    'genus_dominant', 'genus_dominant_pct', 'genus_contam_pct', 'genus_n_taxa', 'genus_breakdown',
    'species_dominant', 'species_dominant_pct', 'species_contam_pct', 'species_n_taxa', 'species_breakdown',
]
# Only include columns that exist
output_cols = [c for c in output_cols if c in df.columns]
df[output_cols].to_csv(RESULTS_DIR / 'kraken2_v5hq_analysis.tsv', sep='\t', index=False)
print(f"  Saved: {RESULTS_DIR / 'kraken2_v5hq_analysis.tsv'}")

# Save summary JSON
phylum_resolved = df[df['phylum_dominant'] != 'unresolved']
genus_resolved = df[df['genus_dominant'] != 'unresolved']

summary = {
    'description': '93 MAGs classified as HQ by V5 but NOT by CheckM2',
    'context': 'V5 says HQ (comp>=90%, cont<=5%), CheckM2 downgrades',
    'kraken2_database': 'k2_standard_08gb_20240605 (pre-built 8 GB standard database)',
    'n_mags_analyzed': len(df),
    'checkm2_downgrade_reasons': {
        'low_completeness_lt90': int((df['checkm2_completeness'] < 90).sum()),
        'high_contamination_gt5': int((df['checkm2_contamination'] > 5).sum()),
        'both': int(((df['checkm2_completeness'] < 90) & (df['checkm2_contamination'] > 5)).sum()),
    },
    'classification_stats': {
        'mean_classified_pct': round(float(df['classified_pct'].mean()), 2),
        'median_classified_pct': round(float(df['classified_pct'].median()), 2),
    },
    'phylum_level_contamination': {
        'n_resolved': int(len(phylum_resolved)),
        'n_contaminated_gt0': int((phylum_resolved['phylum_contam_pct'] > 0).sum()),
        'n_contaminated_gt1': int((phylum_resolved['phylum_contam_pct'] > 1).sum()),
        'n_contaminated_gt5': int((phylum_resolved['phylum_contam_pct'] > 5).sum()),
        'n_contaminated_gt10': int((phylum_resolved['phylum_contam_pct'] > 10).sum()),
        'n_clean_le5': int((phylum_resolved['phylum_contam_pct'] <= 5).sum()),
        'mean_contamination_pct': round(float(phylum_resolved['phylum_contam_pct'].mean()), 2),
        'median_contamination_pct': round(float(phylum_resolved['phylum_contam_pct'].median()), 2),
        'n_multi_phylum': int((phylum_resolved['phylum_n_taxa'] >= 2).sum()),
    },
    'genus_level_contamination': {
        'n_resolved': int(len(genus_resolved)),
        'n_contaminated_gt5': int((genus_resolved['genus_contam_pct'] > 5).sum()),
        'n_clean_le5': int((genus_resolved['genus_contam_pct'] <= 5).sum()),
        'mean_contamination_pct': round(float(genus_resolved['genus_contam_pct'].mean()), 2),
        'median_contamination_pct': round(float(genus_resolved['genus_contam_pct'].median()), 2),
    },
    'categories': {
        'cat_a_kraken2_agrees_v5_clean': int(len(cat_a)),
        'cat_b_kraken2_finds_contamination': int(len(cat_b)),
        'cat_c_unresolved': int(len(cat_c)),
    },
    'correlations': {},
}

# Add correlations if available
if not np.isnan(r_v5_phylum):
    summary['correlations']['kraken2_phylum_vs_v5_cont'] = {
        'pearson_r': round(float(r_v5_phylum), 4),
        'pearson_p': float(p_v5_phylum),
        'spearman_rho': round(float(sr_v5_phylum), 4),
        'spearman_p': float(sp_v5_phylum),
    }
if not np.isnan(r_ck_phylum):
    summary['correlations']['kraken2_phylum_vs_checkm2_cont'] = {
        'pearson_r': round(float(r_ck_phylum), 4),
        'pearson_p': float(p_ck_phylum),
        'spearman_rho': round(float(sr_ck_phylum), 4),
        'spearman_p': float(sp_ck_phylum),
    }
if not np.isnan(r_v5_genus):
    summary['correlations']['kraken2_genus_vs_v5_cont'] = {
        'pearson_r': round(float(r_v5_genus), 4),
        'pearson_p': float(p_v5_genus),
        'spearman_rho': round(float(sr_v5_genus), 4),
        'spearman_p': float(sp_v5_genus),
    }
if not np.isnan(r_ck_genus):
    summary['correlations']['kraken2_genus_vs_checkm2_cont'] = {
        'pearson_r': round(float(r_ck_genus), 4),
        'pearson_p': float(p_ck_genus),
        'spearman_rho': round(float(sr_ck_genus), 4),
        'spearman_p': float(sp_ck_genus),
    }

with open(RESULTS_DIR / 'kraken2_v5hq_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {RESULTS_DIR / 'kraken2_v5hq_summary.json'}")

# ============================================================================
# Final Summary
# ============================================================================
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

n_clean = int((phylum_resolved['phylum_contam_pct'] <= 5).sum())
n_contam = int((phylum_resolved['phylum_contam_pct'] > 5).sum())
n_resolved_total = len(phylum_resolved)

print(f"""
  Context: {len(df)} MAGs classified as HQ by V5 but NOT by CheckM2
  V5 says: completeness >= 90%, contamination <= 5% (all HQ)
  CheckM2 says: NOT HQ (mainly due to lower completeness estimates)
  Kraken2 database: k2_standard_08gb_20240605 (8 GB pre-built)

  Classification:
    Mean classified fraction: {df['classified_pct'].mean():.1f}%
    Median classified fraction: {df['classified_pct'].median():.1f}%

  PHYLUM-LEVEL contamination (cross-phylum = definitive contamination):
    Resolved: {n_resolved_total}/{len(df)} MAGs
    Clean (<= 5% phylum contam): {n_clean}/{n_resolved_total} ({n_clean/n_resolved_total*100:.1f}%)
    Contaminated (> 5% phylum contam): {n_contam}/{n_resolved_total} ({n_contam/n_resolved_total*100:.1f}%)
    Multi-phylum (>= 2 phyla): {(phylum_resolved['phylum_n_taxa'] >= 2).sum()}
    Mean contamination: {phylum_resolved['phylum_contam_pct'].mean():.1f}%
    Median contamination: {phylum_resolved['phylum_contam_pct'].median():.1f}%

  GENUS-LEVEL contamination:
    Resolved: {len(genus_resolved)}/{len(df)} MAGs
    Contaminated (> 5%): {(genus_resolved['genus_contam_pct'] > 5).sum()} / {len(genus_resolved)}
    Clean (<= 5%): {(genus_resolved['genus_contam_pct'] <= 5).sum()} / {len(genus_resolved)}

  V5 vs Kraken2 agreement:
    V5 says ALL {len(df)} MAGs have contamination <= 5%
    Kraken2 confirms (phylum contam <= 5%): {n_clean}
    Kraken2 disagrees (phylum contam > 5%): {n_contam}

  INTERPRETATION:
    If most MAGs are taxonomically clean by Kraken2 -> V5 is correct,
    CheckM2 underestimates completeness for these MAGs.
    If many show Kraken2 contamination -> V5 may miss some contamination.
""")

print("Done!")
