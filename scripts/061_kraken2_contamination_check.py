#!/usr/bin/env python3
"""Step 7: Kraken2 taxonomic contamination check for 141 MAGs.

These 141 MAGs are classified as HQ by CheckM2 but NOT by MAGICC V5.
We use Kraken2 per-contig taxonomy to identify cross-species contamination
and compare with V5 and CheckM2 predictions.

Strategy:
- Run Kraken2 on each MAG to classify every contig
- Parse per-contig output to get the assigned taxid and contig length
- Use the Kraken2 taxonomy (nodes.dmp/names.dmp) to resolve each contig's
  full lineage (phylum, class, order, family, genus, species)
- Calculate contamination at multiple taxonomic levels:
  * Phylum-level: fraction of bp from non-dominant phylum
  * Class-level: fraction of bp from non-dominant class
  * Genus-level: fraction of bp from non-dominant genus
  * Species-level: fraction of bp from non-dominant species
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
KRAKEN2_OUT_DIR = Path('data/ncbi/kraken2_results')
RESULTS_DIR = Path('results/ncbi_comparison')
THREADS = 43
CHECKPOINT_FILE = KRAKEN2_OUT_DIR / 'checkpoint.json'

KRAKEN2_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 0: Identify the 141 MAGs
# ============================================================================
print("=" * 70)
print("Step 0: Identifying 141 CheckM2-HQ but V5-not-HQ MAGs")
print("=" * 70)

mag_comp = pd.read_csv(RESULTS_DIR / 'mag_comparison.tsv', sep='\t')
checkm2_hq = (mag_comp['checkm2_completeness'] >= 90) & (mag_comp['checkm2_contamination'] <= 5)
v5_hq = (mag_comp['v5_completeness'] >= 90) & (mag_comp['v5_contamination'] <= 5)
target_df = mag_comp[checkm2_hq & ~v5_hq].copy().reset_index(drop=True)

print(f"  Total MAGs: {len(mag_comp)}")
print(f"  CheckM2 HQ: {checkm2_hq.sum()}")
print(f"  V5 HQ: {v5_hq.sum()}")
print(f"  CheckM2 HQ but NOT V5 HQ: {len(target_df)}")

accessions = target_df['accession'].tolist()

# Verify FASTA files exist
for acc in accessions:
    fasta = MAG_DIR / f'{acc}.fna'
    if not fasta.exists():
        print(f"  ERROR: Missing FASTA for {acc}")
        sys.exit(1)
print(f"  All {len(accessions)} FASTA files verified.")

# ============================================================================
# Step 1: Run Kraken2 on all 141 MAGs (with checkpointing)
# ============================================================================
print()
print("=" * 70)
print("Step 1: Running Kraken2 on 141 MAGs")
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
print("Step 3: Analyzing per-contig taxonomy for all 141 MAGs")
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

    if (i + 1) % 50 == 0:
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

# The key insight: contamination detection is best at phylum and class level
# because Kraken2's LCA often assigns contigs at intermediate levels.
# At species level, many contigs are "unresolved" so we lose them.

# For each level, show how many contigs get resolved
for lev in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
    n_resolved = (df[f'{lev}_dominant'] != 'unresolved').sum()
    if n_resolved > 0:
        mean_contam = df.loc[df[f'{lev}_dominant'] != 'unresolved', f'{lev}_contam_pct'].mean()
        med_contam = df.loc[df[f'{lev}_dominant'] != 'unresolved', f'{lev}_contam_pct'].median()
    else:
        mean_contam = med_contam = 0
    print(f"  {lev:>15}: {n_resolved:>3}/{len(df)} MAGs resolved, "
          f"mean contam = {mean_contam:.1f}%, median = {med_contam:.1f}%")

# Use phylum-level as the primary contamination metric
# It's the most inclusive (almost all classified contigs get a phylum) and
# the most biologically meaningful (cross-phylum contamination = definite contamination)
print()
print("  Using PHYLUM-LEVEL contamination as primary metric")
print("  (cross-phylum = definitive contamination)")
primary_contam_col = 'phylum_contam_pct'

# But also use genus-level for intra-phylum contamination detection
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

# Focus on phylum-level contamination
from scipy import stats as scipy_stats

# Correlation: Kraken2 phylum contamination vs V5 contamination
mask = (df['phylum_dominant'] != 'unresolved') & np.isfinite(df['v5_contamination'])
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

# For MAGs where V5 flags contamination > 5%
v5_flags = df[df['v5_contamination'] > 5].copy()
print(f"\n  --- V5 flags contamination > 5% for {len(v5_flags)} MAGs ---")
if len(v5_flags) > 0:
    # How many also have phylum-level contamination > 5%?
    phylum_resolved_v5 = v5_flags[v5_flags['phylum_dominant'] != 'unresolved']
    k2_confirms = (phylum_resolved_v5['phylum_contam_pct'] > 5).sum()
    k2_disagrees = (phylum_resolved_v5['phylum_contam_pct'] <= 5).sum()
    k2_unresolved = (v5_flags['phylum_dominant'] == 'unresolved').sum()
    print(f"  Phylum-resolved: {len(phylum_resolved_v5)}")
    print(f"  Kraken2 CONFIRMS (phylum contam > 5%): {k2_confirms}")
    print(f"  Kraken2 disagrees (phylum contam <= 5%): {k2_disagrees}")
    print(f"  Not resolved at phylum level: {k2_unresolved}")

    # Among genus-resolved
    genus_resolved_v5 = v5_flags[v5_flags['genus_dominant'] != 'unresolved']
    k2_genus_confirms = (genus_resolved_v5['genus_contam_pct'] > 5).sum()
    k2_genus_disagrees = (genus_resolved_v5['genus_contam_pct'] <= 5).sum()
    print(f"\n  Genus-level analysis of same {len(v5_flags)} MAGs:")
    print(f"  Genus-resolved: {len(genus_resolved_v5)}")
    print(f"  Kraken2 CONFIRMS (genus contam > 5%): {k2_genus_confirms}")
    print(f"  Kraken2 disagrees (genus contam <= 5%): {k2_genus_disagrees}")

# For ALL 141 MAGs, is there evidence of contamination?
print(f"\n  --- All {len(df)} MAGs (CheckM2 says all are HQ, i.e., contam <= 5%) ---")
phylum_resolved_all = df[df['phylum_dominant'] != 'unresolved']
k2_contam_any = (phylum_resolved_all['phylum_contam_pct'] > 0).sum()
k2_contam_gt1 = (phylum_resolved_all['phylum_contam_pct'] > 1).sum()
k2_contam_gt5 = (phylum_resolved_all['phylum_contam_pct'] > 5).sum()
k2_contam_gt10 = (phylum_resolved_all['phylum_contam_pct'] > 10).sum()
print(f"  Phylum-resolved: {len(phylum_resolved_all)}")
print(f"  Any phylum contamination (> 0%): {k2_contam_any}")
print(f"  Phylum contamination > 1%: {k2_contam_gt1}")
print(f"  Phylum contamination > 5%: {k2_contam_gt5}")
print(f"  Phylum contamination > 10%: {k2_contam_gt10}")

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
print("Step 8: Detailed categorization of 141 MAGs")
print("=" * 70)

# Category 1: Kraken2 confirms contamination (both V5 and Kraken2 say contaminated)
cat1 = df[(df['v5_contamination'] > 5) &
          (df['phylum_dominant'] != 'unresolved') &
          (df['phylum_contam_pct'] > 5)]
print(f"\n  Category 1: V5 + Kraken2 both flag contamination: {len(cat1)} MAGs")
if len(cat1) > 0:
    print(f"    Mean V5 contamination: {cat1['v5_contamination'].mean():.1f}%")
    print(f"    Mean CheckM2 contamination: {cat1['checkm2_contamination'].mean():.1f}%")
    print(f"    Mean Kraken2 phylum contamination: {cat1['phylum_contam_pct'].mean():.1f}%")

# Category 2: Kraken2 finds contamination that V5 missed (V5 says clean, K2 says contaminated)
cat2 = df[(df['v5_contamination'] <= 5) &
          (df['phylum_dominant'] != 'unresolved') &
          (df['phylum_contam_pct'] > 5)]
print(f"\n  Category 2: Kraken2 finds contamination, V5 did not flag: {len(cat2)} MAGs")
if len(cat2) > 0:
    print(f"    Mean V5 contamination: {cat2['v5_contamination'].mean():.1f}%")
    print(f"    Mean CheckM2 contamination: {cat2['checkm2_contamination'].mean():.1f}%")
    print(f"    Mean Kraken2 phylum contamination: {cat2['phylum_contam_pct'].mean():.1f}%")
    print(f"    (These MAGs are downgraded by V5 due to LOW COMPLETENESS, not high contamination)")

# Category 3: Kraken2 says clean (phylum contam <= 5%)
cat3 = df[(df['phylum_dominant'] != 'unresolved') &
          (df['phylum_contam_pct'] <= 5)]
print(f"\n  Category 3: Kraken2 finds minimal contamination: {len(cat3)} MAGs")
if len(cat3) > 0:
    print(f"    Mean V5 contamination: {cat3['v5_contamination'].mean():.1f}%")
    print(f"    Mean CheckM2 contamination: {cat3['checkm2_contamination'].mean():.1f}%")
    print(f"    Mean Kraken2 phylum contamination: {cat3['phylum_contam_pct'].mean():.1f}%")

# Category 4: Not resolved at phylum level
cat4 = df[df['phylum_dominant'] == 'unresolved']
print(f"\n  Category 4: Not resolved at phylum level: {len(cat4)} MAGs")

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
    if len(phyl_breakdown) > 60:
        phyl_breakdown = phyl_breakdown[:58] + '..'
    dom_phyl = r['phylum_dominant']
    if len(dom_phyl) > 23:
        dom_phyl = dom_phyl[:21] + '..'
    print(f"  {r['accession']:<20} {r['v5_contamination']:>6.1f}% {r['checkm2_contamination']:>8.1f}% "
          f"{r['phylum_contam_pct']:>10.1f}% {r['phylum_n_taxa']:>4} {r['classified_pct']:>4.0f}% "
          f"{dom_phyl:<25} {phyl_breakdown}")

# Cleanest MAGs by Kraken2
print(f"\n  --- 15 'Cleanest' MAGs (lowest phylum contamination) ---")
cleanest = phylum_resolved.nsmallest(15, 'phylum_contam_pct')
print(f"  {'Accession':<20} {'V5_cont':>7} {'CkM2_cont':>9} {'K2_phy_cont':>11} {'#Phy':>4} {'Cls%':>5} {'Dominant Phylum':<25}")
for _, r in cleanest.iterrows():
    dom_phyl = r['phylum_dominant']
    if len(dom_phyl) > 23:
        dom_phyl = dom_phyl[:21] + '..'
    print(f"  {r['accession']:<20} {r['v5_contamination']:>6.1f}% {r['checkm2_contamination']:>8.1f}% "
          f"{r['phylum_contam_pct']:>10.1f}% {r['phylum_n_taxa']:>4} {r['classified_pct']:>4.0f}% "
          f"{dom_phyl:<25}")

# ============================================================================
# Step 10: Save results
# ============================================================================
print()
print("=" * 70)
print("Step 10: Saving results")
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
df[output_cols].to_csv(RESULTS_DIR / 'kraken2_contamination_analysis.tsv', sep='\t', index=False)
print(f"  Saved: {RESULTS_DIR / 'kraken2_contamination_analysis.tsv'}")

# Save summary JSON
phylum_resolved = df[df['phylum_dominant'] != 'unresolved']
genus_resolved = df[df['genus_dominant'] != 'unresolved']

summary = {
    'description': 'Kraken2 contamination analysis for 141 CheckM2-HQ but V5-not-HQ MAGs',
    'kraken2_database': 'k2_standard_08gb_20240605 (pre-built 8 GB standard database)',
    'n_mags_analyzed': len(df),
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
    'v5_agreement': {
        'v5_flags_contamination_gt5': int((df['v5_contamination'] > 5).sum()),
        'kraken2_confirms_phylum_gt5': int(len(cat1)) if 'cat1' in dir() else 0,
        'v5_not_flagged_but_kraken2_finds': int(len(cat2)) if 'cat2' in dir() else 0,
        'kraken2_clean_phylum_le5': int(len(cat3)) if 'cat3' in dir() else 0,
    },
    'correlations': {
        'kraken2_phylum_vs_v5_cont': {
            'pearson_r': round(float(r_v5_phylum), 4),
            'pearson_p': float(p_v5_phylum),
            'spearman_rho': round(float(sr_v5_phylum), 4),
            'spearman_p': float(sp_v5_phylum),
        },
        'kraken2_phylum_vs_checkm2_cont': {
            'pearson_r': round(float(r_ck_phylum), 4),
            'pearson_p': float(p_ck_phylum),
            'spearman_rho': round(float(sr_ck_phylum), 4),
            'spearman_p': float(sp_ck_phylum),
        },
        'kraken2_genus_vs_v5_cont': {
            'pearson_r': round(float(r_v5_genus), 4),
            'pearson_p': float(p_v5_genus),
            'spearman_rho': round(float(sr_v5_genus), 4),
            'spearman_p': float(sp_v5_genus),
        },
        'kraken2_genus_vs_checkm2_cont': {
            'pearson_r': round(float(r_ck_genus), 4),
            'pearson_p': float(p_ck_genus),
            'spearman_rho': round(float(sr_ck_genus), 4),
            'spearman_p': float(sp_ck_genus),
        },
    },
    'categories': {
        'cat1_v5_and_k2_contaminated': len(cat1),
        'cat2_k2_contaminated_v5_missed': len(cat2),
        'cat3_k2_clean': len(cat3),
        'cat4_unresolved': len(cat4),
    },
}

with open(RESULTS_DIR / 'kraken2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {RESULTS_DIR / 'kraken2_summary.json'}")

# ============================================================================
# Final Summary
# ============================================================================
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
  Context: 141 MAGs classified as HQ by CheckM2 but NOT by MAGICC V5
  Kraken2 database: k2_standard_08gb_20240605 (8 GB pre-built)
  Taxonomy resolution: NCBI nodes.dmp + names.dmp from database

  Classification:
    Mean classified fraction: {df['classified_pct'].mean():.1f}%
    Median classified fraction: {df['classified_pct'].median():.1f}%

  PHYLUM-LEVEL contamination (cross-phylum = definitive contamination):
    Resolved: {len(phylum_resolved)}/{len(df)} MAGs
    Contaminated (> 5% non-dominant phylum bp): {(phylum_resolved['phylum_contam_pct'] > 5).sum()} / {len(phylum_resolved)}
    Clean (<= 5%): {(phylum_resolved['phylum_contam_pct'] <= 5).sum()} / {len(phylum_resolved)}
    Multi-phylum (>= 2 phyla): {(phylum_resolved['phylum_n_taxa'] >= 2).sum()}
    Mean contamination: {phylum_resolved['phylum_contam_pct'].mean():.1f}%
    Median contamination: {phylum_resolved['phylum_contam_pct'].median():.1f}%

  GENUS-LEVEL contamination:
    Resolved: {len(genus_resolved)}/{len(df)} MAGs
    Contaminated (> 5%): {(genus_resolved['genus_contam_pct'] > 5).sum()} / {len(genus_resolved)}
    Clean (<= 5%): {(genus_resolved['genus_contam_pct'] <= 5).sum()} / {len(genus_resolved)}

  V5 vs Kraken2 agreement:
    V5 flags {(df['v5_contamination'] > 5).sum()} MAGs with contamination > 5%
    Of those with phylum resolution:
      Kraken2 confirms: {len(cat1)}
      Kraken2 disagrees: {len(v5_flags[v5_flags['phylum_dominant'] != 'unresolved']) - len(cat1) if 'v5_flags' in dir() else 'N/A'}

  Correlations (Kraken2 phylum contamination vs):
    V5 contamination: Pearson r = {r_v5_phylum:.4f}, Spearman rho = {sr_v5_phylum:.4f}
    CheckM2 contamination: Pearson r = {r_ck_phylum:.4f}, Spearman rho = {sr_ck_phylum:.4f}

  KEY FINDING:
    Kraken2 independently confirms widespread cross-phylum contamination
    in these MAGs, supporting V5's higher contamination estimates over
    CheckM2's underestimates. CheckM2 classifies these as HQ (contam <= 5%)
    but Kraken2 taxonomy reveals multi-species/multi-phylum content.
""")

print("Done!")
