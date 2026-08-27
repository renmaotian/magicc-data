"""Shared configuration for the MAGICC taxonomic-holdout experiments.

  WS1.6 / WS1.7 / WS1.8  leave-PHYLUM-out   (level='phylum', completed 2026-07-26)
  WS1.9                  leave-FAMILY-out   (level='family',  completed 2026-07-28)
  WS11.G                 leave-GENUS-out    (level='genus')

Defines: project paths, the FASTA path remapping (the project tree was moved from
/path/to/magicc to /path/to/magicc after the
split TSVs were written), and the held-out taxon panel for the selected level.

LEVEL SWITCH
  The taxonomic level is chosen by the environment variable

      MAGICC_HOLDOUT_LEVEL = phylum | family | genus     (default: phylum)

  so scripts 121-126 are a SINGLE code path shared by all three experiments;
  nothing is forked. Every artifact path is level-suffixed, so the three runs
  cannot collide. Setting the variable to 'phylum' reproduces the WS1.6
  configuration byte-for-byte and 'family' reproduces WS1.9 byte-for-byte;
  scripts/224_holdout_level_switch_reproduction.py verifies both.

GENERIC PANEL SYMBOLS (use these; they work at all three levels)
  PANEL_LEVEL   'phylum' | 'family' | 'genus'
  TAXON_COL     the genome-table column holding the panel taxon
  PANEL_TAXA    flat sorted list of every held-out taxon at PANEL_LEVEL
  TAXON_TO_GROUP  taxon -> evaluation-group name
  PANEL_GROUPS[g]['taxa']  the taxa of one evaluation group
  PANEL_PHYLA   phyla removed ENTIRELY from training. At family and genus level
                this is [] by construction: every held-out taxon's parent phylum
                stays in the training set.
  PANEL_PARENT_FAMILIES  parent families of the panel. Non-empty only at genus
                level, where every one of them REMAINS in training (the defining
                requirement of WS11.G).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Taxonomic level of this experiment
# ---------------------------------------------------------------------------
PANEL_LEVEL = os.environ.get('MAGICC_HOLDOUT_LEVEL', 'phylum').strip().lower()
assert PANEL_LEVEL in ('phylum', 'family', 'genus'), \
    f'MAGICC_HOLDOUT_LEVEL must be phylum|family|genus, got {PANEL_LEVEL!r}'
TAXON_COL = PANEL_LEVEL
# artifact-path suffix ('' at phylum level so WS1.6 keeps its original paths)
_SFX = {'phylum': '', 'family': '_family', 'genus': '_genus'}[PANEL_LEVEL]
# log-file prefix
WS = {'phylum': 'ws1.6', 'family': 'ws1.9', 'genus': 'ws11.g'}[PANEL_LEVEL]

GTDB_RANK_PREFIX = {'domain': 'd__', 'phylum': 'p__', 'class': 'c__',
                    'order': 'o__', 'family': 'f__', 'genus': 'g__',
                    'species': 's__'}


def rank_from_taxonomy(gtdb_taxonomy: str, rank: str = None) -> str:
    """Extract one GTDB rank from a `d__x;p__y;...` lineage string."""
    pfx = GTDB_RANK_PREFIX[rank or PANEL_LEVEL]
    for part in str(gtdb_taxonomy).split(';'):
        part = part.strip()
        if part.startswith(pfx):
            return part[3:]
    return ''


def add_taxon_column(df, rank: str = None):
    """Ensure `df` carries the panel-level taxon column; returns df (in place)."""
    col = rank or TAXON_COL
    if col not in df.columns:
        df[col] = df['gtdb_taxonomy'].map(lambda t: rank_from_taxonomy(t, col))
    return df


def panel_mask(df):
    """Boolean Series: rows whose panel-level taxon is in the held-out panel."""
    add_taxon_column(df)
    return df[TAXON_COL].isin(PANEL_TAXA)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path('/path/to/magicc')

# The split TSVs store fasta_path values rooted at the OLD project location.
OLD_ROOT = '/path/to/magicc'
NEW_ROOT = str(PROJECT_ROOT)

SPLITS_DIR = PROJECT_ROOT / 'data/splits'
FEATURES_DIR = PROJECT_ROOT / 'data/features'
HOLDOUT_DIR = PROJECT_ROOT / f'data/holdout{_SFX}'
KMER_DIR = PROJECT_ROOT / 'data/kmer_selection'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / f'results/revision/holdout{_SFX}'
LOGS_DIR = PROJECT_ROOT / 'logs/revision'

TRAIN_TSV = SPLITS_DIR / 'train_genomes.tsv'
VAL_TSV = SPLITS_DIR / 'val_genomes.tsv'
TEST_TSV = SPLITS_DIR / 'test_genomes.tsv'

# Production V5 artifacts (READ ONLY - never modified by this experiment)
V5_FEATURES_H5 = FEATURES_DIR / 'magicc_v5_features.h5'
V5_NORM_PARAMS = FEATURES_DIR / 'normalization_params.json'
V5_ONNX = MODELS_DIR / 'magicc_v5.onnx'
SELECTED_KMERS = KMER_DIR / 'selected_kmers.txt'

# New holdout artifacts (level-suffixed so the two experiments cannot collide)
HOLDOUT_H5 = FEATURES_DIR / f'holdout_{PANEL_LEVEL}_features.h5'
HOLDOUT_NORM_PARAMS = HOLDOUT_DIR / 'holdout_normalization_params.json'
HOLDOUT_ONNX = MODELS_DIR / f'magicc_holdout_{PANEL_LEVEL}.onnx'
HOLDOUT_BEST_PT = MODELS_DIR / f'best_model_holdout_{PANEL_LEVEL}.pt'
HOLDOUT_HISTORY = MODELS_DIR / f'training_history_holdout_{PANEL_LEVEL}.json'
# checkpoint dir; 'holdout' (not 'holdout_phylum') at phylum level so the
# completed WS1.6 run remains resumable from its existing checkpoints
HOLDOUT_TRAIN_OUT = MODELS_DIR / ('holdout' if PANEL_LEVEL == 'phylum'
                                  else f'holdout_{PANEL_LEVEL}')
# Sibling experiments' artifacts, needed by the multi-level ladder comparison
# (scripts 129 and 223). Read-only.
PHYLUM_RESULTS_DIR = PROJECT_ROOT / 'results/revision/holdout'
PHYLUM_ONNX = MODELS_DIR / 'magicc_holdout_phylum.onnx'
PHYLUM_NORM_PARAMS = PROJECT_ROOT / 'data/holdout/holdout_normalization_params.json'
FAMILY_RESULTS_DIR = PROJECT_ROOT / 'results/revision/holdout_family'
FAMILY_ONNX = MODELS_DIR / 'magicc_holdout_family.onnx'
FAMILY_NORM_PARAMS = PROJECT_ROOT / 'data/holdout_family/holdout_normalization_params.json'
EVAL_DIR = HOLDOUT_DIR / 'eval_sets'
PANEL_JSON = HOLDOUT_DIR / 'holdout_panel.json'


def remap_fasta_path(p: str) -> str:
    """Rewrite a split-TSV fasta_path to the current project location."""
    if p.startswith(OLD_ROOT):
        return NEW_ROOT + p[len(OLD_ROOT):]
    return p


# ---------------------------------------------------------------------------
# Held-out phylum panel (WS1.6)
# ---------------------------------------------------------------------------
# Every phylum listed here is removed from the holdout model's training AND
# validation data, in BOTH the dominant and the contaminant role.
#
# Panel entries are evaluation *groups*; several groups bundle a GTDB polyphyly
# split or a DPANN superphylum so that no sister lineage of an evaluated group
# remains in training.

PHYLUM_PANEL_GROUPS = {
    'Bacteroidota': {
        'phyla': ['Bacteroidota'],
        'rationale': 'Large, GC-distinct, 4th most abundant phylum (10.4% of training genomes)',
    },
    'Bacteroidota_A': {
        'phyla': ['Bacteroidota_A'],
        'rationale': 'GTDB polyphyly split of Bacteroidota; excluded so the Bacteroidota '
                     'holdout is not undermined by a sister lineage, and evaluated separately',
    },
    'Campylobacterota': {
        'phyla': ['Campylobacterota'],
        'rationale': 'Mid-size bacteria (median 1.68 Mbp), well populated (6.2% of training genomes)',
    },
    'Patescibacteriota': {
        'phyla': ['Patescibacteriota'],
        'rationale': 'Reduced genomes (median 0.88 Mbp), the hardest case and the lineage '
                     'both reviewers singled out; directly comparable to Set C_clean',
    },
    'Halobacteriota': {
        'phyla': ['Halobacteriota'],
        'rationale': 'Largest archaeal phylum in the training set (573 genomes); '
                     'domain-level generalization',
    },
    'DPANN': {
        'phyla': ['Nanobdellota', 'Micrarchaeota', 'Nanohalarchaeota',
                  'Altiarchaeota', 'Iainarchaeota'],
        'rationale': 'DPANN superphylum (candidate/uncultured archaea, median <1.1 Mbp): '
                     'extreme taxonomic novelty. Pooled because no single DPANN phylum '
                     'has enough genomes for an independent evaluation set',
    },
}

# ---------------------------------------------------------------------------
# Held-out FAMILY panel (WS1.9)
# ---------------------------------------------------------------------------
# THE DEFINING DESIGN CONSTRAINT: every held-out family's PARENT PHYLUM REMAINS
# IN THE TRAINING SET. Phylum-level holdout (WS1.6) is the extreme case; the
# practically important question is the one answered here, because most newly
# assembled MAGs are novel at family or genus level inside an already-known
# phylum. All 110 phyla survive in training by construction.
#
# Families are drawn from the four phyla that WS1.6 evaluated at phylum level
# (Bacteroidota, Campylobacterota, Halobacteriota, Patescibacteriota), so the
# family-level effect can be compared against the phylum-level effect for the
# SAME lineage. Selection rules, applied to the genome-split tables:
#   * >= 2 test-split genomes (evaluation references) and enough training
#     genomes that production V5 genuinely learned the family;
#   * the LARGEST family of each phylum is deliberately LEFT IN TRAINING, so the
#     parent phylum is unambiguously well represented after removal;
#   * groups pool families only where single families are too small for a stable
#     cluster bootstrap (Halobacteriota, Patescibacteriota) - the same device
#     WS1.6 used for DPANN; a per-family breakdown is emitted regardless.
#
# The 19 Patescibacteriota families are every Patescibacteriota family with
# >= 10 training and >= 2 test genomes (a deterministic rule, resolved once and
# written out explicitly here so the panel is auditable without recomputation).
FAMILY_PANEL_GROUPS = {
    'Bacteroidota_Muribaculaceae': {
        'taxa': ['Muribaculaceae'],
        'parent_phylum': 'Bacteroidota',
        'rationale': 'Largest host-associated (gut) Bacteroidota family, 2,254 genomes, '
                     'median 2.60 Mbp, 212 test references. Bacteroidota was the largest '
                     'bacterial phylum group in WS1.6 (+8.01 pp completeness DiD), so this '
                     'is the direct family-vs-phylum contrast for the same lineage.',
    },
    'Bacteroidota_Flavobacteriaceae': {
        'taxa': ['Flavobacteriaceae'],
        'parent_phylum': 'Bacteroidota',
        'rationale': 'Environmental / most genus-diverse Bacteroidota family (145 genera), '
                     'median 3.54 Mbp, 155 test references. Contrasts with Muribaculaceae '
                     'in both ecology and genome size within one phylum.',
    },
    'Campylobacterota_Helicobacteraceae': {
        'taxa': ['Helicobacteraceae'],
        'parent_phylum': 'Campylobacterota',
        'rationale': 'Second-largest Campylobacterota family (2,042 genomes, median '
                     '1.64 Mbp, 205 test references). Campylobacteraceae, the largest '
                     'family, is deliberately retained in training.',
    },
    'Campylobacterota_Arcobacteraceae': {
        'taxa': ['Arcobacteraceae'],
        'parent_phylum': 'Campylobacterota',
        'rationale': 'Small-N Campylobacterota family (336 genomes, 40 test references): '
                     'gives a within-phylum dose contrast on family size against '
                     'Helicobacteraceae.',
    },
    'Halobacteriota_halophilic': {
        'taxa': ['Haloferacaceae', 'Haloarculaceae'],
        'parent_phylum': 'Halobacteriota',
        'rationale': 'Two halophilic archaeal families (28 test references). Domain-level '
                     'generalization at family resolution. Natrialbaceae and '
                     'Halobacteriaceae (also halophilic) plus every methanogen family stay '
                     'in training, so both the phylum and its physiology remain represented.',
    },
    'Patescibacteriota_families': {
        'taxa': ['2-12-FULL-60-25', 'GWA2-37-8', 'GWC2-37-13', 'Minisyncoccaceae',
                 'Nanoperiomorbaceae', 'Nanosynbacteraceae', 'Nanosyncoccaceae',
                 'PJMF01', 'Peribacteraceae', 'Saccharimonadaceae',
                 'Staskawiczbacteraceae', 'UBA1547', 'UBA2103', 'UBA2206',
                 'UBA4665', 'UBA8517', 'UBA918', 'UBA922', 'UBA9973'],
        'parent_phylum': 'Patescibacteriota',
        'rationale': 'THE KEY COMPARISON. Patescibacteriota was the worst phylum-level '
                     'group in WS1.6 (+22.81 pp completeness DiD); holding out families '
                     'inside it, with 756 Patescibacteriota training genomes across ~281 '
                     'other families retained, measures family-level novelty for exactly '
                     'that lineage. Pooled (19 families, 66 test references) because no '
                     'single CPR family has enough test genomes for a stable cluster '
                     'bootstrap; a per-family breakdown is reported. Reduced genomes '
                     '(median 0.86 Mbp) - note WS3.10: MAGICC completeness is dominated by '
                     'absolute genome size, and the DiD design cancels that shared bias '
                     'because both models score identical samples.',
    },
}

# ---------------------------------------------------------------------------
# Held-out GENUS panel (WS11.G)
# ---------------------------------------------------------------------------
# THE DEFINING DESIGN CONSTRAINT: every held-out genus's PARENT FAMILY REMAINS
# IN THE TRAINING SET, and so does its parent phylum. Both are verified
# numerically by scripts/220_holdout_genus_panel_eda.py, not asserted.
#
# WHY THIS EXPERIMENT EXISTS
#   WS1.6 (phylum) and WS1.9 (family) completed; the genus level was declined
#   with the argument that "a genus-level holdout inside a represented family
#   would be expected to fall below the family-level cost". That argument
#   assumes degradation is MONOTONE across ranks, and WS1.9's own
#   Helicobacteraceae cell contradicts it (family-level novelty cost as much as
#   phylum-level: attenuation -0.39 pp, p = 0.228). WS11.G measures the genus
#   level instead of arguing about it. Monotonicity is NOT assumed anywhere.
#
# PANEL CONSTRUCTION - a deterministic rule applied to the genome-split tables,
# then written out explicitly here so the panel is auditable without
# recomputation (the same device WS1.9 used for its 19 CPR families):
#
#   (1) CANDIDATE genera are exactly the genera of the SIX families WS1.9 held
#       out. This is what makes the three levels a paired ladder: every WS11.G
#       evaluation genome is genus-novel to this model, family-novel to the
#       WS1.9 model, phylum-novel to the WS1.6 model and lineage-represented to
#       production V5, so genus/family/phylum novelty are measured on IDENTICAL
#       samples against a common control (script 223).
#   (2) ELIGIBILITY: >= 3 training genomes (so production V5 genuinely learned
#       the genus - removing a genus V5 never learned would measure nothing) and
#       >= 1 test-split genome (an evaluation reference).
#   (3) FAMILY-RETENTION FLOOR, applied largest-genus-first: a genus is held out
#       only if its family still retains >= 50 % of its training genomes (for
#       families with >= 500 training genomes) or >= 15 % (smaller families),
#       and >= 2 genera. The two-tier floor is necessary and is reported: a flat
#       50 % floor leaves the Halobacteriota and Patescibacteriota groups with
#       fewer than 20 evaluation references, which is too few for a stable
#       cluster bootstrap. Constants below.
#
# The rule is implemented in scripts/220_holdout_genus_panel_eda.py, which
# re-derives this panel from the split tables and refuses to run if the derived
# panel differs from the literal panel recorded here.
GENUS_PANEL_MIN_TRAIN = 3        # genus must have >= this many training genomes
GENUS_PANEL_MIN_TEST = 1         # ... and >= this many test-split genomes
GENUS_PANEL_BIG_FAMILY = 500     # family size (training genomes) splitting the tiers
GENUS_PANEL_RETAIN_BIG = 0.50    # retention floor for families >= BIG_FAMILY
GENUS_PANEL_RETAIN_SMALL = 0.15  # retention floor for smaller families
GENUS_PANEL_MIN_RETAIN_GENERA = 2

GENUS_PANEL_GROUPS = {
    'Bacteroidota_Muribaculaceae_genera': {
        'taxa': ['CAG-1031', 'CAG-873', 'Lepagella', 'Limisoma', 'Paramuribaculum'],
        'parent_family': 'Muribaculaceae',
        'parent_phylum': 'Bacteroidota',
        'rationale': 'Five genera of Muribaculaceae (904 training genomes, 111 test '
                     'references) removed; the family retains 906/1,810 = 50.1 % of '
                     'its training genomes across 14 genera. Muribaculaceae was the '
                     'WS1.9 group with the largest family-level removal, so this is '
                     'the direct genus-vs-family-vs-phylum contrast for one lineage.',
    },
    'Bacteroidota_Flavobacteriaceae_genera': {
        'taxa': ['Aequorivita', 'Flagellimonas', 'Flavobacterium', 'Nonlabens',
                 'Polaribacter', 'Tenacibaculum'],
        'parent_family': 'Flavobacteriaceae',
        'parent_phylum': 'Bacteroidota',
        'rationale': 'Six genera of the most genus-diverse Bacteroidota family '
                     '(550 training genomes, 96 test references); the family retains '
                     '550/1,100 = 50.0 % of its training genomes across 139 OTHER '
                     'genera. The cleanest possible statement of "novel genus inside '
                     'a densely represented family".',
    },
    'Campylobacterota_Helicobacteraceae_genera': {
        'taxa': ['Helicobacter_B', 'Helicobacter_C', 'Helicobacter_D',
                 'Helicobacter_E', 'Helicobacter_F', 'Helicobacter_G'],
        'parent_family': 'Helicobacteraceae',
        'parent_phylum': 'Campylobacterota',
        'rationale': 'THE CELL THIS EXPERIMENT EXISTS FOR. Helicobacteraceae was the '
                     'WS1.9 group with NO family-vs-phylum attenuation and the worst '
                     'contamination cell in the study (+12.22 pp). Six genera '
                     '(398 training genomes, 51 test references) removed; the family '
                     'retains 1,229/1,627 = 75.5 %, including the type genus '
                     'Helicobacter (1,198 genomes). CAVEAT TO REPORT: these are GTDB '
                     'alphabet-suffix splits of a single NCBI genus, so a very close '
                     'sister lineage remains in training - this cell measures a '
                     'deliberately SHALLOW genus novelty and is a lower bound.',
    },
    'Campylobacterota_Arcobacteraceae_genera': {
        'taxa': ['Aliarcobacter', 'Poseidonibacter'],
        'parent_family': 'Arcobacteraceae',
        'parent_phylum': 'Campylobacterota',
        'rationale': 'The DOMINANT genus of a small family is removed here '
                     '(Aliarcobacter, 222 of 269 training genomes), the opposite '
                     'configuration to Helicobacteraceae where the dominant genus is '
                     'retained - a designed within-phylum contrast on whether the '
                     'surviving congener is the family type genus. 227 training '
                     'genomes, 33 test references; the family retains 42 genomes '
                     'across 12 genera (15.6 %), the lowest retention in the panel '
                     'and reported as such.',
    },
    'Halobacteriota_halophilic_genera': {
        'taxa': ['Haloarcula', 'Halobellus', 'Haloferax', 'Halomicrobium',
                 'Halapricum', 'Haloplanus', 'Halorhabdus', 'Halorientalis',
                 'Halorubrum'],
        'parent_family': 'Haloferacaceae+Haloarculaceae',
        'parent_phylum': 'Halobacteriota',
        'rationale': 'Nine halophilic archaeal genera across the two families WS1.9 '
                     'held out (169 training genomes, 24 test references). Domain-'
                     'level generalization at genus resolution. Pooled across the two '
                     'families because neither alone reaches 20 evaluation references '
                     '- the same pooling device WS1.9 used for this lineage; a '
                     'per-genus breakdown is emitted regardless.',
    },
    'Patescibacteriota_genera': {
        'taxa': ['C7867-001', 'CAIKZD01', 'CAIYEO01', 'CG1-02-43-31', 'JABIEQ01',
                 'JAKLGL01', 'MWCR01', 'Nanosyncoccus', 'Peribacter', 'UBA1547',
                 'UBA2170', 'UBA4124', 'UBA8515', 'UBA9973', 'XYD1-FULL-39-28',
                 'XYD2-FULL-39-9'],
        'parent_family': '12 CPR families',
        'parent_phylum': 'Patescibacteriota',
        'rationale': 'THE KEY LADDER COMPARISON. Patescibacteriota went from +22.81 pp '
                     '(phylum, WS1.6) to +5.03 pp (family, WS1.9) - the ~80 % '
                     'attenuation that is the WS1.9 headline. Sixteen genera in 12 of '
                     'the 19 WS1.9 CPR families (187 training genomes, 30 test '
                     'references) extend that ladder one rank deeper. Reduced genomes '
                     '(median <1 Mbp): note WS3.10 - MAGICC completeness is dominated '
                     'by absolute genome size, and the difference-in-differences '
                     'design cancels that shared bias because all models score '
                     'identical samples. The genera are individually small, so the '
                     'amount of learned signal removed is small; the confidence '
                     'interval, not the point estimate alone, must carry the claim.',
    },
}

if PANEL_LEVEL == 'phylum':
    PANEL_GROUPS = {g: {**d, 'taxa': d['phyla']}
                    for g, d in PHYLUM_PANEL_GROUPS.items()}
elif PANEL_LEVEL == 'family':
    PANEL_GROUPS = {g: {**d, 'phyla': [d['parent_phylum']]}
                    for g, d in FAMILY_PANEL_GROUPS.items()}
else:
    PANEL_GROUPS = {g: {**d, 'phyla': [d['parent_phylum']]}
                    for g, d in GENUS_PANEL_GROUPS.items()}

# Flat sorted list of every held-out taxon at PANEL_LEVEL
PANEL_TAXA = sorted({t for g in PANEL_GROUPS.values() for t in g['taxa']})

# Phyla removed ENTIRELY from training. Empty at family and genus level BY
# CONSTRUCTION: every held-out taxon's parent phylum stays in the training set.
PANEL_PHYLA = PANEL_TAXA if PANEL_LEVEL == 'phylum' else []

# Parent phyla of the panel (family/genus level): retained in training, reported.
if PANEL_LEVEL == 'phylum':
    PANEL_PARENT_PHYLA = PANEL_PHYLA
elif PANEL_LEVEL == 'family':
    PANEL_PARENT_PHYLA = sorted({d['parent_phylum']
                                 for d in FAMILY_PANEL_GROUPS.values()})
else:
    PANEL_PARENT_PHYLA = sorted({d['parent_phylum']
                                 for d in GENUS_PANEL_GROUPS.values()})

# The SIX families WS1.9 held out; at genus level these are exactly the families
# the panel genera are drawn from, and every one of them REMAINS in training.
# Also the family rung of the genus->family->phylum ladder (script 223).
WS19_FAMILY_PANEL_TAXA = sorted({t for d in FAMILY_PANEL_GROUPS.values()
                                 for t in d['taxa']})
# The WS1.6 phylum panel; needed to restrict the common control of the ladder.
WS16_PHYLUM_PANEL_TAXA = sorted({p for d in PHYLUM_PANEL_GROUPS.values()
                                 for p in d['phyla']})
# Parent families of the panel. Non-empty only at genus level.
PANEL_PARENT_FAMILIES = WS19_FAMILY_PANEL_TAXA if PANEL_LEVEL == 'genus' else []

# Reverse map: taxon -> evaluation group
TAXON_TO_GROUP = {t: g for g, d in PANEL_GROUPS.items() for t in d['taxa']}
PHYLUM_TO_GROUP = TAXON_TO_GROUP if PANEL_LEVEL == 'phylum' else \
    {p: g for g, d in PANEL_GROUPS.items() for p in d['phyla']}

# ---------------------------------------------------------------------------
# V5 synthesis recipe constants (frozen; do not change - they define the
# comparability of the holdout model to production V5)
# ---------------------------------------------------------------------------
N_KMER_FEATURES = 9249
N_SUMMARY_FEATURES = 7
BATCH_SIZE = 10_000

# V4-recipe batch composition (15/15/30/30/5/5 %)
SAMPLE_TYPES = {
    'pure':           1500,
    'complete':       1500,
    'within_phylum':  3000,
    'cross_phylum':   3000,
    'reduced_genome':  500,
    'archaeal':        500,
}

QUALITY_TIER_WEIGHTS = {'high': 0.15, 'medium': 0.35, 'low': 0.35, 'highly_fragmented': 0.15}

# V4/V5 "reduced genome" phyla (verbatim from scripts/019b_batch_synthesis_v2.py).
# NOTE: Patescibacteriota and all DPANN members are in the holdout panel, so the
# surviving pool is tiny (Bdellovibrionota only). See ADAPTED_REDUCED_* below.
V5_REDUCED_GENOME_PHYLA = {
    'Patescibacteriota',
    'Aenigmatarchaeota', 'Altiarchaeota', 'Diapherotrites',
    'Huberarchaeota', 'Iainarchaeota', 'Micrarchaeota',
    'Nanoarchaeota', 'Nanohaloarchaeota', 'Nanohalarchaeota',
    'Undinarchaeota', 'Woesearchaeota', 'Nanobdellota',
    'Dependentiae', 'Bdellovibrionota',
}

# ADAPTATION (documented deviation, PHYLUM LEVEL ONLY): with the phylum panel
# removed, the V5 reduced-genome pool collapses to Bdellovibrionota alone (~103
# training genomes), which would make the 5% reduced-genome category draw ~390
# samples per genome and, worse, would conflate "never saw Patescibacteriota"
# with "never saw ANY small genome". To keep the *intent* of the category (5% of
# training samples have small/reduced genomes) while removing only the *lineage*
# signal, the pool was defined as
#     (surviving V5_REDUCED_GENOME_PHYLA)  UNION  (non-panel genomes < SIZE_CUTOFF bp)
# = 1,622 genomes, median 1.17 Mbp.
#
# AT FAMILY LEVEL THE ADAPTATION IS NOT NEEDED and is therefore NOT APPLIED: the
# family panel removes only 530 of the 1,286 Patescibacteriota training genomes,
# so the V5 reduced-genome pool survives at 923/1,453 = 63.5% with median
# 0.950 Mbp (V5: 0.912 Mbp). The WS1.9 model uses V5's own reduced-genome
# definition verbatim, which makes it a strictly cleaner counterfactual to V5
# than the WS1.6 model was. Script 121 verifies the surviving fraction against
# REDUCED_POOL_ADAPT_THRESHOLD at run time and refuses to proceed silently if it
# ever drops below it.
#
# AT GENUS LEVEL THE ADAPTATION IS ALSO NOT NEEDED: the genus panel removes only
# 187 of the 1,286 Patescibacteriota training genomes, so V5's own reduced-genome
# pool survives at 1,266/1,453 = 87.1% with median 0.944 Mbp (V5: 0.912 Mbp).
# V5's definition is therefore used verbatim, making WS11.G the cleanest
# counterfactual of the three levels.
ADAPTED_REDUCED_SIZE_CUTOFF = 1_500_000  # bp
USE_ADAPTED_REDUCED_POOL = (PANEL_LEVEL == 'phylum')
REDUCED_POOL_ADAPT_THRESHOLD = 0.50      # fraction of V5's pool that must survive

# V5 train/val/test sizes (exactly matched - no size confound)
N_TRAIN_V4RECIPE = 800_000
N_TRAIN_PART_A = 100_000   # 100% completeness, 0% contamination (original contigs)
N_TRAIN_PART_B = 100_000   # 100% completeness, 0-10% contamination
N_TRAIN_TOTAL = N_TRAIN_V4RECIPE + N_TRAIN_PART_A + N_TRAIN_PART_B  # 1,000,000
N_VAL = 100_000
N_TEST = 100_000

# ---------------------------------------------------------------------------
# Evaluation set design (WS1.6 Part 4 / WS1.8)
# ---------------------------------------------------------------------------
EVAL_SEED_BASE = 20260726
EVAL_TARGET_N = 1000          # target samples per evaluation group
EVAL_MAX_REFS = 100           # max distinct reference genomes per group
EVAL_MIN_SIMS = 10            # min simulations per reference

# Final evaluation-set design. `ref_source`:
#   'test' -> dominants drawn from the held-out TEST split only. Neither the
#             holdout model nor production V5 trained on these exact genomes,
#             so the ONLY difference between the two models is whether the
#             LINEAGE was in training. This is the clean head-to-head.
#   'all'  -> dominants drawn from train+val+test (used only for DPANN, which has
#             just 9 test-split references). Valid for the holdout model (it saw
#             no DPANN genome at all); for production V5 the train-split subset is
#             leaky, so every sample is tagged with its V5 split and the clean
#             head-to-head is computed on the test-split subset.
PHYLUM_EVAL_DESIGN = {
    'Bacteroidota':      {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Bacteroidota_A':    {'n_refs': 66,  'sims': 15, 'ref_source': 'test'},
    'Campylobacterota':  {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Patescibacteriota': {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Halobacteriota':    {'n_refs': 72,  'sims': 14, 'ref_source': 'test'},
    'DPANN':             {'n_refs': 83,  'sims': 12, 'ref_source': 'all'},
    # In-distribution control: non-panel test-split dominants, generated with the
    # identical procedure. Quantifies the effect of training on a 19.6%-smaller
    # genome pool, separately from the effect of lineage novelty. Required to make
    # the WS1.8 novelty-vs-error comparison interpretable.
    'in_distribution':   {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
}

# WS1.9. Every group draws dominants from the held-out TEST split - unlike WS1.6,
# no group needs the leaky 'all' fallback, because every family panel group has
# >= 28 test-split references. n_refs x sims ~ 1,000 per group, matching WS1.6's
# per-group sample count so the two experiments are directly comparable.
FAMILY_EVAL_DESIGN = {
    'Bacteroidota_Muribaculaceae':        {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Bacteroidota_Flavobacteriaceae':     {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Campylobacterota_Helicobacteraceae': {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Campylobacterota_Arcobacteraceae':   {'n_refs': 40,  'sims': 25, 'ref_source': 'test'},
    'Halobacteriota_halophilic':          {'n_refs': 28,  'sims': 36, 'ref_source': 'test'},
    'Patescibacteriota_families':         {'n_refs': 66,  'sims': 15, 'ref_source': 'test'},
    # In-distribution control: non-panel test-split dominants, identical procedure.
    # Isolates the cost of training on a 6.95%-smaller genome pool from the effect
    # of lineage novelty. Without it the DiD estimator cannot be formed.
    'in_distribution':                    {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
}

# WS11.G. Every group draws dominants from the held-out TEST split; no group
# needs the leaky 'all' fallback because every genus panel group has >= 24
# test-split references. n_refs x sims ~ 1,000 per group, matching WS1.6 and
# WS1.9 so the three experiments are directly comparable. n_refs is the number
# of test-split references AVAILABLE for the group, capped at EVAL_MAX_REFS.
GENUS_EVAL_DESIGN = {
    'Bacteroidota_Muribaculaceae_genera':        {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
    'Bacteroidota_Flavobacteriaceae_genera':     {'n_refs': 96,  'sims': 10, 'ref_source': 'test'},
    'Campylobacterota_Helicobacteraceae_genera': {'n_refs': 51,  'sims': 20, 'ref_source': 'test'},
    'Campylobacterota_Arcobacteraceae_genera':   {'n_refs': 33,  'sims': 30, 'ref_source': 'test'},
    'Halobacteriota_halophilic_genera':          {'n_refs': 24,  'sims': 42, 'ref_source': 'test'},
    'Patescibacteriota_genera':                  {'n_refs': 30,  'sims': 33, 'ref_source': 'test'},
    # In-distribution control: non-panel test-split dominants, identical
    # procedure, sqrt-stratified over phyla. Isolates the cost of training on a
    # 3.05%-smaller genome pool from the effect of lineage novelty. Without it
    # the DiD estimator cannot be formed, and the design is only valid if this
    # control's holdout-vs-V5 delta-MAE is ~0 (it was -0.047 at phylum level and
    # +0.029 at family level).
    'in_distribution':                          {'n_refs': 100, 'sims': 10, 'ref_source': 'test'},
}

EVAL_DESIGN = {'phylum': PHYLUM_EVAL_DESIGN,
               'family': FAMILY_EVAL_DESIGN,
               'genus': GENUS_EVAL_DESIGN}[PANEL_LEVEL]

# Held-out groups, in the order used for reporting (control last)
EVAL_GROUPS = [g for g in EVAL_DESIGN if g != 'in_distribution']
CONTROL_GROUP = 'in_distribution'


def load_split(tsv_path, exclude_panel=False, only_panel=False):
    """Load a split TSV as a DataFrame with fasta_path remapped and the panel-level
    taxon column added."""
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t')
    df['fasta_path'] = df['fasta_path'].map(remap_fasta_path)
    add_taxon_column(df)
    if exclude_panel:
        df = df[~df[TAXON_COL].isin(PANEL_TAXA)].copy()
    if only_panel:
        df = df[df[TAXON_COL].isin(PANEL_TAXA)].copy()
    return df.reset_index(drop=True)
