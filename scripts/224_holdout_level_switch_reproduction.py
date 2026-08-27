#!/usr/bin/env python3
"""
WS11.G - prove that adding the `genus` level to MAGICC_HOLDOUT_LEVEL did not
change what `phylum` (WS1.6) and `family` (WS1.9) do.

WHY THIS EXISTS
  WS11.G extends a SHARED code path (scripts 121-125 + holdout_lib/config.py)
  rather than forking it, so the two completed experiments must be untouched.
  Only config.py changed; the ten other files in scripts/holdout_lib/ are
  byte-identical to the FROZEN_SHA256_WS1.9.txt record (verified separately).

THE PRIMARY TEST - CONFIG EQUIVALENCE
  For each of phylum and family, everything the pipeline reads out of config.py
  is computed twice: once against the pre-WS11.G config.py (kept verbatim at
  scripts/holdout_lib/config.py.pre_ws11g.bak) and once against the current one.
  Both must agree exactly, including the SHA-256 of the reference-genome
  accession list produced by `select_refs()` from script 123 for every
  evaluation group. `select_refs` is the most seed-sensitive step in the whole
  pipeline (np.random.default_rng seeded from C.EVAL_SEED_BASE and CRC-32
  stable_hash) and it exercises config.py end to end, at no measurable CPU cost.

  The reference config is loaded from a throwaway COPY of scripts/holdout_lib/,
  so the live tree is never modified.

THE SECONDARY, INFORMATIONAL COMPARISON - vs the recorded runs
  The same accession lists are also compared against the references actually
  present in each completed run's eval_sets/<group>/metadata.tsv. Two mismatches
  are EXPECTED, both pre-existing and both independent of WS11.G:

    (a) WS1.6 phylum groups that SUBSAMPLE their references (Bacteroidota,
        Campylobacterota, Patescibacteriota) were drawn with `abs(hash(group))`,
        which Python salts per process. That reproducibility defect was found and
        fixed in WS1.9 by switching to CRC-32 (see `stable_hash` in script 123,
        which documents it). WS1.6's actual references are recorded verbatim in
        its metadata.tsv, so that run stays auditable. Phylum groups that use ALL
        available references (Bacteroidota_A, Halobacteriota, DPANN) never call
        .sample() and therefore do reproduce.

    (b) `in_distribution` at both levels: the sqrt-proportional allocator breaks
        a largest-remainder tie between equal-count rare phyla, and swapping one
        single-genome stratum reorders every downstream .sample() draw. 92 of 100
        references still match at family level; the allocation differs by exactly
        one genome.

  Neither is caused by, or affected by, the genus level - which is precisely what
  the primary test demonstrates.

Usage
  python scripts/224_holdout_level_switch_reproduction.py
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / 'results/revision/holdout_genus'
REF_CONFIG = HERE / 'holdout_lib/config.py.pre_ws11g.bak'


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _load123(libdir):
    """Import script 123 with `libdir` (holding a holdout_lib package) first."""
    sys.path.insert(0, str(libdir))
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        'gen_eval_sets', HERE / '123_generate_holdout_eval_sets.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fingerprint(libdir):
    """Everything the pipeline reads out of config.py, for the current level."""
    g123 = _load123(libdir)
    C = sys.modules['holdout_lib.config']

    fp = {'level': C.PANEL_LEVEL, 'ws': C.WS,
          'panel_taxa': sorted(C.PANEL_TAXA),
          'panel_phyla': sorted(C.PANEL_PHYLA),
          'panel_parent_phyla': sorted(C.PANEL_PARENT_PHYLA),
          'panel_groups': {g: sorted(d['taxa']) for g, d in C.PANEL_GROUPS.items()},
          'eval_design': C.EVAL_DESIGN,
          'eval_groups': list(C.EVAL_GROUPS),
          'control_group': C.CONTROL_GROUP,
          'paths': {'results_dir': str(C.RESULTS_DIR),
                    'holdout_dir': str(C.HOLDOUT_DIR),
                    'features_h5': str(C.HOLDOUT_H5),
                    'onnx': str(C.HOLDOUT_ONNX),
                    'norm_params': str(C.HOLDOUT_NORM_PARAMS),
                    'train_out': str(C.HOLDOUT_TRAIN_OUT),
                    'eval_dir': str(C.EVAL_DIR),
                    'panel_json': str(C.PANEL_JSON)},
          'synthesis': {'n_train': C.N_TRAIN_TOTAL, 'n_val': C.N_VAL,
                        'n_test': C.N_TEST, 'sample_types': C.SAMPLE_TYPES,
                        'batch_size': C.BATCH_SIZE,
                        'n_kmer_features': C.N_KMER_FEATURES,
                        'quality_tier_weights': C.QUALITY_TIER_WEIGHTS,
                        'adapted_reduced_pool': C.USE_ADAPTED_REDUCED_POOL,
                        'adapted_reduced_cutoff': C.ADAPTED_REDUCED_SIZE_CUTOFF,
                        'reduced_pool_threshold': C.REDUCED_POOL_ADAPT_THRESHOLD,
                        'v5_reduced_phyla': sorted(C.V5_REDUCED_GENOME_PHYLA),
                        'eval_seed_base': C.EVAL_SEED_BASE,
                        'eval_target_n': C.EVAL_TARGET_N,
                        'eval_max_refs': C.EVAL_MAX_REFS,
                        'eval_min_sims': C.EVAL_MIN_SIMS}}

    # deterministic reference selection, per evaluation group
    dfs = {}
    for s, p in [('train', C.TRAIN_TSV), ('val', C.VAL_TSV), ('test', C.TEST_TSV)]:
        d = pd.read_csv(p, sep='\t')
        d['fasta_path'] = d.fasta_path.map(C.remap_fasta_path)
        C.add_taxon_column(d)
        d['split'] = s
        dfs[s] = d
    allg = pd.concat(dfs.values(), ignore_index=True)
    in_panel = dfs['test'][C.TAXON_COL].isin(C.PANEL_TAXA)
    pools = {'panel_test': dfs['test'][in_panel].copy(),
             'panel_all': allg[allg[C.TAXON_COL].isin(C.PANEL_TAXA)].copy(),
             'nonpanel_test': dfs['test'][~in_panel].copy()}

    sel = {}
    for group, spec in C.EVAL_DESIGN.items():
        refs = g123.select_refs(group, dict(spec), pools, _Log())
        got = sorted(refs.ncbi_accession)
        e = {'n': len(got),
             'sha256': hashlib.sha256('\n'.join(got).encode()).hexdigest()}
        rec = C.EVAL_DIR / group / 'metadata.tsv'
        if rec.exists():
            want = sorted(pd.read_csv(rec, sep='\t',
                                      usecols=['dominant_accession'])
                          .dominant_accession.unique())
            e['recorded_n'] = len(want)
            e['matches_recorded'] = (got == want)
            e['overlap_with_recorded'] = len(set(got) & set(want))
            e['subsamples'] = len(got) < len(
                pools['panel_test' if group != C.CONTROL_GROUP
                      else 'nonpanel_test'])
        else:
            e['matches_recorded'] = None
        sel[group] = e
    fp['reference_selection'] = sel
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--_worker', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_libdir', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        os.environ['MAGICC_HOLDOUT_LEVEL'] = a._worker
        print('@@JSON@@' + json.dumps(fingerprint(Path(a._libdir)), default=str))
        return

    # throwaway copy of holdout_lib carrying the PRE-WS11.G config.py
    tmp = Path(tempfile.mkdtemp(prefix='ws11g_refcfg_'))
    try:
        shutil.copytree(HERE / 'holdout_lib', tmp / 'holdout_lib',
                        ignore=shutil.ignore_patterns('__pycache__', '*.bak', '*.ws11g'))
        shutil.copy2(REF_CONFIG, tmp / 'holdout_lib/config.py')

        def run(level, libdir):
            env = dict(os.environ, MAGICC_HOLDOUT_LEVEL=level, PYTHONHASHSEED='0')
            p = subprocess.run([sys.executable, __file__, '--_worker', level,
                                '--_libdir', str(libdir)],
                               capture_output=True, text=True, env=env)
            if p.returncode != 0:
                print(p.stdout[-2000:]); print(p.stderr[-3000:], file=sys.stderr)
                sys.exit(f'{level} @ {libdir} crashed')
            line = [l for l in p.stdout.splitlines() if l.startswith('@@JSON@@')][-1]
            return json.loads(line[len('@@JSON@@'):])

        print('=' * 78)
        print('LEVEL-SWITCH REPRODUCTION CHECK (WS11.G)')
        print('PRIMARY TEST: does adding the `genus` level change what `phylum`')
        print('(WS1.6) and `family` (WS1.9) compute?  It must not.')
        print('=' * 78)

        report, ok = {}, True
        for level in ['phylum', 'family']:
            cur = run(level, HERE)
            ref = run(level, tmp)
            equal = (json.dumps(cur, sort_keys=True, default=str) ==
                     json.dumps(ref, sort_keys=True, default=str))
            report[level] = {'config_equivalence': equal, 'current': cur}
            ok &= equal
            print(f'\n--- {level}  (WS {cur["ws"]}, {len(cur["panel_taxa"])} panel '
                  f'taxa -> {Path(cur["paths"]["results_dir"]).name}) ---')
            print(f'  pre-WS11.G config vs current config: '
                  f'{"IDENTICAL" if equal else "*** DIFFERENT ***"}')
            if not equal:
                for k in cur:
                    if json.dumps(cur[k], sort_keys=True, default=str) != \
                            json.dumps(ref[k], sort_keys=True, default=str):
                        print(f'    differs in: {k}')
            print('  per-group reference selection '
                  '(sha256 of the sorted accession list):')
            for g, e in cur['reference_selection'].items():
                mark = {True: 'reproduces recorded run',
                        False: 'differs from recorded run (see docstring)',
                        None: 'no recorded run'}[e['matches_recorded']]
                ov = (f'  overlap {e["overlap_with_recorded"]}/{e["recorded_n"]}'
                      if e.get('recorded_n') else '')
                print(f'    {g:<38} n={e["n"]:>4}  {e["sha256"][:16]}  '
                      f'{mark}{ov}')

        # genus level: record the fingerprint for provenance
        gen = run('genus', HERE)
        report['genus'] = {'config_equivalence': None, 'current': gen}
        print(f'\n--- genus  (WS {gen["ws"]}, {len(gen["panel_taxa"])} panel taxa '
              f'-> {Path(gen["paths"]["results_dir"]).name}) ---')
        for g, e in gen['reference_selection'].items():
            print(f'    {g:<38} n={e["n"]:>4}  {e["sha256"][:16]}')

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'level_switch_reproduction.json').write_text(
            json.dumps(report, indent=2, default=str))
        print(f'\nwrote {OUT / "level_switch_reproduction.json"}')
        print('\nVERDICT: ' + (
            'PASS - the genus level changes NOTHING at phylum or family level.'
            if ok else 'FAIL - the level switch altered a completed experiment.'))
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
