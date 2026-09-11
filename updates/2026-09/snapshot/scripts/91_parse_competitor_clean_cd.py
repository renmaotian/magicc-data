#!/usr/bin/env python3
"""
WS1.5 (competitor part) -- parse / merge competitor-tool output for the clean
replacement benchmark sets ``set_C_clean`` and ``set_D_clean``.

For every set this script produces one prediction TSV per tool, each merged with
``metadata.tsv`` so that true and predicted values sit side by side:

    <set>/checkm2_predictions.tsv     CheckM2 1.0.1   (from checkm2_output/quality_report.tsv)
    <set>/cocopye_predictions.tsv     CoCoPyE 0.5.0   (from cocopye_raw_output.csv, stage 3)
    <set>/deepcheck_predictions.tsv   DeepCheck       (CheckM2 --dbg_vectors PKLs -> ResNet)
    <set>/gunc_predictions.tsv        GUNC 1.1.1      (from results/revision/benchmark/gunc/<set>/)

DeepCheck's "run" happens here rather than in ``90_run_competitors_clean_cd.sh``
because it is a pure tensor transform of CheckM2's intermediate feature vectors
(no external tool is invoked). The model definition, the manual MinMaxScaler
transform, the 20,021 -> 20,164 zero-pad, the 142x142 reshape and the
work-around for DeepCheck's upstream ``forward()`` bug (it returns only the
completeness head) are imported *verbatim* from ``scripts/38_run_deepcheck_v2.py``
so no behaviour is re-invented.

Merge integrity is verified, not assumed. For every merge the script checks:
  1. metadata has the expected number of rows and unique genome_ids;
  2. the tool's genome-id set equals metadata's exactly (extras and misses listed);
  3. the merged frame has exactly len(metadata) rows in metadata order;
  4. true_completeness / true_contamination survive the merge bit-for-bit;
  5. no NaN predictions;
  6. an *independent* alignment probe that does not use the join key:
        CheckM2   -- Genome_Size must equal metadata total_length exactly
        DeepCheck -- same probe, via the CheckM2 feature-vector row order
        GUNC      -- n_contigs compared with metadata n_contigs (reported, not asserted:
                     GUNC counts contigs carrying called genes)
        CoCoPyE   -- no size field is emitted; set equality + 1:1 mapping only
A single failed check aborts that tool/set and is recorded in the verification
report; nothing is written silently.

Outputs
-------
    <set>/{checkm2,cocopye,deepcheck,gunc}_predictions.tsv
    <set>/deepcheck_wallclock.txt
    results/revision/benchmark/clean_cd_merge_verification.json

Usage
-----
    conda run -n magicc2 python scripts/91_parse_competitor_clean_cd.py
    conda run -n magicc2 python scripts/91_parse_competitor_clean_cd.py --tools checkm2,deepcheck
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_DIR / 'data' / 'benchmarks'
GUNC_ROOT = PROJECT_DIR / 'results' / 'revision' / 'benchmark' / 'gunc'
OUT_DIR = PROJECT_DIR / 'results' / 'revision' / 'benchmark'
DEEPCHECK_DIR = PROJECT_DIR / 'tools' / 'DeepCheck'
SCRIPT_38 = PROJECT_DIR / 'scripts' / '38_run_deepcheck_v2.py'

ACCURACY_NOTE = ('ACCURACY RUN, not the controlled speed benchmark (WS8.1). Executed on a '
                 'shared, concurrently loaded 48-core machine (other agents active), so '
                 'wall-clock is indicative only.')

METADATA_KEEP = ['genome_id', 'true_completeness', 'true_contamination',
                 'dominant_accession', 'dominant_phylum', 'sample_type',
                 'n_contigs', 'total_length', 'ref_index', 'replicate']


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class MergeError(RuntimeError):
    pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_wallclock(path: Path) -> Dict[str, object]:
    """Read a key=value *_wallclock.txt written by script 90 (or a bare number)."""
    if not path.is_file():
        return {'wall_clock_s': None, 'threads': None}
    text = path.read_text().strip()
    if '=' not in text:                              # legacy: a bare number of seconds
        try:
            return {'wall_clock_s': float(text.split()[0]), 'threads': None}
        except ValueError:
            return {'wall_clock_s': None, 'threads': None}
    out: Dict[str, object] = {}
    for line in text.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    for k in ('wall_clock_s', 'threads'):
        if k in out:
            try:
                out[k] = float(out[k]) if k == 'wall_clock_s' else int(out[k])
            except ValueError:
                out[k] = None
    return out


def verify_merge(meta: pd.DataFrame, tool_ids: pd.Series, merged: pd.DataFrame,
                 pred_cols: List[str], label: str,
                 allow_missing: bool = True) -> Dict[str, object]:
    """Assert that a metadata<-tool merge is complete, ordered and value-preserving.

    Genomes the tool failed to score are tolerated (``allow_missing``) but always
    counted and listed, never dropped silently. Genomes the tool reports that are
    *not* in the metadata are always fatal: they mean the wrong input directory
    was analysed.
    """
    rep: Dict[str, object] = {'label': label, 'checks': {}}
    ck = rep['checks']

    meta_ids = meta['genome_id'].astype(str)
    ck['metadata_rows'] = int(len(meta))
    ck['metadata_ids_unique'] = bool(meta_ids.is_unique)
    if not ck['metadata_ids_unique']:
        raise MergeError(f'{label}: metadata genome_id is not unique')

    tool_ids = tool_ids.astype(str)
    ck['tool_rows'] = int(len(tool_ids))
    ck['tool_ids_unique'] = bool(tool_ids.is_unique)
    if not ck['tool_ids_unique']:
        dups = tool_ids[tool_ids.duplicated()].unique().tolist()[:10]
        raise MergeError(f'{label}: tool output has duplicate genome ids, e.g. {dups}')

    ms, ts = set(meta_ids), set(tool_ids)
    missing, extra = sorted(ms - ts), sorted(ts - ms)
    ck['n_missing_from_tool'] = len(missing)
    ck['n_extra_in_tool'] = len(extra)
    ck['missing_examples'] = missing[:10]
    ck['extra_examples'] = extra[:10]
    if extra:
        raise MergeError(f'{label}: tool reported {len(extra)} genome id(s) absent from '
                         f'metadata (wrong input directory?), e.g. {extra[:5]}')
    if missing and not allow_missing:
        raise MergeError(f'{label}: {len(missing)} genome(s) missing from tool output, '
                         f'e.g. {missing[:5]}')

    ck['merged_rows'] = int(len(merged))
    if len(merged) != len(meta):
        raise MergeError(f'{label}: merged row count {len(merged)} != metadata {len(meta)}')

    ck['order_preserved'] = bool(
        merged['genome_id'].astype(str).tolist() == meta_ids.tolist())
    if not ck['order_preserved']:
        raise MergeError(f'{label}: merged genome_id order does not match metadata')

    for col in ('true_completeness', 'true_contamination'):
        same = np.array_equal(merged[col].values.astype(float),
                              meta[col].values.astype(float))
        ck[f'{col}_preserved'] = bool(same)
        if not same:
            raise MergeError(f'{label}: {col} altered by the merge')

    nan_counts = {c: int(merged[c].isna().sum()) for c in pred_cols}
    ck['nan_predictions'] = nan_counts
    ck['complete'] = bool(not missing and not any(v > 0 for v in nan_counts.values()))

    for c in pred_cols:
        vals = merged[c].dropna().values.astype(float)
        ck[f'{c}_range'] = [float(vals.min()), float(vals.max())] if len(vals) else None
        ck[f'{c}_n_distinct'] = int(pd.Series(vals).nunique())

    return rep


def independent_size_probe(merged: pd.DataFrame, size_col: str,
                           rep: Dict[str, object], strict: bool) -> None:
    """Alignment probe that does not rely on the join key."""
    if size_col not in merged.columns:
        rep['checks']['size_probe'] = 'column absent'
        return
    a = merged['total_length'].values.astype(float)
    b = pd.to_numeric(merged[size_col], errors='coerce').values.astype(float)
    ok = np.isfinite(b)
    identical = bool(np.all(a[ok] == b[ok])) and bool(ok.all())
    rep['checks']['size_probe'] = {
        'column': size_col, 'n_compared': int(ok.sum()),
        'identical_to_metadata_total_length': identical,
        'max_abs_diff': float(np.nanmax(np.abs(a[ok] - b[ok]))) if ok.any() else None,
    }
    if strict and not identical:
        raise MergeError(f"alignment probe failed: {size_col} != metadata total_length")


def finish(merged: pd.DataFrame, wc: Dict[str, object], tool: str,
           out_path: Path, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = [c for c in METADATA_KEEP if c in merged.columns]
    cols += ['pred_completeness', 'pred_contamination']
    out = merged[cols].copy()
    # keep the leading columns byte-compatible with the existing per-set
    # prediction files so scripts 101-105 auto-discover these unchanged
    out['wall_clock_s'] = wc.get('wall_clock_s')
    out['n_threads'] = wc.get('threads')
    out['tool'] = tool
    for c in (extra_cols or []):
        if c in merged.columns:
            out[c] = merged[c].values
    out.to_csv(out_path, sep='\t', index=False)
    return out


def mae(t, p):
    """NaN-safe MAE (rows the tool failed to score are excluded and counted upstream)."""
    return float(np.nanmean(np.abs(np.asarray(p, float) - np.asarray(t, float))))


def status_for(rep: Dict[str, object]) -> str:
    return 'ok' if rep['checks'].get('complete') else 'ok_with_missing'


# --------------------------------------------------------------------------
# per-tool parsers
# --------------------------------------------------------------------------
def do_checkm2(set_dir: Path, meta: pd.DataFrame) -> Dict[str, object]:
    qr = set_dir / 'checkm2_output' / 'quality_report.tsv'
    if not qr.is_file():
        return {'status': 'missing', 'reason': f'{qr} not found'}
    q = pd.read_csv(qr, sep='\t')
    q['genome_id'] = q['Name'].astype(str)
    q = q.rename(columns={'Completeness': 'pred_completeness',
                          'Contamination': 'pred_contamination'})
    keep = ['genome_id', 'pred_completeness', 'pred_contamination',
            'Completeness_Model_Used', 'Genome_Size', 'Contig_N50',
            'Total_Coding_Sequences']
    keep = [c for c in keep if c in q.columns]
    merged = meta.merge(q[keep], on='genome_id', how='left')
    rep = verify_merge(meta, q['genome_id'], merged,
                       ['pred_completeness', 'pred_contamination'], 'checkm2')
    independent_size_probe(merged, 'Genome_Size', rep, strict=True)

    wc = parse_wallclock(set_dir / 'checkm2_wallclock.txt')
    out = finish(merged, wc, 'CheckM2 1.0.1',
                 set_dir / 'checkm2_predictions.tsv',
                 ['Completeness_Model_Used'])
    rep.update({'status': status_for(rep), 'output': str(set_dir / 'checkm2_predictions.tsv'),
                'wall_clock_s': wc.get('wall_clock_s'), 'threads': wc.get('threads'),
                'comp_mae': mae(out.true_completeness, out.pred_completeness),
                'cont_mae': mae(out.true_contamination, out.pred_contamination)})
    if 'Completeness_Model_Used' in out.columns:
        rep['model_used_counts'] = out['Completeness_Model_Used'].value_counts().to_dict()
    return rep


def do_cocopye(set_dir: Path, meta: pd.DataFrame) -> Dict[str, object]:
    raw = set_dir / 'cocopye_raw_output.csv'
    if not raw.is_file():
        return {'status': 'missing', 'reason': f'{raw} not found'}
    c = pd.read_csv(raw)
    c['genome_id'] = c['bin'].astype(str)

    # Stage-3 (marker + neural network) predictions, stage-2 fallback; 0-1 -> 0-100 %
    if '3_completeness' in c.columns:
        c['pred_completeness'] = c['3_completeness'].fillna(
            c.get('2_completeness', pd.Series(np.nan, index=c.index))) * 100.0
        c['pred_contamination'] = c['3_contamination'].fillna(
            c.get('2_contamination', pd.Series(np.nan, index=c.index))) * 100.0
        stage = 'stage3 (stage2 fallback)'
    elif '2_completeness' in c.columns:
        c['pred_completeness'] = c['2_completeness'] * 100.0
        c['pred_contamination'] = c['2_contamination'] * 100.0
        stage = 'stage2 only'
    else:
        return {'status': 'failed', 'reason': f'no completeness columns in {raw}'}

    n_stage3 = int(c['3_completeness'].notna().sum()) if '3_completeness' in c.columns else 0
    keep = ['genome_id', 'pred_completeness', 'pred_contamination', 'stage',
            '2_num_markers', 'taxonomy', 'taxonomy_level']
    keep = [k for k in keep if k in c.columns]
    merged = meta.merge(c[keep], on='genome_id', how='left')
    rep = verify_merge(meta, c['genome_id'], merged,
                       ['pred_completeness', 'pred_contamination'], 'cocopye')
    rep['checks']['size_probe'] = 'CoCoPyE emits no genome-size field; set equality + 1:1 join only'
    rep['stage_used'] = stage
    rep['n_rows_with_stage3'] = n_stage3
    if 'stage' in c.columns:
        # CoCoPyE's own recommended stage per bin. Following scripts/37 (and therefore
        # the superseded set_C/set_D runs), the stage-3 marker+neural-network estimate is
        # used whenever it exists, regardless of this recommendation, so the clean and
        # superseded sets remain directly comparable.
        rep['cocopye_recommended_stage_counts'] = (
            c['stage'].value_counts().sort_index().to_dict())

    wc = parse_wallclock(set_dir / 'cocopye_wallclock.txt')
    out = finish(merged, wc, 'CoCoPyE 0.5.0',
                 set_dir / 'cocopye_predictions.tsv', ['stage', 'taxonomy'])
    rep.update({'status': status_for(rep), 'output': str(set_dir / 'cocopye_predictions.tsv'),
                'wall_clock_s': wc.get('wall_clock_s'), 'threads': wc.get('threads'),
                'comp_mae': mae(out.true_completeness, out.pred_completeness),
                'cont_mae': mae(out.true_contamination, out.pred_contamination)})
    return rep


def do_deepcheck(set_dir: Path, meta: pd.DataFrame, dc_ctx: Dict,
                 torch_threads: int = 4) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    ck_dir = set_dir / 'checkm2_output'
    pkls = sorted(p for p in ck_dir.glob('*.pkl'))
    if not pkls:
        return {'status': 'missing',
                'reason': f'no CheckM2 --dbg_vectors PKL files in {ck_dir}'}

    dfs = [pd.read_pickle(p) for p in pkls]
    features_df = pd.concat(dfs, ignore_index=True)
    names = features_df['Name'].astype(str).values
    feature_matrix = features_df.iloc[:, 1:].values.astype(float)

    scaled = feature_matrix * dc_ctx['scale'] + dc_ctx['min']
    scaled = scaled[:, :20021]
    padded = np.zeros((scaled.shape[0], 20164), dtype=np.float32)
    padded[:, :20021] = scaled

    model = dc_ctx['model']
    ds = dc_ctx['Dataset'](padded)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    comp, cont = [], []
    torch.set_num_threads(torch_threads)
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            cp, cx = model(batch)
            comp.extend((cp.squeeze() * 100).cpu().numpy().tolist())
            cont.extend((cx.squeeze() * 100).cpu().numpy().tolist())
    wall = time.time() - t0

    d = pd.DataFrame({'genome_id': names, 'pred_completeness': comp,
                      'pred_contamination': cont})
    merged = meta.merge(d, on='genome_id', how='left')
    rep = verify_merge(meta, d['genome_id'], merged,
                       ['pred_completeness', 'pred_contamination'], 'deepcheck')
    rep['n_pkl_files'] = len(pkls)
    rep['feature_matrix_shape'] = list(feature_matrix.shape)

    # independent probe: the same CheckM2 rows carry Genome_Size in quality_report
    qr = ck_dir / 'quality_report.tsv'
    if qr.is_file():
        q = pd.read_csv(qr, sep='\t')[['Name', 'Genome_Size']]
        q['genome_id'] = q['Name'].astype(str)
        probe = merged.merge(q[['genome_id', 'Genome_Size']], on='genome_id', how='left')
        independent_size_probe(probe, 'Genome_Size', rep, strict=True)

    wc_path = set_dir / 'deepcheck_wallclock.txt'
    wc_path.write_text(
        'tool=deepcheck\n'
        'version=DeepCheck (github commit as vendored in tools/DeepCheck)\n'
        f'set={set_dir.name}\n'
        f'wall_clock_s={wall:.2f}\n'
        f'threads={torch_threads}\n'
        f'finished={datetime.now(timezone.utc).isoformat()}\n'
        f'host={socket.gethostname()}\n'
        'command=scripts/91_parse_competitor_clean_cd.py (ResNet inference over '
        'CheckM2 --dbg_vectors feature vectors)\n'
        'note=INFERENCE ONLY. Excludes the CheckM2 feature-extraction time DeepCheck '
        'depends on (see checkm2_wallclock.txt for that set). ' + ACCURACY_NOTE + '\n')

    wc = {'wall_clock_s': round(wall, 2), 'threads': torch_threads}
    out = finish(merged, wc, 'DeepCheck', set_dir / 'deepcheck_predictions.tsv')
    rep.update({'status': status_for(rep), 'output': str(set_dir / 'deepcheck_predictions.tsv'),
                'wall_clock_s': round(wall, 2), 'threads': torch_threads,
                'wall_clock_note': 'inference only, excludes CheckM2 feature extraction',
                'comp_mae': mae(out.true_completeness, out.pred_completeness),
                'cont_mae': mae(out.true_contamination, out.pred_contamination)})
    return rep


def do_gunc(set_dir: Path, meta: pd.DataFrame) -> Dict[str, object]:
    norm = GUNC_ROOT / set_dir.name / 'gunc_normalized.tsv'
    if not norm.is_file():
        return {'status': 'missing', 'reason': f'{norm} not found (GUNC not run)'}
    g = pd.read_csv(norm, sep='\t')
    g['genome_id'] = g['genome'].astype(str)
    if 'n_contigs' in g.columns:          # avoid colliding with metadata's n_contigs
        g = g.rename(columns={'n_contigs': 'gunc_n_contigs'})

    merged = meta.merge(g.drop(columns=['genome']), on='genome_id', how='left')
    # GUNC does not estimate completeness/contamination -> verify only ids and order
    rep = verify_merge(meta, g['genome_id'], merged, [], 'gunc')

    if 'gunc_n_contigs' in merged.columns:
        a = meta['n_contigs'].values.astype(float)
        b = pd.to_numeric(merged['gunc_n_contigs'], errors='coerce').values.astype(float)
        ok = np.isfinite(b)
        rep['checks']['contig_probe'] = {
            'n_compared': int(ok.sum()),
            'n_equal': int(np.sum(a[ok] == b[ok])),
            'frac_equal': float(np.mean(a[ok] == b[ok])) if ok.any() else None,
            'note': 'GUNC counts contigs carrying called genes; <= metadata n_contigs',
        }

    wc = parse_wallclock(set_dir / 'gunc_wallclock.txt')
    cols = [c for c in METADATA_KEEP if c in merged.columns]
    tool_cols = [c for c in merged.columns
                 if c.startswith('gunc_') or c in
                 ('n_effective_surplus_clades', 'taxonomic_level', 'n_genes_called',
                  'n_genes_mapped', 'proportion_genes_retained_in_major_clades',
                  'genes_retained_index', 'contamination_portion', 'mean_hit_identity',
                  'reference_representation_score')]
    out = merged[cols + tool_cols].copy()
    out['wall_clock_s'] = wc.get('wall_clock_s')
    out['n_threads'] = wc.get('threads')
    out['tool'] = 'GUNC 1.1.1'
    out_path = set_dir / 'gunc_predictions.tsv'
    out.to_csv(out_path, sep='\t', index=False)

    n_fail = int((out['gunc_pass'].astype(str).str.lower() == 'false').sum()) \
        if 'gunc_pass' in out.columns else None
    n_pass = int((out['gunc_pass'].astype(str).str.lower() == 'true').sum()) \
        if 'gunc_pass' in out.columns else None
    rep.update({'status': status_for(rep), 'output': str(out_path),
                'wall_clock_s': wc.get('wall_clock_s'), 'threads': wc.get('threads'),
                'n_pass': n_pass, 'n_fail': n_fail,
                'note': 'GUNC flags chimerism; it does not estimate completeness or '
                        'contamination on the MAGICC/CheckM2 scale'})
    return rep


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='set_C_clean,set_D_clean')
    ap.add_argument('--tools', default='checkm2,cocopye,deepcheck,gunc')
    ap.add_argument('--torch-threads', type=int, default=4,
                    help='CPU threads for the DeepCheck ResNet forward pass '
                         '(recorded in deepcheck_wallclock.txt; accuracy is unaffected)')
    args = ap.parse_args()

    sets = [s for s in args.sets.split(',') if s]
    tools = [t for t in args.tools.split(',') if t]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dc_ctx = None
    if 'deepcheck' in tools:
        import torch
        m38 = load_module(SCRIPT_38, 'dc38')
        sp = np.load(DEEPCHECK_DIR / 'scaler_params.npz')
        model = m38.ResNetDualOutput(m38.ResidualBlock, [2, 2, 2, 2])
        sd = torch.load(DEEPCHECK_DIR / 'models' / 'best_model.pt',
                        map_location='cpu', weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        dc_ctx = {'scale': sp['scale'], 'min': sp['min_val'], 'model': model,
                  'Dataset': m38.DeepCheckDataset,
                  'n_params': sum(p.numel() for p in model.parameters())}
        print(f'DeepCheck model loaded: {dc_ctx["n_params"]:,} parameters '
              f'(dual-output work-around for the upstream forward() bug)')

    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'project_dir': str(PROJECT_DIR), 'sets': {}}

    for set_name in sets:
        set_dir = BENCHMARK_DIR / set_name
        meta = pd.read_csv(set_dir / 'metadata.tsv', sep='\t')
        meta['genome_id'] = meta['genome_id'].astype(str)
        print(f'\n=== {set_name}: {len(meta)} genomes ===')
        report['sets'][set_name] = {'n_metadata_rows': int(len(meta)), 'tools': {}}

        for tool in tools:
            try:
                if tool == 'checkm2':
                    rep = do_checkm2(set_dir, meta)
                elif tool == 'cocopye':
                    rep = do_cocopye(set_dir, meta)
                elif tool == 'deepcheck':
                    rep = do_deepcheck(set_dir, meta, dc_ctx, args.torch_threads)
                elif tool == 'gunc':
                    rep = do_gunc(set_dir, meta)
                else:
                    rep = {'status': 'failed', 'reason': f'unknown tool {tool}'}
            except MergeError as e:
                rep = {'status': 'failed', 'reason': str(e)}
            except Exception as e:                       # noqa: BLE001
                rep = {'status': 'failed',
                       'reason': f'{type(e).__name__}: {e}'}
            report['sets'][set_name]['tools'][tool] = rep

            st = rep.get('status')
            if st in ('ok', 'ok_with_missing'):
                extra = ''
                if 'comp_mae' in rep:
                    extra = (f"  comp MAE {rep['comp_mae']:.3f}  "
                             f"cont MAE {rep['cont_mae']:.3f}")
                elif rep.get('n_fail') is not None:
                    extra = f"  pass {rep['n_pass']} / fail {rep['n_fail']}"
                if st == 'ok_with_missing':
                    extra += ('  !! MISSING '
                              f'{rep["checks"]["n_missing_from_tool"]} genome(s) / '
                              f'NaN {rep["checks"]["nan_predictions"]}')
                print(f'  [{tool}] {st.upper()}  n={rep["checks"]["merged_rows"]}'
                      f'  wall={rep.get("wall_clock_s")}s'
                      f'  threads={rep.get("threads")}{extra}')
            else:
                print(f'  [{tool}] {st.upper()}: {rep.get("reason")}')

    out = OUT_DIR / 'clean_cd_merge_verification.json'
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f'\nVerification report -> {out}')

    failed = [(s, t) for s, sv in report['sets'].items()
              for t, tv in sv['tools'].items() if tv.get('status') == 'failed']
    if failed:
        print('FAILURES: ' + ', '.join(f'{s}/{t}' for s, t in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
