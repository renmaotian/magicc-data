#!/usr/bin/env python3
"""
WS4.1 / WS4.2 -- reusable GUNC runner for MAGICC benchmark sets.

Runs GUNC (chimerism / contamination detection by gene-level taxonomic
discordance) on a directory of genome FASTAs and emits a *normalized* TSV so
that every downstream workstream reads GUNC results through one schema.

GUNC lives in its own conda environment (``gunc_env``) because it pins
prodigal/diamond; this script prepends that environment's ``bin`` to ``PATH``
so it can be invoked from any environment (e.g. ``magicc2``).

Usage
-----
    python scripts/76_run_gunc.py \
        --input-dir data/ncbi/mags \
        --output-dir results/revision/gunc/ncbi_mags \
        --extension .fna --threads 8

    # genomes listed in a text file instead of a directory
    python scripts/76_run_gunc.py --input-list paths.txt --output-dir out/

Outputs (inside ``--output-dir``)
--------------------------------
    gunc_output/
        GUNC.<db>.maxCSS_level.tsv   raw GUNC verdict, one row per genome
        gene_calls/                  prodigal .faa per genome + merged query
        diamond_output/              raw DIAMOND blastp hits
        gunc_output/                 per-genome <genome>.<db>.all_levels.tsv
                                     (scores at every taxonomic rank; GUNC nests
                                     this second 'gunc_output' dir itself)
    gunc_normalized.tsv              normalized results (schema below)
    gunc_run_summary.json            versions, command, wall-clock, peak RSS
    gunc_run.log                     GUNC's own phase-timing log

Normalized TSV schema
---------------------
Required columns (in this order), consumed by later workstreams:
    genome                      genome name (input basename, extensions stripped)
    gunc_css                    clade_separation_score at maxCSS taxonomic level
    gunc_pass                   True/False -- GUNC's pass.GUNC verdict
    n_effective_surplus_clades  effective number of surplus clades
    taxonomic_level             taxonomic level at which maxCSS was observed

Additional GUNC columns are appended (n_genes_called, n_genes_mapped,
n_contigs, proportion_genes_retained_in_major_clades, genes_retained_index,
contamination_portion, mean_hit_identity, reference_representation_score) so
that WS4.6's evidence table does not require a re-run.

Notes
-----
* GUNC *flags* contamination; it does not quantify completeness/contamination
  on the same scale as MAGICC or CheckM2. Report it as a detector, not an
  estimator.
* Genomes whose gene count falls below ``--min-mapped-genes`` receive no score;
  they appear in the normalized TSV with empty ``gunc_css`` and
  ``gunc_pass = NA``.
* gzip-compressed inputs are transparently staged (decompressed) into a
  temporary directory, because prodigal cannot read gzip.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = PROJECT_DIR / 'tools' / 'gunc_db'
DEFAULT_GUNC_ENV = 'gunc_env'

FASTA_EXTENSIONS = ('.fasta', '.fna', '.ffn', '.fas', '.fsa', '.fa')

#: GUNC maxCSS_level.tsv column -> normalized column
REQUIRED_COLUMNS = [
    ('genome', 'genome'),
    ('clade_separation_score', 'gunc_css'),
    ('pass.GUNC', 'gunc_pass'),
    ('n_effective_surplus_clades', 'n_effective_surplus_clades'),
    ('taxonomic_level', 'taxonomic_level'),
]
EXTRA_COLUMNS = [
    'n_genes_called', 'n_genes_mapped', 'n_contigs',
    'proportion_genes_retained_in_major_clades', 'genes_retained_index',
    'contamination_portion', 'mean_hit_identity',
    'reference_representation_score',
]


# --------------------------------------------------------------------------
# Environment / tool resolution
# --------------------------------------------------------------------------
def resolve_gunc_env_bin(env_name: str) -> Optional[Path]:
    """Locate the ``bin`` directory of a named conda environment."""
    candidates: List[Path] = []
    conda_base = os.environ.get('CONDA_PREFIX_1') or os.environ.get('CONDA_PREFIX')
    if conda_base:
        candidates.append(Path(conda_base) / 'envs' / env_name / 'bin')
    for base in (Path.home() / 'anaconda3', Path.home() / 'miniconda3',
                 Path.home() / 'mambaforge', Path.home() / 'miniforge3'):
        candidates.append(base / 'envs' / env_name / 'bin')
    for c in candidates:
        if (c / 'gunc').is_file():
            return c
    return None


def build_env(env_name: str, db_file: Path) -> Dict[str, str]:
    """Return an environment dict with the GUNC env on PATH and GUNC_DB set."""
    env = os.environ.copy()
    bindir = resolve_gunc_env_bin(env_name)
    if bindir is not None:
        env['PATH'] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    elif shutil.which('gunc') is None:
        sys.exit(
            f"ERROR: 'gunc' not found. Conda env '{env_name}' has no gunc binary "
            f"and gunc is not on PATH.\n"
            f"Install with: mamba create -y -n {env_name} -c conda-forge -c bioconda gunc"
        )
    env['GUNC_DB'] = str(db_file)
    # Keep BLAS/OpenMP from oversubscribing beyond --threads
    return env


def tool_versions(env: Dict[str, str]) -> Dict[str, str]:
    """Capture exact version strings for the Methods section."""
    out = {}
    for name, cmd in [('gunc', ['gunc', '--version']),
                      ('diamond', ['diamond', '--version']),
                      ('prodigal', ['prodigal', '-v'])]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
            text = (p.stdout + p.stderr).strip().splitlines()
            out[name] = next((ln.strip() for ln in text if ln.strip()), 'unknown')
        except Exception as e:  # pragma: no cover - diagnostic only
            out[name] = f'error: {e}'
    return out


def resolve_db(db_arg: Optional[str]) -> Path:
    """Find the GUNC diamond database file.

    Always returns an *absolute* path: ``gunc run`` is launched with
    ``cwd=<output-dir>``, so a relative ``--db`` would not resolve there.
    """
    if db_arg:
        p = Path(db_arg).expanduser().resolve()
        if p.is_dir():
            hits = sorted(p.glob('*.dmnd'))
            if not hits:
                sys.exit(f"ERROR: no *.dmnd file in {p}")
            return hits[0].resolve()
        if not p.is_file():
            sys.exit(f"ERROR: GUNC database not found: {p}")
        return p
    if os.environ.get('GUNC_DB') and Path(os.environ['GUNC_DB']).is_file():
        return Path(os.environ['GUNC_DB']).resolve()
    hits = sorted(DEFAULT_DB_DIR.glob('*.dmnd'))
    if not hits:
        sys.exit(
            f"ERROR: no GUNC database found in {DEFAULT_DB_DIR}.\n"
            f"Download with: gunc download_db -db progenomes_2.1 {DEFAULT_DB_DIR}"
        )
    return hits[0].resolve()


# --------------------------------------------------------------------------
# Input staging
# --------------------------------------------------------------------------
def genome_name(path: str) -> str:
    """Strip .gz and a recognised FASTA extension from a basename."""
    name = os.path.basename(path)
    if name.lower().endswith('.gz'):
        name = name[:-3]
    # Protein gene-call files written by GUNC are '<genome>.genecalls.faa'
    if name.lower().endswith('.genecalls.faa'):
        return name[:-len('.genecalls.faa')]
    if name.lower().endswith('.faa'):
        return name[:-len('.faa')]
    low = name.lower()
    for ext in FASTA_EXTENSIONS:
        if low.endswith(ext) and len(name) > len(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def collect_inputs(input_dir: Optional[str], input_list: Optional[str],
                   extension: str) -> List[Path]:
    """Return the list of genome FASTA paths to analyse."""
    if input_list:
        lp = Path(input_list)
        if not lp.is_file():
            sys.exit(f"ERROR: genome list file not found: {input_list}")
        paths: List[Path] = []
        missing: List[str] = []
        for lineno, raw in enumerate(lp.read_text().splitlines(), start=1):
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            cand = Path(os.path.expanduser(s))
            if not cand.is_absolute():
                cand = (Path.cwd() / cand) if (Path.cwd() / cand).exists() \
                    else (lp.parent / cand)
            if not cand.is_file():
                missing.append(f"  line {lineno}: {s}")
            else:
                paths.append(cand)
        if missing:
            sys.exit(f"ERROR: {len(missing)} listed path(s) not found:\n"
                     + "\n".join(missing))
        return paths

    d = Path(input_dir)
    if not d.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")
    if extension.lower() in ('auto', 'any', '*'):
        out = []
        for entry in sorted(d.iterdir()):
            if not entry.is_file():
                continue
            stem = entry.name[:-3] if entry.name.lower().endswith('.gz') else entry.name
            if any(stem.lower().endswith(e) for e in FASTA_EXTENSIONS):
                out.append(entry)
        return out
    if not extension.startswith('.'):
        extension = '.' + extension
    return [e for e in sorted(d.iterdir())
            if e.is_file() and (e.name.endswith(extension)
                                or e.name.endswith(extension + '.gz'))]


def stage_inputs(paths: List[Path], staging: Path,
                 suffix: str = '.fa') -> Tuple[Path, str]:
    """
    Symlink (or decompress) inputs into a flat staging directory with a single
    uniform extension, which is what GUNC's ``-e`` expects (``.fa`` for
    nucleotide FASTA, ``.faa`` for pre-computed gene calls).

    Returns (staging_dir, suffix).
    """
    staging.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = staging / (genome_name(str(p)) + suffix)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if p.name.lower().endswith('.gz'):
            with gzip.open(p, 'rb') as fin, open(dest, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        else:
            os.symlink(os.path.abspath(str(p)), dest)
    return staging, suffix


# --------------------------------------------------------------------------
# Result normalization
# --------------------------------------------------------------------------
def normalize_results(gunc_out_dir: Path, all_genomes: List[str],
                      out_tsv: Path) -> Dict[str, int]:
    """Convert GUNC's maxCSS_level.tsv into the normalized schema."""
    hits = sorted(gunc_out_dir.glob('*maxCSS_level.tsv'))
    if not hits:
        sys.exit(f"ERROR: no *maxCSS_level.tsv in {gunc_out_dir}; GUNC likely failed")
    src = hits[0]

    with open(src) as f:
        header = f.readline().rstrip('\n').split('\t')
        rows = [dict(zip(header, ln.rstrip('\n').split('\t')))
                for ln in f if ln.strip()]

    missing_cols = [c for c, _ in REQUIRED_COLUMNS if c not in header]
    if missing_cols:
        sys.exit(f"ERROR: {src} is missing expected column(s): {missing_cols}\n"
                 f"Found: {header}")

    by_genome = {r['genome']: r for r in rows}
    out_header = [new for _, new in REQUIRED_COLUMNS] + \
                 [c for c in EXTRA_COLUMNS if c in header]

    n_pass = n_fail = n_unscored = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, 'w') as f:
        f.write('\t'.join(out_header) + '\n')
        for g in all_genomes:
            r = by_genome.get(g)
            if r is None:
                n_unscored += 1
                f.write('\t'.join([g, '', 'NA', '', 'NA']
                                  + [''] * (len(out_header) - 5)) + '\n')
                continue
            verdict = r.get('pass.GUNC', '').strip()
            if verdict in ('True', 'true', 'TRUE'):
                n_pass += 1
            elif verdict in ('False', 'false', 'FALSE'):
                n_fail += 1
            else:
                n_unscored += 1
            vals = [r.get(old, '') for old, _ in REQUIRED_COLUMNS]
            vals += [r.get(c, '') for c in EXTRA_COLUMNS if c in header]
            f.write('\t'.join(vals) + '\n')

    return {'n_input': len(all_genomes), 'n_scored': len(rows),
            'n_pass': n_pass, 'n_fail': n_fail, 'n_unscored': n_unscored,
            'source': str(src)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Run GUNC on a genome set and emit a normalized TSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input-dir', help='Directory of genome FASTA files')
    src.add_argument('--input-list', help='Text file of genome paths, one per line')
    ap.add_argument('--output-dir', required=True, help='Output directory')
    ap.add_argument('--extension', '-e', default='auto',
                    help='Extension filter for --input-dir (default: auto). '
                         'The .gz form is matched too')
    ap.add_argument('--threads', '-t', type=int, default=8,
                    help='CPU threads for GUNC/diamond (default: 8)')
    ap.add_argument('--db', default=None,
                    help=f'GUNC diamond DB file or directory (default: {DEFAULT_DB_DIR})')
    ap.add_argument('--gunc-env', default=DEFAULT_GUNC_ENV,
                    help=f'Conda env containing gunc (default: {DEFAULT_GUNC_ENV})')
    ap.add_argument('--gene-calls', action='store_true',
                    help='Inputs are pre-computed prodigal protein FASTA '
                         '(.faa) rather than nucleotide assemblies; passes -g '
                         'to gunc run and skips gene calling. Use with '
                         '--extension .genecalls.faa on a previous run\'s '
                         'gunc_output/gene_calls directory.')
    ap.add_argument('--sensitive', action='store_true',
                    help='Pass --sensitive to gunc run')
    ap.add_argument('--detailed', action='store_true', default=True,
                    help='Emit all-taxlevel scores (default: on)')
    ap.add_argument('--no-detailed', dest='detailed', action='store_false')
    ap.add_argument('--contig-taxonomy', action='store_true',
                    help='Also emit per-contig taxonomic assignments')
    ap.add_argument('--min-mapped-genes', type=int, default=11,
                    help='Skip scoring below this many mapped genes (GUNC default: 11)')
    ap.add_argument('--temp-dir', default=None,
                    help='Scratch directory (default: <output-dir>/tmp)')
    ap.add_argument('--keep-staging', action='store_true',
                    help='Keep the staged input directory for debugging')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the GUNC command and exit')
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gunc_out = out_dir / 'gunc_output'
    gunc_out.mkdir(exist_ok=True)
    temp_dir = Path(args.temp_dir) if args.temp_dir else out_dir / 'tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    db_file = resolve_db(args.db)
    env = build_env(args.gunc_env, db_file)
    versions = tool_versions(env)

    inputs = collect_inputs(args.input_dir, args.input_list, args.extension)
    if not inputs:
        sys.exit(f"ERROR: no genome files found "
                 f"(input={args.input_dir or args.input_list}, ext={args.extension})")
    genomes = [genome_name(str(p)) for p in inputs]

    stage_suffix = '.faa' if args.gene_calls else '.fa'
    staging = Path(tempfile.mkdtemp(prefix='gunc_stage_', dir=str(temp_dir)))
    stage_inputs(inputs, staging, stage_suffix)

    cmd = ['gunc', 'run', '-d', str(staging), '-e', stage_suffix,
           '-r', str(db_file), '-t', str(args.threads),
           '-o', str(gunc_out), '--temp_dir', str(temp_dir),
           '--min_mapped_genes', str(args.min_mapped_genes)]
    if args.gene_calls:
        cmd.append('-g')
    if args.detailed:
        cmd.append('--detailed_output')
    if args.sensitive:
        cmd.append('--sensitive')
    if args.contig_taxonomy:
        cmd.append('--contig_taxonomy_output')

    print(f"GUNC:      {versions.get('gunc')}")
    print(f"DIAMOND:   {versions.get('diamond')}")
    print(f"Database:  {db_file} ({db_file.stat().st_size / 1e9:.2f} GB)")
    print(f"Genomes:   {len(inputs)}")
    print(f"Threads:   {args.threads}")
    print(f"Command:   {' '.join(cmd)}")

    if args.dry_run:
        shutil.rmtree(staging, ignore_errors=True)
        return 0

    log_path = out_dir / 'gunc_run.log'
    t0 = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.run(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, cwd=str(out_dir))
    wall = time.time() - t0

    # Peak RSS of the child process tree (ru_maxrss is in KiB on Linux)
    try:
        import resource
        peak_rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 / 1024
    except Exception:
        peak_rss_gb = None

    if proc.returncode != 0:
        print(f"ERROR: gunc run exited {proc.returncode}; see {log_path}",
              file=sys.stderr)
        print(log_path.read_text()[-3000:], file=sys.stderr)
        if not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        return proc.returncode

    stats = normalize_results(gunc_out, genomes, out_dir / 'gunc_normalized.tsv')

    summary = {
        'gunc_version': versions.get('gunc'),
        'diamond_version': versions.get('diamond'),
        'prodigal_version': versions.get('prodigal'),
        'database_file': str(db_file),
        'database_bytes': db_file.stat().st_size,
        'database_gb': round(db_file.stat().st_size / 1e9, 3),
        'command': ' '.join(cmd),
        'input': args.input_dir or args.input_list,
        'threads': args.threads,
        'gene_calls_reused': bool(args.gene_calls),
        'sensitive': args.sensitive,
        'min_mapped_genes': args.min_mapped_genes,
        'wall_clock_s': round(wall, 2),
        's_per_genome': round(wall / max(len(inputs), 1), 3),
        'peak_rss_gb_children': (round(peak_rss_gb, 3)
                                 if peak_rss_gb is not None else None),
        'normalized_tsv': str(out_dir / 'gunc_normalized.tsv'),
        **stats,
    }
    (out_dir / 'gunc_run_summary.json').write_text(json.dumps(summary, indent=2))

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone in {wall:.1f}s ({wall / max(len(inputs),1):.1f}s/genome, "
          f"{args.threads} threads)")
    print(f"  scored={stats['n_scored']}  pass={stats['n_pass']}  "
          f"fail={stats['n_fail']}  unscored={stats['n_unscored']}")
    print(f"  {out_dir / 'gunc_normalized.tsv'}")
    print(f"  {out_dir / 'gunc_run_summary.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
