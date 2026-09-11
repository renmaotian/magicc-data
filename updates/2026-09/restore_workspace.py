#!/usr/bin/env python3
"""Restore a scientific workspace from committed files; no network or inference."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRODUCTION_SHA = 'b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--production-model', type=Path)
    args = ap.parse_args()
    target = args.destination.resolve()
    assert target != REPO and REPO not in target.parents, 'Choose a separate workspace'
    assert not target.exists() or not any(target.iterdir()), 'Destination must be new or empty'
    target.mkdir(parents=True, exist_ok=True)
    records = {}

    def copy(src, rel, role):
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records[str(rel)] = {'path': str(rel), 'source': str(src.relative_to(REPO)) if src.is_relative_to(REPO) else 'user-supplied production model',
                             'sha256_before_path_adaptation': sha(dst), 'role': role}

    for source, dest in [('results', 'results/revision'), ('splits', 'data/splits'), ('cami2', 'results/revision/cami2')]:
        for src in sorted((REPO / source).rglob('*')):
            if src.is_file():
                copy(src, Path(dest) / src.relative_to(REPO / source), 'Existing public input/result')
    aliases = {'set_A': 'set_A_v2', 'set_B': 'set_B_v2', 'set_H': 'set_H_ncbi'}
    for cohort in sorted((REPO / 'benchmark').iterdir()):
        if not cohort.is_dir():
            continue
        for src in sorted(cohort.rglob('*')):
            if src.is_file():
                copy(src, Path('data/benchmarks') / aliases.get(cohort.name, cohort.name) / src.relative_to(cohort), 'Existing public benchmark metadata/prediction')
    with (REPO / 'scripts/SCRIPT_MAPPING.tsv').open() as f:
        mapping = list(csv.DictReader(f, delimiter='\t'))
    substitutions = {}
    restored_scripts = set()
    for row in mapping:
        src = REPO / row['deposited_path']
        rel = Path(row['working_tree_path'])
        if not src.is_file() or not str(rel).startswith('scripts/'):
            continue
        copy(src, rel, 'Previously public script with original filename restored')
        restored_scripts.add(str(rel))
        if src.name != rel.name:
            substitutions[src.name] = rel.name
    rename_re = re.compile('|'.join(re.escape(x) for x in sorted(substitutions, key=len, reverse=True)))
    for rel in restored_scripts:
        dst = target / rel
        if dst.suffix in {'.py', '.sh'}:
            old = dst.read_text()
            text = rename_re.sub(lambda m: substitutions[m.group()], old)
            if text != old:
                dst.write_text(text)
                records[rel]['script_name_adaptation'] = 'Reversed recorded public numbering inside restored legacy code'
    with (HERE / 'SCIENTIFIC_FILE_MANIFEST.tsv').open() as f:
        snapshot = list(csv.DictReader(f, delimiter='\t'))
    for row in snapshot:
        src = HERE / 'snapshot' / row['path']
        assert src.stat().st_size == int(row['bytes']) and sha(src) == row['sha256'], row['path']
        copy(src, Path(row['path']), 'Exact September scientific snapshot overlay')
    # Adapt only literal path roots in executable text, never numerical tables.
    for rel, rec in records.items():
        dst = target / rel
        if dst.suffix in {'.py', '.sh', '.yaml', '.yml'}:
            old = dst.read_text()
            text = old.replace('/media/Data_1/tianrm/projects/magicc2', str(target)).replace('/path/to/magicc', str(target))
            if text != old:
                dst.write_text(text)
                rec['root_path_adaptation'] = 'Historical root literals replaced by this destination; numerical code unchanged'
        rec['restored_sha256'] = sha(dst)
    if args.production_model:
        src = args.production_model.resolve()
        assert sha(src) == PRODUCTION_SHA, 'Incorrect production model'
        copy(src, Path('models/magicc_v5.onnx'), 'Unchanged production V5 supplied by caller')
        records['models/magicc_v5.onnx']['restored_sha256'] = PRODUCTION_SHA
    manifest = {'status': 'COMMITTED_SCIENTIFIC_FILES_RESTORED', 'destination': str(target), 'files': list(records.values()),
                'pending_external_inputs': ['Exact NCBI reference FASTAs', 'Separate core-gene input archive',
                    'Research ONNX release assets for direct recorded-model inference', 'Analysis-specific third-party/raw inputs'],
                'note': 'No inference, retraining, network request, or change to the public repository performed.'}
    (target / 'RESTORE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Restored {len(records)} scientific files to {target}; see RESTORE_MANIFEST.json for path adaptations and external inputs.')


if __name__ == '__main__':
    main()
