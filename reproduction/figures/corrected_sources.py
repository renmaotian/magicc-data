#!/usr/bin/env python3
"""Finite, hashed source overrides for the resubmission5 comparator correction.

Figure code may retain historical logical paths, but accuracy reads resolve to
the explicitly recorded corrected artifact. Every actual tabular/JSON input is
recorded for the portable reproduction archive. Writes are never redirected.
"""
from __future__ import annotations
import atexit,builtins,hashlib,io,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PKG=Path(__file__).resolve().parents[1]
MAP=ROOT/'results/revision/cocopye_stage_resubmission5/source_overrides.json'
_OPEN=builtins.open;_IO_OPEN=io.open;_INSTALLED=False;_SEEN={};_OVERRIDES=None
def sha(path):
    h=hashlib.sha256()
    with _OPEN(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def _mapping():
    global _OVERRIDES
    if _OVERRIDES is None:
        with _OPEN(MAP) as f:m=json.load(f)
        assert m['status']=='ANALYSIS_ARTIFACTS_READY', 'Corrected statistical sources are not ready'
        if m.get('scope')=='MAIN_1_2_3_ONLY':assert Path(sys.argv[0]).stem=='make_figures_1_3','Only Figures1–3 are ready in this intermediate source manifest'
        if m.get('scope')=='FIGURES_ONLY_NOVELTY_TABLE_PENDING':assert Path(sys.argv[0]).stem in {'make_figures_1_3','make_figure_4','make_figure_5','make_supp_figures','make_palette_cvd_check'},'Figure sources are ready; the novelty table/workbook gate remains pending'
        _OVERRIDES=m['overrides']
        for original,entry in _OVERRIDES.items():
            actual=ROOT/entry['path'];assert actual.is_file(),actual
            assert sha(actual)==entry['sha256'],f'Corrected source changed after verification: {actual}'
    return _OVERRIDES
def resolve_source(path):
    if not isinstance(path,(str,bytes,os.PathLike)):return path
    try:p=Path(os.fsdecode(path)).absolute();logical=str(p.relative_to(ROOT))
    except (ValueError,TypeError):return path
    if logical.startswith(('results/revision/','data/benchmarks/','data/features/','data/kmer_selection/')):
        entry=_mapping().get(logical)
        actual=ROOT/entry['path'] if entry else p
        if actual.is_file() and (actual.suffix in {'.tsv','.csv','.json','.txt'} or actual.name.endswith('.tsv.gz')):
            key=str(actual.relative_to(ROOT))
            if key not in _SEEN:_SEEN[key]={'path':key,'sha256':sha(actual),'bytes':actual.stat().st_size,'logical_sources':[logical],'stage_corrected':entry is not None}
            elif logical not in _SEEN[key]['logical_sources']:_SEEN[key]['logical_sources'].append(logical)
        return str(actual)
    return path
def _wrap(opener):
    def wrapped(file,mode='r',*args,**kwargs):
        if isinstance(mode,str) and not any(x in mode for x in 'wax+'):
            file=resolve_source(file)
        return opener(file,mode,*args,**kwargs)
    return wrapped
def flush_manifest():
    if not _SEEN:return
    dest=PKG/'internal_qa/source_reads';dest.mkdir(parents=True,exist_ok=True)
    name=Path(sys.argv[0]).stem or 'interactive'
    with _OPEN(dest/(name+'.json'),'w') as f:json.dump({'reader':name,'inputs':list(sorted(_SEEN.values(),key=lambda x:x['path']))},f,indent=2);f.write('\n')
def install():
    global _INSTALLED
    if _INSTALLED:return
    _mapping();builtins.open=_wrap(_OPEN);io.open=_wrap(_IO_OPEN);_INSTALLED=True;atexit.register(flush_manifest)
