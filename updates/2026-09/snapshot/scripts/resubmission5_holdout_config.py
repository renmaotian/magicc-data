"""Private configuration overlay; leaves completed holdout experiments untouched."""
import importlib.util
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/holdout_resubmission5'
os.environ['MAGICC_HOLDOUT_LEVEL']='family'
from holdout_lib import config as C

def load_script(number, label):
    path=next((ROOT/'scripts').glob(f'{number}_*.py'))
    spec=importlib.util.spec_from_file_location(label,path)
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[label]=mod
    spec.loader.exec_module(mod)
    return mod

def configure(variant='holdout',selection=False):
    # Resolve from the installed checkout; original input tables contain both
    # historical project roots, neither of which should be required by a reader.
    C.PROJECT_ROOT=ROOT;C.SPLITS_DIR=ROOT/'data/splits';C.KMER_DIR=ROOT/'data/kmer_selection'
    C.FEATURES_DIR=ROOT/'data/features'
    C.TRAIN_TSV=C.SPLITS_DIR/'train_genomes.tsv';C.VAL_TSV=C.SPLITS_DIR/'val_genomes.tsv';C.TEST_TSV=C.SPLITS_DIR/'test_genomes.tsv'
    C.V5_ONNX=ROOT/'models/magicc_v5.onnx';C.V5_NORM_PARAMS=ROOT/'data/features/normalization_params.json'
    C.SELECTED_KMERS=ROOT/'data/kmer_selection/selected_kmers.txt'
    def remap(path):
        path=str(path)
        for old in ['/home/tianrm/projects/magicc2','/media/Data_1/tianrm/projects/magicc2']:
            if path.startswith(old+'/'):return str(ROOT)+path[len(old):]
        return path
    C.remap_fasta_path=remap
    spec=json.loads((OUT/'panel_design.json').read_text())
    manifest=__import__('pandas').read_csv(OUT/'lineage_selection_manifest.tsv',sep='\t')
    C.PANEL_LEVEL='family';C.TAXON_COL='family';C.PANEL_TAXA=spec['families'] if variant=='holdout' else []
    C.PANEL_PHYLA=[];C.PANEL_PARENT_PHYLA=spec['parent_phyla']
    C.PANEL_GROUPS={r.family:{'taxa':[r.family],'phyla':[r.phylum],'parent_phylum':r.phylum} for r in manifest.itertuples()}
    C.TAXON_TO_GROUP={f:f for f in spec['families']}
    C.EVAL_DESIGN={r.family:{'n_refs':r.eval_references,'sims':10,'ref_source':'test'} for r in manifest.itertuples()}
    C.EVAL_DESIGN['in_distribution']={'n_refs':100,'sims':10,'ref_source':'test'}
    C.EVAL_GROUPS=list(spec['families']);C.CONTROL_GROUP='in_distribution'
    C.EVAL_SEED_BASE=spec['seed'];C.USE_ADAPTED_REDUCED_POOL=False
    C.HOLDOUT_DIR=OUT/variant/'data';C.HOLDOUT_DIR.mkdir(parents=True,exist_ok=True)
    C.RESULTS_DIR=OUT/('features' if selection else variant)
    C.RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    C.LOGS_DIR=OUT/'logs';C.LOGS_DIR.mkdir(parents=True,exist_ok=True)
    C.WS=f'resubmission5_{variant}'
    C.HOLDOUT_H5=C.HOLDOUT_DIR/'features.h5'
    C.HOLDOUT_NORM_PARAMS=C.HOLDOUT_DIR/'normalization_params.json'
    C.PANEL_JSON=OUT/'panel_design.json'
    C.EVAL_DIR=OUT/'evaluation'
    C.MODELS_DIR=OUT/variant/'models';C.MODELS_DIR.mkdir(parents=True,exist_ok=True)
    C.HOLDOUT_TRAIN_OUT=C.MODELS_DIR
    C.HOLDOUT_BEST_PT=C.MODELS_DIR/'best_model.pt'
    C.HOLDOUT_ONNX=C.MODELS_DIR/'model.onnx'
    C.HOLDOUT_HISTORY=C.MODELS_DIR/'training_history.json'
    if not selection:
        C.SELECTED_KMERS=OUT/'features/selected_kmers_holdout.txt'
        C.N_KMER_FEATURES=len(C.SELECTED_KMERS.read_text().splitlines())
    return C
