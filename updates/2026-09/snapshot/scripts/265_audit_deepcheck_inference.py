#!/usr/bin/env python3
"""Bounded, read-only DeepCheck audit: source, all benchmark schemas, eight probes.

Never overwrites benchmark predictions. The unmodified upstream ResNet's fc2
forward hook exposes its already-computed contamination tensor, independently
checking the local dual-return compatibility implementation.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
DC=ROOT/'tools/DeepCheck'
OUT=ROOT/'results/revision/holdout_resubmission5/deepcheck_audit'
PRIMARY_SETS=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E']
SETS=PRIMARY_SETS+['set_F','set_G','motivating_v2/set_A','motivating_v2/set_B','motivating_v2/set_C']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    commit=subprocess.check_output(['git','-C',str(DC),'rev-parse','HEAD'],text=True).strip()
    tracked=['model.py','process_data.py','train.py','multi_train.py','README.md','scaler.sav','feature_names.pkl',
             'evaluating macro genome assemblies!.ipynb','models/best_model.pt']
    for name in tracked:
        archived=subprocess.check_output(['git','-C',str(DC),'show',f'HEAD:{name}'])
        if name=='models/best_model.pt' and archived.startswith(b'version https://git-lfs.github.com/spec/v1'):
            expected=[x.split()[-1].split(':')[-1] for x in archived.decode().splitlines() if x.startswith('oid sha256:')][0]
            assert sha(DC/name)==expected
        else:assert hashlib.sha256(archived).hexdigest()==sha(DC/name),name
    with open(DC/'scaler.sav','rb') as f:scaler=pickle.load(f)
    with open(DC/'feature_names.pkl','rb') as f:names=pickle.load(f)
    params=np.load(DC/'scaler_params.npz')
    assert len(names)==21241 and np.array_equal(params['scale'],scaler.scale_) and np.array_equal(params['min_val'],scaler.min_)
    # sklearn 0.23.2 predates the clip option. Its affine behavior corresponds
    # to clip=False under the locally installed sklearn 1.3.1.
    old_clip_present=hasattr(scaler,'clip');compatible_scaler=copy.copy(scaler)
    if not old_clip_present:compatible_scaler.clip=False
    predictions={s:pd.read_csv(ROOT/f'data/benchmarks/{s}/deepcheck_predictions.tsv',sep='\t') for s in SETS}
    primary=pd.concat([d.assign(benchmark=s) for s,d in predictions.items() if s in PRIMARY_SETS],ignore_index=True)
    primary=primary[(primary.true_completeness>=50)&(primary.true_completeness<=100)
                    &(primary.true_contamination>=0)&(primary.true_contamination<=primary.true_completeness+1e-6)]
    probes=[]
    for label,frame,column,maximum in [
        ('primary_contamination_max',primary,'pred_contamination',True),
        ('primary_contamination_min',primary,'pred_contamination',False),
        ('all_E_contamination_max',predictions['set_E'].assign(benchmark='set_E'),'pred_contamination',True),
        ('all_E_contamination_min',predictions['set_E'].assign(benchmark='set_E'),'pred_contamination',False),
        ('B_completeness_max',predictions['set_B_v2'].assign(benchmark='set_B_v2'),'pred_completeness',True),
        ('G_contamination_max',predictions['set_G'].assign(benchmark='set_G'),'pred_contamination',True),
        ('motivating_C_contamination_max',predictions['motivating_v2/set_C'].assign(benchmark='motivating_v2/set_C'),'pred_contamination',True),
        ('D_clean_contamination_min',predictions['set_D_clean'].assign(benchmark='set_D_clean'),'pred_contamination',False)]:
        row=frame.loc[frame[column].idxmax() if maximum else frame[column].idxmin()]
        probes.append(dict(label=label,benchmark=row.benchmark,genome_id=row.genome_id,
                           archived_comp=float(row.pred_completeness),archived_cont=float(row.pred_contamination)))
    wanted={(p['benchmark'],p['genome_id']) for p in probes};feature_rows={};schemas=[]
    for s in SETS:
        for path in sorted((ROOT/f'data/benchmarks/{s}/checkm2_output').glob('*.pkl')):
            frame=pd.read_pickle(path)
            assert frame.columns[0]=='Name' and list(frame.columns[1:])==names,path
            for _,row in frame[frame.Name.isin([g for group,g in wanted if group==s])].iterrows():
                key=(s,row.Name);assert key not in feature_rows
                feature_rows[key]=row.iloc[1:].to_numpy(dtype=np.float64)
            schemas.append({'benchmark':s,'path':str(path.relative_to(ROOT)),'rows':len(frame),'features':len(frame.columns)-1,
                            'all_feature_names_and_order_match':True,'sha256':sha(path)})
    assert wanted==set(feature_rows)
    raw=np.stack([feature_rows[(p['benchmark'],p['genome_id'])] for p in probes])
    manual=raw*params['scale']+params['min_val'];official_scaling=compatible_scaler.transform(raw)
    assert np.array_equal(manual,official_scaling)
    padded=np.zeros((len(raw),20164),dtype=np.float32);padded[:,:20021]=manual[:,:20021]
    inputs=torch.from_numpy(padded).reshape(-1,1,142,142)
    upstream=module(DC/'model.py','deepcheck_upstream_audit')
    project=module(ROOT/'scripts/38_run_deepcheck_v2.py','deepcheck_project_audit')
    state=torch.load(DC/'models/best_model.pt',map_location='cpu',weights_only=True)
    native=upstream.ResNet(upstream.ResidualBlock,[2,2,2,2]);native.load_state_dict(state);native.eval()
    wrapper=project.ResNetDualOutput(project.ResidualBlock,[2,2,2,2]);wrapper.load_state_dict(state);wrapper.eval()
    # Import only the timing script's class definitions: importing the full
    # script would execute its argparse CLI. No training/inference data changes.
    timing_path=ROOT/'scripts/163_ws8_deepcheck_infer.py'
    timing_tree=ast.parse(timing_path.read_text())
    timing_classes=ast.Module(body=[n for n in timing_tree.body if isinstance(n,ast.ClassDef)],type_ignores=[])
    timing_ns={'torch':torch,'nn':torch.nn,'Dataset':torch.utils.data.Dataset}
    exec(compile(timing_classes,str(timing_path),'exec'),timing_ns)
    timing=timing_ns['ResNetDualOutput'](timing_ns['ResidualBlock'],[2,2,2,2]);timing.load_state_dict(state);timing.eval()
    contamination=[];hook=native.fc2.register_forward_hook(lambda module,args,result:contamination.append(result.detach().clone()))
    with torch.inference_mode():
        nc=native(inputs);nt=contamination[-1];wc,wt=wrapper(inputs);tc,tt=timing(inputs)
    hook.remove()
    assert torch.equal(nc,wc) and torch.equal(nt,wt)
    assert torch.equal(nc,tc) and torch.equal(nt,tt)
    for i,p in enumerate(probes):
        p.update(upstream_comp=float(nc[i,0]*100),upstream_cont=float(nt[i,0]*100),
                 project_comp=float(wc[i,0]*100),project_cont=float(wt[i,0]*100))
        p['comp_archive_difference']=p['upstream_comp']-p['archived_comp'];p['cont_archive_difference']=p['upstream_cont']-p['archived_cont']
    maxdiff=max(max(abs(p['comp_archive_difference']),abs(p['cont_archive_difference'])) for p in probes)
    assert maxdiff<.001, f'Archived-extremum replay mismatch: {maxdiff} pp'
    pd.DataFrame(schemas).to_csv(OUT/'feature_schema_audit.tsv',sep='\t',index=False)
    pd.DataFrame(probes).to_csv(OUT/'bounded_extreme_replay.tsv',sep='\t',index=False)
    inputs_to_hash=[DC/name for name in tracked]+[DC/'scaler_params.npz',ROOT/'scripts/38_run_deepcheck_v2.py',
        ROOT/'scripts/91_parse_competitor_clean_cd.py',ROOT/'scripts/142_collect_realdata_predictions.py',
        ROOT/'scripts/161_ws8_run_one.sh',timing_path,Path(__file__)]
    inputs_to_hash.extend(ROOT/f'data/benchmarks/{s}/deepcheck_predictions.tsv' for s in SETS)
    pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in inputs_to_hash]).to_csv(OUT/'provenance.tsv',sep='\t',index=False)
    summary={'status':'BOUNDED_AUDIT_PASSED','upstream_commit':commit,'n_benchmark_sets':len(SETS),'n_feature_pickle_files':len(schemas),
             'n_feature_rows_schema_checked':sum(r['rows'] for r in schemas),'n_inference_probe_rows':len(probes),
             'scaler_scale_and_min_exact_match':True,'manual_scaling_exactly_equals_original_affine_transform':True,
             'original_scaler_had_clip_attribute':old_clip_present,'upstream_and_project_heads_bitwise_equal':True,
             'upstream_and_timing_wrapper_heads_bitwise_equal':True,
             'head_assignment_evidence':{'upstream_model':'model.py:110-112 names fc1 x_comp and fc2 x_cont; both have checkpoint tensors',
                  'upstream_training':'train.py:34-35 scales completeness/contamination labels by 100; 129-134 assigns first/second returned heads to corresponding losses',
                  'upstream_notebook':'evaluating macro genome assemblies!.ipynb expects outputs_comp, outputs_cont and multiplies each by 100',
                  'checkpoint_loaded_strictly':True,
                  'fc2_weight_sha256':hashlib.sha256(state['fc2.weight'].numpy().tobytes()).hexdigest(),
                  'fc2_bias_sha256':hashlib.sha256(state['fc2.bias'].numpy().tobytes()).hexdigest(),
                  'limit':'Source/checkpoint compatibility and numerical replay establish the intended published head; the original training execution history is not independently reconstructed.'},
             'maximum_absolute_archive_replay_difference_pp':maxdiff,
             'output_processing':'Unbounded linear fc1 completeness/fc2 contamination heads; multiply by 100, with no clipping or further inverse scaling.',
             'compatibility_adaptations':'Expose already-computed fc2 return; retain all 21241 named CheckM2 features for scaler (official read_feature incorrectly drops two nonexistent label columns); use exact affine scaler parameters to avoid old sklearn pickle compatibility.',
             'full_inference_rerun':False,'benchmark_prediction_files_modified':False}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(pd.DataFrame(probes).to_string(index=False))

if __name__=='__main__':main()
