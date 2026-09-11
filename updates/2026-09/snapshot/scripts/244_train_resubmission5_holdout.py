#!/usr/bin/env python3
"""Full-width V5-recipe CPU training; seeded, atomic and epoch-resumable.

Both matched arms have identical architecture/features/seed/optimizer settings.
CPU FP32 replaces unavailable CUDA FP16. Vectorized augmentation has the original
independent 2% masking and N(0, .01) noise distributions. No data/epoch shortening
is used in a full run. All RNG/optimizer/scheduler states survive an interruption.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from resubmission5_holdout_config import OUT,configure
from holdout_lib.model import MAGICCModel

def atomic_save(path,obj):
    tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['holdout','matched_full'],required=True)
    ap.add_argument('--threads',type=int,default=12);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--max-epochs',type=int,default=150)
    args=ap.parse_args();c=configure(args.variant)
    affinity_path=OUT/'runtime_affinity.json'
    affinity_note='Default operating-system affinity.'
    if not args.smoke and affinity_path.exists():
        assignment=json.loads(affinity_path.read_text())
        cpus=set(assignment['arms'][args.variant])
        if cpus <= set(os.sched_getaffinity(0)):
            os.sched_setaffinity(0,cpus)
            affinity_note='Per-arm socket affinity applied before data/model allocation after paired runtime benchmark.'
        else:
            affinity_note='Archived host CPU assignment unavailable; kept operating-system affinity.'
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    h5=c.HOLDOUT_DIR/('smoke_features.h5' if args.smoke else 'features.h5')
    out=c.MODELS_DIR/('smoke' if args.smoke else 'full');out.mkdir(parents=True,exist_ok=True)
    norm=c.HOLDOUT_DIR/('smoke_normalization_params.json' if args.smoke else 'normalization_params.json')
    manifest=c.HOLDOUT_DIR/('smoke_build_manifest.json' if args.smoke else 'build_manifest.json')
    assert manifest.exists()
    datahash=json.loads(manifest.read_text())['features_h5_sha256']
    assert sha(h5)==datahash, 'Training HDF5 bytes differ from the completed build manifest.'
    with h5py.File(h5,'r') as f:
        assert f.attrs['normalized'] and json.loads(f.attrs['panel_taxa'])==c.PANEL_TAXA
        assert f.attrs['features_sha256']==sha(c.SELECTED_KMERS)
        train={k:torch.from_numpy(f['train'][k][:]) for k in ('kmer_features','assembly_features','labels')}
        val={k:torch.from_numpy(f['val'][k][:]) for k in ('kmer_features','assembly_features','labels')}
    n=len(train['labels']);nv=len(val['labels'])
    if not args.smoke:assert (n,nv)==(1000000,100000)
    batch=512 if not args.smoke else 50
    maxepochs=2 if args.smoke else args.max_epochs
    model=MAGICCModel(n_kmer_features=c.N_KMER_FEATURES,n_assembly_features=7,use_gradient_checkpointing=False)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2)
    start=0;best=math.inf;bestepoch=0;bad=0;history=[]
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['data_sha256']==datahash and ck['feature_sha256']==sha(c.SELECTED_KMERS)
        assert ck['config']['normalization_sha256']==sha(norm)
        for key,value in {'seed':args.seed,'threads':args.threads,'batch_size':batch,'max_epochs':maxepochs,
                          'learning_rate':.001,'weight_decay':.0005,'loss_weights':[2.,1.],
                          'mask_rate':.02,'noise_std':.01,'gradient_clip_norm':1.,'patience':20}.items():
            assert ck['config'][key]==value, f'Resume configuration mismatch for {key}'
        model.load_state_dict(ck['model_state_dict']);opt.load_state_dict(ck['optimizer_state_dict']);scheduler.load_state_dict(ck['scheduler_state_dict'])
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng'])
        start=ck['epoch'];best=ck['best_val_loss'];bestepoch=ck['best_epoch'];bad=ck['epochs_without_improvement'];history=ck['history']
        print(f'Resuming {args.variant} after epoch {start}; best={bestepoch} loss={best:.6f}',flush=True)
    config={'variant':args.variant,'n_parameters':sum(p.numel() for p in model.parameters()),'n_features':c.N_KMER_FEATURES,
            'seed':args.seed,'precision':'CPU FP32','threads':args.threads,'batch_size':batch,'max_epochs':maxepochs,'patience':20,
            'learning_rate':.001,'weight_decay':.0005,'loss_weights':[2.,1.],'mask_rate':.02,'noise_std':.01,
            'scheduler':'CosineAnnealingWarmRestarts T_0=10 T_mult=2','gradient_clip_norm':1.,'training_samples':n,'validation_samples':nv,
            'data_sha256':datahash,'feature_sha256':sha(c.SELECTED_KMERS),'normalization_sha256':sha(norm),
            'data_integrity_check':'Full HDF5 SHA256 verified against the completed build manifest at trainer entry.',
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'affinity_note':affinity_note,
            'training_script_sha256':sha(__file__),
            'implementation_notes':'Same hidden layers and output bounds as V5; 9243 rather than 9249 inputs from actual panel-free reselection. Gradient checkpointing disabled as CPU memory ample; no effect on architecture.',
            'augmentation_notes':'Seeded vectorized Torch RNG with original independent masking/noise distributions; all RNG states checkpointed.'}
    atomic_json(out/'training_config.json',config)
    print(json.dumps(config,indent=2),flush=True)
    started=time.time()
    for epoch in range(start,start if bad>=20 else maxepochs):
        t0=time.time();model.train();order=torch.randperm(n);steps=n//batch;total=0.
        for step in range(steps):
            idx=order[step*batch:(step+1)*batch]
            x=train['kmer_features'][idx].clone();a=train['assembly_features'][idx].clone();y=train['labels'][idx]
            x.masked_fill_(torch.rand_like(x)<.02,0.);x.add_(torch.randn_like(x),alpha=.01);a.add_(torch.randn_like(a),alpha=.01)
            opt.zero_grad(set_to_none=True);pred=model(x,a)
            loss=2*((pred[:,0]-y[:,0])**2).mean()+((pred[:,1]-y[:,1])**2).mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch={epoch+1} step={step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            scheduler.step(epoch+step/steps);total+=float(loss.detach())*batch
        model.eval();preds=[]
        with torch.inference_mode():
            for s in range(0,nv,1024):preds.append(model(val['kmer_features'][s:s+1024],val['assembly_features'][s:s+1024]))
        vp=torch.cat(preds).numpy();truth=val['labels'].numpy();error=vp-truth
        mse=(error.astype(np.float64)**2).mean(axis=0);mae=np.abs(error).mean(axis=0);vloss=float(2*mse[0]+mse[1])
        improved=vloss<best
        if improved:best=vloss;bestepoch=epoch+1;bad=0
        else:bad+=1
        record={'epoch':epoch+1,'train_loss':total/(steps*batch),'val_loss':vloss,'val_comp_mae':float(mae[0]),'val_cont_mae':float(mae[1]),
                'val_comp_r2':float(1-mse[0]/truth[:,0].var()),'val_cont_r2':float(1-mse[1]/truth[:,1].var()),
                'seconds':time.time()-t0,'best_epoch':bestepoch,'epochs_without_improvement':bad}
        history.append(record)
        state={'epoch':epoch+1,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
               'scheduler_state_dict':scheduler.state_dict(),'best_val_loss':best,'best_epoch':bestepoch,
               'epochs_without_improvement':bad,'history':history,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
               'data_sha256':datahash,'feature_sha256':config['feature_sha256'],'config':config}
        atomic_save(checkpoint,state)
        if improved:atomic_save(out/'best_model.pt',state)
        atomic_json(out/'training_history.json',history)
        print(json.dumps(record),flush=True)
        if bad>=20:break
    if history and (history[-1]['epoch']>=maxepochs or bad>=20):
        beststate=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(beststate['model_state_dict']);model.eval()
        opath=out/'model.onnx';temporary=out/'model.onnx.tmp'
        torch.onnx.export(model,(torch.randn(1,c.N_KMER_FEATURES),torch.randn(1,7)),str(temporary),opset_version=17,
                          input_names=['kmer_features','assembly_features'],output_names=['predictions'],
                          dynamic_axes={'kmer_features':{0:'batch_size'},'assembly_features':{0:'batch_size'},'predictions':{0:'batch_size'}})
        os.replace(temporary,opath)
        import onnx,onnxruntime as ort
        onnx.checker.check_model(str(opath));so=ort.SessionOptions();so.intra_op_num_threads=args.threads;so.inter_op_num_threads=1
        session=ort.InferenceSession(str(opath),sess_options=so,providers=['CPUExecutionProvider'])
        vk=val['kmer_features'][:min(512,nv)];va=val['assembly_features'][:min(512,nv)]
        with torch.inference_mode():expected=model(vk,va).numpy()
        observed=session.run(None,{'kmer_features':vk.numpy(),'assembly_features':va.numpy()})[0]
        maxdiff=float(np.max(np.abs(expected-observed)));assert maxdiff<1e-4,maxdiff
        report={'status':'TRAINING_COMPLETE','best_epoch':bestepoch,'best_validation_loss':best,'epochs_completed':history[-1]['epoch'],
                'wall_hours_this_invocation':(time.time()-started)/3600,'onnx_sha256':sha(opath),'pytorch_onnx_max_difference':maxdiff,
                'best_model_sha256':sha(out/'best_model.pt'),'training_config':config}
        atomic_json(out/'completion.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
