#!/usr/bin/env python3
"""Measure full-width MAGICC CPU training steps; never generates accuracy claims."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from holdout_lib.model import MAGICCModel

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=12);ap.add_argument('--steps',type=int,default=6)
    ap.add_argument('--affinity',help='Comma-separated physical CPU IDs; applied before model allocation.')
    ap.add_argument('--output-label',help='Optional unique benchmark filename suffix.')
    args=ap.parse_args()
    if args.affinity:os.sched_setaffinity(0,{int(x) for x in args.affinity.split(',')})
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    m=MAGICCModel(use_gradient_checkpointing=False)
    o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005)
    x=torch.randn(512,9249);a=torch.randn(512,7);y=torch.rand(512,2)*50+50
    elapsed=[]
    for i in range(args.steps+1):
        t=time.perf_counter();o.zero_grad(set_to_none=True);p=m(x,a)
        loss=2*((p[:,0]-y[:,0])**2).mean()+((p[:,1]-y[:,1])**2).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
        if i:elapsed.append(time.perf_counter()-t)
    step=sum(elapsed)/len(elapsed)
    out={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'threads':args.threads,'batch':512,
         'steps':args.steps,'affinity':sorted(os.sched_getaffinity(0)),'step_seconds':elapsed,
         'seconds_per_training_step':step,'estimated_training_minutes_per_million_epoch':step*(1000000//512)/60,
         'estimated_90_epoch_training_hours':step*(1000000//512)*90/3600,
         'note':'Estimate excludes validation, loading, augmentation and checkpoint I/O. FP32 CPU, same full-width architecture and optimizer.'}
    dest=Path(__file__).resolve().parents[1]/'results/revision/holdout_resubmission5'
    dest.mkdir(parents=True,exist_ok=True);(dest/f'cpu_benchmark_{args.output_label or args.threads}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
