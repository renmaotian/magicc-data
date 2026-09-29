#!/usr/bin/env python3
"""Editable, code-native Figure2 workflow. No embedded raster artwork.

Counts and architecture are unchanged from the verified resubmission5 workflow.
Lettered ports connect non-adjacent phases without routing through text boxes.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MPath
from figstyle import (FIG_DIR, PHASE_ACCENT, PHASE_BAND, tint, readable_ink,
                      apply_style, save_fig, ARROW_INK)

def make_workflow():
    apply_style()
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['svg.hashsalt']='magicc-resubmission7-workflow'
    fig=plt.figure(figsize=(7.15,7.35))
    ax=fig.add_axes([.025,.035,.95,.945]);ax.set(xlim=(-1,101),ylim=(-5,101));ax.axis('off')
    boxes={};arrows=[];box_texts={}
    def panel(x,y,w,h,label,i):
        c=PHASE_ACCENT[i];b=PHASE_BAND[i]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.2,rounding_size=.7',facecolor=tint(c,.06),edgecolor=c,lw=.7,zorder=0))
        ax.add_patch(FancyBboxPatch((x,y+h-5.2),w,5.2,boxstyle='round,pad=.2,rounding_size=.7',facecolor=b,edgecolor=b,lw=.6,zorder=1))
        ax.text(x+w/2,y+h-2.6,label,ha='center',va='center',fontsize=8,fontweight='bold',color=readable_ink(b))
    def box(key,x,y,w,h,text,i,bold=False,fs=7.1,solid=False):
        c=PHASE_ACCENT[i];fill=PHASE_BAND[i] if solid else tint(c,.15)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.15,rounding_size=.6',facecolor=fill,edgecolor=c,lw=.8,zorder=3))
        box_texts[key]=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.2,color=readable_ink(fill) if solid else '#17212b',zorder=4)
        boxes[key]=[x,y,w,h]
    def route(key,points):
        codes=[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)
        path=MPath(points,codes)
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=8,lw=.85,color=ARROW_INK,zorder=2,shrinkA=0,shrinkB=0))
        arrows.append({'edge':key,'points':points})
    def down(a,b):
        x,y,w,h=boxes[a];xx,yy,ww,hh=boxes[b];route(a+' -> '+b,[(x+w/2,y-.2),(xx+ww/2,yy+hh+.3)])
    def port(letter,x,y,label=None):
        ax.add_patch(Circle((x,y),1.4,facecolor='white',edgecolor='#485566',lw=.85,zorder=5))
        ax.text(x,y,letter,ha='center',va='center',fontsize=7,fontweight='bold',color='#263442',zorder=6)
        if label:ax.text(x+2.2,y,label,va='center',ha='left',fontsize=7,color='#263442')
    for x,label,i in [(0,'Phase 1 · Data Curation',0),(34.5,'Phase 2 · K-mer Selection',1),(69,'Phase 3 · Training Synthesis',2)]:panel(x,49,31,51,label,i)
    box('gtdb',2,84,27,6,'GTDB\n732,475 genomes',0,True)
    box('filters',2,71,27,9,'Quality filters\nComp ≥98%; cont ≤2%\n<100 contigs; N50 >20 kbp\nLongest contig >100 kbp',0,fs=7)
    box('passed',2,62,27,5,'277,183 pass quality filters',0,True,7)
    box('split',2,51,27,7,'100,000 sampled; split first\nRetained train / val / test:\n79,948 / 10,010 / 9,999',0,True,7)
    for a,b in [('gtdb','filters'),('filters','passed'),('passed','split')]:down(a,b)
    port('A',15.5,46.7);route('split -> reference port A',[(15.5,50.8),(15.5,48.2)])
    box('trainrefs',36.5,84,27,6,'2,000 training references\n1,000 bacterial + 1,000 archaeal',1,fs=7)
    box('core',36.5,71,27,9,'Core gene identification\nProdigal + HMMER\n85 bacterial +\n128 archaeal HMMs',1)
    box('kmc',36.5,62,27,5,'9-mer counting (KMC3)',1)
    box('vocab',36.5,51,27,7,'Top 9,249 canonical k-mers\nSelected by prevalence',1,True,7)
    for a,b in [('trainrefs','core'),('core','kmc'),('kmc','vocab')]:down(a,b)
    port('A',50,93.5);route('reference port A -> trainrefs',[(50,92.1),(50,90.2)])
    port('B',50,46.7);route('vocab -> vocabulary port B',[(50,50.8),(50,48.2)])
    box('fragment',71,84,27,6,'References fragmented\nWithin each split; 4 quality tiers',2,fs=7)
    box('contam',71,71,27,9,'Contamination injection\nWithin- and cross-phylum\nDirichlet mixing',2)
    box('synthetic',71,62,27,5,'1,200,000 synthetic genomes',2,True,7)
    box('synthsplit',71,51,27,7,'1,000,000 training\n100,000 validation\n100,000 test',2,True)
    for a,b in [('fragment','contam'),('contam','synthetic'),('synthetic','synthsplit')]:down(a,b)
    port('A',84.5,93.5);route('reference port A -> fragment',[(84.5,92.1),(84.5,90.2)])
    port('C',84.5,46.7);route('synthsplit -> FASTA port C',[(84.5,50.8),(84.5,48.2)])
    panel(0,0,48,43,'Phase 4 · Feature Extraction',3)
    panel(52,0,48,43,'Phase 5 · Neural Network',4)
    box('input',3,25,13,7,'Input FASTA',3,True,7)
    box('count',22,23,23,9,'9,249 canonical 9-mers\nNumba rolling hash',3,fs=7)
    box('summary',22,11,23,7,'7 k-mer summaries',3,fs=7)
    box('norm',3,1.8,13,17,'Normalization\nZ-score\nLog\nMin-max\nRobust',3,fs=7)
    route('input -> count',[(16.2,28.5),(21.8,28.5)])
    route('count -> summary',[(33.5,22.8),(33.5,18.2)])
    route('count -> norm',[(22,25.5),(19,25.5),(19,16),(16.2,16)])
    route('summary -> norm',[(21.8,14.5),(19,14.5),(19,8),(16.2,8)])
    port('C',9.5,35.4);route('FASTA port C -> input',[(9.5,34),(9.5,32.2)])
    port('B',33.5,35.4);route('vocabulary port B -> count',[(33.5,34),(33.5,32.2)])
    # Normalized tensors leave the phase through separate, explicit ports.
    route('norm -> normalized port D',[(9.5,1.6),(9.5,-1)])
    port('D',9.5,-2.5,'Normalized k-mer counts and summaries')
    box('kbranch',56,24,18,9,'K-mer branch\n9,249 → 4,096\n→ 1,024 → 256',4,fs=7)
    box('sbranch',79,24,18,9,'Summary branch\n7 → 32 → 16',4,fs=7)
    box('fusion',61,12,31,7,'Fusion: concat(256 + 16)\n272 → 128 → 64',4,True,7.2)
    box('prediction',61,1.8,31,7,'Completeness [50–100%]\nContamination [0–100%]',4,True,7.3,True)
    port('D',76.5,35.4)
    route('normalized counts -> kbranch',[(75.0,35.4),(65,35.4),(65,33.2)])
    route('normalized summaries -> sbranch',[(78,35.4),(88,35.4),(88,33.2)])
    route('kbranch -> fusion',[(65,23.8),(65,19.2)])
    route('sbranch -> fusion',[(88,23.8),(88,19.2)])
    down('fusion','prediction')
    ax.text(63,-2.5,'Matching letters connect phases' ,ha='center',va='center',fontsize=7,color='#485566')
    out=Path(FIG_DIR);out.mkdir(parents=True,exist_ok=True)
    svg=out/'Figure_2.svg';fig.savefig(svg,format='svg',metadata={'Date':None,'Creator':'MAGICC editable workflow; Matplotlib '+matplotlib.__version__})
    fig.canvas.draw();renderer=fig.canvas.get_renderer();text_audit=[]
    for key,artist in box_texts.items():
        x,y,w,h=boxes[key];bb=artist.get_window_extent(renderer)
        lower=ax.transData.transform((x,y));upper=ax.transData.transform((x+w,y+h))
        inside=bb.x0>=lower[0] and bb.x1<=upper[0] and bb.y0>=lower[1] and bb.y1<=upper[1]
        text_audit.append({'box':key,'text':artist.get_text(),'font_pt':artist.get_fontsize(),'inside_box':bool(inside)})
    assert all(r['inside_box'] for r in text_audit),text_audit
    (out/'workflow_geometry.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'units':'axis coordinates','editable_text':True,'embedded_raster_images':0,'boxes':boxes,'box_text_checks':text_audit,'minimum_box_font_pt':min(t['font_pt'] for t in text_audit),'arrows':arrows,'ports':{'A':'Split references','B':'Selected k-mer vocabulary','C':'Synthetic FASTA assemblies','D':'Normalized k-mer counts and summary features'}},indent=2)+'\n')
    return [str(svg),*save_fig(fig,'Figure_2')]

if __name__=='__main__':make_workflow()
