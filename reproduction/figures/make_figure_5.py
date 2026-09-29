#!/usr/bin/env python3
"""Figure 5: empirical validation, sequence-error robustness and runtime."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import panels_realdata as pr
import panels_timing_current as pt
import figvalues
import pandas as pd
import numpy as np
from figstyle import FIG_DIR, RESULTS, PALETTE, MARKERS, TOOL_LABEL, add_panel_label, apply_style, save_fig

def draw_mixed(fig,spec):
    df=pd.read_csv(os.path.join(RESULTS,'cami2','analysis','cami2_accuracy_by_cohort.tsv'),sep='\t')
    df=df[(df.binset=='mixed') & (df.cohort=='primary_in_domain_scoreable_LEAKAGE_FREE')]
    gs=spec.subgridspec(1,2,wspace=.13)
    ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1],sharey=ax1)
    for ax,dataset,title in [(ax1,'strain_madness','CAMI II strain-madness'),(ax2,'marine','CAMI II marine')]:
        d=df[df.dataset==dataset]
        for i,(tool,label) in enumerate([('magicc','MAGICC_V5'),('checkm2','CheckM2'),('cocopye','CoCoPyE'),('deepcheck','DeepCheck')]):
            for metric,offset,filled in [('completeness',-.14,True),('contamination',.14,False)]:
                r=d[(d.tool==label)&(d.metric==metric)].iloc[0]
                ax.errorbar(r.mae,i+offset,xerr=[[r.mae-r.mae_lo],[r.mae_hi-r.mae]],
                    color=PALETTE[tool],marker=MARKERS[tool],mfc=PALETTE[tool] if filled else 'white',ms=3.6,lw=.7,ls='none',capsize=1.2)
        ax.set_title(title+'\nn = '+str(int(d.n.iloc[0]))+' bins; '+str(int(d.n_clusters.iloc[0]))+' sources',fontsize=5.6,pad=4)
        ax.set_ylim(3.8,-.7);ax.set_xlim(-.5,22);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('MAE (pp)')
    ax1.set_yticks(range(4),[TOOL_LABEL[t] for t in ['magicc','checkm2','cocopye','deepcheck']],fontsize=5.8)
    ax2.tick_params(axis='y',left=False,labelleft=False)
    return ax1,df

def main():
    apply_style()
    d6,d7=pr.load6(),pr.load7()
    fig=plt.figure(figsize=(7.1,8.1))
    gs=fig.add_gridspec(3,2,left=.08,right=.985,top=.94,bottom=.055,hspace=.72,wspace=.36)
    ax=fig.add_subplot(gs[0,0]);pr.draw_fragmentation(ax,d6)
    add_panel_label(ax,'a',x=-.17,y=1.19)
    a,b=pr.draw_known_composition(fig,gs[0,1],d6)
    add_panel_label(a,'b',x=-.42,y=1.19)
    ax=fig.add_subplot(gs[1,0]);pr._dose_panel(ax,d6["sub"],"Substitution rate (% of bases)",5.25,[],title="Set G: uniform substitutions");ax.set_ylim(0,40)
    add_panel_label(ax,'c',x=-.17,y=1.19)
    ax=fig.add_subplot(gs[1,1]);pr._dose_panel(ax,d6["ind"],"Indel rate (% of bases)",5.25,[],title="Set G: indels");ax.set_ylim(0,62)
    add_panel_label(ax,'d',x=-.17,y=1.19)
    ax=fig.add_subplot(gs[2,0]);runs,summ=pt.draw(ax)
    add_panel_label(ax,'e',x=-.17,y=1.19)
    ax,mixed=draw_mixed(fig,gs[2,1])
    add_panel_label(ax,'f',x=-.40,y=1.19)
    fig.legend(handles=pr.tool_legend_handles(),loc='upper center',bbox_to_anchor=(.5,1),ncol=4,fontsize=6.4,handletextpad=.35,columnspacing=1.4)
    save_fig(fig,'Figure_5',out_dir=FIG_DIR)
    pr.verification_report()
    rows=[]
    for key,value in [('empirical_and_robustness',d6),('reduced_genome_anchor',d7),('timing_runs',runs),('timing_summary',summ),('cami_mixed',mixed)]:
        figvalues.flatten(key,value,rows)
    figvalues.write_values(FIG_DIR,'fig5',rows)
    print('Figure 5 current-layout build complete')
if __name__=='__main__':
    main()
