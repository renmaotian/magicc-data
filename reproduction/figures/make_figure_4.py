#!/usr/bin/env python3
"""Figure4: independently trained ten-genus panel and contaminant relatedness.

Refuses to render incomplete/smoke experiments. All ten selected genera enter
both DiD panels; pooled reference distributions retain available-reference
weighting and are explicitly not an equal-genus summary. Old lineage studies
remain supplementary context, never substituted for the new panel.
"""
import os,sys,json
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import panels_taxonomy as pt
import figvalues
from figstyle import RESULTS,FIG_DIR,PALETTE,add_panel_label,apply_style,save_fig,DENOM_NOTE
from figledger import LEDGER,report_verification
H=Path(RESULTS)/'holdout_resubmission7'
STYLES={'MAGICC_matched_full':('Matched full','.45','D'),'MAGICC_holdout':('Genus holdout',PALETTE['holdout'],'s'),'MAGICC_V5':('Production MAGICC',PALETTE['magicc'],'o')}

def load():
    summary=json.loads((H/'evaluation_summary.json').read_text())
    assert summary['status']=='EVALUATION_COMPLETE', 'Only completed full training may appear in Figure4'
    assert summary['n_panel_groups']==10
    panel=pd.read_csv(H/'lineage_selection_manifest.tsv',sep='\t').sort_values('selection_order')
    did=pd.read_csv(H/'did.tsv',sep='\t')
    refs=pd.read_csv(H/'per_reference_errors.tsv',sep='\t')
    head=pd.read_csv(H/'head_to_head_by_group.tsv',sep='\t')
    order=panel.genus.tolist()
    # Config group names may be direct genus strings or carry a genus_ prefix.
    def identify(group):
        candidates=[f for f in order if str(group)==f or str(group).endswith('_'+f)]
        assert len(candidates)==1,(group,candidates)
        return candidates[0]
    did['genus']=did.group.map(identify)
    refs=refs[refs.group!='in_distribution'].copy();refs['genus']=refs.group.map(identify)
    assert len(did)==20 and set(did.genus)==set(order)
    assert (did.groupby('genus').metric.nunique()==2).all()
    assert refs.groupby('reference').genus.nunique().max()==1
    assert set(refs.tool)==set(STYLES)
    assert refs.groupby(['reference','metric']).tool.nunique().min()==3
    for arm in ['holdout','matched_full']:
        done=json.loads((H/arm/'models/full/completion.json').read_text())
        assert done['status']=='TRAINING_COMPLETE' and done['training_config']['training_samples']==1000000
    return summary,panel,did,refs,head,order

def draw_did(ax,did,order,metric,show_labels=True):
    d=did[did.metric==metric].set_index('genus').loc[order]
    ax.axvline(0,color='.5',ls='--',lw=.7)
    ax.errorbar(d.did,np.arange(10),xerr=[d.did-d.ci_low,d.ci_high-d.did],
                color=PALETTE['holdout'],marker='s',ms=3.8,ls='none',lw=.8,capsize=1.8)
    ax.set_yticks(range(10),order if show_labels else ['']*10,fontsize=5.9)
    ax.set_ylim(9.6,-.6)
    ax.set_xlabel('Adjusted difference in MAE (pp)',fontsize=6.4)
    ax.set_title(('Completeness' if metric=='comp' else 'Contamination')+': genus exclusion',fontsize=7,pad=5)
    return d

def draw_pooled(ax,refs):
    handles=[]
    upper_whiskers=[]
    for j,(tool,(label,color,marker)) in enumerate(STYLES.items()):
        pos=np.array([0,1])+(j-1)*.24
        arrays=[refs[(refs.tool==tool)&(refs.metric==m)].mae.to_numpy() for m in ['comp','cont']]
        upper_whiskers.extend(np.quantile(a,.95) for a in arrays)
        box=ax.boxplot(arrays,positions=pos,widths=.19,whis=(5,95),showfliers=False,patch_artist=True,
            medianprops={'color':color,'lw':1.1},boxprops={'edgecolor':color,'facecolor':color,'alpha':.17},
            whiskerprops={'color':color,'lw':.65},capprops={'color':color,'lw':.65})
        ax.scatter(pos,[np.mean(a) for a in arrays],marker=marker,s=16,color=color,zorder=4)
        handles.append(Line2D([],[],marker=marker,color=color,ls='none',ms=3.6,label=label))
    ax.set_xticks([0,1],['Completeness','Contamination'],fontsize=6)
    ax.set_xlim(-.55,1.55);ax.set_ylim(0,max(upper_whiskers)*1.3)
    ax.set_ylabel('Per-reference mean absolute error (pp)',fontsize=6.4)
    ax.set_title('Pooled target references',fontsize=7,pad=5)
    ax.legend(handles=handles,fontsize=5.4,loc='upper left',frameon=False)


def main():
    apply_style();summary,panel,did,refs,head,order=load()
    # Reduce blank space between rows while preserving the panels' physical
    # height and all font sizes; the complete journal legend then fits below.
    fig=plt.figure(figsize=(7.1,8.3))
    gs=fig.add_gridspec(3,2,height_ratios=[1.2,1,.68],left=.16,right=.985,top=.965,bottom=.07,hspace=.40,wspace=.53)
    a=fig.add_subplot(gs[0,0]);draw_did(a,did,order,'comp')
    b=fig.add_subplot(gs[0,1]);draw_did(b,did,order,'cont',False)
    c=fig.add_subplot(gs[1,0]);draw_pooled(c,refs)
    d=fig.add_subplot(gs[1,1]);dd=pt.draw_setF_bias_by_distance(d,fig='4',panel='d')
    e=fig.add_subplot(gs[2,:]);ee=pt.draw_cami_paired_hl(e,fig='4',panel='e')
    for ax,label,x in [(a,'a',-.42),(b,'b',-.18),(c,'c',-.42),(d,'d',-.24),(e,'e',-.14)]:add_panel_label(ax,label,x=x,y=1.12)
    save_fig(fig,'Figure_4',out_dir=FIG_DIR);report_verification('Figure4')
    records=[]
    for key,value in [('panel',panel),('did',did),('per_reference',refs),('head_to_head',head)]:figvalues.flatten(key,value,records)
    for row in LEDGER.rows:records.append((f"{row['figure']}|{row['panel']}|{row['what']}",row['value']))
    figvalues.write_values(FIG_DIR,'fig4',records)
    print('Completed ten-genus Figure4 build')
if __name__=='__main__':main()
