"""Current released-path MAGICC timings and matched comparator measurements."""
import os
import numpy as np
import pandas as pd
from figstyle import RESULTS, PALETTE, MARKERS, LINESTYLES, TOOL_LABEL
TOOLS = ['magicc','checkm2','cocopye','deepcheck']
def load():
    old = pd.read_csv(os.path.join(RESULTS,'speed','matched_thread_runs.tsv'),sep='\t')
    new = pd.read_csv(os.path.join(RESULTS,'speed_v3','matched_thread_runs_v3.tsv'),sep='\t')
    new['input_set'] = new.input_set_canon
    competitors = pd.concat([old,new],ignore_index=True)
    competitors = competitors[(competitors.tool.isin(TOOLS[1:])) & (competitors.cache=='warm')].copy()
    release = pd.read_csv(os.path.join(RESULTS,'speed_v033','matched_thread_runs_v033.tsv'),sep='\t')
    release = release[(release.tool=='magicc_v033') & (release.cache=='warm')].copy()
    release['tool']='magicc'
    runs = pd.concat([release,competitors],ignore_index=True)
    assert (runs.return_code==0).all()
    assert (runs.n_output_rows==runs.n_genomes).all()
    cm = pd.read_csv(os.path.join(RESULTS,'speed_v3','pooled_cell_summary.tsv'),sep='\t')
    cm = cm[cm.tool.isin(TOOLS[1:]) & (cm.cache=='warm')].copy()
    cm['input_set'] = cm.input_set_canon
    mm = pd.read_csv(os.path.join(RESULTS,'speed_v033','cell_summary_v033.tsv'),sep='\t')
    mm = mm[mm.tool=='magicc_v033'].copy()
    mm['tool']='magicc'
    summary = pd.concat([mm,cm],ignore_index=True)
    for r in summary.itertuples():
        v = runs[(runs.tool==r.tool)&(runs.threads==r.threads)&(runs.input_set==r.input_set)]
        assert len(v)==r.n, (r.tool,r.threads,r.input_set,len(v),r.n)
        assert abs(v.wall_clock_s.median()-r.wall_median_s)<.00011
    return runs,summary

def draw(ax,input_set='set_E_full'):
    runs,summary=load()
    for tool in TOOLS:
        s=summary[(summary.tool==tool)&(summary.input_set==input_set)].sort_values('threads')
        ax.plot(s.threads,s.wall_median_s,color=PALETTE[tool],marker=MARKERS[tool],lw=1,ms=3.6,ls=LINESTYLES[tool],label=TOOL_LABEL[tool])
        for row in s.itertuples():
            v=runs[(runs.tool==tool)&(runs.input_set==input_set)&(runs.threads==row.threads)]
            offsets=np.linspace(-.04,.04,len(v))
            ax.scatter(row.threads*(1+offsets),v.wall_clock_s,s=8,marker=MARKERS[tool],facecolor='white',edgecolor=PALETTE[tool],linewidth=.55,zorder=4)
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set_xticks([1,8,16,32],['1','8','16','32']); ax.set_xlim(.85,38)
    ax.set_xlabel('Threads'); ax.set_ylabel('Wall clock (s, log scale)')
    ax.text(.02,.985,'DeepCheck: inference only\n(CheckM2 features required)',transform=ax.transAxes,fontsize=5.2,ha='left',va='top',color='.3')
    ax.set_title('Matched hardware: '+('1,000' if input_set=='set_E_full' else '100')+' genomes',fontsize=6.6,pad=3)
    return runs,summary
