from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from dacf_research.prepare_inputs import read
from dacf_round4.offline import HERE, R3


def main():
    out=HERE/'figures'; out.mkdir()
    audit=read(HERE/'offline/rolling/initial_stops.json')['edge']
    v=read(HERE/'evaluation/rolling.json')
    routes=[r for r in v['routes'] if r['axis']=='edge' and r['strategy'] in {'S0','S1'}]
    fig, axs=plt.subplots(3,1,figsize=(11,8),layout='constrained',gridspec_kw={'height_ratios':[1.5,1,2.4]})
    axis=axs[0]
    for row in audit['candidates']:
        seconds=row['pts']/600
        axis.scatter(seconds,row['gap'],s=70 if row['pts']==2700 else 30, c='#aa5836' if row['local']['eligible'] else '#8798a3')
    axis.axhline(audit['global_limit'],ls='--',color='#2b6a86',label='Global stop limit')
    axis.set(xlim=(-.05,5.6),ylabel='Edge distance',title='Rolling clip | candidate gaps and global stop');axis.legend()
    axis=axs[1]
    first=read(R3/'run/rolling/initial.json')['initial_pts']
    axis.scatter([p/600 for p in first],[0]*len(first),marker='|',s=120,color='#4a5965')
    for i,r in enumerate(routes,1):
        axis.scatter([p/600 for p in r['natural']['points']],[i]*len(r['natural']['points']),s=70,color='#aa5836')
    axis.set(xlim=(-.05,5.6),yticks=[0,1,2],yticklabels=['Initial nominees','S0 natural','S1 natural'],xlabel='Original seconds')
    axis=axs[2]
    actual=read(HERE/'offline/rolling/receipts.json')['2700']
    with Image.open(actual['path']) as source:
        axis.imshow(source)
    axis.set_title('Exact supplementary original | PTS 2700 / 4.500 s');axis.axis('off')
    fig.savefig(out/'rolling_local_stop.png',dpi=145);plt.close(fig)
    rows=read(HERE/'scale_diagnostics/summary.json')['cases']
    fig,axs=plt.subplots(2,2,figsize=(10.8,6.4),layout='constrained')
    for ax,(case,items) in zip(axs.flat,rows.items()):
        x=[r['maximum_gap_seconds'] for r in items]
        ax.plot(x,[r['sum_delta']['edge'] for r in items],'o-',label='Edge path increment')
        ax.plot(x,[r['sum_delta']['gray'] for r in items],'s-',label='Gray path increment')
        ax.invert_xaxis();ax.set(title=case,xlabel='Maximum remaining gap (s)',ylabel='Per-level path increment');ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=8);fig.savefig(out/'path_scales.png',dpi=145);plt.close(fig)


if __name__=='__main__':main()
