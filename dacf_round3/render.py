"""Plot source time, independent change axes, evidence density, and actual additions."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import read


def render(run, review, case, out):
    folder = run/case
    initial = read(folder/'initial.json')
    clock = dict(zip(initial['source_pts'],initial['source_seconds'],strict=True))
    score = read(review/case/'score.json')
    rho = read(review/case/'density.json')
    receipts = read(folder/'receipts.json')
    arms = [('E0',None),('E1',None),('E2',None),('E3','gray'),('E4','gray'),('E5','edge'),('E6','edge')]
    fig = plt.figure(figsize=(13,11),layout='constrained')
    layout = fig.add_gridspec(6,4,height_ratios=[1.6,1,1,1.1,1,1.6])
    ax = fig.add_subplot(layout[0:2,:])
    for i,key in enumerate(arms,1):
        arm = next(r for r in score['routes'] if (r['arm'],r['channel'])==key)
        selected = arm['equal']['selected_pts']
        ax.scatter([clock[p] for p in selected],[i]*len(selected),s=28,label=None,color='#0078a4')
    ax.scatter([clock[p] for p in initial['initial_pts']],[0]*len(initial['initial_pts']),marker='|',color='#46380c',s=90)
    for role in score['registered_roles']:
        a,b = [clock[p] for p in role['pts']]
        if not score['initial']['hits'][role['id']]:
            ax.axvspan(a-0.013,b+0.013,color='#d06942',alpha=.20)
            ax.text((a+b)/2,7.5,role['id'],fontsize=8,ha='center')
    labels = ['Initial nominations',*(f'{arm} {ch or "time"}' for arm,ch in arms)]
    ax.set_yticks(list(range(8)),labels)
    ax.set_title(f'{case}: actual selected originals by source time (equal prefix, up to 3)')
    ax.set_ylim(-.5,8); ax.grid(axis='x',alpha=.25)
    for i,ch in enumerate(('gray','hsv','edge','clip')):
        panel = fig.add_subplot(layout[2+i//2,(i%2)*2:(i%2)*2+2],sharex=ax)
        parts = {r['component'] for r in rho[ch]}
        for part in parts:
            sub = [r for r in rho[ch] if r['component']==part]
            panel.plot([clock[r['pts']] for r in sub],[r['coord'] for r in sub],color='#0078a4',lw=1.4)
            seeds = [r for r in sub if r['nomination']]
            panel.scatter([clock[r['pts']] for r in seeds],[r['coord'] for r in seeds],s=6,color='#67491c')
        panel.set_ylabel(ch+' path');panel.grid(alpha=.25)
    for i,ch in enumerate(('time','gray')):
        panel = fig.add_subplot(layout[4,i*2:i*2+2],sharex=ax)
        for scale in range(3):
            rows = [r for r in rho[ch] if r['at_three_scales'] is not None]
            panel.plot([clock[r['pts']] for r in rows],[r['at_three_scales'][scale] for r in rows],label=f'{(4,8,16)[scale]}%')
        panel.set_ylabel(f'{ch} evidence KDE');panel.set_xlabel('Original seconds')
        if panel.lines: panel.legend(loc='upper left',fontsize=7)
    seen = []
    for condition in score['routes']:
        for role in condition['equal']['new_roles']:
            hits = condition['equal']['role_pts'][role]
            if hits and hits[0] not in [p for p,_ in seen]:
                seen.append((hits[0],role))
    seen = seen[:4]
    for p in next(r for r in score['routes'] if r['arm']=='E1')['equal']['selected_pts']:
        if p not in [v for v,_ in seen] and len(seen)<4:
            seen.append((p,'E1 sample'))
    for i,(p,role) in enumerate(seen):
        panel = fig.add_subplot(layout[5,i])
        with Image.open(receipts[str(p)]['path']) as image:
            image.thumbnail((450,350));panel.imshow(np.asarray(image))
        panel.set_title(f'{role} | PTS {p}\n{clock[p]:.4f} s',fontsize=8); panel.axis('off')
    fig.savefig(out/f'{case}.png',dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for key in ('run','review','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    a = parser.parse_args()
    a.out.mkdir()
    for case in ('doorway','rolling','new_laptop','new_control'):
        render(a.run,a.review,case,a.out)


if __name__=='__main__':
    main()
