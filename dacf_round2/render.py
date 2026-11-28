"""Static view of where the study would actually read original frames."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction

from dacf_research.prepare_inputs import ROOT, read
from dacf_round2.frozen_inputs import HERE


def main():
    lock=read(HERE/'lock.json')['cases']
    score={v['case']:v for v in read(HERE/'evaluation/summary.json')}
    names=['doorway','rolling','separate_fixture']
    fig,ax=plt.subplots(len(names),1,figsize=(12,6),layout='constrained')
    for a,name in zip(ax,names,strict=True):
        case=lock[name]
        m=read(ROOT/case['index']/'manifest.json')
        q=score[name]; ticks=float(Fraction(m['time_base']))
        for state,y,color in [('simple_gap_baseline',0,'#a07032'),('paired',1,'#256eac')]:
            line=[(start*ticks,(end-start)*ticks) for start,end in q[state]['union_pts']]
            a.broken_barh(line,(y+.1,.6),facecolors=color,alpha=.9)
        for row in q['role_checks']:
            if name=='doorway' and row['id']!='D06': continue
            lo,hi=row['reference_pts']; center=(lo+hi)/2*ticks
            a.plot(center,2.36,'o',color='#a62f56',ms=5)
            if lo!=hi:a.hlines(2.36,lo*ticks,hi*ticks,color='#a62f56',lw=5)
        a.set_title(f'{name.replace("_"," ")}  |  paired {100*q["paired"]["time_fraction"]:.1f}%  |  all gaps {100*q["simple_gap_baseline"]["time_fraction"]:.1f}%',loc='left')
        a.set_yticks([.4,1.4,2.36],['All catalogue gaps','Paired investigation','Reviewed target'])
        a.set_ylim(0,2.7);a.set_xlim(m['start'],m['end']);a.grid(axis='x',color='.88');a.set_xlabel('Source seconds')
    fig.savefig(HERE/'evaluation/investigation_timeline.png',dpi=145)
    plt.close(fig)

if __name__=='__main__':
    main()
