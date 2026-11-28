"""Separately priced nested source probes, used only after the offline predicts."""
from itertools import pairwise
from pathlib import Path

import cv2

from dacf_research.field import METRICS, clip_encoder
from dacf_research.prepare_inputs import ROOT, read, write
from dacf_round3.run import Experiment, nearest
from dacf_round4.offline import HERE, verify


def main():
    lock=read(HERE/'lock.json'); verify(lock)
    out=HERE/'scale_diagnostics'; out.mkdir()
    cv2.setNumThreads(1)
    clip=clip_encoder()
    scopes={'doorway':(10.413,11.620),'rolling':(4.4,4.6),'new_control':(9.56,10.62),'static':(.2,1.30)}
    final={}
    for case,(lo,hi) in scopes.items():
        dest=out/case; dest.mkdir()
        task=Experiment(case,lock['cases'][case],clip,read(ROOT/'dacf_research/protocol.json'),dest)
        points=sorted({nearest(task.native,task.times,t) for t in (lo,hi)})
        eligible=[p for p in task.native if points[0]<p<points[-1]]
        for p in points: task.add(p)
        def estimate(a,b): return task.bank.metric(task.ranks[a],task.ranks[b])
        trace=[]
        for level in range(1,4):
            refinements=[]
            for a,b in pairwise(points):
                legal=[p for p in eligible if a<p<b]
                if not legal: continue
                p=nearest(legal,task.times,(task.times[a]+task.times[b])/2)
                r,_=task.add(p)
                d1,d2,old=estimate(a,p),estimate(p,b),estimate(a,b)
                ref={'pts':[a,p,b],'before_seconds':task.times[b]-task.times[a], 'root':r['id'],
                     'terms':[d1,d2,old],'delta':{ch:d1[ch]+d2[ch]-old[ch] if None not in (d1[ch],d2[ch],old[ch]) else None for ch in METRICS}}
                refinements.append(ref)
            points=sorted(set(points)|{r['pts'][1] for r in refinements})
            trace.append({'level':level,'actual_pts':points,'new_probes':len(refinements),'refinements':refinements,'maximum_gap_seconds':max(task.times[b]-task.times[a] for a,b in pairwise(points)),
                          'sum_delta':{ch:sum(r['delta'][ch] for r in refinements if r['delta'][ch] is not None) for ch in METRICS}})
        final[case]=trace
        write(dest/'receipts.json',task.reader.receipts)
    write(out/'summary.json',{'manual_after_prediction':True,'cases':final})


if __name__=='__main__':main()
