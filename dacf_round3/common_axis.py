"""Compare all routes on the same measured, post-prediction sample union."""
from collections import Counter, defaultdict
from statistics import median

import cv2

from dacf_research.field import METRICS, clip_encoder
from dacf_research.prepare_inputs import ROOT, read, write
from dacf_round3.run import Experiment, max_gap
from dacf_round3.score import materialize
from dacf_round3.setup import HERE, verify


def main():
    lock = read(HERE/'lock.json')
    verify(lock)
    cv2.setNumThreads(1)
    encoder = clip_encoder()
    cfg = read(ROOT/'dacf_research/protocol.json')
    root = HERE/'common_axis'
    root.mkdir()
    aggregate = defaultdict(list)
    for case,registration in lock['cases'].items():
        directory = HERE/'run'/case
        out = root/case
        out.mkdir()
        experiment = Experiment(case,registration,encoder,cfg,out)
        receipts = read(directory/'receipts.json')
        materialize(experiment,receipts)
        points = sorted(set(experiment.coarse)|set(map(int,receipts)))
        coords = {ch:experiment.axis(points,experiment.initial,ch) for ch in METRICS}
        eval_rows = read(HERE/'evaluation'/case/'score.json')
        projections = []
        for row in eval_rows['routes']:
            gap = {ch:max_gap(axis['positions'],experiment.initial|set(row['equal']['selected_pts'])) for ch,axis in coords.items()}
            projections.append({'arm':row['arm'],'channel':row['channel'],'equal':row['equal_comparable'],'gaps':gap})
        time = next(p for p in projections if p['arm']=='E1')
        for p in projections:
            if p['equal'] and time['equal']:
                key = (p['arm'],p['channel'])
                aggregate[key].append({ch:p['gaps'][ch]-time['gaps'][ch] for ch in METRICS})
        write(out/'common.json',{'points':len(points),'coords':coords,'routes':projections})
        print(case,len(points),'shared measured positions',flush=True)
    rows=[]
    for (arm,channel), comparisons in aggregate.items():
        stats={}
        for ch in METRICS:
            values=[r[ch] for r in comparisons]
            sign=Counter('smaller' if v< -1e-8 else 'larger' if v>1e-8 else 'tied' for v in values)
            stats[ch]=dict(sign)|{'median_difference_vs_E1':median(values)}
        rows.append({'arm':arm,'channel':channel,'paired_equal_cases':len(comparisons),'stats':stats})
    write(root/'summary.json',{'scope':'Post-prediction union of coarse and all actual supplemental PTS; no dense reference pixels or tuning. Axes kept independent. Paired with E1 in same video.', 'rows':rows})


if __name__=='__main__':
    main()
