"""Optional offline policy evaluation. Installation never invokes it automatically."""
import argparse
import cv2

from dacf_research.field import METRICS,clip_encoder
from dacf_research.prepare_inputs import ROOT,read,write
from dacf_round4.offline import Experiment,run_route
from dacf_round5.setup import HERE,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=('S0','S1'),default='S0');a=p.parse_args()
    lock=read(HERE/'lock.json');verify(lock)
    cv2.setNumThreads(1)
    out=HERE/f'preset-{a.policy}';out.mkdir()
    cfg=read(ROOT/'dacf_research/protocol.json');rule=read(ROOT/'dacf_round4/protocol.json'); encoder=clip_encoder()
    rows=[]
    for case,r in lock['cases'].items():
        target=out/case;target.mkdir()
        exp=Experiment(case,r,encoder,cfg,target)
        routes=[run_route(exp,a.policy,ch,rule) for ch in METRICS]
        write(target/'routes.json',routes);write(target/'receipts.json',exp.reader.receipts)
        rows.append({'case':case,'first_nominees':len(exp.initial),'natural_by_axis':{r['channel']:r['natural'] for r in routes},
                     'exact_probe_receipts':len(exp.reader.receipts)})
    write(out/'summary.json',{'policy':a.policy,'scope':'Optional actual S0/S1 run with original R4 stop and equal comparison traces. No effect on installed Agent source permissions.', 'cases':rows})


if __name__=='__main__':main()
