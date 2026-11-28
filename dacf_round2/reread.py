"""Operator-selected local original reread; no automatic delivery."""
import argparse
from pathlib import Path

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.evaluate import decisive_read
from dacf_round2.frozen_inputs import HERE
from evidenweft_phase1.prepare import select_time
from evidenweft_phase1.source_review import timeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--region',action='append',default=[])
    p.add_argument('--start',type=float); p.add_argument('--end',type=float)
    p.add_argument('--fps',type=float)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.fps is not None and not 0<a.fps<float('inf'): p.error('Give a positive finite density.')
    if not a.region and (a.start is None or a.end is None): p.error('Give a region ID or both start/end seconds.')
    if a.region and (a.start is not None or a.end is not None): p.error('Use region IDs or time bounds.')
    if not a.region and not a.start<=a.end: p.error('Start must be at or before end.')
    lock=read(HERE/'lock.json')
    c=lock['cases'].get(a.case)
    if not c: p.error('Unknown case. Choose a case in the locked study.')
    src=ROOT/c['source']; index=read(ROOT/c['index']/'manifest.json')
    assert digest(src)==index['video_sha256']==lock['inputs'][c['source']]
    stream, src_rows=timeline(src)
    coarse={r['pts']:r for r in read(ROOT/c['pixels']/'frame_receipts.json')}
    if a.region:
        root=HERE/'replay' if c['cohort']=='replay' else HERE/'independent_execution/independent'
        plan=read(root/a.case/'regions.json')
        available={r['id']:r['bounds_pts'] for r in plan['regions']}
        if set(a.region)-available.keys(): p.error('Unknown region ID for this source.')
        spans=[available[key] for key in a.region]
        eligible=[r for r in src_rows if any(lo<=r['pts']<=hi for lo,hi in spans)]
    else:
        eligible=[r for r in src_rows if a.start<=r['seconds']<=a.end]
    if not eligible: p.error('No original frame in requested range.')
    if a.fps is not None:
        wanted=[]
        # Separate disjoint source spans so a sampling clock won't demand phantom frames in gaps.
        for r in eligible:
            if not wanted or r['frame']!=wanted[-1][-1]['frame']+1:
                wanted.append([])
            wanted[-1].append(r)
        eligible=[r for frames in wanted for r in select_time(frames,min(len(frames),1+int((frames[-1]['seconds']-frames[0]['seconds'])*a.fps)))]
    a.out.mkdir(parents=True,exist_ok=False)
    receipts,seconds=decisive_read(src,stream,src_rows,[],a.out,index,coarse,{r['pts'] for r in eligible})
    assert len(receipts)==len(eligible)
    write(a.out/'result.json',{'case':a.case,'regions':a.region,'custom_seconds':[a.start,a.end],'requested_fps':a.fps,
         'source_sha256':index['video_sha256'],'independent_originals':list(receipts.values()),'wall_seconds':seconds})
    print(a.case,len(receipts),'actual originals',seconds,flush=True)


if __name__=='__main__':
    main()
