"""Replay Phase 1's actual nomination pipeline before DACF prediction."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import HERE, ROOT, digest, read, write
from evidenweft_phase1.prepare import verify


PLUGIN = ROOT / 'evidenweft_phase1/frozen/visureach'
sys.path.insert(0, str(PLUGIN / 'vendor/P2_24'))
from if16_research.common import sequence, frame_key
from if16_research.nominate import nominate
from v05.engine import select_expression


def replay(case, role, rubric):
    phase = ROOT/'evidenweft_phase1'
    folder = phase/'inventory'/case/'index'
    c = read(folder/'catalog.json')
    source = c['source']
    clip = HERE/'inputs'/f'{case}.mp4'
    assert digest(clip) == source['sha256']
    m = read(folder/'manifest.json')
    frames = [r | {'t':r['seconds'],'time_base':m['time_base'],'sample':i} for i,r in enumerate(m['pool'])]
    width = c['coordinate_contract']['analysis_width']; height = c['coordinate_contract']['analysis_height']
    picks = [r['source_frame'] for r in frames]
    raw = subprocess.check_output(['ffmpeg','-v','error','-xerror','-noautorotate','-i',str(clip),'-map','0:v:0','-vf',
        f"select='{select_expression(picks)}',scale={width}:{height},format=rgb24",'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    n = width*height*3
    assert len(raw)==len(frames)*n
    with TemporaryDirectory() as work:
        for i,r in enumerate(frames):
            p = Path(work)/f'{i}.png'
            Image.frombytes('RGB',(width,height),raw[i*n:(i+1)*n]).save(p)
            r['image']=str(p)
        seq=sequence(frames)
    channels, fg, motion = nominate(seq,source)
    replayed = {k:len(v) for k,v in channels.items() if v}
    assert replayed==m['nomination_counts']
    by_pt = {r['pts']:r for r in c['frames']}
    by_sample={}
    for observations in channels.values():
        for v in observations:
            by_sample.setdefault(v['sample'],[]).append(v)
    for i,r in enumerate(frames):
        novel={(x['dimension_id'],x['role'],tuple(x['evidence_root_ids'])) for x in by_sample.get(i,[])}
        frozen={(x['dimension_id'],x['role'],tuple(x['evidence_root_ids'])) for x in by_pt.get(r['pts'],{}).get('nominations',[])}
        assert novel==frozen
    lo,hi=rubric['slots'][role]['pts']
    interval=[i for i,r in enumerate(frames) if lo<=r['pts']<=hi]
    rows=[]
    for i in range(max(0,interval[0]-2),min(len(frames),interval[-1]+3)):
        r=frames[i]
        def edges(m):
            if m is None:
                return None
            return {k: m[k] for k in ('raw_mad','registration_valid','motion_supported','phase_response','overlap','dx','dy','residual_ratio')}
        rows.append({'pts':r['pts'],'time_base':r['time_base'],'seconds':r['seconds'],'sample':i,'inside_locked_role':i in interval,
            'source_frame':r['source_frame'],'nominations':by_sample.get(i,[]),
            'quality':{k:seq[i][k] for k in ('sharpness','contrast')},
            'foreground_neighborhood':fg[i], 'adjacent_from_previous':edges(motion.get(i)),'adjacent_to_next':edges(motion.get(i+1))})
    eq=read(phase/'evaluation_only/equal_count'/case/'delivery_score.json')
    return {'case':case,'role':role,'source':source['sha256'],'source_role_interval':rubric['slots'][role],
        'replayed_nomination_counts':replayed,
        'delivered_at_equal_inventory_count':{x['mode']:x['slots'].get(role,[]) for x in eq['deliveries'] if x['requested']=='inventory'},
        'coarse_role_pts':[frames[i]['pts'] for i in interval],'nearby_raw_measurements':rows}


def main():
    verify()
    reference=read(ROOT/'evidenweft_phase1/evaluation_only/rubric.json')['cases']
    results=[replay(case,role,reference[case]) for case,role in [('bench','B10'),('doorway','D06')]]
    write(HERE/'phase1_attribution.json',results)
    for r in results:
        print(r['case'],r['coarse_role_pts'],r['delivered_at_equal_inventory_count'])
        for f in r['nearby_raw_measurements']:
            if f['inside_locked_role']:
                print(f['pts'],f['foreground_neighborhood']['state'],f['adjacent_from_previous'],[(x['dimension_id'],x['role']) for x in f['nominations']])


if __name__=='__main__':
    main()
