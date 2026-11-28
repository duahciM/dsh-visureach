"""Run the locked DACF manual or automatic diagnostic arm on actual source PTS."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank, METRICS, clip_encoder, crossing, detect_groups, direct_group, local_presence, path_curve
from dacf_research.freeze import PLUGIN, validate
from dacf_research.prepare_inputs import HERE, ROOT, digest, read, write
from evidenweft_phase1.source_review import native_stream, timeline, WITHDRAWN


def baseline(source, case, out, protocol):
    if case in {'bench','doorway'}:
        key=ROOT/'evidenweft_phase1/inventory'/case/'index'
    else:
        key=HERE/'shared_catalogues'/case
        if not (key/'manifest.json').is_file():
            key.mkdir(parents=True, exist_ok=True)
            with (out/'indexer.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),
                  '--video',str(source),'--out',str(key),'--fps',str(protocol['fps']),'--keep-analysis'],
                  stdout=log,stderr=subprocess.STDOUT,check=True)
    m=read(key/'manifest.json')
    c=read(key/'catalog.json')
    assert m['video_sha256']==c['source']['sha256']==digest(source)
    return m,c


def raw_frames(source, stream, timeline_rows, chosen, out, digest_source, width):
    folder=out/'originals'
    folder.mkdir()
    size=stream['width'],stream['height']
    scaled=(width,max(8,round(size[1]*width/size[0])))
    decoded={}
    manifest=[]
    for row,bits in native_stream(source,stream,timeline_rows):
        if row['frame'] not in chosen:
            continue
        rgb_sha=sha256(bits).hexdigest()
        source_image=Image.frombytes('RGB',size,bits)
        path=folder/f'pts-{row["pts"]}.png'
        source_image.save(path)
        cid=f'{digest_source}:{row["frame"]}:{row["pts"]}:{stream["time_base"]}'
        record={**row,'id':cid,'time_base':stream['time_base'],'png':path.relative_to(out).as_posix(),
                'png_sha256':digest(path),'rgb_sha256':rgb_sha,'native_size':size,'analysis_size':scaled}
        manifest.append(record)
        decoded[row['frame']]=record|{'rgb':np.asarray(source_image.resize(scaled,Image.Resampling.BILINEAR))}
    assert set(decoded)==set(chosen)
    write(out/'frame_receipts.json',manifest)
    return decoded


def choose_rank(rows, metric, midpoint):
    if not rows:
        return None
    return min(rows,key=lambda r:(metric(r), abs(r['seconds']-midpoint),r['pts']))['pts']


def pair_field(bank, case, pair, coarse, catalogue):
    a,b=pair['a'],pair['b']
    assert len(a)==len(b)==3 and len(set(a+b))==6 and a==sorted(a) and b==sorted(b) and a[-1]<b[0]
    left,right=a[-1],b[0]
    interior=[q for q in coarse if left<q<right]
    sample=[left,*interior,right]
    config=bank.config
    eps=config['numerical_epsilon']; gap=config['maximum_gap_seconds']
    ab=[bank.metrics_on_pair(x,y) for x,y in product(a,b)]
    separated={ch:median([v['metrics'][ch] for v in ab if v['metrics'][ch] is not None]) if any(v['metrics'][ch] is not None for v in ab) else None for ch in METRICS}
    path={ch:path_curve(bank,sample,ch) for ch in METRICS}
    adjacent_roots=[bank.metrics_on_pair(x,y)['root'] for x,y in zip(sample,sample[1:])]
    trace=[]; points={ch:[] for ch in METRICS}
    for rank in sample:
        f=bank.rows[rank]
        actual=rank not in {left,right}
        pa,pb=direct_group(bank,rank,a),direct_group(bank,rank,b)
        local=local_presence(bank,rank,a,b)
        row={'rank':rank,'pts':f['pts'],'seconds':f['seconds'],'eligible_interior':actual,'frame_root':f['id'],
             'a':pa,'b':pb,'local':local,'anchor_balance':{}}
        for ch in METRICS:
            da,db=(g['channels'][ch]['distance'] for g in (pa,pb))
            p=da/(da+db) if da is not None and db is not None and da+db>eps and separated[ch] is not None and separated[ch]>eps else None
            row['anchor_balance'][ch]=p
            points[ch].append({'pts':f['pts'],'t':f['seconds'],'balance':p,'da':da,'db':db})
        row['figure_status']='available_in_frozen_catalogue' if f['pts'] in catalogue else 'unassessed_without_frozen_nomination'
        trace.append(row)
    middle=(bank.rows[left]['seconds']+bank.rows[right]['seconds'])/2
    inside=[r for r in trace if r['eligible_interior']]
    anchor=choose_rank([r for r in inside if r['anchor_balance']['gray'] is not None], lambda r:abs(r['anchor_balance']['gray']-.5),middle)
    pp={r['pts']:r['P_path'] for r in path['gray']['cumulative']}
    cp=choose_rank([r for r in inside if pp[r['pts']] is not None],lambda r:abs(pp[r['pts']]-.5),middle)
    time=choose_rank(inside,lambda _:0,middle)
    peaks={}
    pos={r['pts']:r['seconds'] for r in trace}
    interior_pts={r['pts'] for r in inside}
    for sigma,values in path['gray']['smoothed'].items():
        possible=[r for r in values if r['pts'] in interior_pts and r['weighted_distance_rate_per_second'] is not None]
        if possible and max(r['weighted_distance_rate_per_second'] for r in possible)>eps:
            peaks[sigma]=max(possible,key=lambda r:(r['weighted_distance_rate_per_second'],-abs(pos[r['pts']]-middle),-r['pts']))['pts']
        else:
            peaks[sigma]=None
    mode=[pos[p] for p in peaks.values() if p is not None]
    smooth=choose_rank(inside,lambda r:abs(r['seconds']-median(mode)),middle) if mode else None
    local=choose_rank([r for r in inside if r['local']['different_cells_with_one_group_supported']],
          lambda r:-len(r['local']['different_cells_with_one_group_supported']),middle)
    chosen={'time_midpoint':time,'gray_anchor_balance':anchor,'gray_path_midpoint':cp,'gray_multiscale':smooth,'disjoint_local_support':local}
    overlays={str(p):bank.figure(r['rank'],catalogue) for p in set(chosen.values()) if p is not None
              for r in trace if r['pts']==p}
    signs=[]; drifts={}
    for i,(first,last) in enumerate(zip(trace,trace[1:])):
        x,y=first['anchor_balance']['gray'], last['anchor_balance']['gray']
        if x is None or y is None or last['seconds']-first['seconds']>gap:
            signs.append({'pts':last['pts'],'direction':'unassessed'})
            continue
        d=y-x
        limit=config['direction_gray_deadband']
        signs.append({'pts':last['pts'],'direction':'toward_B' if d>limit else 'toward_A' if d< -limit else 'within_deadband','delta_P_anchor_gray':d})
    for ch in METRICS:
        pct={r['pts']:r['P_path'] for r in path[ch]['cumulative']}
        signed={p['pts']:p['balance']-pct[p['pts']] for p in points[ch] if p['balance'] is not None and pct[p['pts']] is not None}
        drifts[ch]={'by_pts':signed,'max_abs':max(map(abs,signed.values()),default=None)}
    intersections={ch:crossing(points[ch],eps,config['support_caps'][ch],gap,separated[ch] is not None and separated[ch]>eps) for ch in METRICS}
    reversals=0; previous=None
    for v in signs:
        d=v['direction']
        if d=='unassessed':
            previous=None
        elif d!='within_deadband':
            reversals+=int(previous is not None and previous!=d)
            previous=d
    return {'case':case,'pair':pair,'anchor_pts':{'a':[bank.rows[k]['pts'] for k in a],'b':[bank.rows[k]['pts'] for k in b]},
            'boundary_pts':[bank.rows[left]['pts'],bank.rows[right]['pts']],'ab_group_pair_roots':[v['root'] for v in ab],
            'ab_group_distance_medians':separated,'rows':trace,'adjacent_geometry_roots':adjacent_roots,
            'path':path,'intersections':intersections,'direction_gray':signs,'observed_direction_reversals':reversals,
            'anchor_minus_path':drifts,'per_scale_gray_peaks':peaks,'selections':chosen,'frozen_figure_probes':overlays}


def gz_dump(path, value):
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=4) as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('arm',choices=('manual','automatic'))
    a=parser.parse_args()
    validate(a.arm)
    cfg=read(HERE/'protocol.json')
    explicit=read(HERE/'manual_anchors.json')['cases'] if a.arm=='manual' else None
    if a.arm=='automatic':
        assert (HERE/'run_manual/summary.json').is_file(), 'Complete manual mechanism validation first.'
    root=HERE/f'run_{a.arm}'
    root.mkdir()
    cv2.setNumThreads(1)
    try:
        embed=clip_encoder()
        embedding_note={'status':'enabled','model':'frozen CLIP vision ONNX CPU image embedding, whole-frames'}
    except Exception as e:
        embed=None
        embedding_note={'status':'dependency_failure','reason':str(e)}
    results=[]
    for source in sorted((HERE/'inputs').iterdir()):
        start=monotonic()
        name=source.stem
        out=root/name;out.mkdir()
        manifest,library=baseline(source,name,out,cfg)
        assert manifest['video_sha256']!=WITHDRAWN
        stream,ref=timeline(source)
        assert stream['time_base']==manifest['time_base']
        ranks_by_pts={r['pts']:r['frame'] for r in ref}
        pool=[f['source_frame'] for f in manifest['pool']]
        assert all(ranks_by_pts[f['pts']]==f['source_frame'] for f in manifest['pool'])
        catalogue={r['pts']:r for r in library['frames']}
        anchor_pairs=[]
        if explicit is not None:
            for r in explicit[name]:
                anchor_pairs.append({'id':r['id'],'scope':r['scope'],
                    'a':r.get('a_ranks') or [ranks_by_pts[p] for p in r['a_pts']],
                    'b':r.get('b_ranks') or [ranks_by_pts[p] for p in r['b_pts']]})
        sources=set(pool)|{k for p in anchor_pairs for k in p['a']+p['b']}
        bank=Bank(raw_frames(source,stream,ref,sources,out,manifest['video_sha256'],cfg['analysis_width']),cfg,embed)
        if explicit is None:
            auto=detect_groups(bank,pool)
            anchor_pairs=auto['pairs']
            write(out/'automatic_groups.json',auto)
        report=[pair_field(bank,name,p,pool,catalogue) for p in anchor_pairs]
        write(out/'pair_results.json',report)
        gz_dump(out/'pair_evidence.json.gz',list(bank.evidence.values()))
        gz_dump(out/'local_holdout.json.gz',list(bank.links.values()))
        info={'case':name,'source_sha256':manifest['video_sha256'],'source_frames':len(ref),'coarse':len(pool),
              'baseline_catalogue':len(catalogue),'clip':bank.clip_state,'pair_count':len(report),'distinct_pair_roots':len(bank.evidence),
              'point_holdout_roots':len(bank.links),'observations':sum(len(p['rows']) for p in report),
              'runs_without_interior':sum(not any(r['eligible_interior'] for r in p['rows']) for p in report),
              'compute_seconds':round(monotonic()-start,3),'selections':{p['pair']['id']:p['selections'] for p in report}}
        if explicit is None:
            info.update(stable_groups=len(auto['groups']),rejected_intervals=len(auto['rejected']))
        results.append(info)
        print(a.arm,name,info['pair_count'],info['observations'],info['clip'],info['compute_seconds'],flush=True)
    write(root/'summary.json',{'arm':a.arm,'cases':results,'clip_environment':embedding_note})


if __name__=='__main__':
    main()
