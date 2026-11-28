"""Generate region receipts from the locked, unchanged initial nominations."""
import argparse
from itertools import product
from statistics import median
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank, METRICS, clip_encoder, direct_group, crossing, path_curve, local_presence
from dacf_research.freeze import validate as verify_round1
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump
from dacf_round2.frozen_inputs import HERE
from dacf_round2.regions import discover


def load_rasters(c, manifest):
    receipts=read(c/'frame_receipts.json')
    pts={r['pts']:r for r in receipts}
    pixels={}
    for row in manifest['pool']:
        r=pts[row['pts']]
        img=c/r['png']
        assert digest(img)==r['png_sha256'] and r['frame']==row['source_frame']
        with Image.open(img) as native:
            rgb=np.asarray(native.convert('RGB').resize(tuple(r['analysis_size']),Image.Resampling.BILINEAR))
        pixels[r['frame']]=r|{'rgb':rgb}
    return pixels


def evidence(bank, region, pool, catalogue):
    lo,hi=region['bounds_pts']
    span=[q for q in pool if lo<=bank.rows[q]['pts']<=hi]
    a,b=region.get('left'),region.get('right')
    x,y=(v['prototypes'] if v else [] for v in (a,b))
    direct=[bank.metrics_on_pair(p,q) for p,q in product(x,y) if p!=q]
    cross={ch:median(r['metrics'][ch] for r in direct) if direct and all(r['metrics'][ch] is not None for r in direct) else None for ch in METRICS}
    paths={ch:path_curve(bank,span,ch) for ch in METRICS}
    p_path={ch:{r['pts']:r['P_path'] for r in path['cumulative']} for ch,path in paths.items()}
    rows=[]; points={ch:[] for ch in METRICS}
    epsilon=bank.config['numerical_epsilon']
    for rank in span:
        f=bank.rows[rank]
        left=direct_group(bank,rank,x) if a else None
        right=direct_group(bank,rank,y) if b else None
        local=local_presence(bank,rank,x,y) if a and b and f['pts'] not in (lo,hi) else None
        row={'pts':f['pts'],'seconds':f['seconds'],'source_root':f['id'],'left':left,'right':right,'local':local,
             'phase1_catalogue_member':f['pts'] in catalogue,'channels':{}}
        for ch in METRICS:
            da,db=(v['channels'][ch]['distance'] if v else None for v in (left,right))
            relative=da/(da+db) if None not in (da,db) and da+db>epsilon and (cross[ch] or 0)>epsilon else None
            p=p_path[ch][f['pts']]
            row['channels'][ch]={'relative_anchor':relative,'sampled_path_fraction':p,
                'anchor_minus_path':relative-p if None not in (relative,p) else None}
            points[ch].append({'pts':f['pts'],'t':f['seconds'],'balance':relative,'da':da,'db':db})
        rows.append(row)
    crossings={ch:crossing(z,epsilon,bank.config['support_caps'][ch],bank.config['maximum_gap_seconds'],(cross[ch] or 0)>epsilon)
               for ch,z in points.items()}
    extrema={}
    directions={}
    for ch,p in paths.items():
        extrema[ch]={}
        for scale,vals in p['smoothed'].items():
            key='weighted_distance_rate_per_second'
            extrema[ch][scale]=[{'pts':v['pts'],'kind':'peak' if u[key]<v[key] else 'valley','value':v[key]}
                for u,v,w in zip(vals,vals[1:],vals[2:]) if all(item[key] is not None for item in (u,v,w)) and
                ((u[key]<v[key] and v[key]>w[key]) or (u[key]>v[key] and v[key]<w[key]))]
        delta=[{'pts':v['pts'],'delta':v['balance']-u['balance']} for u,v in zip(points[ch],points[ch][1:])
               if u['balance'] is not None and v['balance'] is not None and v['t']-u['t']<=bank.config['maximum_gap_seconds']]
        directions[ch]=delta
    a_rows=rows[0] if rows else None; b_rows=rows[-1] if rows else None
    figures={}
    for row in (a_rows,b_rows):
        if row and row['phase1_catalogue_member']:
            rank=next(q for q in span if bank.rows[q]['pts']==row['pts'])
            figures[str(row['pts'])]=bank.figure(rank,catalogue)
    return {'region_id':region['id'],'rows':rows,'paths':paths,'crossings':crossings,
            'direct_group_pair_roots':[r['root'] for r in direct],'group_distance_by_channel':cross,
            'channel_direction_steps':directions,'multi_scale_extrema':extrema,'frozen_figure_probes_at_seed':figures}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',choices=('replay','independent'))
    args=parser.parse_args()
    verify_round1('automatic')
    lock=read(HERE/'lock.json')
    assert all(digest(ROOT/p)==v for p,v in lock['inputs'].items())
    if args.cohort=='independent':
        assert (HERE/'replay/summary.json').is_file(), 'Complete replay predictions first.'
    policy=read(HERE/'protocol.json')
    cfg=read(ROOT/'dacf_research/protocol.json')
    root=HERE/args.cohort;root.mkdir()
    cv2.setNumThreads(1)
    try:
        clip=clip_encoder()
    except Exception as error:
        clip=None
        write(root/'clip_startup_failure.json',{'error':str(error)})
    summary=[]
    for name,c in lock['cases'].items():
        if c['cohort']!=args.cohort:
            continue
        start=monotonic()
        out=root/name;out.mkdir()
        idx=ROOT/c['index']
        m=read(idx/'manifest.json'); source=ROOT/c['source']
        lib=read(idx/'catalog.json')
        assert m['video_sha256']==lib['source']['sha256']==digest(source)
        bank=Bank(load_rasters(ROOT/c['pixels'],m),cfg,clip)
        pool=[r['source_frame'] for r in m['pool']]
        catalogue={r['pts']:r for r in lib['frames']}
        result=discover(bank,pool,catalogue,policy)
        assert result['nomination_pts']==list(catalogue)
        regions=result['regions']
        proof=[evidence(bank,r,pool,catalogue) for r in regions]
        result['source_sha256']=m['video_sha256']
        result['nomination_status']='all initial nominees carried through; no insertion into old catalogue'
        write(out/'regions.json',result)
        gz_dump(out/'region_signals.json.gz',proof)
        gz_dump(out/'pair_evidence.json.gz',list(bank.evidence.values()))
        gz_dump(out/'local_holdout.json.gz',list(bank.links.values()))
        rec={'case':name,'catalogue_kept':len(catalogue),'coarse_count':len(pool),'clip':bank.clip_state,
             'paired_leaf':sum(r['kind']=='paired_investigation' for r in regions),
             'only_unknown':sum('unknown' in r['kind'] for r in regions),'recursively_split':sum(bool(r['children']) for r in regions),
             'adjacent_chains':len(result['region_chains']),'recurrence_candidates':len(result['recurrence_candidates']),
             'omitted_opportunities':len(result['omitted']),'geometry_pairs':len(bank.evidence),'seconds':round(monotonic()-start,3)}
        print(args.cohort,rec,flush=True);summary.append(rec)
    write(root/'summary.json',summary)

if __name__=='__main__':
    main()
