"""Frozen replay and local-stop comparison. Predictions have no role files."""
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
import argparse
import shutil
import subprocess
import sys
from time import perf_counter

import cv2

from dacf_research.field import clip_encoder, METRICS
from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import DESKTOP, ROOT, digest, read, write
from dacf_research.run import raw_frames
from dacf_round3.run import Experiment, nearest, window_edges
from dacf_round3.sampling import ranked_gaps
from dacf_round3.setup import verify as verify_r3
from evidenweft_phase1.source_review import timeline, WITHDRAWN

HERE=Path(__file__).resolve().parent
R3=ROOT/'dacf_round3'


def seal():
    previous=read(R3/'lock.json')
    verify_r3(previous)
    cases=previous['cases'].copy()
    reference=HERE/'independent'
    reference.mkdir()
    src=DESKTOP/'8.mp4'
    sig=digest(src)
    assert sig!=WITHDRAWN and sig not in previous['inputs'].values()
    source=reference/'source.mp4'
    shutil.copyfile(src,source)
    index=reference/'index'
    subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'), '--video',str(source),'--out',str(index),'--fps','5'],check=True)
    raw=read(index/'manifest.json')
    stream,clock=timeline(source)
    raw_frames(source,stream,clock,{r['source_frame'] for r in raw['pool']},reference,sig,320)
    cases['independent_pan']={k:p.relative_to(ROOT).as_posix() for k,p in {'source':source,'index':index,'pixels':reference}.items()}|{'cohort':'new'}
    pins=[R3/'lock.json',R3/'validation.json',R3/'run_manifest.json',R3/'evaluation/summary.json',
          HERE/'protocol.json',HERE/'offline.py',source,index/'manifest.json',index/'catalog.json',reference/'frame_receipts.json']
    for case in previous['cases']:
        pins.extend(R3/'run'/case/key for key in ('tree-v2.json','region_graph.json','receipts.json','routes.json','initial.json'))
    write(HERE/'lock.json',{'date':datetime.now().astimezone().isoformat(), 'cases':cases,
                           'hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in pins}})


def verify(lock):
    verify_r3(read(R3/'lock.json'))
    assert all(digest(ROOT/p)==sig for p,sig in lock['hashes'].items())


def local_assess(task, axis, option, rules):
    a,b=option['bounds']
    midpoint=(task.times[a]+task.times[b])/2
    radius=rules['local_radius_seconds']
    component=axis['positions'][a][0]
    edges=[edge for edge in axis['edges'] if edge['applicable'] and axis['positions'][edge['pts'][0]][0]==component
           and abs(sum(task.times[p] for p in edge['pts'])/2-midpoint)<=radius]
    total=sum(edge['value'] for edge in edges)
    fraction=option['gap']/total if total and len(edges)>=2 else None
    eligible=fraction is not None and fraction>rules['local_fraction'] and task.times[b]-task.times[a]>=rules['minimum_probe_span_seconds']
    return {'eligible':eligible,'fraction':fraction,'local_path_total':total,'support_pairs':[e['pts'] for e in edges],
            'state':'local_gap_remaining' if eligible else 'unknown_local_basis' if fraction is None else 'locally_covered_or_below_span'}


def proposals(task,samples,evidence,channel,cfg):
    axis=task.axis(samples,evidence,channel)
    total=axis['total']
    selected, options=ranked_gaps(axis['positions'],samples,evidence,task.times,task.native,task.paired,
                                 [max(1e-12, s*total) for s in (.04,.08,.16)],False)
    assessments=[dict(option,local=local_assess(task,axis,option,cfg)) for option in options]
    global_max=max((o['gap'] for o in options),default=0.)
    return axis, selected, assessments, global_max


def run_route(task, strategy, channel, rules):
    samples=task.coarse.copy()
    keep=task.initial.copy()
    prioritization=keep.copy()
    trace=[]
    natural=None
    if strategy=='endpoint_prior_control':
        prioritization={p for _,a,b in task.paired for p in (a,b)}
    while len(trace)<rules['observation_ceiling']:
        started=perf_counter()
        if strategy=='restricted_time':
            entries=set(p for _,a,b in task.paired for p in task.native[bisect_right(task.native,a):bisect_left(task.native,b)])-keep
            if not entries:
                break
            p=max(entries,key=lambda p:(min(abs(task.times[p]-task.times[q]) for q in keep),-p))
            root,a,b=next((root,a,b) for root,a,b in task.paired if a<p<b)
            option={'root':root,'bounds':[a,b],'pts':p}
            max_gap=None
            axis=None
        else:
            axis, selected, options, max_gap=proposals(task,samples,prioritization,channel,rules)
            options=[q for q in options if q['pts'] not in keep]
            eligible=[q for q in options if q['local']['eligible']]
            if natural is None and (not eligible or strategy=='S0' and max_gap<=.12*axis['total']):
                if strategy!='S0' or not options or max_gap<=.12*axis['total']:
                    natural={'after':len(trace),'reason':'policy_stop','eligible_local_count':len(eligible),
                             'global_gap':max_gap,'global_threshold':.12*axis['total'], 'unresolved_local_count':sum(q['local']['fraction'] is None for q in options)}
            ranked=eligible if strategy=='S1' and eligible else options
            option=max(ranked,key=lambda q:(q['coverage'],q['gap'],-q['pts'])) if ranked else None
        if not option or (natural is not None and len(trace)>=rules['equal_originals']) or (strategy not in {'S0','S1'} and len(trace)>=rules['equal_originals']):
            break
        p=option['pts']
        rank_seconds=perf_counter()-started
        rec,really_new=task.add(p)
        a,b=option['bounds']
        keep.add(p)
        prioritization.add(p)
        if p not in samples:
            samples.insert(bisect_left(samples,p),p)
        row=option|{'receipt':rec['id'],'source_seconds':task.times[p],'decode_cache_miss':really_new,'seconds_selection':rank_seconds,
                    'cold_filter_frames':sum(x['frames_at_input_filter'] or 0 for x in rec['decode_attempts']),
                    'cold_read_seconds':rec['wall_seconds'],'png_bytes':rec['png_bytes'],'forced_equal_after_stop':natural is not None}
        if axis:
            after=task.axis(samples,prioritization,channel)
            m=lambda x,y:task.bank.metric(task.ranks[x],task.ranks[y])[channel]
            pieces=[m(a,p),m(p,b),m(a,b)]
            row.update(original_edges=window_edges(axis['edges'],a,b),revised_edges=window_edges(after['edges'],a,b),
                       coverage_gap_before=option['gap'],path_increment=sum(pieces[:2])-pieces[2] if None not in pieces else None,
                       chord_terms=pieces,legacy_global_stop=max_gap<=.12*axis['total'])
        trace.append(row)
    return {'strategy':strategy,'channel':channel,'steps':trace,'initial_nominees_retained':len(task.initial),
            'natural':natural or {'after':len(trace),'reason':'ceiling_or_exhaustion'}}


def density_bins(task):
    time0=task.times[task.coarse[0]]
    result={}
    for channel in METRICS:
        axis=task.axis(task.coarse,task.initial,channel)
        records={}
        for edge in axis['edges']:
            a,b=edge['pts']
            bin_id=int((task.times[a]-time0)//1)
            row=records.setdefault(bin_id,{'interval_seconds':[time0+bin_id,time0+bin_id+1],'coarse':set(),'distance':0.,'unknown_edges':0,'zero_edges':0})
            row['coarse'].add(a)
            if edge['applicable']:
                row['distance']+=edge['value']
                row['zero_edges']+=edge['value']==0
            else:
                row['unknown_edges']+=1
        rows=[]
        for r in records.values():
            items=sorted(r.pop('coarse'))
            rows.append(r|{'coarse_pts':items,'nomination_pts':[p for p in items if p in task.initial],
                           'density':sum(p in task.initial for p in items)/len(items)})
        median_density=median([r['density'] for r in rows]) if rows else 0.
        median_change=median([r['distance'] for r in rows]) if rows else 0.
        for r in rows:
            r['density_band']='high' if r['density']>median_density else 'low'
            r['change_band']='unknown' if r['unknown_edges'] else 'high' if r['distance']>median_change else 'low'
        result[channel]={'descriptive_only_cut_points':[median_density,median_change],'bins':rows}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seal',action='store_true')
    args=parser.parse_args()
    if args.seal:
        seal()
        return
    lock=read(HERE/'lock.json')
    verify(lock)
    policy=read(HERE/'protocol.json')
    destination=HERE/'offline'
    destination.mkdir()
    cv2.setNumThreads(1)
    encoder=clip_encoder()
    cfg=read(ROOT/'dacf_research/protocol.json')
    for name, registration in lock['cases'].items():
        out=destination/name
        out.mkdir()
        task=Experiment(name,registration,encoder,cfg,out)
        audits={}
        for ch in METRICS:
            axis, _, options, top=proposals(task,task.coarse,task.initial,ch,policy)
            audits[ch]={'global_change_total':axis['total'],'global_limit':axis['total']*.12,'largest_candidate_gap':top,
                        's0_initial_stop':top<=axis['total']*.12,'candidates':options}
        write(out/'initial_stops.json',audits)
        routes=[run_route(task,strategy,ch,policy) for strategy in ('S0','S1','endpoint_prior_control') for ch in METRICS]
        routes.append(run_route(task,'restricted_time',None,policy))
        if name=='independent_pan':
            old=read(R3/'protocol.json')
            write(out/'extension_r3_arms.json',[task.run(arm,ch,old) for arm in old['arms'] for ch in (METRICS if arm in {'E3','E4','E5','E6'} else [None])])
            write(out/'extension_baseline.json',{'plan':list(task.plan.values()),'pts':task.native,'nominees':sorted(task.initial)})
        write(out/'routes.json',routes)
        write(out/'density.json',density_bins(task))
        write(out/'receipts.json',task.reader.receipts)
        print(name,len(task.reader.receipts),'unique exact reads',flush=True)
    verify(lock)


if __name__=='__main__':
    main()
