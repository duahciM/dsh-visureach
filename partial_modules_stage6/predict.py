"""Prepare common originals, reuse DACF inputs, and emit only automatic results."""

from collections import Counter
import gzip
import json
import subprocess
import sys
from time import perf_counter
import traceback

import cv2

from dacf_research.field import Bank, clip_encoder
from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump, raw_frames
from dacf_round2.investigate import evidence
from dacf_round2.regions import discover
from evidenweft_phase1.source_review import timeline
from partial_modules_stage6.envelopes import qualify
from partial_modules_stage6.material import H


def gzip_json(path):
    with gzip.open(path,'rt',encoding='utf8') as handle:
        return json.load(handle)


def warm_bank(bank,folder):
    refs={r['id']:rank for rank,r in bank.rows.items()}
    new_counts={}
    for file,target in [('pair_evidence.json.gz',bank.evidence),('local_holdout.json.gz',bank.links)]:
        rows= gzip_json(folder/file)
        for row in rows:
            a,b=row['root'].removesuffix(':local_heldout').split('>')
            if a not in refs or b not in refs:
                continue
            x,y=refs[a],refs[b]
            target[x,y]=row
            if 'metrics' in row:
                bank.matrix[tuple(sorted((x,y)))]=row['metrics']
        new_counts[file]=len(target)
    return new_counts


def predict(name,spec,out,encoder,encoder_error):
    out.mkdir()
    source=ROOT/spec['source']
    assert digest(source)==spec['sha256']
    cfg=read(ROOT/'dacf_research/protocol.json')
    policy=read(ROOT/'dacf_round2/protocol.json')
    index=ROOT/spec['initial_index'] if spec['initial_index'] else out/'initial_index'
    start=perf_counter()
    if not spec['initial_index']:
        with (out/'indexer.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),
              '--video',str(source),'--out',str(index),'--fps',str(cfg['fps']),'--keep-analysis'],
              check=True,stdout=log,stderr=subprocess.STDOUT)
    initial_time=perf_counter()-start
    signatures={k:digest(index/k) for k in ('manifest.json','catalog.json')}
    original=read(index/'manifest.json')
    catalogue={r['pts']:r for r in read(index/'catalog.json')['frames']}
    assert original['video_sha256']==spec['sha256']
    start=perf_counter()
    info,clock=timeline(source)
    coarse=[r['source_frame'] for r in original['pool']]
    raw=raw_frames(source,info,clock,set(coarse),out,spec['sha256'],cfg['analysis_width'])
    decode_time=perf_counter()-start
    windows=[coarse]
    if name=='long_street':
        windows=[[n for n in coarse if lo<=raw[n]['pts']<=hi] for lo,hi in [(0,60),(360,420),(730,794)]]
    feature_input={r:raw[r] for window in windows for r in window}
    start=perf_counter();bank=Bank(feature_input,cfg,encoder);feature_time=perf_counter()-start
    warm={}
    start=perf_counter()
    if spec['r2_history']:
        history=ROOT/spec['r2_history']
        previous=read(history/'regions.json')
        assert previous['source_sha256']==spec['sha256'] and previous['nomination_pts']==list(catalogue)
        regions=previous['regions']
        warm=warm_bank(bank,history)
        signal_source=str((history/'region_signals.json.gz').relative_to(ROOT))
    else:
        regions=[]
        signals=[]
        for i,win in enumerate(windows):
            local=discover(bank,win,catalogue,policy)
            rename=lambda ident:f'w{i+1}-{ident}'
            for r in local['regions']:
                item=r|{'id':rename(r['id']),'children':[rename(x) for x in r['children']]}
                regions.append(item)
                signals.append(evidence(bank,item,win,catalogue))
        gz_dump(out/'round2_signals.json.gz',signals)
        signal_source=str((out/'round2_signals.json.gz').relative_to(ROOT))
    r2_time=perf_counter()-start
    initial_pair_keys=set(bank.evidence);initial_local_keys=set(bank.links)
    start=perf_counter()
    result=qualify(bank,windows,catalogue,regions,policy,spec['sha256'],spec['extent'])
    elapsed=perf_counter()-start
    if initial_pair_keys!=set(bank.evidence):
        gz_dump(out/'supplement_pairs.json.gz',[v for k,v in bank.evidence.items() if k not in initial_pair_keys])
    if initial_local_keys!=set(bank.links):
        gz_dump(out/'supplement_holdout.json.gz',[v for k,v in bank.links.items() if k not in initial_local_keys])
    assert signatures=={k:digest(index/k) for k in signatures}
    result.update({'sha256':spec['sha256'],'extent':spec['extent'],'time_base':info['time_base'],
        'initial_index':str(index.relative_to(ROOT)),'index_sha':signatures,'first_pts':list(catalogue),
        'pool_pts':[raw[x]['pts'] for x in coarse],
        'evaluated_windows':[[raw[x[0]]['pts'],raw[x[-1]]['pts']] for x in windows],
        'r2_regions':regions,'r2_signal_file':signal_source,'initial_catalogue_was_reused':bool(spec['initial_index']),
        'cost':{'initial_index_seconds':initial_time,'native_decode_seconds':decode_time,'probed_full_source_frames':len(clock),
         'retained_shared_native_frames':len(raw),'new_or_cached_feature_frames':len(feature_input),'feature_seconds':feature_time,
         'existing_dacf_prepare_seconds':r2_time,'envelope_seconds':elapsed, 'cached_pair_and_holdout':warm,
         'new_envelope_pair_count':len(set(bank.evidence)-initial_pair_keys),
         'new_envelope_holdout_count':len(set(bank.links)-initial_local_keys),
         'clip':bank.clip_state,'clip_startup_error':encoder_error}})
    write(out/'auto.json',result)
    return {'case':name,'envelopes':len(result['accepted']),'closed':len(result['appearance_scope']['closed']),
            'compared_regimes':len(result['all_candidates']),'R2_regions':len(regions),
            'unresolved':dict(Counter(reason for r in result['all_candidates'] for reason in r['reasons'])),'cost':result['cost']}


if __name__=='__main__':
    lock=read(H/'freeze.json')
    assert all(digest(ROOT/k)==v for k,v in lock['predictors'].items())
    outputs=H/'automatic';outputs.mkdir()
    cv2.setNumThreads(1)
    try:
        encoder=clip_encoder();error=None
    except (ImportError,OSError,RuntimeError,ValueError) as exc:
        encoder=None;error=f'{type(exc).__name__}: {exc}'
    results=[];failures={}
    for name,spec in read(H/'inputs/cases.json').items():
        cv2.setRNGSeed(730)
        try:
            row=predict(name,spec,outputs/name,encoder,error)
            print(name,row['envelopes'],row['closed'],row['R2_regions'],flush=True)
            results.append(row)
        except (ValueError,AssertionError,KeyError,OSError,RuntimeError,subprocess.CalledProcessError):
            failures[name]=traceback.format_exc()
            print(failures[name],flush=True)
    write(outputs/'summary.json',{'results':results,'failed':failures,'opened_assessor':False})
