"""Use the existing catalog and DACF Round 2 implementations on a source new to Round 2."""

import subprocess
import sys
from time import monotonic

import cv2

from dacf_research.field import Bank, clip_encoder
from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import gz_dump, raw_frames
from dacf_round2.investigate import evidence
from dacf_round2.regions import discover
from evidenweft_phase1.source_review import timeline
from partial_modules_stage5.review_sources import H


if __name__=='__main__':
    out=H/'real_round2';out.mkdir()
    src=H/'inputs/11.mp4'
    lock=read(H/'freeze.json')
    assert digest(src)==lock['input_sources']['real_return']
    with (out/'indexer.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),
            '--video',str(src),'--out',str(out/'index'),'--fps','5','--keep-analysis'],stdout=log,stderr=subprocess.STDOUT,check=True)
    catalog=out/'index/catalog.json';manifest=out/'index/manifest.json'
    before={p.name:digest(p) for p in (catalog,manifest)}
    index=read(manifest); catalogue={r['pts']:r for r in read(catalog)['frames']}
    stream, clock=timeline(src)
    config=read(ROOT/'dacf_research/protocol.json')
    pool=[p['source_frame'] for p in index['pool']]
    start=monotonic()
    raw=raw_frames(src,stream,clock,set(pool),out,index['video_sha256'],config['analysis_width'])
    prep=monotonic()-start
    encoder_error=None
    try:
        encoder=clip_encoder()
    except (ImportError,OSError,RuntimeError,ValueError) as exc:
        encoder=None
        encoder_error=f'{type(exc).__name__}: {exc}'
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    start=monotonic();bank=Bank(raw,config,encoder)
    result=discover(bank,pool,catalogue,read(ROOT/'dacf_round2/protocol.json'))
    assert result['nomination_pts']==list(catalogue)
    signals=[evidence(bank,r,pool,catalogue) for r in result['regions']]
    assert before=={p.name:digest(p) for p in (catalog,manifest)}
    result.update({'source_sha256':index['video_sha256'],'frozen_catalogue_and_pool':before,'time_base':stream['time_base'],
                   'source_bounds_pts':[clock[0]['pts'],clock[-1]['pts']], 'coarse_pts':[index['pool'][i]['pts'] for i in range(len(pool))]})
    write(out/'regions.json',result)
    gz_dump(out/'region_signals.json.gz',signals)
    gz_dump(out/'pair_evidence.json.gz',list(bank.evidence.values()))
    gz_dump(out/'local_holdout.json.gz',list(bank.links.values()))
    write(out/'receipt.json',{'original_full_decode_frames':len(clock),'original_retained_coarse':len(raw),'catalogue_count':len(catalogue),
          'clip':bank.clip_state,'clip_startup_failure':encoder_error,'coarse_native_seconds':prep,
          'evidence_seconds':monotonic()-start,'regions':len(signals)})
