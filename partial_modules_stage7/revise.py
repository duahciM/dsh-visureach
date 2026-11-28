"""Add one actual AMR read as a new evidence version, with no overwrite of automatic results."""

import gzip
import json
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest,read,write
from dacf_research.field import clip_encoder
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline
from partial_modules_stage6.downstream import frames
from partial_modules_stage6.predict import warm_bank
from partial_modules_stage7.dacf import qualify
from partial_modules_stage7.inputs import H
from partial_modules_stage7.local_observation import Layout,local_groups
from partial_modules_stage7.predict import build_feature_input


if __name__=='__main__':
    out=H/'revision-v1';out.mkdir()
    case='real_return'
    old_lock=H/'automatic'/case/'results.json'; baseline=read(old_lock)
    readout=read(H/'one-refinement/result.json')
    before=digest(old_lock)
    assert before==readout['original_v7_auto_sha_unchanged']
    old=read(ROOT/'partial_modules_stage6/automatic'/case/'auto.json')
    spec=read(H/'inputs/cases.json')[case]
    movie=ROOT/spec['source'];info,clock=timeline(movie)
    a=readout['added']
    assert a['pts'] not in baseline['pool_pts'] and spec['sha256']==readout['source_sha256']==digest(movie)
    start=perf_counter()
    verifier=Reader(movie,spec['sha256'],info,clock,out/'source-proof')
    fresh,_=verifier.read(a['pts']); proof_time=perf_counter()-start
    attachments=readout['image_payloads']
    with Image.open(fresh['path']) as independent:
        w,h=independent.size
        pixels=independent.convert('RGB').tobytes()
    hits=[]
    for attach in dict.fromkeys(attachments):
        candidate=H/'one-refinement'/f"{attach.removeprefix('sha256:')}.png"
        with Image.open(candidate) as actual:
            rgb=actual.convert('RGB')
            if rgb.tobytes()==pixels:
                hits.append((attach,np.asarray(rgb.resize((320,max(8,round(h*320/w))),Image.Resampling.BILINEAR))))
    assert len(hits)==1
    attachment,source_rgb=hits[0]
    d=frames(ROOT/spec['stage6'])
    added=fresh|{'rgb':cv2.cvtColor(source_rgb,cv2.COLOR_RGB2BGR)}
    all_rows=sorted(d+[added],key=lambda r:r['pts'])
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    t=perf_counter()
    bank,windows=build_feature_input(all_rows,old['evaluated_windows'],clip_encoder())
    feature_seconds=perf_counter()-t
    r2=read(ROOT/'dacf_round2/protocol.json')
    warm_bank(bank,ROOT/spec['r2_history'])
    model=Layout(bank)
    with gzip.open(H/'automatic'/case/'layout_fit.json.gz','rt',encoding='utf8') as stream:
        model.pairs.update({tuple(r['ranks']):r for r in json.load(stream)})
    initial=len(model.pairs)
    t=perf_counter()
    groups=[local_groups(model,w,bank.config['maximum_gap_seconds'],bank.config['automatic']['dwell_seconds'],
                r2['prototype_neighborhood']['radius_s']) for w in windows]
    catalogue={r['pts']:r for r in read(ROOT/old['initial_index']/'catalog.json')['frames']}
    revised=qualify(bank,model,windows,groups,catalogue,old['r2_regions'],r2,spec['sha256'],spec['extent'],True,True)
    compute=perf_counter()-t
    nearest=sorted((r for r in bank.rows if r!=fresh['frame']),key=lambda r:abs(bank.rows[r]['pts']-a['pts']))[:2]
    addition=[model.compare(fresh['frame'],p) for p in nearest]
    assert before==digest(old_lock)
    write(out/'result.json',{'revision':'one actual extra source sample','previous_sha256':before,'native_verified':fresh,
        'before_LP_envelopes':baseline['arms']['LP']['accepted'],'revised_LP':revised,'new_groups':groups,
        'adjacent_subsegments':addition,'extra_AMR_original_requests':len(attachments),
        'extra_independent_source_reads':1,'verified_attachment_by_exact_RGB':attachment,'feature_frames_in_revision':len(bank.rows),
        'new_layout_pairs':len(model.pairs)-initial,'feature_seconds':feature_seconds,'new_comparisons_seconds':compute,
        'independent_source_proof_seconds':proof_time,'AMR_host_elapsed_seconds':readout['elapsed_ms']/1000})
