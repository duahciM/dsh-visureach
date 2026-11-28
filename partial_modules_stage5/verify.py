"""Revalidate immutable files, nomination rows, and exact source pixels."""

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.source_review import native_stream, timeline
from partial_modules_stage5.review_sources import H


def verify_six():
    base=ROOT/'plugin_offline/motion-source'
    def load(index):
        catalog=read(index/'catalog.json')
        source=read(index/'manifest.json')
        rows=Counter((fr['frame_id'], n['dimension_id'], json.dumps(n,sort_keys=True)) for fr in catalog['frames'] for n in fr['nominations'])
        pool=[tuple(f[k] for k in ('frame_id','pts','source_frame','seconds')) for f in source['pool']]
        return source['video_sha256'],pool,rows
    sha,pool,rows=load(base)
    result={}
    for i in range(1,7):
        dim=f'D{i}'
        other,positions,test=load(ROOT/f'plugin_offline/motion-no-{dim}')
        assert sha==other and pool==positions
        expected=Counter({k:v for k,v in rows.items() if k[1]!=dim})
        assert test==expected
        result[dim]={'own_removed':sum(rows.values())-sum(expected.values()),'all_other_reasons_exact':sum(expected.values())}
    return {'source_sha256':sha,'pool_frame_count':len(pool),'switches':result}


def verify_images():
    host=read(H/'host-verified/result.json')
    probe=host['probe']['pts']
    src=ROOT/'dacf_round6/independent/source.mp4'
    sig=digest(src);assert sig==host['source_sha256']
    info,clock=timeline(src)
    assert probe in [r['pts'] for r in clock]
    assert Fraction(info['time_base'])*probe==Fraction(str(host['probe']['seconds']))
    sha=host['original_attachment_without_scene'].removeprefix('sha256:')
    img=H/'host-verified'/f'{sha}.png'
    assert digest(img)==sha
    with Image.open(img) as stored:
        rgb=stored.convert('RGB').tobytes()
    for row,bits in native_stream(src,info,clock):
        if row['pts']==probe:
            assert rgb==bits
            break
    else:
        raise ValueError('Exact host original not found by sequential decode.')
    q=host['checked_stopped_node']['policy_hint']
    assert q['policy']=='S0' and len(q['statuses'])==4 and all(d['state']=='stopped' for d in q['statuses'].values())
    return {'sha256':sig,'pts':probe,'png_sha256':sha,'independent_exact_native_rgb':True}


if __name__=='__main__':
    lock=read(H/'freeze.json')
    assert all(digest(ROOT/name)==sha for name,sha in lock['history_files'].items())
    assert all(digest(H/name)==sha for name,sha in lock['protocol_code_and_review'].items())
    for name,sha in lock['input_sources'].items():
        path=read(H/'source_review'/name/'receipts.json')['source']
        assert digest(ROOT/path)==sha
    scene=read(H/'scene_scope.json')['cases']
    assert all(not a['automatic_scope']['closed'] for a in scene.values())
    assert {k:len(v['reviewed_scope']['closed']) for k,v in scene.items()}=={'detour':2,'cut':0,'camera':0,'real_return':1}
    manifest,cat=('manifest.json','catalog.json')
    initial=read(H/'real_round2/regions.json')['frozen_catalogue_and_pool']
    assert all(digest(H/'real_round2/index'/name)==initial[name] for name in (manifest,cat))
    comparisons=read(H/'local_replay/summary.json')
    assert comparisons['fixed_catalogue']==initial
    result={'unchanged_historical_files':len(lock['history_files']),'checked_scope_cases':len(scene),
            'first_pool_and_catalogue_unchanged':initial,'independent_frame_proof':verify_images(),
            'six_frozen_publication_switches_byte_equivalent_other_rows':verify_six()}
    write(H/'verified.json',result)
