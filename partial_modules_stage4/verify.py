"""Verify real source attachments and dependency scope on actual Stage 4 measurements."""

from argparse import ArgumentParser

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage2.ledger import Ledger
from partial_modules_stage4.prepare import H


def host_attachments():
    result=read(H/'host/result.json')
    case=read(H/'predictions/new-local/base_receipts.json')
    receipts={r['pts']:r for r in case['native_receipts']}
    assert result['source_sha256']==case['source_sha256'] and result['passed']
    paths={stamp:receipts[stamp]['path'] for stamp in result['pts']}
    actual_payloads={key:H/'host'/f'{key}.png' for key in result['typed_png_sha256']}
    for key,path in actual_payloads.items():
        assert digest(path)==key
    originals={}
    examined=[]
    for item in result['refs']:
        if item['type']!='original':
            continue
        originals[item['id']]=item
        path=paths[item['pts']]
        with Image.open(path) as a:
            rgb=a.convert('RGB')
        matches=[]
        for key,attachment in actual_payloads.items():
            with Image.open(attachment) as saved:
                if saved.size==rgb.size and saved.convert('RGB').tobytes()==rgb.tobytes():
                    matches.append(key)
        assert len(matches)==1
        examined.append({'type':'original','id':item['id'],'pts':item['pts'],'sha':matches[0]})
    for item in result['refs']:
        if item['type']!='derived':
            continue
        parent=originals[item['parent_evidence_id']]
        x,y,w,h=item['native_xywh']
        assert [x,y,w,h]==result['roi'][1]
        remaining=set(actual_payloads)-{r['sha'] for r in examined}
        assert len(remaining)==1
        key=next(iter(remaining))
        with Image.open(paths[parent['pts']]) as source, Image.open(actual_payloads[key]) as retrieved:
            assert source.convert('RGB').crop((x,y,x+w,y+h)).tobytes()==retrieved.convert('RGB').tobytes()
        examined.append({'type':'derived','parent':item['parent_evidence_id'],'pts':parent['pts'],'sha':key})
    assert len(examined)==3
    return examined


def dependency_check(out):
    data=read(H/'predictions/roles/R1P1.json')
    score=read(H/'evaluation.json')['cohorts']['roles']['arms']['R1P1']['box_and_relationship_scores']
    tested=next(x for x in score['assessable_two_time_rows'] if x['max_abs_error_native_px']==0)
    raw=next(x for x in data['m5']['raw_cross_time'] if x['measurement_ref']==tested['measurement_ref'])
    case=read(H/'predictions/roles/region_features.json')
    ref=next(x for x in case['charts'] if x['ref']==raw['reference_geometry'])
    journal=out/'scope.jsonl'; ledger=Ledger(journal,data['source_sha256'])
    ledger.add(ref['ref'],'background_geometry',ref['source_roots'],region_measurement=ref)
    ledger.add(raw['measurement_ref'],'raw_relation',raw['source_roots'],measurement=raw)
    ledger.add('trial-permanent-identity','identity_hypothesis',raw['source_roots'],
               scope='Deliberate reviewer test of a continuous identity claim with two measured endpoints')
    ledger.add('trial-trajectory','identity_dependent_relation',raw['source_roots'],[raw['measurement_ref'],'trial-permanent-identity'])
    ledger.add('unobserved-gap','appearance',raw['source_roots'],status='unassessed')
    try:
        ledger.withdraw('trial-permanent-identity','unobserved-gap','Test whether missing interior samples are negative evidence')
    except ValueError as exc:
        rejected=str(exc)
    else:
        raise AssertionError('Unobserved interior must not count as a refutation.')
    restored=list(Ledger(journal,data['source_sha256']).query())
    assert next(r for r in restored if r['id']==raw['measurement_ref'])['requires']==[]
    assert next(r for r in restored if r['id']=='trial-trajectory')['requires']==[raw['measurement_ref'],'trial-permanent-identity']
    return {'journal_reloaded':restored,'unknown_refutation_rejected':rejected,'assessed_region_difference':tested,
            'existing_withdrawal_test':str(ROOT/'partial_modules_stage3/ledger_check_verified/result.json')}


if __name__=='__main__':
    ap=ArgumentParser();ap.add_argument('--out',default='verification');args=ap.parse_args()
    folder=H/args.out
    if folder.parent!=H:
        raise ValueError('Use a study-local verification directory.')
    folder.mkdir()
    lock=read(H/'freeze.json')
    for group in ('historical','predictor_and_inputs'):
        assert all(digest(ROOT/name)==sha for name,sha in lock[group].items())
    assert digest(ROOT/lock['assessor']['file'])==lock['assessor']['sha256']
    result={'historical_count':len(lock['historical']),'frozen_inputs_count':len(lock['predictor_and_inputs']),
            'host_payloads':host_attachments(),'M6':dependency_check(folder)}
    write(folder/'integrity.json',result)
