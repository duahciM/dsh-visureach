"""Bind a fresh reproduction to the same source, candidate bounds, and coarse pool."""

from argparse import ArgumentParser

from dacf_research.prepare_inputs import digest, read, write
from partial_modules_stage6.material import H


def summary():
    status=read(H/'automatic/summary.json')
    assert not status['opened_assessor'] and not status['failed']
    rows={}
    signatures={}
    for item in status['results']:
        name=item['case']
        path=H/'automatic'/name/'auto.json'
        p=read(path)
        signatures[name]=digest(path)
        rows[name]={k:p[k] for k in ('sha256','evaluated_windows','first_pts','pool_pts','accepted','appearance_scope')}
        rows[name]['candidate_decisions']=[{key:row[key] for key in ('bounds_pts','state','reasons')}
                                           for row in p['all_candidates']]
    return rows,signatures


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    rows,signatures=summary()
    assert rows==read(H/'canonical_predictions.json'), 'Stage 6 prediction decisions differ from the archived run.'
    lock=H/'downstream_lock.json'
    if args.verify:
        expected=read(lock)
        assert expected['automatic_outputs']==signatures and expected['code_sha256']==digest(H/'downstream.py')
    else:
        write(lock,{'code_sha256':digest(H/'downstream.py'),'automatic_outputs':signatures,
                    'scope':'New reproduction with the same source, coarse ranks, decisions and scope.'})
