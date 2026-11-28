"""Check deterministic outcomes before binding window replays to new run artifacts."""

from argparse import ArgumentParser

from dacf_research.prepare_inputs import digest,read,write
from partial_modules_stage7.inputs import H


def binding():
    report=read(H/'automatic/summary.json')
    assert not report['read_assessor'] and not report['failures']
    rows={}; signatures={}
    for name in report['cases']:
        file=H/'automatic'/name/'results.json'
        source=read(file)
        rows[name]={'source':source['source_sha256'],'pool_pts':source['pool_pts'],'first_pts':source['first_pts'],
            'arms':{key:{'accepted':r['accepted'],'closed_scope':r['closed_scope'],'causes':[
                [t['pts'],t['state'],t['cause']] for t in r['audit']]} for key,r in source['arms'].items()}}
        signatures[name]=digest(file)
    return rows,{'automatic':signatures,'source_code':{key:digest(H/key) for key in ('windowed.py','window_run.py')}}


if __name__=='__main__':
    args=ArgumentParser();args.add_argument('--verify',action='store_true');req=args.parse_args()
    stable,output=binding()
    assert stable==read(H/'canonical_predictions.json')
    if req.verify:
        previous=read(H/'window_lock.json')
        assert all(previous[key]==value for key,value in output.items())
    else:
        write(H/'window_lock.json',output|{'state':'fresh equivalent run; fixed before reference values'})
