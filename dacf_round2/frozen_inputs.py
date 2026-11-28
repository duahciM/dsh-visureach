"""Use the frozen production nominator on independent files and lock both cohorts."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from dacf_research.freeze import PLUGIN, validate as verify_round1
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_research.run import raw_frames
from evidenweft_phase1.source_review import timeline

HERE=Path(__file__).resolve().parent


def pin(paths):
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def run_independent_index():
    root=HERE/'independent_cache'
    root.mkdir()
    for src in sorted((HERE/'independent').glob('*.mp4')):
        capture=root/src.stem
        capture.mkdir()
        with (capture/'indexer.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),
                '--video',str(src),'--out',str(capture/'index'),'--fps','5'],stdout=log,stderr=subprocess.STDOUT,check=True)
        index=read(capture/'index/manifest.json')
        stream,frames=timeline(src)
        facts={r['pts']:r['frame'] for r in frames}
        assert all(facts[row['pts']]==row['source_frame'] for row in index['pool'])
        raw_frames(src,stream,frames,{row['source_frame'] for row in index['pool']},capture,index['video_sha256'],320)
        print('independent frozen nomination',src.stem,index['sampled_frame_count'],index['catalog_frame_count'],flush=True)


def registry():
    dev=ROOT/'dacf_research'
    cases={}
    for source in sorted((dev/'inputs').iterdir()):
        name=source.stem
        index=ROOT/'evidenweft_phase1/inventory'/name/'index' if name in {'bench','doorway'} else dev/'shared_catalogues'/name
        cases[name]={'source':source,'index':index,'pixels':dev/'run_automatic'/name,'cohort':'replay'}
    for source in sorted((HERE/'independent').iterdir()):
        name=source.stem
        cache=HERE/'independent_cache'/name
        cases[name]={'source':source,'index':cache/'index','pixels':cache,'cohort':'independent'}
    return cases


def freeze():
    historical=verify_round1('manual')
    cases=registry()
    input_files=[HERE / x for x in ('independent_inputs.py','independent_sources.json','frozen_inputs.py','regions.py','investigate.py','protocol.json')]
    for c in cases.values():
        input_files.extend([c['source'],c['index']/'manifest.json',c['index']/'catalog.json',c['pixels']/'frame_receipts.json'])
    r1=ROOT/'dacf_research'
    retained=[r1/'lock.json',r1/'source_verification.json',r1/'phase1_attribution.json',r1/'run_manifest.json']
    for arm in ('manual','automatic'):
        retained.extend(p for p in (r1/f'run_{arm}').rglob('*') if p.name in
            {'summary.json','pair_results.json','pair_evidence.json.gz','local_holdout.json.gz','automatic_groups.json'})
    references=[HERE/'references/roles.json',ROOT/'dacf_research/evaluation_only/roles.json',
                ROOT/'evidenweft_phase1/evaluation_only/rubric.json',*sorted((HERE/'references').glob('*-source-timeline.json'))]
    write(HERE/'lock.json',{'locked_at':datetime.now().astimezone().isoformat(),
        'previous_baseline':historical['baseline'],'inputs':pin(input_files),'preserved_round1_outputs':pin(retained),
        'reference_hashes_scoring_only':pin(references),
        'cases':{n:{k:v.relative_to(ROOT).as_posix() if isinstance(v,Path) else v for k,v in c.items()} for n,c in cases.items()}})


if __name__=='__main__':
    if '--index' in sys.argv:
        run_independent_index()
    else:
        freeze()
