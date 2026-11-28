import hashlib
import json
import os
from pathlib import Path

from p1.io import dump

PROJECT=Path(__file__).resolve().parents[1]
OLD=PROJECT/'v05_first_run_20261117'
OUT=Path(os.environ.get('V06_OUTPUT_DIR',str(PROJECT/'v06_first_run_20261117'))).resolve()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def paths():
    for p in PROJECT.rglob('*'):
        if p.is_file() and not p.is_relative_to(OUT) and not any(x in p.parts for x in ('.venv','__pycache__','v06')):
            yield p


def main():
    if OUT.exists():raise SystemExit('Refuse overwrite: first-round output already exists')
    OUT.mkdir()
    cases=read(OLD/'run_manifest.json')['cases']
    history={str(p.relative_to(PROJECT)):sha(p) for p in paths()}
    sources={c['path']:sha(c['path']) for c in cases}
    pools={c['id']:{name:sha(OLD/c['id']/name) for name in ('frames.json','scene_ids.json','similarity_matrices.npz','pair_evidence.jsonl','motion_quality.json')} for c in cases}
    dump(OUT/'history_freeze.json',{'historical_files':history,'sources':sources,'candidate_pools':pools,'cases':cases,
         'note':'before v06 predictions; includes all v05 code/output and old history; excludes venv/cache/new v06 code'})
    print('FROZEN',len(history),'historical files and',len(cases),'source/pair pools')


if __name__=='__main__':main()
