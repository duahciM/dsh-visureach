import hashlib
import json
import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
V05=PROJECT/'v05_first_run_20261117'
V06=PROJECT/'v06_first_run_20261117'
OUT=Path(os.environ.get('V07_OUTPUT_DIR',str(PROJECT/'v07_first_run_20261117'))).resolve()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def historical_paths():
    for p in PROJECT.rglob('*'):
        if p.is_file() and not p.is_relative_to(OUT) and not any(x in p.parts for x in ('.venv','__pycache__','v07')):
            yield p
