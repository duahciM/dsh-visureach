import os
from pathlib import Path

from identity_catalog_matching.common import FIRST, IF02, IF03, PROJECT, V05, V06, read, sha, source_ref

PACKAGE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('IF05_OUT',PROJECT/'identity_small_support_spatial_20261118')).resolve()
IF04=PROJECT/'identity_catalog_matching_20261118'
CFG=read(PACKAGE/'protocol.json')


def history_files():
    for p in PROJECT.rglob('*'):
        if p.is_file() and not p.is_relative_to(PACKAGE) and not p.is_relative_to(OUT) and not {'.venv','__pycache__'}.intersection(p.parts):
            yield p
