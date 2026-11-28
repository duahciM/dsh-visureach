import os
from pathlib import Path

from identity_feature_decomposition.common import FIRST, PROJECT, V05, V06, read, sha

PACKAGE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('IF04_OUT', PROJECT/'identity_catalog_matching_20261118')).resolve()
IF02 = PROJECT/'identity_feature_decomposition_20261118'
IF03 = PROJECT/'identity_feature_grouping_20261118'
V08 = PROJECT/'v08_first_run_20261117'
CFG = read(PACKAGE/'protocol.json')


def history_files():
    for path in PROJECT.rglob('*'):
        if (path.is_file() and not path.is_relative_to(PACKAGE) and not path.is_relative_to(OUT)
                and not {'.venv','__pycache__'}.intersection(path.parts)):
            yield path


def source_ref(frame):
    return {k:frame[k] for k in ('sample','source_frame','pts','time_base','t','image')}


def duplicate(hsv, gray, a, b):
    return bool(a != b and hsv[a,b] <= .04 and gray[a,b] <= .02)
