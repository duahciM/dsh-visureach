import os
from pathlib import Path

from identity_feature_decomposition.common import FIRST, PROJECT, V05, V06, read, sha

PACKAGE = Path(__file__).resolve().parent
IF02 = PROJECT/'identity_feature_decomposition_20261118'
V08 = PROJECT/'v08_first_run_20261117'
OUT = Path(os.environ.get('IF03_OUT', str(PROJECT/'identity_feature_grouping_20261118'))).resolve()
CFG = read(PACKAGE/'protocol.json')


def historical_files():
    for path in PROJECT.rglob('*'):
        if path.is_file() and not path.is_relative_to(PACKAGE) and not path.is_relative_to(OUT):
            if not {'.venv','__pycache__'}.intersection(path.parts):
                yield path
