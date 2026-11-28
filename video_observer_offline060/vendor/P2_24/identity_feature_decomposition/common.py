import os
from pathlib import Path

from identity_first.common import PROJECT, V05, V06, read, sha

PACKAGE = Path(__file__).resolve().parent
FIRST = PROJECT / 'identity_first_20261118'
OUT = Path(os.environ.get('IF02_OUT', str(PROJECT / 'identity_feature_decomposition_20261118'))).resolve()
CFG = read(PACKAGE / 'protocol.json')


def history_files():
    for path in PROJECT.rglob('*'):
        if path.is_file() and not path.is_relative_to(PACKAGE) and not path.is_relative_to(OUT):
            if not {'.venv', '__pycache__'}.intersection(path.parts):
                yield path
