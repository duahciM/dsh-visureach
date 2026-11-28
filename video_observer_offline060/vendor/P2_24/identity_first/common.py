import os
from pathlib import Path

from v08.common import PROJECT, V05, V06, V07, read, sha

V08 = PROJECT / 'v08_first_run_20261117'
PACKAGE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('IDENTITY_FIRST_OUT', str(PROJECT / 'identity_first_20261118'))).resolve()
CFG = read(PACKAGE / 'protocol.json')


def history_files():
    for path in PROJECT.rglob('*'):
        if path.is_file() and not path.is_relative_to(OUT) and not path.is_relative_to(PACKAGE):
            if '.venv' not in path.parts and '__pycache__' not in path.parts:
                yield path
