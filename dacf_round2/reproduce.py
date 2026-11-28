"""Replay the sealed offline protocol without replacing any existing run."""
import argparse
from pathlib import Path
from shutil import copyfile
import sys
from unittest.mock import patch

from dacf_research.prepare_inputs import ROOT, digest, read
from dacf_round2 import evaluate, investigate, render, run_independent, verify_round2
from dacf_round2.frozen_inputs import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='New output directory')
    out = parser.parse_args().out.resolve()

    lock = read(HERE / 'lock.json')
    for group in ('inputs', 'preserved_round1_outputs', 'reference_hashes_scoring_only'):
        if not all(digest(ROOT / path) == expected for path, expected in lock[group].items()):
            raise ValueError(f'Sealed input mismatch in {group}')
    out.mkdir()
    (out / 'references').mkdir()
    for relative in ('lock.json', 'protocol.json', 'references/roles.json'):
        copyfile(HERE / relative, out / relative)

    with patch.object(investigate, 'HERE', out), patch.object(sys, 'argv', [sys.argv[0], 'replay']):
        investigate.main()
    with patch.object(investigate, 'HERE', out), patch.object(run_independent, 'HERE', out), patch.object(sys, 'argv', [sys.argv[0]]):
        run_independent.main()
    for stage in (evaluate, verify_round2, render):
        with patch.object(stage, 'HERE', out):
            stage.main()


if __name__ == '__main__':
    main()
