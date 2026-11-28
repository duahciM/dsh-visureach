"""Run the unchanged predictor from a separate output root.

The independent cohort's source folder already occupies the default output name.
This entry keeps the input folder and the original lock byte-for-byte intact.
"""
import shutil
import sys

from dacf_research.prepare_inputs import digest
from dacf_round2.frozen_inputs import HERE
from dacf_round2 import investigate


def main():
    folder=HERE/'independent_execution'
    folder.mkdir()
    (folder/'replay').mkdir()
    for rel in ('lock.json','protocol.json','replay/summary.json'):
        shutil.copyfile(HERE/rel,folder/rel)
        assert digest(HERE/rel)==digest(folder/rel)
    investigate.HERE=folder
    sys.argv=[sys.argv[0],'independent']
    investigate.main()


if __name__=='__main__':
    main()
