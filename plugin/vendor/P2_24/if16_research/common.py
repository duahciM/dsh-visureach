"""Frame identities, frozen inputs and dataset receipts."""
from pathlib import Path

import cv2
import numpy as np

from identity_catalog_matching.common import read, sha
from p1.core import quality
from p1.io import dump

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IF15 = ROOT / "if15_research_20261120"
RULES = read(HERE / "protocol.json")
COMPONENTS = read(ROOT / "if09_research" / "protocol.json")["P2"]


def frame_key(source, row):
    return f'{source["sha256"]}:{row["source_frame"]}:{row["pts"]}:{row["time_base"]}'


def get_meta(case):
    return read(IF15 / "P2" / case / "frames.json")


def frame_pixels(row):
    rgb = cv2.imread(row["image"])
    if rgb is None:
        raise ValueError(f'Image missing at original source frame {row["source_frame"]}')
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return row | {"rgb": rgb, "gray": gray} | quality(gray)


def sequence(rows):
    return [frame_pixels(f) for f in rows]


def index_receipt(out, cases):
    inputs = {}
    image_count = 0
    for case in cases:
        folder = IF15 / "P2" / case
        m = get_meta(case)
        assert sha(m["source"]["path"]) == m["source"]["sha256"]
        paths = [folder / f for f in ("frames.json", "selection.json", "relations.json")]
        paths.extend(Path(r["image"]) for r in m["frames"])
        inputs[case] = {"source": m["source"], "candidate_count": len(m["frames"]),
                        "input_hashes": {str(p): sha(p) for p in paths}}
        image_count += len(m["frames"])
    dump(out / "prediction_lock.json", {"protocol": sha(HERE / "protocol.json"),
        "frozen_input_rows": image_count, "clips": inputs,
        "historical_component_config": sha(ROOT / "if09_research" / "protocol.json"),
        "prior_run": sha(IF15 / "run_manifest.json"), "annotations_loaded": False,
        "g3_source_type": "real recording, user supplied in execution card"})
