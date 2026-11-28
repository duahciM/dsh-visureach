"""Create a read-only manifest over P2-15 and append one separate holdout."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from if17_research.catalog import read, ref, sha
from p1.io import dump
from p215_research.prepare import EXCLUDED_SHA256, ROOT, supplemental
from p215_research.semantic import Clip, indexed_case


HERE=Path(__file__).resolve().parent
OLD=ROOT/"p215_discovery_20261121"
CFG=read(HERE/"protocol.json")
HOLDOUT="flower"
HOLDOUT_URL="https://mdn.github.io/shared-assets/videos/flower.mp4"


def fresh_semantic(out, item):
    target=out/"semantic"/HOLDOUT
    target.mkdir(parents=True,exist_ok=True)
    book=read(item["catalog"]["path"])
    with (OLD/"assets/openimages.csv").open(encoding="utf-8-sig",newline="") as source:
        lexicon=sorted({row[1].casefold() for row in csv.reader(source) if len(row)==2})
    model=Clip(OLD)
    started=perf_counter()
    vocab=model.text([f"a photo of a {word}." for word in lexicon])
    rows,vectors,errors=indexed_case(book,lexicon,vocab,model)
    assert [r["frame_id"] for r in rows]==[r["frame_id"] for r in book["frames"]]
    dump(target/"candidates.json",{"rows":rows,"errors":errors,
         "extractor":read(OLD/"semantic/run_manifest.json")["extractor"]})
    np.save(target/"vectors.npy",vectors,allow_pickle=False)
    names=Counter(tag["word"] for row in rows for node in row["nodes"] for tag in node["visual_hypotheses"])
    dump(target/"metrics.json",{"images":len(rows),"regions":len(vectors),"terms":len(names),
         "edges":sum(len(row["relations"]) for row in rows),"errors":len(errors),
         "elapsed_seconds":perf_counter()-started})


def freeze_catalogs(out, video, card):
    if (out/"inputs_lock.json").exists():
        raise ValueError("The prediction input lock already exists")
    books=read(OLD/"library/manifest.json")
    heldout=sha(video)
    if heldout in {EXCLUDED_SHA256,*(v["source"]["sha256"] for v in books.values())}:
        raise ValueError("Excluded or duplicated holdout source")
    for case, entry in books.items():
        entry["semantic"]=str(OLD/"semantic"/case)
    # Reuse the existing new-video 2 fps and 30 second computational-window procedure.
    fresh=supplemental(out,"fresh",video)
    fresh["semantic"]=str(out/"semantic"/HOLDOUT)
    fresh["basis"]="New MDN holdout with the unchanged P2-15 supplemental nomination procedure"
    fresh_semantic(out,fresh)
    books[HOLDOUT]=fresh
    dump(out/"library/manifest.json",books)
    prior=read(OLD/"public_goals.json")
    goals=prior+[
        {"id":"flower-explore","case":HOLDOUT,"text":"What is worth noticing? Show complementary before, middle and after views with a stable image reference."},
        {"id":"flower-detail","case":HOLDOUT,"text":'When does the yellow center inside the "red flower" become clearly visible? Inspect between samples from the source when necessary.'}
    ]
    dump(out/"public_goals.json",goals)
    checks={}
    for entry in books.values():
        if entry["source"]["sha256"]==EXCLUDED_SHA256:
            raise ValueError("User-excluded source in a reused catalog")
        for path in [Path(entry["semantic"])/name for name in ("candidates.json","vectors.npy","metrics.json")]:
            checks[str(path)]=sha(path)
        for kind in ("source","catalog","index","coverage"):
            record=entry[kind]
            assert sha(record["path"])==record["sha256"]
            checks[record["path"]]=record["sha256"]
        book=read(entry["catalog"]["path"])
        for frame in book["frames"]:
            native=frame["source_image"]
            assert native["path"] not in checks or checks[native["path"]]==native["sha256"]
            checks[native["path"]]=native["sha256"]
    for path in (HERE/"prepare.py",HERE/"navigate.py",HERE/"graph.py",HERE/"protocol.json",
                 ROOT/"p215_research/prepare.py",ROOT/"p215_research/retrieval.py",ROOT/"p215_research/semantic.py",
                 ROOT/"if17_research/browse.py",ROOT/"if17_research/catalog.py",ROOT/"if16_research/nominate.py",
                 OLD/"assets/text.onnx",OLD/"assets/vision.onnx",OLD/"assets/tokenizer.json",OLD/"assets/openimages.csv",
                 OLD/"library/manifest.json",OLD/"public_goals.json",OLD/"P2_15_REPORT.md",OLD/"evaluation.json",
                 OLD/"protocol_lock.json",OLD/"retrieval/manifest.json",out/"library/manifest.json",out/"public_goals.json"):
        checks[str(path.resolve())]=sha(path)
    trials=read(OLD/"retrieval/manifest.json")["trials"]
    checks.update({item["path"]:item["sha256"] for item in trials.values()})
    dump(out/"inputs_lock.json",{"files":checks,"card":ref(card),"holdout":ref(video)|{"url":HOLDOUT_URL},
        "excluded_user_source_sha256":EXCLUDED_SHA256,"frozen_prior_count":len(prior),"frozen_prior_trials":len(trials),
        "reference_storage":"independent evaluator only; no path or reference values passed to the navigator"})
    print("Locked",len(checks),"files with",len(books),"catalogs and",len(goals),"goals",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--video",required=True,type=Path)
    parser.add_argument("--card",required=True,type=Path)
    args=parser.parse_args()
    freeze_catalogs(args.out.resolve(),args.video.resolve(),args.card.resolve())
