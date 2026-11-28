"""Freeze or verify all prior artifacts and the CP-IF16 execution card."""
import argparse
import sys
from pathlib import Path

from identity_catalog_matching.common import read,sha
from if10_research import freeze as engine
from p1.io import dump

HERE=Path(__file__).resolve().parent
CARD=engine.MEDIA.parent/"video-keyframes-research-CP-IF16-unbounded-multidimensional-next.md"


def verify(out):
    recipe=read(out/"history_lock.json")
    old=recipe["history"]
    new=engine.history(out)
    paths={r["path"]:r["sha256"] for r in recipe["media"]+list(recipe["historical_sources"].values())}
    paths[recipe["card"]["path"]]=recipe["card"]["sha256"]
    missing=sorted(old.keys()-new.keys())
    additions=sorted(new.keys()-old.keys())
    changed=sorted(k for k in old.keys()&new.keys() if old[k]!=new[k])
    mismatched=sorted(p for p,h in paths.items() if not Path(p).is_file() or sha(p)!=h)
    r={"historical_files":len(old),"external_sources_and_card":len(paths),"missing":missing,
       "added":additions,"changed":changed,"sources_changed":mismatched,
       "pass":not any((missing,additions,changed,mismatched))}
    dump(out/"history_verification.json",r)
    print("History",r["pass"],r["historical_files"],r["external_sources_and_card"],flush=True)


if __name__=="__main__":
    engine.HERE=HERE
    engine.CARD=CARD
    if "--verify" in sys.argv:
        ap=argparse.ArgumentParser()
        ap.add_argument("--out",required=True,type=Path)
        ap.add_argument("--verify",action="store_true")
        x=ap.parse_args()
        verify(x.out.resolve())
    else:
        engine.main()
