"""Hash-based intake and blinded source review sheets."""
import argparse
from pathlib import Path
from time import monotonic

from identity_catalog_matching.common import read,sha
from if10_research.p3_live import contact
from if12_research.contracts import IF11,reference
from p1.io import dump
from v05.engine import load_source

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MEDIA=Path("C:/Users/Administrator/Desktop/视觉插件/素材")
RULES=read(HERE/"protocol.json")
PRIOR=ROOT/"if14_research_20261120"


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    x=a.parse_args()
    o=x.out.resolve()
    p=o/"intake"
    p.mkdir(exist_ok=False)
    archive=read(IF11/"run_manifest.json")["cases"]
    known={r["source"]["sha256"]:r["case"] for r in archive}
    items=[{"path":str(f),"sha256":sha(f)} for f in sorted(MEDIA.iterdir()) if f.is_file() and f.suffix.lower() in
           (".mp4",".mov",".mkv",".webm",".avi",".m4v")]
    dump(p/"input_lock.json",{"sources":items,"previous":reference(IF11/"run_manifest.json"),
         "protocol":reference(HERE/"protocol.json"),"code":sha(__file__)})
    new=[]
    known_new=set()
    for r in items:
        h=r["sha256"]
        if h in known or h in known_new:
            continue
        known_new.add(h)
        k="new_"+h[:8]
        target=p/k
        target.mkdir()
        started=monotonic()
        fs,src=load_source(r["path"],target,RULES["intake"]["review_fps"],limit=RULES["pool"]["cap"])
        assert src["sha256"]==h
        for n,offset in enumerate(range(0,len(fs),32),1):
            contact(fs[offset:offset+32],target/f"source_review_{n:02d}.png",f"{k} source frames | analysis review",scale=275)
        item={"case":k,"source":src,"clip_ref":reference(target/"frames.json"),"review_count":len(fs),
              "review_seconds":monotonic()-started}
        new.append(item)
        print(k,src["start"],src["end"],len(fs),r["path"],flush=True)
    dump(p/"manifest.json",{"new":new,"historical_duplicates":[r|{"case":known[r["sha256"]]} for r in items if r["sha256"] in known]})
    print(len(new),"new distinct sources",flush=True)


if __name__=="__main__":
    main()
