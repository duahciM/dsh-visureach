"""Independent exact-source positives for the two verified new recordings."""
import argparse
from pathlib import Path

from if15_research.audit import pts_ledger
from if17_research.catalog import read,ref
from p216_research.navigate import checked_json
from p219_research.score import checked
from v05.engine import export_native

from .prepare import OLD

NEW={
    "road_fresh":{"opening_road_geometry": {"windows":[[0,4.1]],"scope":"Overhead dark asphalt and thin white oblique marking in a fixed view, with main central road surface clear","type":"direct_visual_role","example":2},
                  "later_road_occupancy":{"windows":[[5.75,7.15]],"scope":"White vehicle visible across the main road surface with the same fixed border marking","type":"direct_visual_role","example":6.5}},
    "notebook_scene":{"notebook_initial_state":{"windows":[[0,1.85]],"scope":"Upright laptop display relative to the keyboard and desk, with green card below","type":"direct_visual_role","example":1},
                      "notebook_later_state":{"windows":[[3.65,5.85]],"scope":"Reduced lid angle over the keyboard, with the same desk and green card. No contact claim.","type":"direct_visual_role","example":4.75}},
}


def run(out):
    old=read(OLD/"E0_reference_lock.json")
    frozen=checked(old["strict"])["goals"]
    pub=read(out/"frozen_p223_candidate_and_wave_manifest.json")
    goals={g["id"]:g for g in pub["goals"]}
    roles={k:(frozen[k] if k in frozen else {"roles":NEW[k],"control":None}) for k in goals}
    folder=out/"evaluation_only";folder.mkdir(exist_ok=True)
    examples={}
    for gid,v in NEW.items():
        src=checked(pub["library"][goals[gid]["case"]]["catalog"])["source"]
        tb,ledger=pts_ledger(src);times=[float(r["pts_time"]) for r in ledger]
        ids=sorted({min(range(len(times)),key=lambda i:abs(times[i]-x["example"])) for x in v.values()})
        records=[{"sample":i,"source_frame":i,"pts":r["pts"],"time_base":tb,"t":times[i]} for i,r in enumerate(ledger)]
        examples[gid]=export_native(src,records,ids,folder/gid)
    checked_json(folder/"strict_roles.json",{"goals":roles,"historical_untouched":old["strict"]})
    checked_json(folder/"original_examples.json",examples)
    checked_json(out/"E0_reference_lock.json",{"strict":ref(folder/"strict_roles.json"),"broad":old["broad"],"native_examples":ref(folder/"original_examples.json"),
                 "protocol":ref(out/"protocol_lock.json"),"locked_before_fresh_predictions":True,"historical_roles_copied_identically":True,
                 "method":"Independent single source-image review; only positive windows and narrow direct facts; no candidate or agent path can read this file."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);run(p.parse_args().out.resolve())
