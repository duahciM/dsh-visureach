"""Freeze exact historical image deliveries and an evaluation-only original-pixel key."""
import argparse
from pathlib import Path

from if15_research.audit import checks_for
from if17_research.browse import as_pixels,rendered_page
from if17_research.catalog import read,ref,sha
from if18_research.video_tools import CASES,PRIOR
from p1.io import dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
IF18=ROOT/"if18_research_20261120"
PROTOCOL=read(HERE/"protocol.json")
QUESTIONS={
    "g3":"Describe the small object's directly visible positions over source time, and state which portions of the behavior remain unobserved.",
    "short":"Describe the directly visible relationship of the ball, robotic gripper and tray as the clip progresses. Identify whether the ball is visibly held or separate when the frames are sufficient, and which timing remains unknown."
}
PINS={
 "g3":{"history":"g3_B1_1","pts":[2400,2640,2700,2800,2860,3000,3160,3200,3240,3300]},
 "short":{"history":"short_C_1","pts":[3720,87720,153720,165720,189720,195720,201720,207720,213720,219720,303720]}
}
ANSWER_KEY={
 "g3":{
    "right":{"pts":[2640,2700,2800],"pixels":"small red/white shape immediately right of cup, moving left between these separately sampled poses",
             "safe_conclusion":"visible on image right at the cited samples"},
    "sampled_gap":{"pts":[2860,3000,3160],"pixels":"cup and table discernible with no clear red/white target around either side",
                  "safe_conclusion":"not discernible in these discrete samples"},
    "left":{"pts":[3200,3240,3300],"pixels":"near or left of cup boundary at 3200, clear left at 3240 and 3300",
             "safe_conclusion":"visibly left; 3240/3300 unambiguous"},
    "unknown":"identity across gap, full path, physical cause, any unseen trajectory"},
 "short":{
    "table":{"pts":[3720,87720],"pixels":"red ball on bench left of blue tray, gripper approaches but separate"},
    "near_gripper":{"pts":[153720,165720,189720,195720,201720],
                    "pixels":"ball directly under gripper at first and near tray edge later; upper fingers and ball may overlap in projection",
                    "safe_conclusion":"co-visible proximity. Do not require definite 3D contact or claim a release instant."},
    "separate":{"pts":[207720,213720,219720,303720],"pixels":"red ball in blue tray and a visible gap toward raised/open gripper",
                "safe_conclusion":"visibly separated at 6.924/7.124 and later. 7.124 cannot support still held."},
    "unknown":"exact release event/time or force, unseen in-between contact mechanics"}
}


def native_rows(case):
    pin=PINS[case]
    folder=IF18/"sessions"/pin["history"]
    rec=read(folder/"result.json")
    catalog=read(PRIOR/"P2_13"/CASES[case]/"catalog.json")
    source=catalog["source"]
    by_pts={f["pts"]:f for f in catalog["frames"]}
    raw={}
    for a in folder.glob("call_*/original/read_receipt.json"):
        for f in read(a)["returned"]:
            raw[(a.parent.parent.name,f["pts"])]=f
    received={}
    for c in rec["calls"]:
        if not c["returned_to_observer"]:
            continue
        for f in c["frame_receipts"]:
            if f["pts"] in received:
                continue
            key=c["args"].get("range_token",f'call_{c["call"]:02d}')
            img=raw.get((key,f["pts"]))
            if img is None and f["pts"] in by_pts:
                v=as_pixels(by_pts[f["pts"]])
                img=v|{"native_image_sha256":v["source_image"]["sha256"]}
            if img is None:
                continue
            assert sha(img["native_image"])==f["native_png_sha256"]
            assert f["image_sheet_sha256"] in {a["sha256"] for a in c["queued_images"]}
            received[f["pts"]]=img|{"prior_call":c["call"],"prior_delivered_on_turn":c["returned_on_turn"],
                "prior_attachment_sha256":f["image_sheet_sha256"]}
    return source,[received[p] for p in pin["pts"]],ref(folder/"result.json")


def prepare(out):
    folder=out/"fixed_input"
    folder.mkdir(parents=True,exist_ok=False)
    l={}
    evidence=out/"evaluation_only"
    evidence.mkdir(exist_ok=False)
    for case in PROTOCOL["stage_A"]["cases"]:
        d=folder/case
        d.mkdir()
        src,fs,prior=native_rows(case)
        checked=checks_for(src,fs,[],[],[],len(fs))
        assert checked["pass"]
        positions=[]
        last=None
        for i,f in enumerate(fs,1):
            positions.append({"ordinal":f"F{i:02d}","row":(i-1)//4+1,"column":(i-1)%4+1,
                  "source_frame":f["source_frame"],"pts":f["pts"],"time_base":f["time_base"],
                  "seconds":f["t"],"previous_displayed_seconds":last})
            last=f["t"]
        p=d/"actual_frames.png"
        rendered_page(fs,p,positions,0)
        # Appearance data are kept outside the model's prompt. The table only fixes source addressing.
        packet={"case":case,"source":src,"historical_result":prior,"frames":fs,"positions":positions,
                "image":ref(p),"native_audit":checked,"source_scope":"Acknowledged historical original pixels, ordered and re-rendered without mirroring."}
        dump(d/"input.json",packet)
        dump(evidence/f"{case}_reference.json",{"facts":ANSWER_KEY[case],"input":ref(d/"input.json"),"not_for_observers":True})
        l[case]=ref(d/"input.json")
    dump(folder/"manifest.json",{"input_by_case":l,"study_protocol":ref(HERE/"protocol.json"),"one_common_input_per_case":True})


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    x=ap.parse_args()
    prepare(x.out.resolve())
