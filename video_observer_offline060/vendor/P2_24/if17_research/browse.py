"""Chronological source browsing and exact local read, independent of catalog membership."""
import argparse
import json
from collections import defaultdict
from math import ceil,isfinite
from pathlib import Path

from PIL import Image,ImageDraw

from if15_research.audit import pts_ledger
from p1.io import dump
from v05.engine import export_native
from v07.audit import audit_images

from .catalog import read,sha
from .agent import tile_list


def matches(r,field,value):
    if field=="t_range":
        return value[0]<=r["t"]<=value[1]
    if field=="source_sha256":
        return value==r["video_sha256"]
    if field in ("source_frame","pts","frame_id"):
        return value==r[field]
    raise ValueError(field)


def lookup(folder,filters):
    book=read(folder/"catalog.json")
    index=read(folder/"index.json")
    if index["source_sha256"]!=book["source"]["sha256"]:
        raise ValueError("Catalog index source mismatch")
    allow=set(index["timeline"])
    for name in ("dimension","role","root","region_candidate","relationship","uncertainty","window","time_group"):
        val=filters.get(name)
        if val is not None:
            allow.intersection_update(index["filters"][name].get(str(val),[]))
    exact={i:filters[i] for i in ("t_range","source_sha256","source_frame","pts","frame_id") if filters.get(i) is not None}
    found=[r for r in book["frames"] if r["frame_id"] in allow and all(matches(r,k,v) for k,v in exact.items())]
    return book,found


def verify_native(ref):
    if sha(ref["path"])!=ref["sha256"]:
        raise ValueError(f'Native image digest mismatch at PTS {ref["pts"]}')
    return ref


def as_pixels(row):
    ref=verify_native(row["source_image"])
    return row|{"native_image":ref["path"],"image":ref["path"]}


def rendered_page(fs,canvas_path,receipt,start_index):
    w,h=340,round(340*9/16)
    gap=58
    page=Image.new("RGB",(w*4,ceil(len(fs)/4)*(h+gap)+26),"#172631")
    ink=ImageDraw.Draw(page)
    ink.text((9,7),"Chronological video observations | read left to right, then top to bottom",fill="white")
    for i,(f,r) in enumerate(zip(fs,receipt)):
        x=i%4*w
        y=26+i//4*(h+gap)
        with Image.open(f["native_image"]) as p:
            rgb=p.convert("RGB")
            rgb.thumbnail((w,h))
            page.paste(rgb,(x,y))
        ink.text((x+5,y+h+4),f'{r["ordinal"]} | {f["t"]:.6f}s | PTS {f["pts"]} ({f["time_base"]})',fill="white")
        ink.text((x+5,y+h+19),f'Source frame {f["source_frame"]} | match {start_index+i+1}',fill="white")
        a=r["previous_displayed_seconds"]
        if a is not None:
            ink.text((x+5,y+h+34),f'Open unsampled interval ({a:.6f}, {f["t"]:.6f})',fill="#c4daea")
    page.save(canvas_path)


def browse(folder,d,filters,offset,page_size,image,previous_displayed=None):
    book,found=lookup(folder,filters)
    if offset<0 or page_size<=0:
        raise ValueError("Require nonnegative offset and a positive page size")
    if previous_displayed is None and offset>0 and offset<=len(found):
        previous_displayed=found[offset-1]["t"]
    page=found[offset:offset+page_size]
    fs=[as_pixels(r) for r in page] if image else page
    if fs:
        ordinal=tile_list(fs,"T")
        for i,r in enumerate(ordinal):
            r["ordinal"]=f'#{fs[i]["chronological_ordinal"]:04d}'
            previous=fs[i-1]["t"] if i else previous_displayed
            r["previous_displayed_seconds"]=previous
            r["open_gap_from_previous_delivered_image"]=[previous,r["seconds"]] if previous is not None else None
    else:
        ordinal=[]
    d.mkdir(exist_ok=False,parents=True)
    if image and fs:
        rendered_page(fs,d/"page.png",ordinal,offset)
    receipt={"filters":filters,"matched_catalog_frames":len(found),"catalog_total":book["frame_count"],
       "cursor_start":offset,"cursor_next":offset+len(page) if offset+len(page)<len(found) else None,
       "returned_reference_count":len(fs),"source_sha256":book["source"]["sha256"],
       "match_records":page,"time_mapping":ordinal,"actually_rendered_images":len(fs) if image else 0,
       "page_attachment":str(d/"page.png") if image and fs else None,
       "page_scope":"Displayed indices are page-scoped receipts; catalog membership is independent."}
    dump(d/"page_receipt.json",receipt)
    return receipt


def read_original_range(book,d,start,end,fps):
    src=book["source"]
    if sha(src["path"])!=src["sha256"]:
        raise ValueError("Original source digest differs")
    if not all(isfinite(v) for v in (start,end,fps)) or not (src["start"]<=start<end<=src["end"] and fps>0):
        raise ValueError("Require positive rate and a range within the original source")
    tb,pts=pts_ledger(src)
    selection=[]
    next_target=start
    for i,row in enumerate(pts):
        t=float(row["pts_time"])
        if t<start:
            continue
        if t>=end:
            break
        if t+1e-8<next_target:
            continue
        selection.append({"source_frame":i,"pts":row["pts"],"time_base":tb,"t":t,
                          "sample":len(selection),"image":src["path"]})
        next_target=start+(int((t-start)*fps+1e-7)+1)/fps
    d.mkdir(exist_ok=False,parents=True)
    out=export_native(src,selection,list(range(len(selection))),d/"native")
    audit=[v for i in range(0,len(out),16) for v in audit_images(src,out[i:i+16])]
    assert all(r["file_hash_equal"] and r["exact_source_rgb_equal"] for r in audit)
    originally={r["pts"] for r in book["frames"]}
    receipts={"requested_time":[start,end],"requested_fps":fps,"returned":out,
       "source_sha256":src["sha256"],"covered_exact_original_pts":[r["pts"] for r in out],
       "returned_outside_catalog":[r["pts"] for r in out if r["pts"] not in originally],
       "archive_mutation":False,"native_audit":audit,"read_method":"index exact PTS, native original-sequence decode"}
    dump(d/"read_receipt.json",receipts)
    return receipts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--receipt",required=True,type=Path)
    where=ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--case")
    where.add_argument("--long-mode",choices=("fixed","overlap"))
    for flag in ("dimension","role","root","region-candidate","relationship","uncertainty",
                 "window","time-group","frame-id","source-sha256"):
        ap.add_argument("--"+flag)
    ap.add_argument("--pts",type=int)
    ap.add_argument("--source-frame",type=int)
    ap.add_argument("--t-range",nargs=2,type=float)
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--page-size",type=int,default=6)
    ap.add_argument("--render",action="store_true")
    ap.add_argument("--read-range",nargs=3,type=float,metavar=("START","END","FPS"))
    x=ap.parse_args()
    o=x.out.resolve()
    d=o/"P2_13"/x.case if x.case else o/"P2_14"/x.long_mode
    if x.read_range is not None:
        z=read_original_range(read(d/"catalog.json"),x.receipt.resolve(),*x.read_range)
        print(len(z["returned"]),"actual native original frames")
    else:
        filters={k:v for k,v in vars(x).items() if k not in
                 ("out","receipt","case","long_mode","offset","page_size","render","read_range") and v is not None}
        z=browse(d,x.receipt.resolve(),filters,x.offset,x.page_size,x.render)
        print("page",z["returned_reference_count"],"matches",z["matched_catalog_frames"],"next",z["cursor_next"])


if __name__=="__main__":
    main()
