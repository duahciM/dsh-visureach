"""Real native-image intake using the historical session/prompt image schema locally."""

import base64
import json
import sys
from pathlib import Path

from if17_research.catalog import read, ref
from if20_research.direct import inspect_png
from p216_research.navigate import commit

from .freeze import TOKENIZER, VENDOR_TOKENS


PAGE=16


def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"))


def word_counter():
    sys.path.append(str(VENDOR_TOKENS))
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(TOKENIZER))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def interface_receive(block,expected,width,height):
    if set(block)!={"type","mimeType","data"} or block["type"]!="image" or block["mimeType"]!="image/png":
        raise ValueError("Image content block does not conform to the real adapter schema")
    native=inspect_png(base64.b64decode(block["data"],validate=True))
    if (native["sha256"],native["width"],native["height"])!=(expected,width,height):
        raise ValueError("Wrong PNG bytes or image dimensions at observer interface")
    return native


def observe(folder,goal,arm,source,images,stop,typed_pages=None):
    folder.mkdir(exist_ok=True,parents=True)
    result=folder/"receipt.json"
    if result.exists():
        last=read(result)
        if not last["technical_abort"]:
            return last
    tok=word_counter()
    visible=[]
    seen=set()
    notices=[]
    tokens=0
    text_bytes=0
    failure=None
    prior=None
    w,h=source["probe"]["width"],source["probe"]["height"]
    archive_only=0
    page_root=folder/"observer_pages"
    page_root.mkdir(exist_ok=True)
    locations=typed_pages if typed_pages is not None else [images[i:i+PAGE] for i in range(0,len(images),PAGE)]
    for page,group in enumerate(locations,1):
        mapping=[]
        previous=prior
        for n,row in enumerate(group):
            when=row["seconds"]
            mapping.append({"image":len(visible)+n+1,"pts":row["pts"],"time_base":row["time_base"],"source_frame":row["source_frame"],
                            "seconds":when,"previous_image_seconds":previous})
            previous=when
        prompt="Question: "+goal["text"]+"\nOriginal frame timestamps (one row per independent full PNG): "+compact(mapping)
        n_text=len(tok.encode(prompt,add_special_tokens=False).ids)
        page_path=page_root/f"{page:05d}.json"
        if page_path.exists():
            persisted=read(page_path)
            if persisted["prompt"]!=prompt or [r["pts"] for r in persisted["items"]]!=[r["pts"] for r in group]:
                raise ValueError("Observer checkpoint does not match current stream")
            group_records=persisted["items"]
            seen.update(r["pts"] for r in group_records)
            if group_records:
                prior=group_records[-1]["seconds"]
        else:
            partial=page_root/f"interrupted_{page:05d}.json"
            saved=read(partial) if partial.exists() else {"prompt":prompt,"items":[]}
            group_records=saved["items"]
            prefix=len(group_records)
            if saved["prompt"]!=prompt or [r["pts"] for r in group_records]!=[r["pts"] for r in group[:prefix]]:
                raise ValueError("Partial page does not match the resumable stream")
            seen.update(r["pts"] for r in group_records)
            if group_records:
                prior=group_records[-1]["seconds"]
            for original,stamp in zip(group[prefix:],mapping[prefix:],strict=True):
                try:
                    data=Path(original["path"]).read_bytes()
                    block={"type":"image","mimeType":"image/png","data":base64.b64encode(data).decode("ascii")}
                    model_item=interface_receive(block,original["sha256"],w,h)
                except (OSError,ValueError) as exc:
                    failure={"stage":"local_observer_delivery","page":page,"actual_prefix":len(group_records),"error":str(exc)}
                    break
                pts=stamp["pts"]
                record=stamp|{"goal_id":goal["id"],"case":goal["case"],"arm":arm,"page":page,
                    "source_sha256":source["sha256"],"path":original["path"],"attachment":model_item,
                    "delivered_to_interface":True,"sdk_acknowledged_by_model":False,
                    "repeated_visible_pts":pts in seen,"backend_origin":original["kind"],
                    "in_catalog":original["in_catalog"],"in_coarse_pool":original["in_coarse_pool"],
                    "format":"full_native_png","transform":None,"source_cache_reuse":original.get("source_cache_reuse",None)}
                group_records.append(record)
                seen.add(pts)
                prior=stamp["seconds"]
            if failure:
                commit(partial,{"prompt":prompt,"items":group_records,"failure":failure})
            else:
                commit(page_path,{"prompt":prompt,"items":group_records,"proxy_text_tokens":n_text})
        notices.append(ref(page_path) if not failure else ref(page_root/f"interrupted_{page:05d}.json"))
        visible.extend(group_records)
        tokens+=n_text
        text_bytes+=len(prompt.encode())
        if failure:
            break
    archive_only=sum(r["in_catalog"] for r in visible)
    outcome={"goal":goal,"arm":arm,"source_sha256":source["sha256"],"public_range":[source["start"],source["end"]],"items":visible,
             "N_visible":len(visible),"N_unique_pts":len(seen),"N_repeated":len(visible)-len(seen),
             "N_first_batch":sum(r["page"]==1 for r in visible),"N_source_revisit":sum(r["backend_origin"]=="source_window" for r in visible),
             "N_visible_catalog_membership":archive_only,"N_outside_catalog":len(visible)-archive_only,"N_crops_or_resizes":0,
             "metadata_text_tokens_clip_proxy":tokens,"metadata_text_utf8_bytes":text_bytes,"sdk_model_input_tokens":None,
             "delivery":"locally ingested session/prompt native image block, not provider or model acknowledgement",
             "source_stop":stop,"technical_abort":bool(failure) or stop=="technical_abort","failure":failure,"page_receipts":notices}
    commit(result,outcome)
    return outcome
