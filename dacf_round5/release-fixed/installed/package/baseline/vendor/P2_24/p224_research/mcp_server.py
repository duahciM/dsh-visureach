"""Per-session, read-only MCP service for the active video observer."""
import argparse
import base64
import json
from pathlib import Path
import sys
from uuid import uuid4

from if17_research.catalog import read

from .access import Video


COMMON={"start":{"type":"number"},"end":{"type":"number"},"cursor":{"type":"integer","minimum":0},"page_size":{"type":"integer","minimum":1}}
TOOLS={
    "catalog_search":{"name":"catalog_search","description":"Search times and visual candidate metadata; no image. Use mode overview for uniformly spaced visible stations in this range or earliest/latest with cursor for detail.",
      "inputSchema":{"type":"object","properties":COMMON|{"mode":{"type":"string","enum":["overview","earliest","latest"]},"dimension":{"type":"string","enum":["D1","D2","D3","D4","D5","D6"]}},"additionalProperties":False}},
    "delivery_request":{"name":"delivery_request","description":"Read full native PNG frame(s) by exact frame ID or PTS from the shared index. Give the currently needed observable perspective, state or comparison. This tool returns actual images. Use specific frames, not a catalogue enumeration.",
      "inputSchema":{"type":"object","properties":{"frame_ids":{"type":"array","items":{"type":"string"}},"pts":{"type":"array","items":{"type":"integer"}},"reason":{"type":"string"}},"required":["reason"],"additionalProperties":False}},
    "wave_query":{"name":"wave_query","description":"Query only navigation clues from discrete I/R measurements. Returns two sampled candidates, approximate image-space change reasons and unknown gaps. No images, no observed event or verified role.",
      "inputSchema":{"type":"object","properties":COMMON|{"signal":{"type":"string","enum":["all","rise","fall","plateau","equal_height_change","peak","gap","transition"]}},"additionalProperties":False}},
}


def append(log,doc):
    with log.open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(doc,ensure_ascii=False)+"\n")


def serve(args):
    packet=read(args.packet)
    backend=Video(packet)
    available={k:v for k,v in TOOLS.items() if k!="wave_query" or args.wave}
    for line in sys.stdin:
        req=json.loads(line)
        if "id" not in req:
            continue
        m=req["method"]; params=req.get("params",{}); ident=req["id"]
        response={"jsonrpc":"2.0","id":ident}
        if m=="initialize":
            # Request legacy-compatible stdio tool calls with image content support.
            protocol=params.get("protocolVersion","2025-11-25")
            response["result"]={"protocolVersion":protocol,"capabilities":{"tools":{"listChanged":False}},
                                "serverInfo":{"name":"private-frozen-video","version":"0.1.0"}}
        elif m=="ping":
            response["result"]={}
        elif m=="tools/list":
            response["result"]={"tools":list(available.values())}
        elif m=="tools/call":
            name=params.get("name");payload=params.get("arguments",{});stamp=uuid4().hex
            data={"call_ref":stamp,"tool":name,"arguments":payload}
            chunks=[]
            error=False
            try:
                if name not in available:
                    raise ValueError("Tool unavailable in this arm")
                if name=="delivery_request":
                    found=backend.delivery_request(payload.get("frame_ids"),payload.get("pts"),payload.get("reason",""))
                    data["image_offers"]=[r for r,_ in found]
                    reply={"delivery_status":"exact original attachments included in tool result","frames":[{k:v for k,v in r.items() if k not in ("path","native","source_sha256")} for r,_ in found]}
                    chunks=[{"type":"image","mimeType":"image/png","data":base64.b64encode(bits).decode("ascii")} for _,bits in found]
                else:
                    reply=getattr(backend,name)(**payload)
                    data["image_offers"]=[]
            except (ValueError,TypeError,KeyError,OSError) as ex:
                error=True;reply={"error":str(ex),"image_count":0};data["image_offers"]=[]
            encoded=json.dumps({"call_ref":stamp,**reply},ensure_ascii=False,separators=(",", ":"))
            data|={"returned_text_bytes":len(encoded.encode("utf-8")),"returned_metadata":reply,"error":error}
            append(args.log,data)
            response["result"]={"content":[{"type":"text","text":encoded}]+chunks,"isError":error}
        else:
            response["error"]={"code":-32601,"message":"Unsupported MCP method"}
        print(json.dumps(response,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser();p.add_argument("--packet",required=True,type=Path);p.add_argument("--log",required=True,type=Path);p.add_argument("--wave",action="store_true")
    serve(p.parse_args())
