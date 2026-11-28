"""Cache source-bound crop embeddings and retractable image-space index hints."""

import argparse
from collections import Counter
import csv
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from if17_research.catalog import read, ref, sha
from p1.io import dump
from .prepare import ROOT, RULES


S = RULES["semantics"]
MEAN = np.asarray([.48145466, .4578275, .40821073], dtype=np.float32)
STD = np.asarray([.26862954, .26130258, .27577711], dtype=np.float32)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))


def regions(image, observations, contract):
    w, h = image.size
    # Saturation gives an image-only regional proposal, with no preferred hue or class.
    proxy = np.array(image.resize((256, max(8, round(256 * h / w)))))
    ph, pw = proxy.shape[:2]
    saturation = cv2.cvtColor(proxy, cv2.COLOR_RGB2HSV)[:, :, 1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(saturation >= 115))
    candidates = []
    for x, y, rw, rh, area in stats[1:count]:
        if not (12 <= area <= pw * ph * .2 and min(rw, rh) >= 3):
            continue
        candidates.append((int(area), (round(x*w/pw), round(y*h/ph), round(rw*w/pw), round(rh*h/ph)), "saturation_component"))
    candidates.sort(reverse=True)
    for observation in observations:
        box = observation.get("region_xywh_analysis")
        if box and len(box) == 4:
            x, y, rw, rh = box
            ratio = (w / contract["analysis_width"], h / contract["analysis_height"])
            candidates.append((int(rw*rh), (round(x*ratio[0]), round(y*ratio[1]), round(rw*ratio[0]), round(rh*ratio[1])), "m7_region_hint"))
    choices = [((0,0,w,h), "full_frame")]
    for _, (x,y,rw,rh), kind in candidates:
        pad = max(6, min(rw,rh)//5)
        left, top = max(0,x-pad), max(0,y-pad)
        box = (left,top,min(w,x+rw+pad)-left,min(h,y+rh+pad)-top)
        if all(overlap(box,b) / max(1,min(box[2]*box[3], b[2]*b[3])) < .65 for b,_ in choices[1:]):
            choices.append((box,kind))
        if len(choices) >= 3:
            break
    choices.extend([((0,0,round(.62*w),h),"left_context"),
                    ((round(.38*w),0,w-round(.38*w),h),"right_context")])
    return choices[:S["crop_limit"]]


def node_edges(nodes, width, height, root):
    localized = nodes[1:S["relation_nodes"]]
    edges = []
    for i, a in enumerate(localized):
        for b in localized[i+1:]:
            ax,ay,aw,ah=a["region"]["bbox"]
            bx,by,bw,bh=b["region"]["bbox"]
            dx=(ax+aw/2-bx-bw/2)/width
            dy=(ay+ah/2-by-bh/2)/height
            axis = max(abs(dx), abs(dy))
            if axis >= S["min_center_separation_fraction"]:
                predicate = ("left_of" if dx<0 else "right_of") if abs(dx)>=abs(dy) else ("above" if dy<0 else "below")
            elif overlap((ax,ay,aw,ah),(bx,by,bw,bh)):
                predicate = "regions_overlap"
            else:
                continue
            intersection=overlap((ax,ay,aw,ah),(bx,by,bw,bh))
            edges.append({"subject_candidate_id":a["candidate_id"],"object_candidate_id":b["candidate_id"],
                "predicate":predicate,"normalized_offset":[round(dx,5),round(dy,5)],
                "overlap_fraction_of_smaller":round(intersection / max(1,min(aw*ah,bw*bh)),5),
                "separate_proposal_regions":intersection / max(1,min(aw*ah,bw*bh)) < S["max_edge_overlap_for_separation"],
                "scope":"single_frame","evidence_root":root,"status":"hypothesis",
                "uncertainty":"Proposal crop centers and overlap in this image; object boundaries, contact, depth and identity are unknown."})
    return edges


class Clip:
    def __init__(self, out, vision=True):
        assets=out/"assets"
        sys.path.append(str(ROOT/"if13_research_20261120/runtime_onnx"))
        sys.path.append(str(assets/"token_runtime"))
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts=ort.SessionOptions()
        opts.intra_op_num_threads=8
        opts.inter_op_num_threads=1
        self.text_net=ort.InferenceSession(str(assets/"text.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.vision_net=ort.InferenceSession(str(assets/"vision.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]) if vision else None
        self.tokenizer=Tokenizer.from_file(str(assets/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=77)
        self.tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
        cv2.setNumThreads(1)

    def text(self, strings):
        result=[]
        for i in range(0,len(strings),32):
            encoded=self.tokenizer.encode_batch(strings[i:i+32])
            values={"input_ids":np.asarray([e.ids for e in encoded],dtype=np.int64),
                    "attention_mask":np.asarray([e.attention_mask for e in encoded],dtype=np.int64),
                    "position_ids":np.broadcast_to(np.arange(77,dtype=np.int64),(len(encoded),77))}
            output={o.name for o in self.text_net.get_outputs()}
            result.append(self.text_net.run(["text_embeds"] if "text_embeds" in output else None,
                          {x.name:values[x.name] for x in self.text_net.get_inputs()})[0])
        vectors=unit(np.concatenate(result))
        if vectors.ndim!=2:
            raise ValueError("Expected projected CLIP text features")
        return vectors

    def vision(self, pixels):
        batch=np.stack(pixels)
        names={o.name for o in self.vision_net.get_outputs()}
        vectors=self.vision_net.run(["image_embeds"] if "image_embeds" in names else None,
                                    {"pixel_values":batch})[0]
        if vectors.ndim!=2:
            raise ValueError("Expected projected CLIP image features")
        return unit(vectors)


def tensor(image, box):
    x,y,w,h=box
    rgb=ImageOps.pad(image.crop((x,y,x+w,y+h)),(224,224),method=Image.Resampling.BICUBIC,color=(123,117,104))
    return ((np.asarray(rgb,dtype=np.float32)/np.float32(255)-MEAN)/STD).transpose(2,0,1).copy()


def indexed_case(book, lexicon, noun_vectors, model):
    rows=[]
    all_vectors=[]
    errors=[]
    for i in range(0,len(book["frames"]),4):
        images=[]
        queue=[]
        for frame in book["frames"][i:i+4]:
            info=frame["source_image"]
            root=f'{frame["frame_id"]}:semantic_pixels'
            entry={"frame_id":frame["frame_id"],"video_sha256":book["source"]["sha256"],
                   "source_frame":frame["source_frame"],"pts":frame["pts"],"time_base":frame["time_base"],"t":frame["t"],
                   "source_image":info,"root":root,"nodes":[],"relations":[],
                   "m7_roots":sorted({r for obs in frame["observations"] for r in obs["evidence_root_ids"]}),
                   "comparison_frames":sorted({r for obs in frame["observations"] for r in obs["source_comparison_frames"]})}
            rows.append(entry)
            try:
                if sha(info["path"])!=info["sha256"]:
                    raise ValueError("Original PNG changed")
                with Image.open(info["path"]) as file:
                    image=file.convert("RGB")
                props=regions(image,frame["observations"],book["coordinate_contract"])
                images.extend(tensor(image,b) for b,_ in props)
                queue.append((entry,props,image.size))
            except (ValueError,OSError) as exc:
                entry.update(status="unassessable",unknown_reason=str(exc))
                errors.append({"frame_id":frame["frame_id"],"error":str(exc)})
        if not images:
            continue
        vectors=model.vision(images)
        sims=vectors@noun_vectors.T
        at=0
        for entry,props,(w,h) in queue:
            for j,(box,kind) in enumerate(props):
                best=np.argsort(-sims[at],kind="stable")[:S["nouns_per_crop"]]
                words=[{"word":lexicon[k],"cosine":round(float(sims[at,k]),6)} for k in best if sims[at,k]>=S["min_cosine"]]
                entry["nodes"].append({"candidate_id":f'{entry["frame_id"]}:region:{j}',
                    "region":{"bbox":box,"coordinate_system":"original_pixels","shape":"proposal_rect"},
                    "proposal":kind,"visual_hypotheses":words,"status":"hypothesis" if words else "unassessable",
                    "uncertainty":"Cosine ranking of a crop; labels and crop scope are provisional and may include background.",
                    "feature_row":len(all_vectors)})
                all_vectors.append(vectors[at])
                at+=1
            entry.update(status="hypothesis",relations=node_edges(entry["nodes"],w,h,entry["root"]))
    return rows,np.asarray(all_vectors,dtype=np.float32),errors


def run(out):
    manifest=read(out/"library/manifest.json")
    lock=read(out/"protocol_lock.json")
    for item in lock["prediction_inputs"]:
        assert sha(item["path"])==item["sha256"]
    assets=out/"assets"
    with (assets/"openimages.csv").open(encoding="utf-8-sig",newline="") as f:
        lexicon=sorted({row[1].casefold() for row in csv.reader(f) if len(row)==2})
    model=Clip(out)
    noun_vectors=model.text([f"a photo of a {word}." for word in lexicon])
    target=out/"semantic"
    target.mkdir(exist_ok=True)
    stamp={"model":read(assets/"sources.json"),"assets":[ref(assets/name) for name in
           ("vision.onnx","text.onnx","tokenizer.json","openimages.csv")],
           "scope":"Automatic candidate words use an independent generic vocabulary. Arbitrary query text is encoded separately.",
           "preprocess":"Aspect-preserving 224-pixel pad, RGB CLIP normalization. Cosine values are not probabilities."}
    costs={}
    for case,item in manifest.items():
        path=target/case
        path.mkdir(exist_ok=True)
        current=path/"metrics.json"
        if current.exists():
            costs[case]=read(current)
            continue
        start=perf_counter()
        book=read(item["catalog"]["path"])
        rows,vectors,errors=indexed_case(book,lexicon,noun_vectors,model)
        assert [r["frame_id"] for r in rows]==[r["frame_id"] for r in book["frames"]]
        dump(path/"candidates.json",{"rows":rows,"extractor":stamp,"errors":errors})
        np.save(path/"vectors.npy",vectors,allow_pickle=False)
        counts=Counter(word["word"] for row in rows for node in row["nodes"] for word in node["visual_hypotheses"])
        costs[case]={"images":len(rows),"regions":len(vectors),"edges":sum(len(r["relations"]) for r in rows),
                     "elapsed_seconds":perf_counter()-start,"errors":len(errors),"vocabulary_count":len(counts),
                     "bytes":sum(p.stat().st_size for p in path.glob("*"))}
        dump(current,costs[case])
        print(case,costs[case],flush=True)
    dump(target/"run_manifest.json",{"cases":costs,"extractor":stamp,"automatic_lexicon_size":len(lexicon),"frame_membership_unchanged":True})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out.resolve())
