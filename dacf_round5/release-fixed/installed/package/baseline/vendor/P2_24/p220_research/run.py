"""Reexecute visible streams and log candidate, read and delivery separately."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from if17_research.catalog import read, ref, sha
from p215_research.retrieval import handoff
from p216_research.navigate import checked_json, commit, delivery
from p217_research.triggers import Case, make_candidates
from p218_research.observer import observe
from p219_research.choose import choose, index
from p219_research.run import Source as EarlierSource, for_observer, visual_packet

from .prepare import CFG, EXCLUDED_SHA256, OLD
from .visual import coarse_episodes, find_archived_complements, visibility


class Exact(EarlierSource):
    def __init__(self,book):
        super().__init__(book)
        self.coarse_pts={r["pts"] for r in book.coverage}
        self.catalog_pts={r["pts"] for r in book.rows}
        for p in (OLD/"decoded"/book.case).glob("*.json"):
            i=int(p.stem)
            if i not in self.points:
                row=read(p)
                assert row["source_sha256"]==book.source["sha256"] and self.ledger[i]["pts"]==row["pts"]
                checked_json(self.folder/p.name,row)
                self.points[i]=row


def s2_for_fresh(book,source,scope):
    mapped,_,_=choose(book,scope,"S2")
    images=[]
    reasons={}
    seen=set()
    queries=[]
    for key,why in mapped.items():
        row=delivery(book,key,None)
        images.append(row|{"in_catalog":True,"in_coarse_pool":True})
        seen.add(row["pts"])
        reasons[str(row["pts"])]=why
    for family in scope["selected_comparison_groups"]:
        trace,nodes=visual_packet(book,source,family["selected"],source.coarse_pts,source.catalog_pts)
        queries.append(trace)
        for item,why in nodes:
            reasons.setdefault(str(item["pts"]),[]).append({"observation_role":why,"root_ids":trace["trigger"]["comparison"]["basis_root_ids"],
                      "reason":"Frozen S2 compatible source comparison"})
            if item["pts"] not in seen:
                images.append(item)
                seen.add(item["pts"])
    return images,reasons,queries


def historical_s2(row):
    for f in row.values():
        assert sha(f["path"])==f["sha256"]
    images=read(row["image_receipt"]["path"])
    old=read(row["candidate_decisions"]["path"])
    reasons={str(r["pts"]):r["reasons"] for r in old["candidates"] if r["decision"]=="offered_by_role"}
    for pts,infos in old["raw_offer_roles"].items():
        for info in infos:
            reasons.setdefault(pts,[]).append({"observation_role":info["purpose"],"root_ids":info["roots"],"reason":"Frozen S2 source comparison"})
    return [for_observer(r) for r in images["items"]],reasons,old["source_queries"]


def actual_read(book,source,episodes):
    source.ensure([p["source_frame"] for p in episodes])
    items=[]
    for offered in episodes:
        point=source.points[offered["source_frame"]]
        assert point["pts"]==offered["pts"]
        raw=handoff(book,point["native_image"],point["native_image_sha256"],point,None)
        image=raw|{"kind":"source_window","in_catalog":point["pts"] in source.catalog_pts,"in_coarse_pool":point["pts"] in source.coarse_pts}
        items.append((image,offered|{"native_checked":point["rgb_verification"],"observed_region":offered["region_original_px"]}))
    return items


def run(out):
    public=read(out/"input_receipt.json")
    assert all(sha(v["path"])==v["sha256"] for v in public.values())
    inputs=read(public["public"]["path"])
    frozen=read(OLD/"routes_frozen.json")["routes"]
    by_case=defaultdict(list)
    for task in inputs["goals"]:
        by_case[task["case"]].append(task)
    start=perf_counter()

    def one(value):
        case,goals=value
        entry=inputs["library"][case]
        assert entry["source"]["sha256"]!=EXCLUDED_SHA256
        book=Case(out,case,entry)
        source=Exact(book)
        facts,missing=make_candidates(book)
        done={}
        count=[]
        for goal in goals:
            gid=goal["id"]
            traces=out/"traces"/gid
            traces.mkdir(exist_ok=True,parents=True)
            scope=index(book,goal,facts,missing)
            key="S2/"+gid
            initial,reasons,queries=historical_s2(frozen[key]) if key in frozen else s2_for_fresh(book,source,scope)
            base={r["frame_id"]:reasons[str(r["pts"])] for r in book.rows if str(r["pts"]) in reasons}
            b1,checks=visibility(book,scope,base) if goal["intent"]=="single" else ({},[])
            b2,refchecks=find_archived_complements(book,scope,base)
            episodes,course=coarse_episodes(book,scope)
            original=actual_read(book,source,episodes)
            checked_json(traces/"index.json",{"entries":scope["entries"],"all_scope_candidates":scope["unread_eligible_count"],
                "historical_representative_groups":[{"region":g["region"],"channel":g["channel"],"all":g["all_ids"],"selected":g["selected"]["id"]}
                                         for g in scope["selected_comparison_groups"]],
                "windows_without_catalogue_endpoints":missing})
            for arm in CFG["arms"]:
                current=initial[:]
                labels={k:list(v) for k,v in reasons.items()}
                revisit=[]
                if arm in ("B1","B3") and goal["intent"]=="single":
                    current=[]
                    labels={}
                    for frame,why in b1.items():
                        c=delivery(book,frame,None)
                        current.append(c|{"in_catalog":True,"in_coarse_pool":True})
                        labels[str(c["pts"])]=why
                elif arm in ("B2","B3"):
                    seen={r["pts"] for r in current}
                    for frame,why in b2.items():
                        original_item=delivery(book,frame,None)
                        pts=original_item["pts"]
                        labels.setdefault(str(pts),[]).extend(why)
                        if pts not in seen:
                            current.append(original_item|{"in_catalog":True,"in_coarse_pool":True})
                            seen.add(pts)
                    for image,proof in original:
                        pts=image["pts"]
                        labels.setdefault(str(pts),[]).append({"observation_role":"coarse_only_foreground_episode", "root_ids":[proof["root_id"]],
                                   "reason":proof["reason"],"region_original_px":proof["region_original_px"]})
                        revisit.append(proof)
                        if pts not in seen:
                            current.append(image)
                            seen.add(pts)
                folder=out/"streams"/arm/gid
                receipt=observe(folder,goal,arm,book.source,current,"scoped_method_completed" if current else "no_grounded_offer")
                path=traces/f"{arm}.json"
                checked_json(path,{"goal":goal,"arm":arm,"historical_B0_same_order":arm=="B0" and key in frozen,
                            "one_view_selection":{"verified_positive_checks":checks,"selected_frames":list(b1)} if arm in ("B1","B3") else None,
                            "complementary_checks":refchecks if arm in ("B2","B3") else [],"coarse_backend_checks":course if arm in ("B2","B3") else [],
                            "frozen_source_queries":queries,"new_exact_source_checks":revisit,"source_candidates_not_global_observations":True,
                            "offer_reasons_by_pts":labels,"selected_pts_in_order":[r["pts"] for r in current],
                            "delivered_pts_in_order":[r["pts"] for r in receipt["items"]],"identity_and_cause_claimed":False})
                k=arm+"/"+gid
                done[k]={"observer":ref(folder/"receipt.json"),"candidate":ref(path),"index":ref(traces/"index.json")}
                count.append({"goal":goal,"arm":arm,"n":receipt["N_visible"],"source_sha256":book.source["sha256"],
                              "public_range":receipt["public_range"],"technical_abort":receipt["technical_abort"]})
                print(k,receipt["N_visible"],flush=True)
        return done,count
    with ThreadPoolExecutor(max_workers=3) as pool:
        r=list(pool.map(one,by_case.items()))
    checked_json(out/"count_only.json",{"pairs":[x for _,items in r for x in items],"source_role_or_algorithm_pts_exposed":False})
    checked_json(out/"algorithm_lock.json",{"routes":{k:v for part,_ in r for k,v in part.items()},"count_input":ref(out/"count_only.json")})
    commit(out/"backend_timing.json",{"seconds":perf_counter()-start,"visual_budget_metric":False})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out.resolve())
