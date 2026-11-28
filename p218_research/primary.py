"""Re-execute frozen T3/T5 without using the evaluation references."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from if17_research.catalog import read, ref, sha
from p216_research.navigate import checked_json, commit
from p217_research.runner import ExactSource, trial
from p217_research.triggers import Case, make_candidates

from .freeze import EXCLUDED_SHA256, HERE, OLD, PRIMARY
from .observer import observe


class ReuseSource(ExactSource):
    def __init__(self,book):
        super().__init__(book)
        for point in (OLD/"decoded"/book.case).glob("*.json"):
            row=read(point)
            if row["source_sha256"]!=book.source["sha256"]:
                raise ValueError("Frozen decode reuse has the wrong source SHA")
            idx=int(point.stem)
            if self.ledger[idx]["pts"]!=row["pts"] or not row["rgb_verification"]["exact_source_rgb_equal"]:
                raise ValueError("Frozen decode reuse failed PTS or RGB provenance")
            if idx not in self.points:
                checked_json(self.folder/point.name,row)
                self.points[idx]=row


def deliveries(result):
    pages=[]
    for item in result["event_receipts"]:
        if sha(item["path"])!=item["sha256"]:
            raise ValueError("Back-end page was changed")
        images=read(item["path"])["images"]
        if images:
            pages.append([row|{"source_cache_reuse":Path(row["path"]).is_relative_to(OLD)
                               if row["kind"]=="source_window" else None} for row in images])
    return pages


def run(out):
    files=read(out/"protocol_lock.json")["inputs"]
    for p in (out/"public_goals.json",out/"public_library.json",HERE/"primary.py",HERE/"observer.py"):
        assert sha(p)==files[str(p.resolve())]
    library=read(out/"public_library.json")
    groups=defaultdict(list)
    for goal in read(out/"public_goals.json"):
        groups[goal["case"]].append(goal)

    def run_case(case,goals):
        v=library[case]
        assert v["source"]["sha256"]!=EXCLUDED_SHA256
        book=Case(out,case,v)
        decoder=ReuseSource(book)
        facts,missing=make_candidates(book)
        results={}
        counts=[]
        arms=PRIMARY|({"A4_native":"T0"} if case in ("still","return") else {})
        for goal in goals:
            for arm,solver in arms.items():
                backend=trial(book,decoder,goal,solver,facts,missing,out/"frozen_backend",False,None)
                folder=out/"streams"/arm/goal["id"]
                receipt=observe(folder,goal,arm,book.source,[],backend["run_state"],deliveries(backend))
                name=arm+"/"+goal["id"]
                results[name]={"observer":ref(folder/"receipt.json"),"backend":ref(out/"frozen_backend"/case/solver/goal["id"]/"result.json")}
                counts.append({"goal":goal,"arm":arm,"source_sha256":book.source["sha256"],"public_range":receipt["public_range"],
                               "n":receipt["N_visible"],"n_catalog":receipt["N_visible_catalog_membership"],"technical_abort":receipt["technical_abort"]})
                print(name,receipt["N_visible"],receipt["source_stop"],flush=True)
        return results,counts

    start=perf_counter()
    with ThreadPoolExecutor(max_workers=3) as workers:
        parts=list(workers.map(lambda pair:run_case(*pair),groups.items()))
    flattened={k:v for item,_ in parts for k,v in item.items()}
    visible_counts=[row for _,counts in parts for row in counts]
    count_path=out/"primary_visible_counts.json"
    checked_json(count_path,{"pairs":visible_counts,"scope":"count-only baseline handoff with no outcome roles, selected PTS or change windows"})
    checked_json(out/"primary_algorithm_freeze.json",{"routes":flattened,"count_only_export":ref(count_path),
                 "protocol":ref(out/"protocol_lock.json"),"algorithm_pt_data_exposed_to_baseline":False})
    commit(out/"primary_runtime.json",{"seconds":perf_counter()-start,"backend_decode":"immutable exact RGB cache reused with new PTS verification and real delivery"})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    run(parser.parse_args().out.resolve())
