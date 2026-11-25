"""Lossless nomination union and fixed-budget display allocation."""
from collections import Counter, defaultdict

from .common import RULES, frame_key


def group_roots(reasons):
    groups=[]
    for r in reasons:
        roots=set(r["evidence_root_ids"])
        families={r["evidence_family"]}
        dimensions={r["dimension_id"]}
        kept=[]
        for g in groups:
            if roots & g[0] or families & g[1]:
                roots |= g[0]
                families |= g[1]
                dimensions |= g[2]
            else:
                kept.append(g)
        groups=kept+[(roots,families,dimensions)]
    return [{"roots":sorted(r),"families":sorted(f),"dimensions":sorted(d)} for r,f,d in groups]


def merge(fs,src,channels):
    rows={}
    for reasons in channels.values():
        for r in reasons:
            f=fs[r["sample"]]
            k=frame_key(src,f)
            assert r["frame_id"]==k
            if k not in rows:
                rows[k]={"video_sha256":src["sha256"],"sample":r["sample"],"t":f["t"],
                          "source_frame":f["source_frame"],"pts":f["pts"],"time_base":f["time_base"],"frame_id":k,
                          "full_resolution_reference":{"source_path":src["path"],"source_frame":f["source_frame"],
                              "pts":f["pts"],"time_base":f["time_base"]},"nominations":[]}
            rows[k]["nominations"].append(r)
    role_counts=Counter(r["role"] for row in rows.values() for r in row["nominations"])
    for row in rows.values():
        reasons=row["nominations"]
        row.update(nomination_count=len(reasons),
                   image_dimensions=sorted({r["dimension_id"] for r in reasons}),
                   independent_evidence_groups=group_roots(reasons),
                   unique_role_ids=sorted({r["role"] for r in reasons if role_counts[r["role"]]==1}))
    return sorted(rows.values(),key=lambda r:r["t"])


def display(fs,library,channels,budget=8,soft=False):
    selected={}
    t={r["sample"]:r["t"] for r in library}
    selected_reasons=set()
    def record(i,label,r=None):
        selected[i]={"allocation":label,"support":r}
        if r:
            selected_reasons.add((r["role"],tuple(r["evidence_root_ids"])))
    endpoints=[r for r in channels["D6"] if r["role"]=="endpoint"]
    for r in endpoints[:budget]:
        record(r["sample"],"source range endpoint",r)
    radius=RULES["local"]["soft_group_seconds" if soft else "nearby_seconds"]
    d1_times=[]
    for d in RULES["selector"]["sequence"]:
        possible=[r for r in channels[d] if r["sample"] not in selected and
                  (r["role"],tuple(r["evidence_root_ids"])) not in selected_reasons]
        if not possible or len(selected)>=budget:
            continue
        def ordering(r):
            ts=t[r["sample"]]
            other_time=all(abs(ts-u)>radius for u in d1_times) if d=="D1" else True
            separated=all(abs(ts-t[i])>radius for i in selected)
            return other_time,separated if soft else True,r["priority_within_dimension"],-ts
        best=max(possible,key=ordering)
        j=best["sample"]
        if d=="D1":
            d1_times.append(t[j])
        record(j,"reserved dimension slot "+d,best)
    for remaining in [{r["sample"] for r in channels["D6"]},set(t)]:
        while len(selected)<min(budget,len(t)) and remaining-set(selected):
            def gap(j):
                return min((abs(t[j]-t[i]) for i in selected),default=1),-t[j]
            record(max(remaining-set(selected),key=gap),"coverage fill")
    by_frame={row["sample"]:row for row in library}
    for row in library:
        j=row["sample"]
        row["delivered_in_first_batch"]=j in selected
        row["display_reason"]=selected.get(j)
        row["deferred_reason"]=None if j in selected else "display budget and allocated slots; remains queryable"
        if soft:
            nearest=[i for i in selected if abs(t[i]-t[j])<=radius]
            row["nearby_group_id"]=by_frame[min(nearest,key=lambda i:(abs(t[i]-t[j]),t[i]))]["frame_id"] if nearest else None
    return sorted(selected),selected


def reverse_index(library):
    inv=defaultdict(list)
    for frame in library:
        for n in frame["nominations"]:
            inv[n["role"]].append({"frame_id":frame["frame_id"],"t":frame["t"],
                  "roots":n["evidence_root_ids"],"region":n["applicable_region"],
                  "first_batch":frame["delivered_in_first_batch"]})
    return dict(inv)
