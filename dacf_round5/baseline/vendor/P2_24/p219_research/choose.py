"""Goal-scoped proposals over the unchanged D1–D6 archive."""

from collections import defaultdict
from itertools import groupby
from statistics import median

from if17_research.catalog import read
from p217_research.triggers import PARAMS
from p218_research.candidate import route as source_plan


RELEVANT={"scene":{"D4","D5"},"single":{"D1","D3","D4"},"relation":{"D3","D2","D5"},
          "recurrence":{"D1","D3","D4"},"boundary":{"D2","D5"},"open":{"D2","D4","D5"}}
ROLE={"D1":"direct_region_observation","D2":"local_state_comparison","D3":"projected_relation_candidate",
      "D4":"viewpoint_reference","D5":"conflict_or_visibility_check","D6":"independent_time_witness"}


def rank_nominations(row,dimension):
    return max((n["priority_within_dimension"] for n in row["nominations"] if n["dimension_id"]==dimension),default=0.)


def contiguous(rows):
    for _,g in groupby(rows,key=lambda r:r["image_dimensions"]==["D6"]):
        yield list(g)


def index(book,goal,facts,missing):
    intent=goal["intent"]
    rows=book.rows
    middle=book.source["start"]+(book.source["end"]-book.source["start"])*2/3
    eligible=[r for r in rows if goal.get("scope_hint")!="late" or r["t"]>=middle]
    times=sorted({book.source["start"],book.source["end"],*[r["t"] for r in rows if "D6" in r["image_dimensions"]]})
    zones=[[a,c] for a,c in zip(times,times[1:])]
    def region(row):
        return next((j for j,(a,c) in enumerate(zones) if a<=row["t"]<c),len(zones)-1)
    allowed={k["frame_id"] for k in eligible}
    channels={}
    for row in rows:
        key=row["frame_id"]
        channels[key]={"frame":book.stamp(key),"dimension_entries":row["image_dimensions"],"time_region":region(row),
                       "observations":[{"dimension":o["dimension_id"],"role":o["role"],"root_ids":o["evidence_root_ids"],
                                "comparison_ids":o["source_comparison_frames"],"analysis_box":o["region_xywh_analysis"]} for o in row["observations"]],
                       "scope_eligible":key in allowed}
    nominal=source_plan(book,goal,"A3",facts,missing)
    qtype={"D2":"D2","D3":"D3","D4":"D4","D5":"D5","D5_following_bracket":"D5","D6_with_P1_pixels":"D6","D6_with_P1_blend_proxy":"D6"}
    variant=defaultdict(list)
    if intent!="single":
        for q in nominal["proposals"]:
            dim=qtype.get(q["dimension"])
            if (dim not in RELEVANT[intent] and dim!="D6") and q["trigger_kind"]!="explicit_source_request":
                continue
            if q.get("c_id") not in allowed and q["trigger_kind"]!="explicit_source_request":
                continue
            z=region(book.by[q["c_id"]]) if q.get("c_id") else -1
            variant[(z,dim)].append(q)
    qgroups=[]
    for (z,d),items in variant.items():
        def strength(q):
            m=q["comparison"]
            return (bool(q.get("parent_trigger_id")),m.get("crop_mad",m.get("raw_mad",0)), -q["interval"][0])
        qgroups.append({"region":z,"channel":d,"selected":max(items,key=strength),"all_ids":[q["id"] for q in items]})
    return {"goal":goal,"entries":channels,"selected_comparison_groups":qgroups,"unread_eligible_count":len(eligible),
            "partitions":zones,"nominal_proposals":nominal["proposals"],"eligible":eligible,"region":region,
            "comparison_endpoints_outside_library":missing}


def choose(book,scope,arm):
    goal=scope["goal"]
    intent=goal["intent"]
    allowed=scope["eligible"]
    region=scope["region"]
    props=[]
    defer=defaultdict(list)
    def add(key,role,why,root=(),parent=None):
        props.append({"frame_id":key,"observation_role":role,"reason":why,"root_ids":list(root),"comparison_id":parent,
                      "uncertainty":"image-level candidate; physical identity and causality unknown"})

    if intent=="single":
        def location(row):
            if "left frame edge" not in goal["text"].casefold():
                return 0
            return max((320-(v["region_xywh_analysis"][0]+v["region_xywh_analysis"][2]/2)
                        for v in row["observations"] if v["region_xywh_analysis"]),default=0)
        if allowed:
            hit=max(allowed,key=lambda r:(location(r),rank_nominations(r,"D1"),rank_nominations(r,"D3"),r["t"]))
            add(hit["frame_id"],"direct_single_view","The public task asks for a single direct native view in its requested time region.",
                sorted({k for o in hit["observations"] for k in o["evidence_root_ids"]}))
    else:
        # An independent temporal entrance protects a region even when no
        # higher-dimension proposal activates within that region.
        for bucket in contiguous(allowed):
            if all(r["image_dimensions"]==["D6"] for r in bucket) and len(bucket)>1:
                checks=[book.compare(a["frame_id"],c["frame_id"]) for a,c in zip(bucket,bucket[1:])]
                if any(x["raw_mad"]>=PARAMS.activity_mad for x in checks):
                    mid=median(r["t"] for r in bucket)
                    witness=min(bucket,key=lambda r:(abs(r["t"]-mid),r["t"]))
                    add(witness["frame_id"],"independent_time_witness","Locally changing earlier/later D6-only run has its own checkable temporal witness.",
                        [witness["frame_id"]])
        frame_groups=defaultdict(list)
        if intent in ("relation","scene","recurrence"):
            for r in allowed:
                for dim in RELEVANT[intent]&set(r["image_dimensions"]):
                    if dim=="D2" or dim=="D5":
                        continue
                    frame_groups[(region(r),dim)].append(r)
        for (z,dim),items in sorted(frame_groups.items()):
            best=max(items,key=lambda r:(rank_nominations(r,dim), -r["t"]))
            roots=sorted({e for o in best["observations"] if o["dimension_id"]==dim for e in o["evidence_root_ids"]})
            add(best["frame_id"],ROLE[dim],"Relevant measured nomination independently represents this time region and observation role.",roots)
            for item in items:
                if item is not best:
                    defer[item["frame_id"]].append("other discoverable candidate in the current role and time region")
        for group in scope["selected_comparison_groups"]:
            q=group["selected"]
            if not q.get("c_id"):
                continue
            roots=q["comparison"]["basis_root_ids"]
            add(q["c_id"],ROLE.get(group["channel"],"transition_check"),"Measured current observation for the goal-relevant comparison",roots,q["id"])
            if (arm in ("S2","S3") or intent=="boundary") and q["a_id"]!=q["c_id"]:
                add(q["a_id"],"comparable_prestate", "Existing actual comparison endpoint; a similar earlier state can bound a later change.",roots,q["id"])
    sem_proposals=[]
    if arm=="S3" and intent=="open":
        stored=read(book.entry["semantic"]+"/candidates.json")["rows"]
        delivered_regions={region(book.by[x["frame_id"]]) for x in props}
        histogram=defaultdict(set)
        lookup=defaultdict(list)
        for row in stored:
            for node in row["nodes"]:
                box=node["region"]["bbox"]
                if box[2]*box[3]>=.45*book.source["probe"]["width"]*book.source["probe"]["height"]:
                    continue
                for tag in node["visual_hypotheses"]:
                    histogram[tag["word"]].add(row["frame_id"])
                    lookup[row["frame_id"]].append((tag,box,node["candidate_id"]))
        from p215_research.semantic import overlap
        candidate_regions=defaultdict(list)
        for row in allowed:
            key=row["frame_id"]
            boxes=[book.native_box(o["region_xywh_analysis"]) for o in row["observations"] if o["region_xywh_analysis"]]
            for tag,box,uid in lookup[key]:
                matched=any(overlap(a,box)>.2*min(a[2]*a[3],box[2]*box[3]) for a in boxes)
                record={"frame_id":key,"tag":tag["word"],"proposal_region":box,"candidate_root":uid,"image":row["source_image"],
                        "region":region(row),"corroborated_local_region":matched,"weak_word_only":True,"cosine":tag["cosine"]}
                sem_proposals.append(record)
                if matched and len(histogram[tag["word"]])>1 and record["region"] not in delivered_regions:
                    candidate_regions[record["region"]].append(record)
        for rows in candidate_regions.values():
            r=max(rows,key=lambda x:(x["cosine"],x["frame_id"]))
            add(r["frame_id"],"weak_object_region_for_exploration","Repeated weak object word and independent measured local footprint open an otherwise unexplored time region.",[r["candidate_root"]])
            r["delivered_as_weak_seed"]=True
    by=defaultdict(list)
    for row in props:
        row["time_region"]=region(book.by[row["frame_id"]])
        by[row["frame_id"]].append(row)
    return dict(by),dict(defer),sem_proposals
