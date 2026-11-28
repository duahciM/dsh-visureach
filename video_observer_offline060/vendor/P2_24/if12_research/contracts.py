"""IF12 module registry, dependency checks and append-only runtime receipts."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import traceback

from identity_catalog_matching.common import read, sha
from p1.io import dump
from .fingerprints import OPTIONAL, code_assets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IF11 = ROOT/"if11_research_20261119"
IF10 = ROOT/"if10_research_20261119"
IF09 = ROOT/"if09_research_20261119"
IF07 = ROOT/"if07_research_20261119"
RULES = read(HERE/"protocol.json")
PRIOR = read(IF11/"protocol_locked.json")
SPATIAL = read(IF10/"protocol_locked.json")["space"]

@dataclass(frozen=True)
class Module:
    responsibility: str
    entry: str
    required: tuple = ()
    optional: tuple = ()
    maturity: str = "candidate"


REGISTRY = {
    "M0":Module("source pool and time/coordinate space","v05.engine.load_source/resample; if07_research.run.prior_frames; if09_research.run.load_fresh"),
    "M1":Module("pair and motion observation ledger","if09_research.relations.ledger; v05.engine.motion_diagnostics; p14_scene_prototypes.features",("M0",)),
    "M2":Module("visual ID and per-view evidence","identity_small_support_spatial.run; identity_feature_grouping.grouping",("M1",),maturity="shadow"),
    "M3":Module("T1 references and local continuity","if09_research.relations.cover_charts; if10_research.spatial.endpoint_candidates",("M1",),("M2",)),
    "M4":Module("localized difference and retrieval candidate","if11_research.region.analyze; if12_research.regions",("M0","M1","M3"),("M2",)),
    "M5":Module("state and role evidence","if11_research.state.sampled_role_ledger; if09_research.subjects",("M0",),("M1","M4")),
    "M6":Module("independent evidence judgement without source mutation","if12_research.modular.judgements",(),("M2","M3","M4","M5")),
    "M7":Module("fixed-budget frame selection","v05.engine.choose; if11_research.state.role_trial",("M0",),("M4","M5","M6")),
    "M8":Module("query proposals and original-source reads","if11_research.state.windows; v05.engine.resample",("M0",),("M4","M5","M7")),
    "M9":Module("Agent and observation coverage","if11_research.agent.Harness; if12_research.agent",("M7","M8"),maturity="shadow")
}


def signature(obj):
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def reference(p):
    p = Path(p)
    return {"path":str(p),"sha256":sha(p)}


class StageStopped(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


class Pipeline:
    def __init__(self, out, run_id, source_hash, off=(), fail=()):
        self.out = out
        self.out.mkdir(parents=True,exist_ok=True)
        self.run_id = run_id
        self.source_hash = source_hash
        self.off = set(off)
        self.fail = set(fail)
        self.done = {}
        self.data = {}
        self.counter = 0

    def invoke(self, module, action, callback, refs=(), mode="live_compute", params=None, required=None):
        info = REGISTRY[module]
        children = info.required if required is None else required
        deps = {m:self.done[m]["status"] if m in self.done else "not_loaded" for m in children}
        consumed = OPTIONAL.get(action, info.optional)
        opt = {m:self.done[m]["status"] for m in consumed if m in self.done}
        self.counter += 1
        assets = code_assets(ROOT,module,action)
        version = signature([module,action,info.entry,assets])
        missing = [str(p) for p in refs if not Path(p).is_file()]
        direct = [reference(p) for p in refs if Path(p).is_file()]
        upstream = {m:self.done[m]["cache_key"] for m,status in (deps | opt).items() if status=="completed"}
        space = self.done.get("M0",{}).get("coordinate_contract") if module!="M0" else None
        request = {"run_id":self.run_id,"module_id":module,"module_version":version,
                   "config_hash":signature(params),"input_refs":direct,"source_hash":self.source_hash,
                   "budget":(params or {}).get("budget",{}),"execution_mode":mode}
        key = signature(request | {"run_id":None,"upstream":upstream,"coordinate_space":space})
        t = perf_counter()
        val, meta = None, {}
        try:
            if module in self.off:
                raise StageStopped("disabled","module switched off")
            if any(v!="completed" for v in deps.values()):
                raise StageStopped("dependency_missing","required stage did not complete")
            if missing:
                raise StageStopped("dependency_missing","missing input artifacts: "+", ".join(missing))
            if module in self.fail:
                raise RuntimeError("injected module failure")
            val, meta = callback()
            status, reason = "completed", meta.pop("reason",None)
            self.data[module] = val
        except StageStopped as exc:
            status, reason = exc.status, exc.reason
        except Exception:
            status, reason = "failed", traceback.format_exc()
        result = {"request":request,"action":action,"status":status,"reason":reason,
                  "execution_mode":mode if status=="completed" else "not_run" if status in ("disabled","dependency_missing") else mode,
                  "maturity":info.maturity,"cache_key":key,"implementation_assets":assets,
                  "cache_dependencies":upstream,"source_coordinate_space":space,
                  "dependencies":{"required":deps,"optional":opt},
                  "artifact_refs":[],"root_ids":[],"candidate_ids":[],"used_budget":{},
                  "timings":{"wall_seconds":perf_counter()-t}} | meta
        dump(self.out/f"{self.counter:02}_{module}_{action}.json",result)
        self.done[module] = result
        return val
