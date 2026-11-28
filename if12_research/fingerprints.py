"""Source-level fingerprints and action-specific M0–M9 dependencies."""
from pathlib import Path
from functools import lru_cache

from identity_catalog_matching.common import sha

FILES={
    "M0":("if12_research/modular.py","if12_research/fresh.py","p1/io.py","v05/engine.py",
          "if07_research/run.py","if09_research/run.py"),
    "M1":("if12_research/modular.py","if12_research/fresh.py","if09_research/relations.py",
          "if08_research/bridges.py","p14_scene_prototypes/features.py","p1/photometric.py","v05/engine.py"),
    "M2":("if12_research/identity.py","identity_small_support_spatial/run.py","identity_small_support_spatial/ranking.py",
          "identity_catalog_matching/match.py","identity_feature_grouping/grouping.py","identity_feature_grouping/evidence.py"),
    "M3":("if12_research/modular.py","if09_research/relations.py","if10_research/spatial.py"),
    "M4":("if12_research/modular.py","if11_research/region.py","if10_research/spatial.py"),
    "M5":("if12_research/modular.py","if11_research/state.py"),
    "M6":("if12_research/modular.py",),
    "M7":("if12_research/modular.py","if11_research/state.py","v05/engine.py"),
    "M8":("if12_research/modular.py","if11_research/state.py","v05/engine.py"),
    "M9":("if12_research/coverage.py","if12_research/agent.py","if11_research/agent.py","if11_research/dsh_patch.yml"),
}

ACTION_FILES={
   "regional_trials":("if12_research/regions.py",),
   "selection_shadow":("if12_research/selection.py",),
   "Q1":("if12_research/query.py","if12_research/protocol.json"),
   "Q2":("if12_research/query.py","if12_research/protocol.json"),
}

OPTIONAL={
    "frozen_references":(),
    "reference":(),
    "I0":(),
    "regional_trials":(),
    "image_claims":("M4",),
    "state_roles":("M4",),
    "select":("M5",),
    "selection_shadow":("M4",),
    "Q0":("M4","M5"),
    "direct":(),
    "Q1":("M1",),
    "Q2":("M1",),
}


@lru_cache(maxsize=64)
def code_assets(root,module,action):
    files=("if12_research/contracts.py","if12_research/fingerprints.py",*FILES[module],
           *ACTION_FILES.get(action,()))
    return {f:sha(Path(root)/f) for f in files}
