"""One pre-association reference gate; all other functions reuse frozen v06."""
import copy

from v05.engine import public
from v06.engine import CFG as V06_CFG,associate_candidates


def gated_cores(frames,old,matrices,eligibility):
    hsv,gray,sift,support=matrices;active=[];decisions=[]
    for n,candidate in enumerate(old['candidate_cores']):
        kept=[i for i in candidate['refs'] if eligibility[i]['eligible']];rejected=[i for i in candidate['refs'] if i not in kept]
        size_ok=len(candidate['members'])>=4 and frames[max(candidate['members'])]['t']-frames[min(candidate['members'])]['t']>=V06_CFG['minimum_core_span_s']
        span=frames[max(kept)]['t']-frames[min(kept)]['t'] if kept else None
        self_ok=len(kept)>=2 and all(any(i!=j and support[i,j] for j in kept) for i in kept)
        available=bool(size_ok and self_ok and span+1e-9>=V06_CFG['minimum_core_span_s']) if span is not None else False
        decisions.append({'old_candidate_id':n,'old_refs':candidate['refs'],'qualified_refs':kept,'rejected_refs':rejected,'available':available,
            'retained_reference_span_s':span,'original_member_conditions':bool(size_ok),'retained_nonself_support':bool(self_ok),
            'reason':'qualified_original_references_no_replacements' if available else 'too_few_or_too_close_retained_refs_or_no_nonself_support',
            'removed_as_core_only':rejected,'member_pool_changed':False})
        if available:active.append({**copy.deepcopy(candidate),'candidate_id':len(active),'old_candidate_id':n,'refs':kept})
    assoc,logs,relation=associate_candidates(active,support)
    assoc.sort(key=lambda cs:min(i for n in cs for i in active[n]['refs']))
    groups=[]
    for n,cs in enumerate(assoc):
        refs=sorted({i for k in cs for i in active[k]['refs']});units=[]
        for i in refs:
            same=next((u for u in units if hsv[i,u[0]]<=V06_CFG['near_duplicate_hsv'] and gray[i,u[0]]<=V06_CFG['near_duplicate_gray']),None)
            if same is None:units.append([i])
            else:same.append(i)
        groups.append({'scene_id':f'S{n+1:02d}','candidate_ids':cs,'old_candidate_ids':[active[k]['old_candidate_id'] for k in cs],
            'frozen_core_samples':refs,'reference_view_units':units,'prototype_samples':refs,
            'prototype_frames':[{**public(frames[i]),'selection_basis':'original v06 ref passed locked three-edge gate, no replacement'} for i in refs],
            'physical_identity':'unknown','core_revision':0})
    mappings=[]
    for g in old['groups']:
        original=set(g['candidate_ids']);news=[p['scene_id'] for p in groups if original&set(p['old_candidate_ids'])]
        mappings.append({'old_scene_id':g['scene_id'],'old_core_samples':g['frozen_core_samples'],'new_scene_ids':news,
            'retained_core_samples':sorted(set(g['frozen_core_samples'])&{i for p in groups for i in p['frozen_core_samples']}),
            'status':'unavailable' if not news else 'survives_with_original_association_rules','not_a_different_scene_judgment':True})
    return {'groups':groups,'candidate_cores':active,'candidate_qualification':decisions,'association_evidence':logs,'association_relation':relation.tolist(),
            'old_to_new_ids':mappings,'core_promotions':0,'replacement_references':0,
            'freeze_policy':'one qualification pass then original v06 association; ordinary members never become replacement cores'}


def boundary_changes(old,new,id_mappings):
    mapping={m['old_scene_id']:m['new_scene_ids'] for m in id_mappings};used=set();records=[]
    for b in old:
        pool=[(i,c) for i,c in enumerate(new) if i not in used and c['left_scene_id'] in mapping.get(b['left_scene_id'],[]) and c['right_scene_id'] in mapping.get(b['right_scene_id'],[])]
        chosen=min(pool,key=lambda p:(abs(p[1]['midpoint']-b['midpoint']),p[0])) if pool else None
        if chosen:
            i,c=chosen;used.add(i);records.append({'action':'unchanged' if b['bracket']==c['bracket'] else 'changed','old':b,'new':c,'matching':'mapped IDs then nearest midpoint, no truth/reference selection'})
        else:records.append({'action':'withdrawn','old':b,'new':None,'reason':'no surviving mapped two-sided reliable occurrence boundary; not proof old boundary false'})
    records.extend({'action':'new','old':None,'new':b} for i,b in enumerate(new) if i not in used)
    return records
