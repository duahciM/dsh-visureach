"""Audit failure mechanisms without retraining, retuning, or replacing a locked arm."""
from collections import Counter,defaultdict

import cv2
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import read,write
from partial_modules_stage3.material import H
from partial_modules_stage3.observation import structures,iou,CHANGE,NEIGHBOR,ENV,overlaps_inside,Media,scoped_identity
from partial_modules_stage3.evaluate import summarize
from partial_modules_stage3.run import m5


def mechanism(name, sealed):
    base=read(H/'predictions'/name/'base_receipts.json')
    arm=read(H/'predictions'/name/'A3.json'); raw=read(H/'predictions'/name/'A0.json')
    reach=defaultdict(list); assets={}; gray={}
    for r in arm['regions']:reach[r['pts']].append(r)
    for x in base['receipts']:
        with Image.open(x['path']) as p:assets[x['pts']]=np.array(p.convert('RGB'))
        gray[x['pts']]=cv2.cvtColor(assets[x['pts']],cv2.COLOR_RGB2GRAY)
    class Surface:
        images=assets
        # Passing the exact original raster is enough for the structural policy.
    surface=Surface();surface.gray=gray
    source=[];roles=Counter()
    for r in sealed:
        t=r['pts'];box=r['box']; candidates=[q for q in reach[t] if r['role'] in q['roles']]
        b,extra=structures(surface,t)
        sat=cv2.cvtColor(assets[t],cv2.COLOR_RGB2HSV)[:,:,1]
        trials={'pts':t,'episode':r['episode'],'target_role':r['role'],
            'best_structural_iou':max((iou(box,c) for c in b),default=0.0),
            'best_environment_proposal_iou':max((iou(box,q['native_xywh']) for q in reach[t] if ENV in q['roles']),default=0.0),
            'best_correct_role_iou':max((iou(box,q['native_xywh']) for q in candidates),default=0.0),
            'structural_discarded':extra,'chroma_mask_occupied_fraction':float((sat>=55).mean())}
        if r['role']==NEIGHBOR:
            roles['required_static_or_relational']+=1
            roles['initial_structure_overlaps']+=trials['best_structural_iou']>=.5
            roles['stable_chart_proposal_overlaps']+=trials['best_environment_proposal_iou']>=.5
            roles['role_measured']+=trials['best_correct_role_iou']>=.5
        conflicts=[q for q in reach[t] if CHANGE in q['roles'] and overlaps_inside(q['native_xywh'],box)>=.5]
        trials['local_changed_component_in_target_count']=len(conflicts)
        source.append(trials)
    a4=read(H/'diagnostic_manual'/name/'A4.json')['identity']
    flow=Counter(q['flow']['state'] for q in a4['pair_audit'] if q['flow'])
    return {'source_sha256':base['source_sha256'],'all_requested_reference_chain':dict(roles),
            'per_target':source,'reference_alignment_states':dict(Counter(q['state'] for q in base['compensated_change_diagnostics'])),
            'first_arm_identity_reasons':{k:read(H/'predictions'/name/f'{k}.json')['identity']['attempts_by_reason'] for k in ('A0','A1','A2','A3')},
            'hand_supplied_local_fit':a4['attempts_by_reason'],'hand_supplied_nonrigid_outcomes':dict(flow),
            'unassessed_scope_links_remain_unassessed':True}


def correct_manual(case, spec, ref, directory):
    dest=directory/case;dest.mkdir()
    collection=Media(spec,dest)
    inventories=defaultdict(list)
    for r in ref:
        o=collection.region(r['pts'],r['box'],[r['role']],[{'kind':'assessor_diagnostic'}])
        inventories[r['pts']].append(o|{'episode':r['episode']})
    for t in collection.ticks:inventories[t]
    correct=scoped_identity(collection,inventories,'A3-no-flow')
    relation=m5(collection,inventories,correct,manual=True)
    bundle={'arm':'A4-no-flow-matched-role','source_sha256':collection.sha,'regions':sum(list(inventories.values()),[]),
            'identity':correct,'relation':relation,'seconds_identity':None,'seconds_relation':None}
    write(dest/'result.json',bundle)
    return summarize(bundle,ref)


if __name__=='__main__':
    out=H/'diagnostics';out.mkdir()
    sealed=read(H/'references/sealed.json')['targets']
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    inputs=read(H/'inputs/source_inventory.json')
    mechanics={k:mechanism(k,v) for k,v in sealed.items()}
    write(out/'failure_mechanisms.json',mechanics)
    fixed={k:correct_manual(k,inputs[k],ref,out) for k,ref in sealed.items()}
    write(out/'paired_A4_ablation.json',{'fix':'Original A4-no-flow selector inadvertently included environment rows. First run is retained.',
          'scope':'Freeze unchanged. Compare this corrected hand-ROI arm to the original A4. Automatic arms and AMR gate are unchanged.',
          'cases':fixed})
