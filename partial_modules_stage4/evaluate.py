"""Independent evaluation of complete boxes, local evidence and image measurements."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from statistics import median

import numpy as np

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage2.evaluate import source_audit
from partial_modules_stage3.evaluate import summarize
from partial_modules_stage3.observation import CHANGE, ENV, NEIGHBOR, intersect, iou
from partial_modules_stage4.prepare import H


def inside(points, box):
    xy=np.array(points).reshape(-1,2)
    x,y,w,h=box
    return (xy[:,0]>=x)&(xy[:,0]<x+w)&(xy[:,1]>=y)&(xy[:,1]<y+h)


def reference_support(anchors, geometry, selected):
    groups=defaultdict(dict)
    for anchor in anchors:
        groups[anchor['key']][anchor['pts']]=anchor
    anchored_matches=[]
    lines=[]
    box_matches=[]
    available=set()
    for basis in geometry['calibration']:
        a,b=basis['pts']
        for key, labels in groups.items():
            if a in labels and b in labels:
                available.add((a,b,key))
    for row in geometry['feature_patches']+geometry['stable_edges']:
        a,b=row['pts']
        for key, labels in groups.items():
            if a not in labels or b not in labels:
                continue
            is_point=row['kind']=='stable_feature_patch'
            native=(row['native_points_left'],row['native_points_right']) if is_point else row['segments_native']
            left,right=(labels[k]['roi' if is_point else ('edge_roi' if 'edge_roi' in labels[k] else 'roi')] for k in (a,b))
            common=inside(native[0],left)&inside(native[1],right)
            if common.sum() >= (3 if is_point else 2) and common.mean()>=.80:
                ref={'pts':[a,b],'key':key,'calculation_root':row['calculation_root'],
                     'source_roots':row['source_roots'],'in_reference_fraction':float(common.mean()),
                     'participant_pts':[j for j in (a,b) if labels[j]['participant']]}
                (anchored_matches if is_point else lines).append(ref)
    for r in selected:
        for basis in r['basis']:
            if basis['kind'] not in ('registered_edge_region','registered_stable_structure'):
                continue
            a,b=basis['pts']
            for key,labels in groups.items():
                if a in labels and b in labels and all('edge_roi' in labels[t] for t in (a,b)):
                    if min(iou(labels[a]['edge_roi'],basis['old']),iou(labels[b]['edge_roi'],basis['current']))>=.5:
                        box_matches.append({'pts':[a,b],'key':key,'chart_ref':basis['reference_chart']})
    def unique(rows):
        return len({(*r['pts'],r['key']) for r in rows})
    static_pos=sum(x['participant'] for x in anchors)
    scope_pts={(p,r['key']) for r in anchored_matches+lines for p in r['participant_pts']}
    return {'labelled_consecutive_pair_count':len(available),'feature_reference_pairs':unique(anchored_matches),
            'stable_edge_pairs':unique(lines),'verified_complete_reference_pairs':unique(box_matches),
            'participant_endpoints_with_local_support':len(scope_pts),'participant_endpoints_total':static_pos,
            'feature_support_detail':anchored_matches,'line_support_detail':lines,'box_support_detail':box_matches}


def original_hits(result, old_targets, audit):
    found=[]
    for row in audit['cases']:
        selected=[r for r in result['regions'] if r['pts']==row['pts'] and row['role'] in r['roles']]
        found.append({'pts':row['pts'],'label':row['label'], 'kept':any(iou(r['native_xywh'],row['gold_xywh'])>=.5 for r in selected),
                      'exact_from_A0':any(r['id'] in {x['id'] for x in row['original_candidates']} for r in selected)})
    assigned=set(result['identity_follows_role']['identifiers'])
    background=0;dual=0
    for t in old_targets:
        if t['role']!=ENV:
            continue
        eligible=[r for r in result['regions'] if r['id'] in assigned and r['pts']==t['pts'] and iou(t['box'],r['native_xywh'])>=.50]
        background+=bool(eligible)
        dual+=any(ENV in r['roles'] and CHANGE in r['roles'] for r in eligible)
    return {'original_A0_targets_recalled':sum(q['kept'] for q in found),'original_A0_targets':len(found),
            'A0_targets_with_exact_old_proposals':sum(q['exact_from_A0'] for q in found),
            'old_A0_retention_detail':found,'known_background_with_identity_attempt_eligibility':background,
            'known_background_with_independent_change_and_stable_claims':dual}


def state_score(rows, golds):
    output=[]
    for label in golds:
        matches=[];box=label['roi']
        for r in rows:
            if r['pts']!=label['pts']:
                continue
            for item in r['basis']:
                if item['kind']!='compensated_local_difference':
                    continue
                patch=item['explicit_roi']; overlap=intersect(box,patch)
                if overlap/(box[2]*box[3])>=.20 and overlap/(patch[2]*patch[3])>=.65:
                    matches.append({'patch':patch,'originating_proposal':r['id'],'chart_ref':item['calculation_root']})
        output.append(label|{'matched':bool(matches),'regional_support':matches})
    return output


def empty_geometry(geometry):
    return geometry|{'feature_patches':[],'stable_edges':[]}


def compare_case(work,labels,trace):
    geo=read(work/'region_features.json')
    full=geo['extra_geometry']
    common=read(work/'base_receipts.json')
    arms={}
    source_root=ROOT/'partial_modules_stage3/predictions'/work.name
    old=read(source_root/'A3.json') if (source_root/'A3.json').is_file() else None
    if old:
        legacy=read(work/'R0P0.json')
        key=lambda p:{r['id']:sorted(r['roles']) for r in p}
        assert key(legacy['regions'])==key(old['regions']),work
        assert [r['views'] for r in legacy['identity_follows_role']['links']]==[r['views'] for r in old['identity']['links']],work
    for arm in ('R0P0','R0P1','R1P0','R1P1'):
        row=read(work/f'{arm}.json')
        report=summarize({'regions':row['regions'],'arm':arm,'source_sha256':row['source_sha256'],
                          'identity':row['identity_follows_role'],'relation':row['m5'],
                          'seconds_identity':row['seconds_identity'],'seconds_relation':row['seconds_relations_both']},labels['legacy_targets'])
        fixed=summarize({'regions':row['regions'],'arm':arm,'source_sha256':row['source_sha256'],
                         'identity':row['identity_follows_role'],'relation':row['m5_fixed_identity'],
                         'seconds_identity':row['seconds_identity'],'seconds_relation':row['seconds_relations_both']},labels['legacy_targets'])
        report['identity_fixed_interpretations']=dict(Counter(q['interpretation_scope'] for q in row['m5_fixed_identity']['raw_cross_time']))
        report['identity_follows_role_interpretations']=dict(Counter(q['interpretation_scope'] for q in row['m5']['raw_cross_time']))
        primary=full if arm.startswith('R1') else empty_geometry(full)
        record={'box_and_relationship_scores':report,'fixed_identity_valid_relation_diffs':fixed['within_8px_real_projected_endpoint_differences'],
                'reference_scores':reference_support(labels['local_anchors'],primary,row['regions']),
                'state_patches':state_score(row['regions'],labels['state_regions']),
                'role_audit':original_hits(row,labels['legacy_targets'],trace),
                'fixed_identity_count':row['same_identity_for_all_conditions']['eligible_original_count']}
        if arm in ('R0P0','R1P1'):
            no_flow=read(work/f'{arm}-no-flow.json')
            score=summarize({'regions':row['regions'],'arm':arm+'-no-flow','source_sha256':row['source_sha256'],
                          'identity':no_flow['identity'],'relation':row['m5'],
                          'seconds_identity':no_flow['seconds_identity'],'seconds_relation':row['seconds_relations_both']},labels['legacy_targets'])
            record['frozen_rigid_only']={k:score[k] for k in ('confirmed_links','identity_probes','compute_seconds')}
        arms[arm]=record
    cost={k:v for k,v in common.items() if k.startswith('seconds')}
    native=common['native_receipts']
    cost.update({'exact_originals':len(native),'png_bytes':sum(r['png_bytes'] for r in native),
                 'decode_filter_inputs':sum(a['frames_at_input_filter'] or 0 for r in native for a in r['decode_attempts'])})
    return {'stage3_A3_exact_regression':bool(old),'arms':arms,'cost':cost},native


def gates(scores):
    arm={c:k['arms']['R1P1'] for c,k in scores.items()}
    keep={k:arm[k]['role_audit']['original_A0_targets_recalled']==arm[k]['role_audit']['original_A0_targets']
          for k in ('roles','street-repeat')}
    local={'retained_original':keep,'controlled_ref_local_support_at_least_half':
           arm['roles']['reference_scores']['participant_endpoints_with_local_support']>=5}
    for k in ('new-pan','new-local'):
        t=arm[k]['reference_scores']
        local[k+'_at_least_one_reference_edge']=bool(t['feature_reference_pairs'] or t['stable_edge_pairs'])
    full={}
    for k in ('external-1','external-2'):
        t=arm[k]['box_and_relationship_scores']
        full[k+'_coverage']=t['coverage']>=.5
        result=t['median_center_abs_error_native_px']
        full[k+'_center']=result is not None and result<=8
    full['real_linked_role_pairs']=sum(arm[k]['box_and_relationship_scores']['within_8px_real_projected_endpoint_differences']
                                      for k in ('street-repeat','external-1','external-2','new-pan','new-local'))>=2
    return {'local':local,'full':full,'passed_local':all(keep.values()) and all(v for k,v in local.items() if k!='retained_original'),
            'passed_full':all(full.values()),'new_AMR_run':False}


if __name__=='__main__':
    ap=ArgumentParser();ap.add_argument('--prediction-dir',default='predictions');ap.add_argument('--out',default='evaluation.json');args=ap.parse_args()
    prediction=H/args.prediction_dir
    assert (prediction/'completed.json').is_file()
    frozen=read(H/'freeze.json');reference=ROOT/frozen['assessor']['file']
    assert frozen['assessor']['sha256']==digest(reference)
    labels=read(reference);sources=read(H/'inputs/source_inventory.json');audit=read(H/'phase3_role_audit.json')
    scores={};frames=[]
    for case,spec in sources.items():
        scores[case],bundle=compare_case(prediction/case,labels[case],audit.get(case,{'cases':[]}))
        frames.append(bundle)
    checked=source_audit(frames,[ROOT/x['path'] for x in sources.values()])
    output=H/args.out
    if output.exists() or output.parent!=H:
        raise FileExistsError(output)
    write(output,{'gates':gates(scores),'cohorts':scores,'source_pixel_check':checked})
