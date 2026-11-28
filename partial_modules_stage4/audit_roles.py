"""Read the frozen Phase 3 results and trace individual lost correct proposals."""

from collections import Counter

from dacf_research.prepare_inputs import ROOT, read, write
from partial_modules_stage3.observation import CHANGE, NEIGHBOR, iou
from partial_modules_stage4.prepare import H


def trace(case, labels):
    folder = ROOT / 'partial_modules_stage3/predictions' / case
    baseline, original = (read(folder / f'{arm}.json') for arm in ('A0', 'A3'))
    registrations = {r['pts'][1]:r for r in read(folder / 'base_receipts.json')['compensated_change_diagnostics']}
    exact_by_id = {r['id']:r for r in original['regions']}
    rows = []
    for target in labels:
        role, pts, box = (target[k] for k in ('role', 'pts', 'box'))
        if role not in (CHANGE, NEIGHBOR):
            continue
        def matching(data, require_role):
            return sorted(({'iou':iou(r['native_xywh'],box), **r} for r in data['regions']
                           if r['pts']==pts and (not require_role or role in r['roles'])),
                          key=lambda r: (-r['iou'],r['id']))
        before = [r for r in matching(baseline,True) if r['iou'] >= .5]
        if not before:
            continue
        by_role = matching(original,True)
        geometry = matching(original,False)
        nominees = []
        for candidate in before:
            exact = exact_by_id.get(candidate['id'])
            nominees.append({'id':candidate['id'],'xywh':candidate['native_xywh'],'old_iou':candidate['iou'],
                             'exact_A3_roles':exact['roles'] if exact else [],
                             'exact_A3_identity_eligible':candidate['id'] in original['identity']['identifiers'],
                             'source_root':candidate['source_root']})
        recovered = bool(by_role and by_role[0]['iou'] >= .5)
        good_geometry = bool(geometry and geometry[0]['iou'] >= .5)
        all_exact_absent = not any(r['exact_A3_roles'] for r in nominees)
        diagnosis = ('recovered' if recovered else 'role_only_mismatch' if good_geometry else
                     'exact_old_proposal_removed_no_other_valid_geometry' if all_exact_absent else 'novel_geometry_invalid')
        rows.append({'pts':pts,'label':target['episode'],'role':role,'gold_xywh':box,'original_candidates':nominees,
                     'registered_comparison':registrations.get(pts,{}).get('state','no_previous_sample'),
                     'best_any_A3':geometry[0] if geometry else None,'best_correct_role_A3':by_role[0] if by_role else None,
                     'diagnosis':diagnosis})
    counts = Counter(r['diagnosis'] for r in rows)
    by_registration = Counter(r['registered_comparison'] for r in rows if r['diagnosis']!='recovered')
    return {'counts':dict(counts),'lost_by_compensation_applicability':dict(by_registration),'cases':rows}


if __name__ == '__main__':
    targets = read(ROOT / 'partial_modules_stage3/references/sealed.json')['targets']
    result = {k:trace(k,v) for k,v in targets.items()}
    write(H / 'phase3_role_audit.json', result)
    print({k:v['counts'] for k,v in result.items()})
