"""Trace the counterintuitive evidence gained by the deliberately shifted control."""

from dacf_research.prepare_inputs import read, write
from partial_modules_stage3.observation import iou
from partial_modules_stage6.material import H


if __name__=='__main__':
    spec=read(H/'assessment/sealed.json')['reel']
    roles={r['id']:r for r in spec['roles']}
    grades=read(H/'evaluation.json')['cases']['reel']['all_source_arms']
    manual=grades['MANUAL']['role_detail'];wrong=grades['WRONG']['role_detail']
    supplement=read(H/'observations/reference/reel/WRONG.json')
    rows=[r for g in supplement['nomination_groups'] for r in g['nominations']]
    gaps=[(a[1],b[0]) for a,b in zip(spec['bounds'],spec['bounds'][1:])]
    def phase(t):
        return next((i for i,(a,b) in enumerate(gaps) if a<t<b),None)
    previous=[]
    for key,value in wrong.items():
        if not value['assessed_native_ROI_support_pts'] or manual[key]['assessed_native_ROI_support_pts']:
            continue
        for item in rows:
            if item['pts'] not in value['assessed_native_ROI_support_pts'] or item['dimension_id']=='D6':
                continue
            if not item['applicable_region'] or iou(item['applicable_region'],roles[key]['native_roi'])<.2:
                continue
            pts=[int(source.split(':')[-2]) for source in item['comparison_frame_ids']]
            previous.append({'assessment_role':key,'pts':item['pts'],'native_source_frame_reference':item['source_root'],
              'basis':item['role'],'roi':item['applicable_region'],'calculation_id':item['calculation_id'],
              'comparison_pts':pts,'support_in_same_reviewed_stable_context':all(phase(p)==phase(item['pts']) for p in pts)})
    write(H/'wrong_scope_diagnostics.json',{'source':'reel','extra_wrong_ROI_support':previous,
        'manual_hits_lost_with_wrong_scope':[key for key,value in manual.items()
            if value['non_D6_source_evidence'] and not wrong[key]['non_D6_source_evidence']]})
