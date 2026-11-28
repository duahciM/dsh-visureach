"""Contact view of exact source ROI evidence and contrasting failures."""

from PIL import Image, ImageDraw, ImageFont

from dacf_research.prepare_inputs import read
from partial_modules_stage4.prepare import H


def panel(canvas, index, case, stamp, heading, boxes, points=()):
    base=read(H/'predictions'/case/'base_receipts.json')['native_receipts']
    source=next(x for x in base if x['pts']==stamp)
    x,y=(index%2)*560+12,(index//2)*340+25
    with Image.open(source['path']) as original:
        sx,sy=540/original.width,304/original.height
        visual=original.convert('RGB').resize((540,304))
    paint=ImageDraw.Draw(visual)
    for box,color in boxes:
        u,v,w,h=box
        paint.rectangle((u*sx,v*sy,(u+w)*sx,(v+h)*sy),outline=color,width=3)
    for px,py in points:
        u,v=px*sx,py*sy
        paint.ellipse((u-3,v-3,u+3,v+3),fill='#7efffe')
    canvas.paste(visual,(x,y))
    ImageDraw.Draw(canvas).text((x,y-19),f'{heading}  |  PTS {stamp}',font=ImageFont.load_default(size=15),fill='white')


if __name__=='__main__':
    sheet=Image.new('RGB',(1120,1020),'#1a2028')
    audit=read(H/'phase3_role_audit.json')['street-repeat']['cases']
    street=next(x for x in audit if x['diagnosis']=='exact_old_proposal_removed_no_other_valid_geometry')
    panel(sheet,0,'street-repeat',street['pts'],'Street before role filtering',[(street['original_candidates'][0]['xywh'],'#f5e580')])
    panel(sheet,1,'street-repeat',street['pts'],'Street: best remaining A3 geometry',[(street['best_any_A3']['native_xywh'],'#f48392')])
    agg=read(H/'evaluation.json')['cohorts']
    control=agg['roles']['arms']['R1P1']['reference_scores']
    data=read(H/'predictions/roles/region_features.json')['extra_geometry']
    best=next(r for r in control['feature_support_detail'] if r['key']=='rail' and 1200 in r['pts'])
    supported=next(r for r in data['feature_patches'] if r['calculation_root']==best['calculation_root'])
    idx=supported['pts'].index(1200)
    panel(sheet,2,'roles',1200,'Controlled stationary reference: local features',
          [([330,173,72,67],'#f5e580')],supported[('native_points_left','native_points_right')[idx]])
    host=read(H/'host/result.json')
    panel(sheet,3,'new-local',host['pts'][1],'New real window: small print serving as local reference',[(host['roi'][1],'#f5e580')])
    review=read(H/'inputs/sealed_assessment.json')['new-pan']['local_anchors'][:2]
    for index,row in enumerate(review,start=4):
        panel(sheet,index,'new-pan',row['pts'],'Pan: visible review ROI, geometry abstained',[(row['roi'],'#f48392')])
    sheet.save(H/'exact_examples.png')
