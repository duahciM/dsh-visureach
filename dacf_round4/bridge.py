"""Trusted host-side source refinement. The model cannot supply filesystem paths."""
import json
from pathlib import Path
import sys

import cv2
from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round3.tree import spatial, temporal
from dacf_round4.offline import HERE, verify


def execute(packet):
    freeze=read(HERE/'lock.json')
    verify(freeze)
    case=freeze['cases'][packet['case']]
    # The allowlisted experiment copy must have the exact frozen source bytes.
    assert digest(Path(packet['authorized'])) == digest(ROOT/case['source'])
    out=Path(packet['out'])
    node=packet['node']
    cv2.setNumThreads(1)
    if packet['mode']=='time':
        temporal(case,node,out)
        data=read(out/'temporal.json')
        data['lookup']='Use amr_query with a child ID or delivery_request with these actual PTS.'
        originals=data['source_originals']
        data['new_probe_pts']=originals[1]['pts']
    else:
        points=packet['pts']
        region=packet['native_xywh']
        spatial(case,node,points,region,out)
        data=read(out/'spatial.json')
        data['lookup']='Use amr_query with a child ID and detail=proof for validated maps or unknowns.'
        if packet.get('crop_pts') is not None:
            r=next((r for r in data['source_originals'] if r['pts']==packet['crop_pts']),None)
            if r is None:
                raise ValueError('crop_pts must equal one of the supplied real PTS.')
            x,y,w,h=region
            target=out/'crop.png'
            with Image.open(r['path']) as source:
                source.crop((x,y,x+w,y+h)).save(target)
            data['crop']={'path':str(target),'source_root':r['id'],'pts':r['pts'],'native_xywh':region,
                          'sha256':digest(target),'width':w,'height':h,'kind':'derived_crop_unscaled'}
    write(out/'bridge_result.json',data)
    return data


if __name__ == '__main__':
    print(json.dumps(execute(read(Path(sys.argv[1]))),ensure_ascii=True))
