"""Reconcile real worker PNGs and crops against exact original source RGB."""
import json
from hashlib import sha256
from PIL import Image

from dacf_research.prepare_inputs import digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    engine=H/'engineering'; run=read(engine/'result.json')
    originals={}; derived={}
    for file in (engine/'cache/sessions').glob('*.jsonl'):
        for line in file.read_text('utf8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['event'] in ('delivery_request','source_resample'):
                for f in row['frames']:originals[f['attachment_sha256']]=f
            elif row['event']=='derived_image':derived[row['attachment_sha256']]=row
    target=H/'inputs/control/source.mkv'; info,clock=timeline(target)
    wanted={f['pts'] for f in originals.values()}
    px={}
    for r,values in native_stream(target,info,clock):
        if r['pts'] in wanted:px[r['pts']]=values
        if set(px)==wanted:break
    proof=[]
    for actual,path in run['transport_files'].items():
        assert digest(engine/path)==actual
        with Image.open(engine/path) as decoded:
            d=decoded.convert('RGB')
            if actual in originals:
                a=originals[actual]
                assert d.tobytes()==px[a['pts']]
                proof.append({'kind':'original','pts':a['pts'],'hash':actual,'rgb_sha256':sha256(d.tobytes()).hexdigest()})
            else:
                crop=derived[actual]
                parent=next(a for a in originals.values() if a['evidence_id']==crop['parent_evidence_id'])
                x,y,w,h=crop['native_xywh']
                origin=Image.frombytes('RGB',(info['width'],info['height']),px[parent['pts']])
                cropped=origin.crop((x,y,x+w,y+h))
                if crop.get('scale',1)!=1:cropped=cropped.resize((crop['width'],crop['height']),Image.Resampling.LANCZOS)
                assert cropped.tobytes()==d.tobytes()
                proof.append({'kind':'derived','parent':crop['parent_evidence_id'],'parent_pts':parent['pts'],'hash':actual,'native_xywh':crop['native_xywh']})
    assert len(proof)==4
    write(engine/'transport_verified.json',{'source_sha256':run['source_sha256'],'verified':proof})
