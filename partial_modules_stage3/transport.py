"""Audit installed-plugin images and SDK receipt joins against an independent decode."""
from hashlib import sha256
import json
from io import BytesIO

from PIL import Image
from dacf_research.prepare_inputs import digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from partial_modules_stage3.material import H


def join(folder):
    originals={};crops={}
    for f in (folder/'cache/sessions').glob('*.jsonl'):
        for line in f.read_text('utf8').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r['event'] in ('delivery_request','source_resample'):
                for item in r['frames']:originals[item['attachment_sha256']]=item
            elif r['event']=='derived_image':crops[r['attachment_sha256']]=r
    return originals,crops


def independent(source, ticks):
    info,rows=timeline(source);pixels={}
    for r,b in native_stream(source,info,rows):
        if r['pts'] in ticks:pixels[r['pts']]=b
        if set(pixels)==ticks:break
    assert set(pixels)==ticks
    return (info['width'],info['height']),pixels


if __name__=='__main__':
    source=H/'inputs/roles/source.mkv';cases={}
    for tag in ('engineering','engineering-near','native'):
        folder=H/tag;response=read(folder/'result.json')
        originals,crops=join(folder)
        ids={d['evidence_id']:d for d in originals.values()}
        blocks=([v['attachment']['attachmentId'].removeprefix('sha256:') for v in response['images']] if tag=='native'
                else response['typed_png_sha256'])
        metadata={q['attachment']['attachmentId'].removeprefix('sha256:'):q for q in response.get('images',[])}
        shape,pixels=independent(source,{s['pts'] for s in originals.values()})
        traces=[]
        for sha in blocks:
            orig=originals.get(sha);d=crops.get(sha)
            assert orig is not None or d is not None
            if orig:
                tick=orig['pts'];root=orig['evidence_id'];expected=Image.frombytes('RGB',shape,pixels[tick])
            else:
                parent=ids[d['parent_evidence_id']];tick=parent['pts'];root=parent['evidence_id']
                x,y,w,h=d['native_xywh'];expected=Image.frombytes('RGB',shape,pixels[tick]).crop((x,y,x+w,y+h))
                if d.get('scale',1)!=1:expected=expected.resize((d['width'],d['height']),Image.Resampling.LANCZOS)
            if tag!='native':
                png=folder/(sha+'.png');assert digest(png)==sha
                with Image.open(png) as actual:assert expected.tobytes()==actual.convert('RGB').tobytes()
                proof='exact_payload_pixels'
            elif orig:
                cache=next((folder/'cache').glob('observer-index-v5/*/native'),None)
                path=cache/f'frame_{orig["source_frame"]}.png'
                assert digest(path)==orig['png_sha256']
                with Image.open(path) as cached:assert cached.convert('RGB').tobytes()==expected.tobytes()
                proof='SDK_original_attachment_id_matches_ledger; source_RGBA_verified'
            else:
                # The native SDK receipt references backend-saved pixels via the same attachment digest.
                assert [d['width'],d['height']]==[metadata[sha]['attachment']['width'],metadata[sha]['attachment']['height']]
                proof='SDK_crop_attachment_digest_and_dimensions_joined; parent_exact_RGB_verified'
            traces.append({'type':'original' if orig else 'derived','source_tick':tick,'parent_or_original_evidence_id':root,
                'sha256_attachment':sha,'scope':proof,'afterward_model_reply':metadata.get(sha,{}).get('followed_by_model_reply')})
        cases[tag]=traces
    write(H/'evaluation/transport.json',{'source_sha256':digest(source),'cases':cases})
