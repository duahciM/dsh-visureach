"""Stress the raw M5 appearance channel against local change and shared exposure."""
import subprocess
from hashlib import sha256
import numpy as np
from PIL import Image

from dacf_research.prepare_inputs import digest,read,write
from evidenweft_phase1.source_review import native_stream,timeline
from dacf_round3.acquire import Reader
from partial_modules_stage1.study import HERE,patch_color


if __name__=='__main__':
    out=HERE/'photometric_control';out.mkdir()
    ev=read(HERE/'first_run/paired/evidence.json'); start=ev['source_receipts'][0]; candidates=ev['views'][str(start['pts'])]
    source=out/'source.mkv'
    roi=next(r['native_xywh'] for r in candidates if r['role']=='candidate')
    ref=next(r['native_xywh'] for r in candidates if r['role']=='reference')
    with Image.open(start['path']) as p: original=np.array(p.convert('RGB'))
    w,h=original.shape[1],original.shape[0];x,y,bw,bh=roi
    patch=original.copy();patch[y+12:y+bh-12,x+12:x+bw-12]=np.clip(
        patch[y+12:y+bh-12,x+12:x+bw-12].astype(np.int16)+46,0,255).astype(np.uint8)
    exposure=np.clip(original.astype(np.int16)+24,0,255).astype(np.uint8)
    raw=[original,patch,original,exposure,original]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
         '-r','4','-i','-','-an','-c:v','ffv1','-pix_fmt','bgr0',str(source)],input=b''.join(a.tobytes() for a in raw),check=True,timeout=35)
    stream,clock=timeline(source)
    assert len(clock)==len(raw)
    sha=digest(source)
    write(out/'lock.json',{'video_sha256':sha,'original_reference_root':start['id'],'roi':roi,'context_control_roi':ref,
          'declared_order':['baseline','target photometric modification','return','whole image exposure modification','return'],
          'scope':'Small derived synthetic control. Raw luma is inspected separately in each native region.'})
    reader=Reader(source,sha,stream,clock,out/'native')
    output=[]
    for position,r in enumerate(clock):
        rec,_=reader.read(r['pts'])
        with Image.open(rec['path']) as p: pixels=np.array(p.convert('RGB'))
        assert sha256(raw[position].tobytes()).hexdigest()==rec['rgb_sha256']
        output.append({'pts':r['pts'],'source_root':rec['id'],'object_luma':patch_color(pixels,roi)['mean_luma_8bit'],
                       'reference_luma':patch_color(pixels,ref)['mean_luma_8bit'],'png_sha256':rec['png_sha256']})
    assert {r['pts']:sha256(p).hexdigest() for r,p in native_stream(source,stream,clock)}=={p:r['rgb_sha256'] for p,r in reader.receipts.items()}
    base=output[0]
    for r in output:
        r.update(delta_object_8bit=r['object_luma']-base['object_luma'],delta_context_8bit=r['reference_luma']-base['reference_luma'])
    write(out/'result.json',{'measurements':output,'actual_source_receipts':list(reader.receipts.values()),
      'scope':'Both operations can change raw target luma. The separate reference channel exposes different applicability; no automatic physical-state decision.'})
