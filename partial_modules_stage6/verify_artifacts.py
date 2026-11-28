"""Verify the source-only insert and new AMR probe against independent decoding."""

from hashlib import sha256

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.source_review import native_stream, timeline
from partial_modules_stage6.material import H


if __name__=='__main__':
    spec=read(H/'inputs/cases.json')['reel'];source=ROOT/spec['source']
    assert digest(source)==spec['sha256']
    checked=read(H/'one-boundary-probe/result.json')
    auto=read(H/'automatic/reel/auto.json'); assessment=read(H/'assessment/sealed.json')['reel']
    assert digest(H/'automatic/reel/auto.json')==checked['envelope_sha_unchanged']
    insert=assessment['subcoarse_insert']['visible_only_at_pts']
    probe=checked['coarse_pool_did_not_have']['pts']
    assert all(p not in auto['pool_pts'] for p in (probe,insert))
    independent=[]
    info,clock=timeline(source)
    for frame,bits in native_stream(source,info,clock):
        if frame['pts'] in (probe,insert):
            independent.append({'frame':frame,'sha256_rgb':sha256(bits).hexdigest()})
    native=H/'assessment/reel/native'/f'pts-{insert}.png'
    with Image.open(native) as p:
        assert next(r['sha256_rgb'] for r in independent if r['frame']['pts']==insert)==sha256(p.convert('RGB').tobytes()).hexdigest()
    known=[]
    for key in checked['actual_image_payload_digests']:
        path=H/'one-boundary-probe'/f'{key}.png'
        assert digest(path)==key
        with Image.open(path) as src:
            known.append(sha256(src.convert('RGB').tobytes()).hexdigest())
    assert next(r['sha256_rgb'] for r in independent if r['frame']['pts']==probe) in known
    assert len(checked['summary']['originals'])==3
    write(H/'pixel_audit.json',{'source_sha256':spec['sha256'],'independently_decoded_out_of_pool':independent,
        'sampled_attachments':len(checked['summary']['originals']),'unique_pngs':len(known),
        'same_round_envelope_sha_unchanged':checked['envelope_sha_unchanged'],'human_review_insert_png':str(native)})
