"""Join SDK/provider images to immutable source and time accounting across all native attempts."""
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest,read,write
from dacf_round7.run import H,PROTOCOL
from dacf_round5.native_audit import structural_png
from evidenweft_phase1.source_review import timeline,native_stream


def lines(path):
    return [json.loads(l) for l in path.read_text('utf8').splitlines() if l.strip()] if path.is_file() else []


def audit_attempt(p):
    folder=p.parent
    r=read(p); events=read(folder/'events.json')
    wire=lines(folder/'provider.jsonl.effective'); controls=lines(folder/'calls.jsonl')
    calls={x['callId']:x for x in r['calls'] if x.get('callId')}
    decoded=defaultdict(list); derivatives={}
    for f in (folder/'cache/sessions').glob('*.jsonl'):
        for entry in lines(f):
            if entry['event'] in ('source_resample','delivery_request'):
                for frame in entry['frames']:
                    decoded[frame['attachment_sha256']].append(frame)
            if entry['event']=='derived_image':
                derivatives[entry['attachment_sha256']]=entry
    native=next((folder/'cache/observer-index-v5').glob('*/native'),None)
    by_id={rr['evidence_id']:rr for rrset in decoded.values() for rr in rrset}
    images=[]
    for im in r['images']:
        attached=im['attachment']['attachmentId'].removeprefix('sha256:')
        matched=by_id.get(im['evidence_id'])
        info={'sdk_followed_by_reply':im['followed_by_model_reply'],'transport_sha256':attached,'bytes':im['attachment']['bytes']}
        if matched is None and attached in decoded and len({a['pts'] for a in decoded[attached]})==1:
            matched=decoded[attached][0]
        if matched and matched['attachment_sha256']==attached:
            img=native/f"frame_{matched['source_frame']}.png"
            assert digest(img)==matched['png_sha256']
            with Image.open(img) as d:
                rgb=sha256(d.convert('RGB').tobytes()).hexdigest()
            info.update(kind='original',pts=matched['pts'],evidence_id=matched['evidence_id'],rgb_sha256=rgb)
        elif attached in derivatives:
            d=derivatives[attached]; parent=by_id[d['parent_evidence_id']]
            local=native/f"frame_{parent['source_frame']}.png"
            assert digest(local)==parent['png_sha256']==d['parent_png_sha256']
            x,y,w,h=d['native_xywh']
            with Image.open(local) as pic:
                crop=pic.convert('RGB').crop((x,y,x+w,y+h))
                if d.get('scale',1)!=1:
                    crop=crop.resize((d['width'],d['height']), Image.Resampling.LANCZOS)
                b=BytesIO();crop.save(b,format='PNG')
            assert sha256(structural_png(b.getvalue())).hexdigest()==attached
            info.update(kind='derived',parent=parent['evidence_id'],pts=parent['pts'],derived_id=d['derived_id'])
        else:
            info.update(kind='unjoined',candidate_pts=sorted({a['pts'] for a in decoded[attached]}))
        images.append(info)
    answers=[x for x in events if x['type']=='assistant/message']
    def receipts():
        for result in r['tool_results']:
            e=events[result['event_index']]
            start=next((x['time'] for x in events if x['type']=='tool/call' and x['data'].get('callId')==result['call']),None)
            yield {'name':calls.get(result['call'],{}).get('name'), 'success':not result['error'], 'attachments':result['images'],
                'text_bytes':result['metadata_bytes'],'sdk_call_to_result_ms':e['time']-start if start else None,
                'later_assistant':result['followed_by_model_reply']}
    attempt_tool_res=list(receipts())
    paired=[]
    for i, request in enumerate(wire):
        if i < len(answers):
            end=answers[i]['time']
            if end>=request['at_ms'] and (i+1==len(wire) or end<wire[i+1]['at_ms']):
                paired.append(end-request['at_ms'])
    unique_wire={t['tool_use_id']:t['content'] for v in wire for t in v['tool_results']}
    return {'path':p.relative_to(H).as_posix(),'stop':r['stop'],'answered':bool(r['answer_text']),'total_seconds':r['seconds'],
        'usage':r['usage'],'wire_requests':len(wire),'wire_unique_result_blocks':len(unique_wire),
        'wire_image_blocks':sum(c.get('type')=='image' for v in unique_wire.values() if isinstance(v,list) for c in v),
        'wire_encoded_image_chars':sum((c.get('encoded_length') or 0) for v in unique_wire.values() if isinstance(v,list) for c in v if c.get('type')=='image'),
        'wire_first_tool_hash':wire[0]['tool_hash'] if wire else None,
        'request_to_assistant_event_ms':paired,'last_paired_request_to_assistant_ms':paired[-1] if paired and len(paired)==len(wire) else None,
        'host_handlers':[{k:c.get(k) for k in ('tool','elapsed_ms','reason','status','output_image_count')} for c in controls if c.get('event')=='tool_compute'],
        'tool_receipts':attempt_tool_res,'native_cache_pngs':len(list(native.glob('*.png'))) if native else 0,'images':images}


def main():
    paths=sorted(p for p in H.glob('*/*/result.json') if (p.parent/'events.json').is_file())
    rows=[audit_attempt(p) for p in paths]
    originals=defaultdict(set)
    for row in rows:
        for im in row['images']:
            if im['kind']=='original':originals[im['pts']].add(im['rgb_sha256'])
    if originals:
        source=ROOT/PROTOCOL['source']; st,clock=timeline(source); proved=set()
        for sample,rgb in native_stream(source,st,clock):
            if sample['pts'] in originals:
                assert originals[sample['pts']]=={sha256(rgb).hexdigest()}
                proved.add(sample['pts'])
        assert proved==set(originals)
    totals=Counter()
    for row in rows:totals.update(row['usage'])
    write(H/'actual_cost_and_delivery.json',{'runs':rows,'total_usage':dict(totals),'independent_rgb_verified_pts':sorted(originals),
      'timing_limits':'Tool host elapsed includes cache/decode/compute and host handling where applicable. SDK call-to-result can overlap other work. Provider wait mixes send, queue, network, generation and event delivery. No isolated transmission or private reasoning timing; cached PNG count is not decoder-frame count. Invoiced price was not collected.'})

if __name__=='__main__':main()
