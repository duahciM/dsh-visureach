"""Audit serialized tools, assembled prompt, receipts and costs. No automatic semantic scoring."""
import argparse,json,re
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
from math import ceil
from pathlib import Path

from PIL import Image
from dacf_research.prepare_inputs import ROOT,read,write,digest
from evidenweft_phase1.source_review import timeline,native_stream
from dacf_round5.native_audit import structural_png

H=Path(__file__).resolve().parent
EXTRAS={'amr_query','amr_refine','investigation_regions'}

def js_lines(p):return [json.loads(t) for t in p.read_text('utf8').splitlines() if t.strip()] if p.is_file() else []

def audit(batch, output_name='summary.json'):
    output=[]
    proof=defaultdict(lambda:defaultdict(set))
    guide=read(H/'guide.json')
    design=read(batch/'design.json')
    for case,c,arm in design['jobs']:
        d=batch/('-'.join((case,c,arm))); r=read(d/'result.json')
        events=read(d/'events.json')
        system=[e['data']['message'] for e in events if e['type']=='system/message']
        sys=''.join(p['text'] for m in system for p in m['content'] if p['type']=='text')
        req=js_lines(d/'request-audit.jsonl')
        idx={}; crops={}
        for f in (d/'cache/sessions').glob('*.jsonl'):
            for row in js_lines(f):
                if row.get('video_sha256')!=r['source_sha256']:continue
                if row['event'] in {'delivery_request','source_resample'}: idx.update({f['evidence_id']:f for f in row['frames']})
                if row['event']=='derived_image':crops[row['attachment_sha256']]=row
        native=next((d/'cache/observer-index-v5').glob('*/native'),None)
        info={};orig=[];crop=[];unknown=[]; deliveries=[]
        payloads={a['event_index']:a.get('text','') for a in r['tool_results']}
        slots=Counter()
        for a in r['images']:
            png=a['attachment']['attachmentId'].removeprefix('sha256:')
            candidates=[v for v in idx.values() if v['attachment_sha256']==png]
            event=a['event_index']; ids=re.findall(r'(?m)^\d+\. .*?\bevidence_id=(ev-[a-f0-9]+)',payloads.get(event,''))
            pos=slots[event];slots[event]+=1
            rr=idx.get(a['evidence_id'] or (ids[pos] if pos<len(ids) else None)) or (candidates[0] if candidates and len({v['pts'] for v in candidates})==1 else None)
            if rr and rr['attachment_sha256']==png:
                p=native/f"frame_{rr['source_frame']}.png"; assert digest(p)==rr['png_sha256']
                with Image.open(p) as decoded: proof[case][rr['pts']].add(sha256(decoded.convert('RGB').tobytes()).hexdigest())
                orig.append(rr['pts'])
                deliveries.append({'pts':rr['pts'],'later_message':a['followed_by_model_reply'],
                    'seconds_since_first_event':(events[event]['time']-events[0]['time'])/1000,
                    'model_tool_calls_before_delivery':sum(e['type']=='tool/call' for e in events[:event])})
                if a['followed_by_model_reply']:info[rr['evidence_id']]=rr['pts']
            elif png in crops:
                rr=crops[png]; parent=idx[rr['parent_evidence_id']]
                p=native/f"frame_{parent['source_frame']}.png"
                assert digest(p)==parent['png_sha256']==rr['parent_png_sha256']
                x,y,w,h=rr['native_xywh']
                with Image.open(p) as im:
                    patch=im.convert('RGB').crop((x,y,x+w,y+h))
                    if rr.get('scale',1)!=1:patch=patch.resize((rr['width'],rr['height']),Image.Resampling.LANCZOS)
                    wire=BytesIO();patch.save(wire,format='PNG')
                assert sha256(structural_png(wire.getvalue())).hexdigest()==png
                crop.append({'id':rr['derived_id'],'parent':rr['parent_evidence_id'],'pts':parent['pts'],
                             'with_later_model_message':a['followed_by_model_reply'],'independent_crop_pixels_verified':True})
            else: unknown.append({'hash':png,'event':a['event_index'],'candidate_pts':[v['pts'] for v in candidates]})
        calls=Counter(c['name'] for c in r['calls'])
        claims=[]
        if isinstance(r['answer'],dict) and isinstance(r['answer'].get('facts'),list):
            for fact in r['answer']['facts']:
                if isinstance(fact,dict):
                    claims.append(fact|{'audited_original_pts':[info[id_] for id_ in fact.get('evidence_ids',[]) if id_ in info]})
        control=js_lines(d/'calls.jsonl')
        output.append({'case':case,'condition':c,'arm':arm,'stop':r['stop'],'answer_parseable':isinstance(r['answer'],dict),'seconds':r['seconds'],
            'guide_found_in_assembled_system':{k:txt in sys for k,txt in guide.items()},'system_characters':len(sys),
            'estimated_system_tokens_dsh_four_chars':(ceil(len(sys)/4)+4 if sys else None),
            'first_input_usage':next((e['data'].get('usage') for e in events if e['type']=='assistant/message'),None),
            'tools_hash':req[0]['toolHash'] if req else None,'tool_schema_count':len(req[0]['tools']) if req else None,
            'all_tool_counts':dict(calls),'extra_calls':{k:calls[k] for k in EXTRAS if calls[k]},
            'tool_errors':[row['text'] for row in r['tool_results'] if row['error']],
            'actual_original_attachment_count':len(orig),'repeated_originals':len(orig)-len(set(orig)),
            'different_native_pts':sorted(set(orig)),'originals_with_later_message':len(set(info.values())),
            'delivery_progress':deliveries,'host_tool_ms':sum(a['elapsed_ms'] for a in control if a.get('event')=='tool_compute'),
            'optional_hint_count':sum(a.get('event')=='related_hint' for a in control),
            'frames_in_native_cache':len(list(native.glob('*.png'))) if native else 0,
            'derived_attachment_count':len(crop),'unjoined':unknown,'claim_evidence':claims,
            'verified_crop_receipts':crop,
            'usage':r['usage'],'bytes_images':sum(im['attachment']['bytes'] for im in r['images']),
            'host_amr_refinements':[a for a in control if a.get('event')=='amr_refinement'],
            'public_observation_writes':[a.get('args') for a in control if a.get('tool')=='observation_record' and (a.get('args') or {}).get('action') in {'create','revise'} and a.get('status')=='computed']})
    rgb={}
    for case,hashes in proof.items():
        source=H/'independent/source.mp4' if case=='independent' else ROOT/'dacf_round5/inputs/fixture/source.mp4'
        info,clock=timeline(source);verified=set()
        for row,bits in native_stream(source,info,clock):
            if row['pts'] in hashes:
                assert hashes[row['pts']]=={sha256(bits).hexdigest()}
                verified.add(row['pts'])
        assert verified==hashes.keys()
        rgb[case]=len(verified)
    write(batch/output_name,{'attempts':output,'independent_native_rgb_pts':rgb})
    print(batch,len(output),rgb)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('batch');p.add_argument('--output', default='summary.json');a=p.parse_args();audit(H/a.batch,a.output)
