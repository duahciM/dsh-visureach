"""Join true SDK attachments, tool envelopes, ledgers, source pixels, and answer citations."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image,ImageDraw

from dacf_research.prepare_inputs import read,write,digest
from dacf_round5.setup import HERE
from evidenweft_phase1.source_review import timeline,native_stream


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def structural_png(data):
    chunks=[data[:8]]
    offset=8
    while offset<len(data):
        end=offset+12+int.from_bytes(data[offset:offset+4],'big')
        assert end<=len(data)
        if data[offset+4:offset+8] in {b'IHDR',b'PLTE',b'tRNS',b'IDAT',b'IEND'}:
            chunks.append(data[offset:end])
        offset=end
    return b''.join(chunks)


def audit(root,label):
    roles=read(HERE/'roles.json')
    source_expected=defaultdict(lambda:defaultdict(set))
    source_frames=defaultdict(lambda:defaultdict(set))
    pictures={}
    attempts=[]
    for folder in sorted(root.iterdir()):
        if not (folder/'result.json').is_file(): continue
        result=read(folder/'result.json'); events=read(folder/'events.json')
        case=result['item']['source']
        indexed={}; derived={}
        sd=result['source_sha256']
        native=next((folder/'cache/observer-index-v5').glob('*/native'),None)
        for ledger in (folder/'cache/sessions').glob('*.jsonl'):
            for row in lines(ledger):
                if row.get('video_sha256')!=sd: continue
                if row['event'] in ('delivery_request','source_resample'):
                    indexed.update({ref['evidence_id']:ref for ref in row['frames']})
                if row['event']=='derived_image':
                    derived[row['derived_id']]=row
        tool_receipts={r['event_index']:r for r in result['tool_results']}
        imagegroups=defaultdict(list)
        for im in result['images']: imagegroups[im['event_index']].append(im)
        delivered=[]; crop_images=[]; unjoined=[]; first_by={}; seen={}; seen_crops={}
        first_event=events[0]['time'] if events else 0
        for event_i, blocks in imagegroups.items():
            receipt_text=tool_receipts[event_i]['text']
            ids=re.findall(r'(?m)^\d+\. .*?\bevidence_id=(ev-[a-f0-9]+)',receipt_text)
            for index, image in enumerate(blocks):
                payload=image['attachment']
                signature=payload['attachmentId'].removeprefix('sha256:')
                ev=ids[index] if index<len(ids) else image['evidence_id']
                row=indexed.get(ev)
                if row and row['attachment_sha256']==signature:
                    src_png=native/f'frame_{row["source_frame"]}.png'
                    assert digest(src_png)==row['png_sha256']
                    with Image.open(src_png) as original:
                        assert original.size==(payload['width'],payload['height'])
                        source_expected[case][row['pts']].add(sha256(original.convert('RGB').tobytes()).hexdigest())
                    source_frames[case][row['pts']].add((row['source_frame'],row['time_base']))
                    pictures[(case,row['pts'])]=src_png
                    item={'source_pts':row['pts'],'seconds':row['seconds'],'evidence_id':ev,'event_index':event_i,
                          'later_message':image['followed_by_model_reply'],'bytes':payload['bytes'],
                          'sdk_relative_seconds':(events[event_i]['time']-first_event)/1000,
                          'model_tool_calls_through_delivery':sum(e['type']=='tool/call' for e in events[:event_i])}
                    delivered.append(item)
                    if item['later_message']:
                        seen[ev]=item
                        for role in roles[case]['roles']:
                            if role['seconds'][0]-.000002<=item['seconds']<=role['seconds'][1]+.000002:
                                first_by.setdefault(role['id'],item)
                    continue
                crop=next((r for id_,r in derived.items() if id_ in receipt_text and r['attachment_sha256']==signature),None)
                if crop:
                    parent=indexed[crop['parent_evidence_id']]
                    path=native/f'frame_{parent["source_frame"]}.png'
                    assert digest(path)==parent['png_sha256']==crop['parent_png_sha256']
                    assert crop['width']==payload['width'] and crop['height']==payload['height']
                    x,y,w,h=crop['native_xywh']
                    with Image.open(path) as raw:
                        proof=raw.convert('RGB').crop((x,y,x+w,y+h))
                        if crop.get('scale',1)!=1:
                            proof=proof.resize((crop['width'],crop['height']),Image.Resampling.LANCZOS)
                        wire=BytesIO(); proof.save(wire,format='PNG')
                    assert sha256(structural_png(wire.getvalue())).hexdigest()==signature
                    crop_images.append(crop|{'later_message':image['followed_by_model_reply'],'bytes':payload['bytes'],'independent_parent_crop_verified':True})
                    if image['followed_by_model_reply']:
                        seen_crops[crop['derived_id']]={'pts':parent['pts'],'native_xywh':crop['native_xywh'],'parent_evidence_id':crop['parent_evidence_id']}
                else:
                    unjoined.append({'event':event_i,'signature':signature,'text_prefix':receipt_text[:150]})
        answer=result.get('answer')
        claims=[]
        if isinstance(answer,dict):
            for row in answer.get('facts',[]):
                if not isinstance(row,dict):continue
                refs=[seen[ev] for ev in row.get('evidence_ids',[]) if ev in seen]
                claims.append(row|{'verified_pts':[ref['source_pts'] for ref in refs],
                    'verified_derived_citations':[seen_crops[e] for e in row.get('evidence_ids',[]) if e in seen_crops],
                    'unknown_ids':[e for e in row.get('evidence_ids',[]) if e not in seen and e not in seen_crops]})
        control=lines(folder/'calls.jsonl') if (folder/'calls.jsonl').exists() else []
        attempts.append({'item':result['item'],'stop':result['stop'],'provider':result['provider'],'reasoning':result['reasoning'],
                         'structured_final':isinstance(answer,dict) and result['stop']['kind']=='completed','wall_seconds':result['wall_seconds'],'usage':result['usage'],
                         'original_deliveries':len(delivered),'different_received_originals':len({r['source_pts'] for r in seen.values()}),
                         'repeat_original_deliveries':len(delivered)-len({r['source_pts'] for r in delivered}),'derived_deliveries':crop_images,'unjoined':unjoined,
                         'original_nearest_locations':first_by,'exact_witnesses_received':[role['id'] for role in roles[case]['roles'] if role['exact_pts'] in {r['source_pts'] for r in seen.values()}],
                         'claims':claims,'summary':answer.get('summary') if isinstance(answer,dict) else None,
                         'public_intermediate_messages':[c.get('text','') for e in events if e['type']=='assistant/message' for c in e['data']['message']['content']
                                                         if c['type']=='text' and c.get('text') and c['text']!=result['answer_text']],
                         'model_tool_calls':[r['name'] for r in result['calls']],'tool_errors':[r['text'] for r in result['tool_results'] if r['error']],
                         'host_subcalls':Counter(r.get('tool') for r in control if r['event']=='tool_compute' and r['reason']=='amr_refine'),
                         'host_refinements':[r for r in control if r['event']=='amr_refinement'],
                         'host_tool_compute_ms':sum(r['elapsed_ms'] for r in control if r['event']=='tool_compute'),
                         'session_native_cache_files':len(list(native.glob('*.png'))) if native and native.is_dir() else 0,
                         'wire_image_bytes':sum(im['attachment']['bytes'] for im in result['images']),
                         'wire_metadata_bytes':sum(row['metadata_bytes'] for row in result['tool_results'])})
    verification={}
    for case,expected in source_expected.items():
        source=HERE/'inputs'/case/'source.mp4'; stream,clock=timeline(source); seen=set()
        for row,bits in native_stream(source,stream,clock):
            if row['pts'] in expected:
                assert expected[row['pts']]=={sha256(bits).hexdigest()}
                assert source_frames[case][row['pts']]=={(row['frame'],stream['time_base'])}
                seen.add(row['pts'])
        assert seen==set(expected)
        verification[case]={'unique_original_pts_rgb':len(seen),'decode_for_audit_only':len(clock)}
        p=sorted(set(p for name,p in pictures if name==case))
        subs=sorted({min(p,key=lambda q:abs(q-role['exact_pts'])) for role in roles[case]['roles']})
        canvas=Image.new('RGB',(900,((len(subs)+2)//3)*177),'#172331')
        draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(subs):
            x,y=i%3*300,i//3*177
            with Image.open(pictures[case,p]) as im:
                im.thumbnail((286,154));canvas.paste(im,(x+4,y+2))
            draw.text((x+4,y+159),f'Actually delivered original | PTS {p}',fill='white')
        canvas.save(HERE/f'{label}-{case}-sampled-originals.png')
    write(HERE/f'{label}-audit.json',{'attempts':attempts,'verification':verification})
    print('Audited',len(attempts),'native sessions and',verification,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--engineering',action='store_true');v=p.parse_args()
    audit(HERE/('native-engineering' if v.engineering else 'native'),v.label)
