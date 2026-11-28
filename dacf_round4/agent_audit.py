"""Independent originals, derived crops, claim links and native costs."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import digest, read, write
from dacf_round4.offline import HERE
from evidenweft_phase1.source_review import native_stream, timeline


def jsonlines(path):
    return [json.loads(r) for r in path.read_text(encoding='utf-8').splitlines() if r.strip()]


def audit(root, dest, prefix):
    source=root/'public/source.mp4'
    source_sha=digest(source)
    groups=[(k,root/k) for k in ('A0','A1','A2')]
    extra=root/'global_stop_only/A2'
    if (extra/'result.json').exists(): groups.append(('A2_S2',extra))
    repaired=root/'transport_retry/A2'
    if (repaired/'result.json').exists(): groups.append(('A2_transport_repair',repaired))
    expected=defaultdict(set)
    summaries=[]
    image_map={}
    for key, arm in groups:
        result=read(arm/'result.json')
        native=next((arm/'cache/observer-index-v5').glob('*/native'),None)
        idx={}
        for ledger in (arm/'cache/sessions').glob('*.jsonl'):
            for r in jsonlines(ledger):
                if r['event'] not in {'delivery_request','source_resample'}: continue
                assert r['video_sha256']==source_sha
                for frame in r['frames']:
                    idx[(frame['evidence_id'],'sha256:'+frame['attachment_sha256'])]=frame
        queries=[]; refinements=[]; crops={}
        if (arm/'calls.jsonl').exists():
            for event in jsonlines(arm/'calls.jsonl'):
                if event.get('event')=='research_query': queries.append(event)
                if event.get('event')!='research_refinement': continue
                if event['status']=='completed':
                    raw=read(Path(event['receipt']))
                    for original in raw['source_originals']:
                        assert original['source_sha256']==source_sha and digest(Path(original['path']))==original['png_sha256']
                        expected[original['pts']].add(original['rgb_sha256'])
                    if 'crop' in raw:
                        crop=raw['crop']; parent=next(r for r in raw['source_originals'] if r['id']==crop['source_root'])
                        x,y,w,h=crop['native_xywh']
                        with Image.open(parent['path']) as p, Image.open(crop['path']) as c:
                            assert p.crop((x,y,x+w,y+h)).tobytes()==c.tobytes()
                        assert digest(Path(crop['path']))==crop['sha256']
                        crops[event['crop']['attachment_id']]=crop
                    event['saved_correspondence_states']=[c['comparison']['excluded_context']['state'] for child in raw['children'] for c in child.get('comparisons',[])]
                refinements.append(event)
        observed={}; first=0; derived=0; unmatched=[]
        for image in result['images']:
            hash_=image['attachment']['attachmentId']
            ev=image['evidence_id']
            row=idx.get((ev,hash_))
            if row is not None:
                path=native/f'frame_{row["source_frame"]}.png'
                assert digest(path)==row['png_sha256']
                with Image.open(path) as f:
                    assert f.size==(image['attachment']['width'],image['attachment']['height'])
                    expected[row['pts']].add(sha256(f.convert('RGB').tobytes()).hexdigest())
                image_map.setdefault(row['pts'],path)
                first+=1
                if image['followed_by_model_reply']: observed[ev]=row['pts']
            elif hash_ in crops:
                derived+=1
            else:
                unmatched.append(hash_)
        claims=[]
        answer=result.get('answer')
        if isinstance(answer,dict):
            for r in answer.get('facts',[]):
                ids=r.get('evidence_ids',[])
                claims.append(r|{'citation_pts':[observed[e] for e in ids if e in observed], 'unresolved_citations':[e for e in ids if e not in observed]})
        totals={'arm':key,'stop':result['stop'],'structured_final':isinstance(answer,dict) and result['stop']['kind']=='completed',
                'provider':result['provider'],'reasoning':result['reasoning'],'usage':result['usage'],'wall_seconds':result['wall_seconds'],
                'attached_originals':first,'attached_derived':derived,'unmatched':unmatched,
                'distinct_originals_with_later_reply':len(set(observed.values())),'actual_original_pts':sorted(set(observed.values())),
                'claims':claims,'summary':answer.get('summary') if isinstance(answer,dict) else None,
                'research_queries':queries,'research_refinements':refinements,
                'actual_tool_calls':[c['name'] for c in result['calls']],
                'sdk_tool_errors':[r['text'] for r in result['tool_results'] if r['error']],
                'attachment_bytes':sum(a['attachment'].get('bytes',0) for a in result['images']),
                'tool_metadata_bytes':sum(a['metadata_bytes'] for a in result['tool_results'])}
        summaries.append(totals)
    stream, clock=timeline(source)
    seen=set()
    for row,pixels in native_stream(source,stream,clock):
        p=row['pts']
        if p in expected:
            assert expected[p]=={sha256(pixels).hexdigest()}
            seen.add(p)
    assert seen==set(expected)
    write(dest,{'source_sha256':source_sha,'verified_rgb_source_pts_in_union':len(seen),'attempts':summaries})
    images=[p for p in (2520,2580,2640,2660,2700,2720,2740,2800,2840,2860,2900,3000,3100,3160,3200,3220,3300) if p in image_map]
    if images:
        panel=Image.new('RGB',(1000,((len(images)+3)//4)*166),'#18222d')
        painter=ImageDraw.Draw(panel)
        for i,p in enumerate(images):
            x,y=i%4*250,i//4*166
            with Image.open(image_map[p]) as f:
                f.thumbnail((240,139)); panel.paste(f,(x+4,y))
            painter.text((x+4,y+145),f'{p} | {p/600:.4f}s',fill='white')
        panel.save(root/f'{prefix}-consulted-originals.png')
    print('Verified',len(seen),'different source PTS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prefix',default='all')
    a=p.parse_args()
    root=HERE/'agent'; audit(root,root/f'{a.prefix}-audit.json',a.prefix)
