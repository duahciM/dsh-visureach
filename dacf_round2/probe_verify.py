"""Audit only actual DSH attachments and source citations. Does not judge semantics."""
from hashlib import sha256
import json
from pathlib import Path
import sys

from PIL import Image

from dacf_research.prepare_inputs import digest, read, write
from evidenweft_phase1.source_review import timeline, native_stream

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'agent_probe'
    source=root/'public/video.mp4'
    stream,rows=timeline(source)
    by_pts={r['pts']:r for r in rows}
    original_by_pts={}
    answers=[]
    attempts=[('basic',root/'basic'),('region_metadata',root/'region_metadata')]
    retry=root/'retry_longer_watchdog/region_metadata'
    if (retry/'result.json').is_file():
        attempts.append(('region_metadata_retry',retry))
    for variant,folder in attempts:
        result=read(folder/'result.json')
        calls={c['name'] for c in result['calls']}
        indexed={}
        native_dir=next((folder/'cache/observer-index-v5').glob('*/native'),None)
        for ledger in (folder/'cache/sessions').glob('*.jsonl'):
            for line in ledger.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                r=json.loads(line)
                if r['event'] not in {'source_resample','delivery_request'}: continue
                assert r['video_sha256']==digest(source)
                for item in r['frames']:
                    indexed[(item['evidence_id'],item['attachment_sha256'])]=item
        verified={}
        count=0
        lacked=[]
        for attachment in result['images']:
            pic=attachment['attachment']
            target=indexed.get((attachment['evidence_id'],pic['attachmentId'].removeprefix('sha256:')))
            if not target:
                lacked.append(attachment['evidence_id'])
                continue
            p=native_dir/f'frame_{target["source_frame"]}.png'
            bypts=by_pts[target['pts']]
            assert digest(p)==target['png_sha256'] and bypts['frame']==target['source_frame']
            assert target['time_base']==stream['time_base']
            with Image.open(p) as opened:
                assert opened.size==(stream['width'],stream['height'])
                rg=sha256(opened.convert('RGB').tobytes()).hexdigest()
            original_by_pts.setdefault(target['pts'],set()).add(rg)
            count+=1
            if attachment['followed_by_model_reply']:
                verified[target['evidence_id']]={'pts':target['pts'],'seconds':bypts['seconds'],'path':str(p),'image_hash':rg}
        claims=[]
        if isinstance(result.get('answer'),dict):
            for row in result['answer'].get('facts',[]):
                if not isinstance(row,dict): continue
                kept=[verified[i] for i in row.get('evidence_ids',[]) if i in verified]
                claims.append({'claim':row.get('claim'),'kind':row.get('kind'),'validated_citations':kept,
                    'contains_cited_D06_pts':any(6517<=r['pts']<=6786 for r in kept)})
        hints=[]
        if (folder/'calls.jsonl').is_file():
            hints=[r['regionIds'] for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()
                   if (r:=json.loads(line)).get('event')=='investigation_query']
        checks={k:result[k] for k in ('stop','provider','reasoning','usage','wall_seconds')}
        answers.append({'condition':variant,**checks,'observed_tool_names':sorted(calls),'region_query_ids':hints,
           'matched_attachments':count,'unmatched_attachments':lacked,'distinct_source_images_with_following_model_message':len(verified),
           'matched_D06_pts':sorted({r['pts'] for r in verified.values() if 6517<=r['pts']<=6786}),'trace_of_claim_citations':claims,
           'final_answer_available':isinstance(result.get('answer'),dict) and result['stop']['kind']=='completed'})
    independent=set()
    if original_by_pts:
        for r,bits in native_stream(source,stream,rows):
            if r['pts'] not in original_by_pts: continue
            assert original_by_pts[r['pts']]=={sha256(bits).hexdigest()}
            independent.add(r['pts'])
    assert independent==set(original_by_pts)
    output=root/(sys.argv[1] if len(sys.argv)>1 else 'source_and_cost_audit.json')
    write(output,{'source_sha256':digest(source),'different_native_source_pts_verified':len(independent),
         'conditions':answers})
    print('Audited',len(independent),'actual native PTS',flush=True)

if __name__=='__main__':
    main()
