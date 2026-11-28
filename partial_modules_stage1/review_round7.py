"""Independent condition-masked scoring and real SDK/provenance checks."""
from collections import Counter
from hashlib import sha256
import json

from dacf_research.prepare_inputs import read,write
from partial_modules_stage1.round7_retest import HERE,SOURCE,launch
from dacf_round7 import receipts
from evidenweft_phase1.source_review import native_stream,timeline

def main():
    folder=HERE/'round7_retest'; lock=read(folder/'lock.json')
    blinded={job:read(folder/job/'result.json')['answer'] for job,v in lock['assignments'].items() if v['task']=='blind'}
    prompt=('Score two anonymous answers to the same four tool questions. Use the fixed criteria below. '
            'For each item give 0 incorrect, 1 partial, 2 complete. Check arguments and suitable alternatives, not mere tool names. '
            'Point out genuine misuse, including use of a catalog-only graph for crop lineage or A/B as a synonym for comparing foreground and context, and do not infer actual execution. '
            'Return compact JSON {"summary":"brief","scores":{"id":[2,1,2,2]},"issues":{"id":["short specific issues"]}}.\n'
            +json.dumps({'rubric':lock['rubric'],'answers':blinded},ensure_ascii=False))
    judgment=folder/'masked-review'/'result.json'
    rated=read(judgment) if judgment.is_file() else launch(folder/'masked-review',prompt,plain=True,watchdog=220)
    # The existing Round 7 reader only uses H to make its returned path relative.
    receipts.H=HERE
    rgb={}
    results=[]; cost=Counter()
    for ident,what in lock['assignments'].items():
        data=read(folder/ident/'result.json'); cost.update(data['usage'])
        row={'id':ident,**what,'stop':data['stop'],'seconds':data['seconds'],'calls':[c['name'] for c in data['calls']],
             'schema_successful_operations':{}, 'provider_original_pts':[], 'sdk_error_text':[r['text'] for r in data['tool_results'] if r['error']]}
        if what['task']=='operation':
            p=receipts.audit_attempt(folder/ident/'result.json')
            calls={v['callId']:v['name'] for v in data['calls']}
            events=read(folder/ident/'events.json')
            for result in data['tool_results']:
                message=events[result['event_index']]['data']['message']
                call_id=message.get('toolCallId') or message.get('source',{}).get('callId')
                name=calls.get(call_id)
                if name and not result['error']:
                    row['schema_successful_operations'][name]=row['schema_successful_operations'].get(name,0)+1
            row['provider_image_blocks']=p['wire_image_blocks']
            row['provider_original_pts']=[i['pts'] for i in p['images'] if i['kind']=='original']
            for image in p['images']:
                if image['kind']=='original':
                    prior=rgb.setdefault(image['pts'],image['rgb_sha256'])
                    assert prior==image['rgb_sha256']
            row['verified_crops']=[i for i in p['images'] if i['kind']=='derived']
            row['unjoined_image_count']=sum(i['kind']=='unjoined' for i in p['images'])
        results.append(row)
    cost.update(rated['usage'])
    stream,clock=timeline(SOURCE)
    verified=set()
    for row,pixels in native_stream(SOURCE,stream,clock):
        pt=row['pts']
        if pt in rgb:
            assert sha256(pixels).hexdigest()==rgb[pt]
            verified.add(pt)
    assert verified==set(rgb)
    write(folder/'summary-verified.json',{'runs':results,'blind_rating':rated['answer'],'blind_rating_stop':rated['stop'],
         'independent_rgb_verified_pts':sorted(verified),'all_attempt_usage':dict(cost)})

if __name__=='__main__':main()
