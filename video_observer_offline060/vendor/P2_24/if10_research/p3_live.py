"""Live, bounded observation sessions using authenticated Codex CLI. No scripted answers."""
import argparse,json,os,random,re,shutil,subprocess,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image,ImageDraw
from p1.io import dump
from v05.engine import resample
from identity_catalog_matching.common import read,sha
from .freeze import ROOT

MODEL='gpt-6.2-sol'
CASE='local_object'
TASKS=['where the prominent small high-contrast vertical form is initially','where that form is late relative to the earlier image',
       'whether there is evidence of an intermediate sideways position','whether the surrounding pattern stays largely in place']
START="""You are an independent video observer for a bounded experiment. You see 8 actual frames of one short silent visual sequence as a timestamped montage, in time order. Each tile is one frame. Use only the image attachment and its stamped times. Do NOT call tools, browse, inspect files, or infer unseen actions. The evaluation asks: {tasks}. First describe only supported visual facts; cite times, preserve uncertainties, and then independently request up to TWO intervals of 0.2 to 1.1 seconds each to actually reread the original at 10 fps. The source span is [{start},{end}]. A facilitator will return permitted frames. Output only JSON {{\"facts\":[{{\"claim\":\"...\",\"times\":[0.0]}}],\"uncertain\":[\"...\"],\"retrieval\":[{{\"start\":0.0,\"end\":1.0,\"why\":\"...\"}}]}}. Use [] if no query. A planned query is not evidence."""
FOLLOW="""The facilitator now supplies any actual 10 fps original-source reads you requested. Use their attached, timestamped frame tiles and your first montage: answer each earlier question, then give a short final description. If no image was returned or a question remains uncertain, say that. No more queries. Do not call tools, open files, or use unrelated information. Log of what the facilitator actually ran: {log}. Output only JSON {{\"facts\":[{{\"claim\":\"...\",\"times\":[0.0]}}],\"uncertain\":[\"...\"],\"final_narration\":\"...\"}}."""
JUDGE="""You are a method-blind reviewer of four independent observation packets from the same video. Method names are withheld and opaque packet IDs do not identify methods. Attachments contain a sealed anchor strip of the known source and each packet's initial images and actual returned source images. Score only claims that are true relative to the anchor and grounded in frames actually visible to this observer at that stage. Sparse times need not be exact if visually justified. Four fact IDs: initial = small upright high-contrast candidate starts in left part; final = it is much farther right at late time; mid = visually observed intermediate horizontal position, not merely interpolation; background = the broader patterned background is largely fixed (relative displacement, not just camera motion). Don't reward a guess without cited or identifiable direct frames. Fixed unresolved questions count = 4 minus correctly grounded facts; unsupported definite claims count separately. After-first and after-query from facts only; final from actual final prose only (not silently credit separate facts). Missing/failed output is not zero: mark unscorable. Read numeric timestamps and avoid overrating detailed but false claims. No tools or reading files. Return only JSON {{\"scores\":[{{\"id\":\"...\",\"scorable\":true,\"first\":[\"initial\"],\"after\":[\"initial\"],\"final\":[\"initial\"],\"unsupported_first\":0,\"unsupported_after\":0,\"rationale\":\"...\"}}]}}. The first attachment is the score anchor. Later attachments are in the same packet order as the text below; each packet has its first montage followed by the returned query montages, whose counts are in reads. Review packet text and actual facilitator results: {packet}"""


def contact(fs,path,title,scale=350):
    from math import ceil
    w=scale;h=round(w*9/16);cols=4
    canvas=Image.new('RGB',(w*cols,(h+21)*ceil(len(fs)/cols)+24),'#152530');ink=ImageDraw.Draw(canvas)
    ink.text((6,4),title,fill='white')
    for n,r in enumerate(fs):
        x=n%cols*w;y=24+n//cols*(h+21)
        with Image.open(r.get('native_image',r['image'])) as p:
            p=p.convert('RGB');p.thumbnail((w,h))
            canvas.paste(p,(x,y))
        ink.text((x+5,y+h+1),f'{r["t"]:.4f} s | PTS {r["pts"]}',fill='white')
    canvas.save(path)


def invoke(prompt,attachments,destination,session=None):
    basis=['codex','exec']
    if session:
        basis+=['resume','--skip-git-repo-check','--ignore-user-config','--json','-m',MODEL,
                '-c','sandbox_mode="read-only"','-c','approval_policy="never"']
    else:
        basis+=['--skip-git-repo-check','--ignore-user-config','--json','-s','read-only','-m',MODEL,
                '-c','approval_policy="never"']
    for p in attachments:basis+=['-i',str(p.resolve())]
    if session:basis+=[session]
    basis+=['-']
    t=time.perf_counter()
    try:result=subprocess.run(basis,input=prompt,encoding='utf-8',errors='replace',capture_output=True,cwd=destination.parent,timeout=140)
    except Exception as exc:
        dump(destination.with_suffix('.error.json'),{'error':repr(exc),'seconds':time.perf_counter()-t});return {}
    destination.write_text(result.stdout,encoding='utf8');destination.with_suffix('.stderr.log').write_text(result.stderr,encoding='utf8')
    rows=[]
    for line in result.stdout.splitlines():
        try:rows.append(json.loads(line))
        except ValueError:continue
    sessions=[r.get('thread_id') for r in rows if r.get('type')=='thread.started']
    outputs=[r['item']['text'] for r in rows if r.get('type')=='item.completed' and r.get('item',{}).get('type')=='agent_message']
    tools=[r['item'].get('type') for r in rows if r.get('type')=='item.completed' and r.get('item',{}).get('type') not in
           ('agent_message','reasoning','todo_list')]
    usage=[r.get('usage') for r in rows if r.get('type')=='turn.completed']
    reply=None
    if outputs:
        s=outputs[-1].strip()
        try:
            reply=json.loads(s)
        except ValueError:
            j=s.find('{');k=s.rfind('}')
            if j>=0 and k>j:
                try:reply=json.loads(s[j:k+1])
                except ValueError:pass
    return {'exit_code':result.returncode,'thread_id':sessions[-1] if sessions else session,
      'reply':reply if isinstance(reply,dict) else None,'tool_events':tools,'usage':usage,
      'seconds':time.perf_counter()-t,'roles_messages':len(outputs),'attached':[str(z) for z in attachments]}


def query_requests(reply,destination,src,meta):
    requested=reply.get('retrieval',[]) if reply else []
    returned=[]; images=[]
    if not isinstance(requested,list):return [{'state':'invalid_type'}],images
    for j,r in enumerate(requested):
        if j>=2:
            returned.append({'requested':r,'state':'budget_rejected'});continue
        try:
            a=float(r['start']);b=float(r['end'])
            if not (src['start']<=a<b<=src['end'] and .2-1e-7<=b-a<=1.1+1e-7):
                returned.append({'requested':r,'state':'outside_frozen_time_or_width_limit'});continue
        except Exception:
            returned.append({'requested':r,'state':'invalid'});continue
        try:
            d=destination/f'query_{j+1}';resample(src['path'],[a,b],d,previous_frames=meta['frames'],fps=10,budget=8)
            result=read(d/'keyframes.json')['frames']
            t=read(d/'resample_trace.json')
            if result:
                chart=destination/f'returned_{j+1}.png'
                contact(result,chart,f'Query {j+1} | {a:.3f} to {b:.3f} | actual original source frames')
                images.append(chart)
            returned.append({'requested':r,'state':'read_original_video','t':[p['t'] for p in result],
                'image_count':len(result),'trace':str(d/'resample_trace.json'),
                'distinct_candidates':t['candidate_count'],'outside_coarse':len(t['new_source_frames_vs_coarse'])})
        except Exception as exc:returned.append({'requested':r,'state':'failed_to_read','reason':repr(exc)})
    return returned,images


def observer(cid,d,start,src,meta):
    init=d/'initial.png'
    first=invoke(start,[init],d/'initial.jsonl')
    if not first.get('reply') or first.get('tool_events') or first.get('exit_code')!=0:
        v={'id':cid,'first':first,'after':{},'reads':[],'completed':False,'reason':'initial tool/json/transport failure'}
        dump(d/'result.json',v);return v
    reads,photos=query_requests(first['reply'],d,src,meta)
    simple=[{k:v for k,v in r.items() if k!='trace'} for r in reads]
    prompt=FOLLOW.format(log=json.dumps(simple,ensure_ascii=True))
    nxt=invoke(prompt,photos,d/'followup.jsonl',first['thread_id']) if first.get('thread_id') else {}
    v={'id':cid,'first':first,'after':nxt,'reads':reads,'completed':bool(nxt.get('reply')) and not bool(nxt.get('tool_events')) and nxt.get('exit_code')==0}
    dump(d/'result.json',v);return v


def judges(o,results,anchor):
    r=o/'blind_review';r.mkdir()
    attachments=[anchor]; packet=[]
    for e in results:
        d=o/e['id']
        attachments.append(d/'initial.png')
        attachments+=list(sorted(d.glob('returned_*.png')))
        packet.append({'id':e['id'],'after_first':e['first'].get('reply'),'after_query_and_final':e['after'].get('reply'),
            'reads':[q for q in e['reads'] if q['state']=='read_original_video'],'completed':e['completed'],
            'images': [p.name for p in sorted(d.glob('returned_*.png'))]})
    # Blind packet logs must not reveal any method or research artifacts via trace paths.
    for row in packet:
        for p in row['reads']:p.pop('trace',None)
    t=JUDGE.format(packet=json.dumps(packet,ensure_ascii=True))
    dump(r/'opaque_packets.json',packet)
    with ThreadPoolExecutor(2) as ex:
        futures=[ex.submit(invoke,t,attachments,r/f'judge_{i+1}.jsonl') for i in range(2)]
        out=[p.result() for p in futures]
    dump(r/'replies.json',out)
    valid=[]
    roles={'initial','final','mid','background'}
    for review in out:
        q=review.get('reply') or {}
        if not isinstance(q.get('scores'),list) or review.get('tool_events'):continue
        classified={e.get('id'):e for e in q['scores'] if isinstance(e,dict)}
        clean={}
        for item in results:
            k=item['id']
            e=classified.get(k,{})
            if not item['completed'] or not e.get('scorable'):continue
            if not all(isinstance(e.get(key),list) and set(e[key])<=roles for key in ('first','after','final')):continue
            clean[k]=e
        valid.append(clean)
    common=set.intersection(*(set(v) for v in valid)) if len(valid)==2 else set()
    d=[]
    for i in sorted(common):
        a,b=(x[i] for x in valid)
        for stage in ('first','after','final'):
            if set(a[stage])!=set(b[stage]):d.append({'id':i,'stage':stage,'scorer_a':a[stage],'scorer_b':b[stage]})
        for k in ('unsupported_first','unsupported_after'):
            if a.get(k)!=b.get(k):d.append({'id':i,'stage':k,'scorer_a':a.get(k),'scorer_b':b.get(k)})
    third=None;settlement={}
    if d:
        just=JUDGE.format(packet=json.dumps(packet,ensure_ascii=True))
        just+='\nThird independent arbitration, only for the previously disputed packet+field below. Assess without seeing other grades. Return the same format. Disputed fields: '+json.dumps([{k:v for k,v in x.items() if k in ('id','stage')} for x in d])
        third=invoke(just,attachments,r/'judge_arbitrator.jsonl'); rr=third.get('reply') or {}
        if not third.get('tool_events') and isinstance(rr.get('scores'),list):
            settlement={w.get('id'):w for w in rr['scores'] if isinstance(w,dict) and w.get('scorable')}
    final=[]
    for i in sorted(common):
        a,b=(x[i] for x in valid)
        record={'id':i}
        for stage in ('first','after','final','unsupported_first','unsupported_after'):
            wa,wb=a.get(stage),b.get(stage)
            agree=set(wa)==set(wb) if isinstance(wa,list) and isinstance(wb,list) else wa==wb
            q=wa if agree else settlement.get(i,{}).get(stage)
            if stage in ('first','after','final') and (not isinstance(q,list) or not set(q)<=roles):q=None
            record[stage]=q
        record['fixed_remaining_after']=4-len(record['after']) if record['after'] is not None else None
        record['new_grounded_fact_count']=len(set(record['after'])-set(record['first'])) if record['after'] is not None and record['first'] is not None else None
        record['final_loss_count']=len(set(record['after'])-set(record['final'])) if record['after'] is not None and record['final'] is not None else None
        final.append(record)
    return {'blind_reviews':len(valid),'dual_scored':len(common),'first_disagreements':d,
            'third_attempt':third,'arbitrated_scores':final,'actual_raw_reviewer_outputs':out}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--trial-dir',default='P3_live');a=p.parse_args()
    o=a.out.resolve(); d=o/a.trial_dir; d.mkdir()
    s=o/'cases'/CASE
    meta=read(s/'frames.json');src=meta['source']
    f=read(s/'selection.json')['arms']
    arms=['V','uniform','S2','K_roles'];ids=['R-166','R-249','R-451','R-623']
    random.Random(104031).shuffle(ids)
    mapping=dict(zip(ids,arms))
    for k,arm in mapping.items():
        cd=d/k;cd.mkdir(); contact(f[arm],cd/'initial.png',f'{k} | 8 source images (pairs left to right, then down)')
    start=START.format(tasks='; '.join(TASKS),start=src['start'],end=src['end'])
    data=meta['frames']; notes=[min(range(len(data)),key=lambda i:abs(data[i]['t']-t)) for t in (.2,3,5.8)]
    anchor=d/'review_only_anchor.png';contact([data[i] for i in notes],anchor,'Scoring reference only | independent known source frames')
    dump(d/'evaluation_reference_sealed.json',{'source':src['sha256'],'visual_truth':'locally moving synthetic form, surrounding fixed pattern; no causal claim',
           'review_anchor_frames':[data[i] for i in notes],'roles':['initial','final','mid','background']})
    dump(d/'p3_lock.json',{'prediction_packets_sha':{k:sha(d/k/'initial.png') for k in ids},'mapping_private':mapping,
       'code_sha':sha(Path(__file__)),'model':MODEL,'protocol':'two same-session turns, authenticated actual agent',
       'max_queries':2,'fixed_query_10_fps':True,'method_blind_judges':True,'source_development_only':True,
       'oracle_or_method_data_shared_with_observers':False,'observer_start_prompt':start,'followup_prompt_template':FOLLOW,
       'judging_prompt_template':JUDGE,'scoring_reference_hash':sha(d/'evaluation_reference_sealed.json')})
    with ThreadPoolExecutor(2) as ex:
        jobs=[ex.submit(observer,k,d/k,start,src,meta) for k in ids]
        result=[x.result() for x in jobs]
    dump(d/'observers.json',result)
    if any(x['completed'] for x in result):
        q=judges(d,result,anchor)
    else:
        q={'status':'not_run_no_completed_observer','blind_reviews':0,'dual_scored':0}
    dump(d/'smoke_summary.json',{'live_initial_sessions':sum(bool(x['first'].get('thread_id')) for x in result),
      'same_session_completed_observers':sum(x['completed'] for x in result),
      'source_reads_executed':sum(r['state']=='read_original_video' for v in result for r in v['reads']),
      'new_independent_trial':'blocked: no new source beyond frozen five directory videos','explicit_no_method_superiority_claim':True,
      'judges':q})
    print('live P3',sum(x['completed'] for x in result),q.get('dual_scored',0),flush=True)

if __name__=='__main__':main()
