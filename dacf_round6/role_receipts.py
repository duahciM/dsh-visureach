"""Join exact witness and surrounding positive sample windows without claiming visual interpretation."""
import argparse
from pathlib import Path

from PIL import Image,ImageDraw
from dacf_research.prepare_inputs import ROOT,read,write
from dacf_round6.summarize import H
from evidenweft_phase1.source_review import timeline
from dacf_round3.acquire import Reader


def main():
    parser=argparse.ArgumentParser();parser.add_argument('batch');parser.add_argument('--references',action='store_true');a=parser.parse_args()
    ref=read(H/'references.json')
    if a.references:
        shelf=H/'source-review';shelf.mkdir()
    focus={}
    for case,d in ref.items():
        source=H/'independent/source.mp4' if case=='independent' else ROOT/'dacf_round5/inputs/fixture/source.mp4'
        st,clock=timeline(source)
        focus[case]={k:{'exact':clock[v['witness']]['pts'],'window':[clock[p]['pts'] for p in v['frames']],
                          'time':[clock[p]['seconds'] for p in v['frames']]} for k,v in d['roles'].items()}
        if a.references:
            from dacf_research.prepare_inputs import digest
            reader=Reader(source,digest(source),st,clock,shelf/case)
            strip=Image.new('RGB',(950,((len(d['roles'])+2)//3)*201),'#16212b');g=ImageDraw.Draw(strip)
            for i,(id_,entry) in enumerate(focus[case].items()):
                found,_=reader.read(entry['exact'])
                with Image.open(found['path']) as src:
                    src.thumbnail((306,179))
                    x,y=i%3*315,i//3*201;strip.paste(src,(x+2,y));g.text((x+3,y+179),f'{id_} | PTS {entry["exact"]}',fill='white')
            strip.save(shelf/f'{case}.png');write(shelf/f'{case}.json',reader.receipts)
    folder=H/a.batch
    if not (folder/'summary.json').exists():return
    batch=read(folder/'summary.json')
    out=[]
    for item in batch['attempts']:
        cite={p for r in item['claim_evidence'] for p in r['audited_original_pts']}
        seen=set(item['different_native_pts']); ws=focus[item['case']]
        hit=lambda window,where:bool(any(window[0]<=x<=window[1] for x in where))
        covered=[k for k,r in ws.items() if hit(r['window'],seen)]
        out.append({'condition':item['condition'],'arm':item['arm'],'case':item['case'],
             'exact_witness_received':[k for k,r in ws.items() if r['exact'] in seen],
             'positive_window_received':covered,'positive_window_cited':[k for k,r in ws.items() if hit(r['window'],cite)],
             'complete_delivery_groups':[g for g in ref[item['case']]['groups'] if set(g)<=set(covered)],
             'evidence_boundary':'Receipt and citation window counts. They do not infer what the model understood or what happened between originals.'})
    write(folder/'window_scores.json',{'roles':focus,'attempts':out})

if __name__=='__main__':main()
