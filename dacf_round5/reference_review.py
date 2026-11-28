"""Confirm score witnesses independently from source pixels, after prediction launch."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

from dacf_research.prepare_inputs import read,write
from dacf_round3.acquire import Reader
from dacf_round5.setup import HERE
from evidenweft_phase1.source_review import timeline


def main():
    out=HERE/'references';out.mkdir()
    targets=read(HERE/'roles.json')
    for case in ('fixture','arm_clip'):
        source=HERE/'inputs'/case/'source.mp4'
        plan=read(source.with_name('plan.json'))
        stream,clock=timeline(source)
        reader=Reader(source,plan['sha256'],stream,clock,out/case)
        refs=[]; roles=targets[case]['roles']
        sheet=Image.new('RGB',(960,((len(roles)+2)//3)*190),'#192736')
        painter=ImageDraw.Draw(sheet)
        for i,r in enumerate(roles):
            receipt,_=reader.read(r['exact_pts'])
            assert r['seconds'][0]<=receipt['seconds']<=r['seconds'][1]
            with Image.open(receipt['path']) as im:
                if r['id']=='F_GREEN': assert im.getpixel((154,104))[1]>140
                x,y=320*(i%3),190*(i//3); im.thumbnail((309,173)); sheet.paste(im,(x+4,y+2))
            painter.text((x+4,y+173),f'{r["id"]} | exact PTS {r["exact_pts"]} | {receipt["seconds"]:.3f}s',fill='white')
            refs.append(r|{'source_receipt':receipt})
        write(out/f'{case}.json',refs)
        sheet.save(out/f'{case}.png')
    diag=read(HERE/'append/diagnostic.json')
    static=diag['matched_static']['channels']['hsv'];moving=diag['unrelated_high']['channels']['hsv']
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    a=[static['prefix_global_threshold'],static['new_threshold_without_join'],moving['new_threshold_without_join']]
    ax.bar(['Original prefix','+ static tail','+ unrelated high-change tail'],a,color=['#49758a','#83a8b2','#c17b46'])
    ax.axhline(static['prefix_candidate_gap'],ls='--',label='Unchanged prefix candidate gap',color='#252f36')
    ax.legend();ax.set_ylabel('HSV distance, original S0 threshold');ax.set_title('Same audited prefix, different global denominators')
    fig.savefig(HERE/'append/global_stop_scope.png',dpi=145)


if __name__=='__main__':main()
