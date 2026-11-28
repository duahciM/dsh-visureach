"""Bind frozen replays and prepare two source-only transfer windows."""

from pathlib import Path

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, WITHDRAWN, digest, read, write
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline

H=Path(__file__).resolve().parent
OLD=ROOT/'partial_modules_stage6'


def sheet(cells,path):
    canvas=Image.new('RGB',(1200,((len(cells)+4)//5)*153),'#15242c')
    ink=ImageDraw.Draw(canvas)
    for index,row in enumerate(cells):
        x,y=index%5*240,index//5*153
        with Image.open(row['path']) as image:
            frame=image.convert('RGB');frame.thumbnail((235,125));canvas.paste(frame,(x,y))
        ink.text((x+3,y+127),f"PTS {row['pts']} | {row['seconds']:.2f} s",fill='white')
    canvas.save(path)


def source_window(name,source,maximum):
    info,clock=timeline(ROOT/source)
    sha=digest(ROOT/source);assert sha!=WITHDRAWN
    limited=[r for r in clock if r['seconds']<=maximum]
    out=H/'inputs'/name;out.mkdir()
    rd=Reader(ROOT/source,sha,info,clock,out/'native')
    chosen=limited[::max(1,round(len(limited)/18))]
    if chosen[-1]!=limited[-1]:chosen.append(limited[-1])
    sheet([rd.read(r['pts'])[0] for r in chosen],out/'contact.png')
    write(out/'review_receipts.json',list(rd.receipts.values()))
    return {'source':source,'sha256':sha,'time_base':info['time_base'],'extent':[limited[0]['pts'],limited[-1]['pts']],
            'raw_window_seconds':maximum,'observations_not_used_to_tune':True}


if __name__=='__main__':
    specs=read(OLD/'inputs/cases.json')
    for name in specs:
        specs[name]['stage6']=f'partial_modules_stage6/automatic/{name}'
    specs['new_recording']=source_window('new_recording','partial_modules_stage7/inputs/dudek.webm',8.0)
    specs['reverse_transfer']=source_window('reverse_transfer','dacf_research/inputs/reverse_doorway.mp4',13.0)
    write(H/'inputs/cases.json',specs)
