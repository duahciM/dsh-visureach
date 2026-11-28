"""Make source-only review sheets and preserve exact images used by the reviewer."""

from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, WITHDRAWN, digest, write
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline, native_stream

H=Path(__file__).resolve().parent


def review(name, path, frequency, rows):
    output=H/'source_review'/name
    output.mkdir(parents=True)
    info,clock=timeline(path)
    count=len(clock)
    if rows:
        selected=set(rows)
    else:
        selected=set(range(0,count,max(1,round(count/frequency))))|{count-1}
    sha=digest(path)
    assert sha!=WITHDRAWN
    reader=Reader(path,sha,info,clock,output/'native')
    cells=[]
    for i in sorted(selected):
        addr=clock[i]; receipt,_=reader.read(addr['pts']); cells.append(receipt)
    sheet=Image.new('RGB',(1200,((len(cells)+4)//5)*154),'#182230')
    draw=ImageDraw.Draw(sheet)
    for i,receipt in enumerate(cells):
        x,y=(i%5)*240,(i//5)*154
        with Image.open(receipt['path']) as original:
            img=original.convert('RGB');img.thumbnail((234,126))
        sheet.paste(img,(x+3,y+2))
        frame=receipt['source_frame']; pts=receipt['pts']
        draw.text((x+5,y+132),f"Frame {frame} · PTS {pts} · {clock[frame]['seconds']:.2f}s",fill='white')
    sheet.save(output/'contact.png')
    write(output/'receipts.json',{'sha256':sha,'source':path.relative_to(ROOT).as_posix(),
        'clock':clock,'receipt_rows':cells})


if __name__=='__main__':
    copy=H/'inputs';copy.mkdir()
    local=Path('C:/Users/Administrator/Desktop/视觉插件/素材/11.mp4')
    target=copy/'11.mp4';copy2(local,target)
    assert digest(local)==digest(target)
    review('real_return',target,20,None)
    review('detour',ROOT/'dacf_research/inputs/detour.mkv',None,list(range(49)))
    review('cut',ROOT/'dacf_research/inputs/cut.mkv',None,[0,4,8,12,13,14,15,16,17,22,29])
    review('camera',ROOT/'partial_modules_stage3/inputs/camera/source.mkv',None,[0,16,24,28,31,32,33,36,40,48,51,52,53,56,60,72,79])
