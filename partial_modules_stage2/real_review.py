"""Build an assessor-only paper review of the original street footage."""
from pathlib import Path
from PIL import Image,ImageDraw
from evidenweft_phase1.source_review import native_stream,timeline

H=Path(__file__).resolve().parent
if __name__=='__main__':
    path=Path('C:/Users/Administrator/Desktop/视觉插件/素材/10.avi')
    info,clock=timeline(path)
    rows=[14,18,22,26,30,34,38,42,46,50,54,58]
    sheet=Image.new('RGB',(1280,774),'#101b26'); draw=ImageDraw.Draw(sheet)
    for s,pixels in native_stream(path,info,clock):
        tick=s['pts']
        if tick not in rows:continue
        n=rows.index(tick); x0=(n%4)*320;y0=(n//4)*258
        im=Image.frombytes('RGB',(info['width'],info['height']),pixels).resize((320,240))
        d=ImageDraw.Draw(im)
        for x in range(80,768,80):
            u=x*320/768;d.line((u,0,u,240),fill='#90a8a0');d.text((u,0),str(x),fill='white',stroke_width=1,stroke_fill='#202030')
        for y in range(80,576,80):
            v=y*240/576;d.line((0,v,320,v),fill='#90a8a0');d.text((0,v),str(y),fill='white',stroke_width=1,stroke_fill='#202030')
        sheet.paste(im,(x0,y0));draw.text((x0+4,y0+241),f'tick {tick} | {s["seconds"]:.1f}s',fill='white')
        if tick==rows[-1]:break
    sheet.save(H/'real_frame_review.png')
