"""Render two exact evidence frames that have different Figure 2 applicability."""
from PIL import Image,ImageDraw
from dacf_research.prepare_inputs import read
from partial_modules_stage3.material import H

if __name__=='__main__':
    native={r['pts']:r['path'] for r in read(H/'predictions/roles/base_receipts.json')['receipts']}
    boxes=read(H/'references/sealed.json')['targets']['roles']
    sheet=Image.new('RGB',(1024,365),'#15202b')
    for i,run in enumerate(('engineering','engineering-near')):
        r=read(H/run/'result.json'); t=r['pts'][-1]
        with Image.open(native[t]) as p:img=p.convert('RGB').resize((490,274))
        paint=ImageDraw.Draw(img)
        rr=[row for row in boxes if row['pts']==t and row['episode'] in ('rail','target_before')]
        for item in rr:
            x,y,w,h=item['box']; sx,sy=490/800,274/448
            color='#ffcf74' if item['episode']=='rail' else '#e57c92'
            paint.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline=color,width=3)
        col=i*512+11;sheet.paste(img,(col,34)); painter=ImageDraw.Draw(sheet)
        fig=r['region_figure']['summary']
        painter.text((col,11),f'Actual source frame at tick {t} (time base 1/1000)',fill='white')
        painter.text((col,320),f'Figure 2: {fig["measured_pairs"]} projected pair(s)',fill='#ffd075')
    painter.text((11,344),'Pink: observed region   Gold: stable referent   Measurements are scoped to source-image projection.',fill='#cfdee8')
    sheet.save(H/'evaluation/projected_endpoints.png')
