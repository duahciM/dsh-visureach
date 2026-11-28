"""Draw the actual envelope arms and source PTS on a shared time axis."""

from fractions import Fraction

from PIL import Image, ImageDraw, ImageFont

from dacf_research.prepare_inputs import read
from partial_modules_stage6.material import H


def strip(photo,start,case):
    draw=ImageDraw.Draw(photo)
    suite=read(H/'evaluation.json')['cases'][case]
    auto=read(H/'automatic'/case/'auto.json')
    ref=read(H/'assessment/sealed.json')[case]
    tb=Fraction(auto['time_base'])
    lo,hi=auto['extent']
    def location(pts):
        return 206+round(892*(pts-lo)/(hi-lo))
    draw.text((26,start),{'reel':'Controlled reel', 'real_return':'Real room return'}[case],
              font=ImageFont.load_default(size=22),fill='#19272d')
    color={'Manual reference':'#35698b','Automatic proposal':'#ad5828','Shifted control':'#ac475b',
           'Frozen first nomination':'#25465c','5-FPS source pool':'#aeb8bc'}
    bands=[('Manual reference',ref['bounds']),
           ('Automatic proposal',[r['pts'] for r in auto['accepted']]),
           ('Shifted control',read(H/'observations/reference'/case/'WRONG.json')['reference_envelopes'])]
    for i,(label,windows) in enumerate(bands):
        y=start+46+i*35
        draw.text((26,y-7),label,font=ImageFont.load_default(size=13),fill='#223746')
        draw.line((206,y,1098,y),fill='#d2dce0',width=1)
        for a,b in windows:
            draw.rounded_rectangle((location(a),y-8,location(b),y+8),radius=3,fill=color[label])
    for i,(label,times) in enumerate([('5-FPS source pool',auto['pool_pts']),('Frozen first nomination',auto['first_pts'])]):
        y=start+152+i*30
        draw.text((26,y-8),label,font=ImageFont.load_default(size=13),fill='#223746')
        for p in times:
            draw.line((location(p),y-3,location(p),y+6),fill=color[label],width=2)
    if ref.get('subcoarse_insert'):
        p=ref['subcoarse_insert']['visible_only_at_pts']
        x=location(p); y=start+155
        draw.ellipse((x-4,y-4,x+4,y+4),fill='#b81d67')
        draw.text((x-58,y+12),'source-only insert',font=ImageFont.load_default(size=12),fill='#b81d67')
    for t in range(0,int(float(tb*hi))+1,2):
        x=location(t/float(tb))
        draw.text((x-7,start+218),str(t)+'s',font=ImageFont.load_default(size=12),fill='#5f6f73')
    counts=suite['envelope_counts']
    draw.text((26,start+247),f"Closed scopes · manual {counts['manual_closed']}   /   automatic {counts['auto_closed']}   /   shifted {counts['wrong_closed']}",
              font=ImageFont.load_default(size=14),fill='#223746')


if __name__=='__main__':
    image=Image.new('RGB',(1140,620),'#f5f8f9')
    strip(image,15,'reel')
    strip(image,323,'real_return')
    image.save(H/'envelope_timeline.png')
