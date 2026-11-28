"""Render a small source-backed sheet for one real M2 false bridge and the same-time conflict."""
from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import read
from partial_modules_stage1.fixtures import HERE


if __name__=='__main__':
    out=HERE/'first_run/paired'
    a=read(out/'A3.json'); store=read(out/'evidence.json')
    conflict=next(v for v in a['m2'] if v['outcome']=='coappearance_conflict')
    views={row['view_key']:row for values in store['views'].values() for row in values}
    row=next(v for v in a['m2'] if v['episodes']==['g0','g3'])
    calc=next(v for v in store['local_calculations'] if v['comparison_ref']==row['independent_comparison_refs'][0])
    samples=[('Simultaneous separate regions',conflict['counterevidence'][0]['views']),
             ('Lookalike false bridge: earlier',calc['views'][:1]),('Lookalike false bridge: later',calc['views'][1:])]
    page=Image.new('RGB',(1032,270),'#101c29'); drawing=ImageDraw.Draw(page)
    paths={r['pts']:r['path'] for r in store['source_receipts']}
    for i,(label,keys) in enumerate(samples):
        frames=[views[k] for k in keys];tick=frames[0]['pts']
        with Image.open(paths[tick]) as image: thumb=image.convert('RGB').resize((330,194),Image.Resampling.LANCZOS)
        for view in frames:
            x,y,w,h=view['native_xywh']; ink=ImageDraw.Draw(thumb)
            ink.rectangle((x*330/640,y*194/376,(x+w)*330/640,(y+h)*194/376),outline='#ffde81',width=3)
            ink.text((x*330/640+2,y*194/376+3),view['episode'],fill='#ffffff',stroke_width=2,stroke_fill='#101c29')
        x=i*344+7;page.paste(thumb,(x,26));drawing.text((x+1,6),label,fill='#f3f7fc');drawing.text((x+1,230),f'PTS {tick}  |  native box overlay',fill='#e4edf6')
    drawing.text((8,253),'Physical identity was not assigned by the visual matcher. Original source receipts are stored separately.',fill='#c3cdd9')
    page.save(HERE/'evaluation/lookalike_counterexample.png')
