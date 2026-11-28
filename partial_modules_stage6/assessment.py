"""Prepare independent source observations and condition-blind evaluation roles."""

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, read, write
from dacf_round3.acquire import Reader
from evidenweft_phase1.source_review import timeline
from partial_modules_stage6.material import H


def ref(name,lo,hi,roi=None):
    return {'id':name,'pts':[lo,hi],'native_roi':roi}


if __name__=='__main__':
    d=H/'assessment';d.mkdir()
    old=read(ROOT/'partial_modules_stage5/reviewed_bounds.json')
    records={k:{'bounds':[r['pts'] for r in old.get(k,[]) if r['kind']!='presentation_change_only'],
                'roles':[],'relations':[],'controls':list(old.get(k,[]))}
              for k in read(H/'inputs/cases.json')}
    records['reel']={'bounds':[[2500,2600],[6500,7600],[11500,11600]],
        'subcoarse_insert':{'bound':[13200,13400],'visible_only_at_pts':13300},
        'roles':[ref('B_panel_before',2800,3300,[214,22,30,24]),ref('B_panel_on',3400,4600,[214,22,30,24]),
          ref('B_panel_after',4800,6000,[214,22,30,24]),ref('B_slider_before',3000,4500,[101,103,28,25]),
          ref('B_slider_near',5200,6100,[149,102,62,30]),ref('C_panel_before',7800,8500,[26,55,36,36]),
          ref('C_panel_on',8700,10300,[26,55,36,36]),ref('C_panel_after',10600,11200,[26,55,36,36]),
          ref('C_tab_before',7800,9400,[174,93,25,16]),ref('C_tab_near',10100,11100,[172,122,31,33]),
          ref('open_early',0,1800),ref('open_late',11900,12900),ref('source_only_short_insert',13300,13300)],
        'relations':[['B_panel_before','B_panel_on','B_panel_after'],['B_slider_before','B_slider_near'],
                     ['C_panel_before','C_panel_on','C_panel_after'],['C_tab_before','C_tab_near']],
        'controls':[]}
    records['real_return']['roles']=[ref('bed_before',0,559),ref('wall_first',1676,2793),ref('sofa_initial',3910,4700),
        ref('sofa_and_table',4800,6200),ref('wall_return',7821,8379),ref('bed_return',10055,11028)]
    records['real_return']['relations']=[['bed_before','wall_first','sofa_initial'],
        ['sofa_and_table','wall_return','bed_return']]
    for key,data in read(H/'inputs/cases.json').items():
        directory=d/key;directory.mkdir()
        info,clock=timeline(ROOT/data['source'])
        selected={r['pts'] for r in clock[::max(1,len(clock)//14)]}
        if key=='reel':
            selected |= {2400,2500,2600,3300,3400,4000,4700,4800,5600,6500,7000,7600,8400,8700,9400,10500,10600,11500,
                         11600,13200,13300,13400}
        before=Reader(ROOT/data['source'],data['sha256'],info,clock,directory/'native')
        receipt=[before.read(pts)[0] for pts in sorted(selected)]
        sheet=Image.new('RGB',(1100,((len(receipt)+4)//5)*147),'#202c3a')
        for i,r in enumerate(receipt):
            with Image.open(r['path']) as image:
                view=image.convert('RGB');view.thumbnail((218,123))
            x,y=i%5*220,i//5*147;sheet.paste(view,(x,y))
            ImageDraw.Draw(sheet).text((x+3,y+125),f"PTS {r['pts']} ({r['seconds']:.3f}s)",fill='white')
        sheet.save(directory/'contact.png')
        write(directory/'receipts.json',receipt)
    records['camera']['evaluation_scope']='Constructed background retexture and viewpoint changes; no separate document-scene role.'
    records['real_pan']['evaluation_scope']='Only registered fixed-space pan example; reviewed frames need not exhaust all real content.'
    records['long_street']['evaluation_scope']='Long mostly fixed view with local movement. Reviewed time windows are negative context controls, not dense annotations.'
    write(d/'sealed.json',records)
