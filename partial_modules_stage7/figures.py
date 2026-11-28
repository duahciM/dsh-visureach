"""Draw the three replay frames and the actual change footprints by window."""

from PIL import Image, ImageDraw, ImageFont

from dacf_research.prepare_inputs import read
from partial_modules_stage7.inputs import H,OLD


if __name__=='__main__':
    result=read(H/'evaluation.json')['window_cases']['reel']['targeted_replay_after_scoring']
    source=OLD/'automatic/reel'
    receipt={r['pts']:r for r in read(source/'frame_receipts.json')}
    sheet=Image.new('RGB',(1104,806),'#182329')
    for i,key in enumerate(('4400','9200','9400')):
        with Image.open(source/receipt[int(key)]['png']) as frame:
            native=frame.convert('RGB').resize((344,194))
        for j,choice in enumerate(('A','B','WRONG')):
            candidate=native.copy()
            proof=result[key][choice][0]
            draw=ImageDraw.Draw(candidate)
            for x,y,w,h in proof['projected_M4_candidate_bboxes']:
                draw.rectangle((x*344/320,y*194/180,(x+w)*344/320,(y+h)*194/180),outline='#ffff8c',width=3)
            left,top=15+j*366,25+i*264
            sheet.paste(candidate,(left,top+24))
            ink=ImageDraw.Draw(sheet)
            ink.text((left,top),f'{choice}  |  source PTS {key}',font=ImageFont.load_default(size=17),fill='white')
            phases=','.join(str(v) for v in proof['registered_refs'])
            ink.text((left,top+224),'Registered refs: '+phases,font=ImageFont.load_default(size=11),fill='#dbebec')
    sheet.save(H/'comparison_windows.png')
