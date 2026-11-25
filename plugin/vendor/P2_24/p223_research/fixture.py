"""Rendered positive-control source images and targeted real-image checks."""
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw

from if17_research.catalog import ref, sha
from p216_research.navigate import checked_json

from .features import figure1, graph
from .timeline import timeline


POS={"A":(65,72),"B":(255,72),"C":(155,175),"P":(155,30),"G":(157,215)}
COLORS={"A":"#d76843","B":"#328bc1","C":"#62aa6b","P":"#eaad14","G":"#ac9d86"}
FIXTURES={"F1":("",()),"F2":("AB",("AB",)),"F3":("ABC",("AB","AC","BC")),"F4":("ABC",("AB","AC","BC")),
          "spectator_C":("ABC",("AB",)),"floor_ABC":("ABCG",("AG","BG","CG")),"AB_repeat":("AB",("AB","AB")),
          "A_with_apple":("ABP",("AP",)),"B_with_apple":("ABP",("BP",))}
LINK=(57,198,207)


def mark(a,b):
    x,y=POS[a];w,z=POS[b]
    return (x+w)//2,(y+z)//2


def run(out):
    folder=out/"relation_fixtures"
    folder.mkdir(exist_ok=True)
    results={}
    pixels=[]
    for n,(name,(symbols,edges)) in enumerate(FIXTURES.items()):
        pic=Image.new("RGB",(320,258),(26,32,40))
        d=ImageDraw.Draw(pic)
        for a,b in edges:
            d.line([POS[a],POS[b]],fill=(93,107,124),width=3)
        if name=="F1":
            d.rectangle((29,160,290,220),fill=(89,79,75))
        for symbol in symbols:
            x,y=POS[symbol]
            d.ellipse((x-14,y-14,x+14,y+14),fill=COLORS[symbol])
            d.text((x-3,y-5),symbol,fill="white")
        for a,b in edges:
            x,y=mark(a,b)
            d.rectangle((x-3,y-3,x+3,y+3),fill=LINK)
        d.text((9,238),name+"  |  visible relation marks",fill="white")
        dest=folder/(name+".png")
        pic.save(dest)
        root=sha(dest)
        def color(s):
            hx=COLORS[s].lstrip("#")
            return tuple(int(hx[i:i+2],16) for i in (0,2,4))
        with Image.open(dest) as sample:
            placed=[s for s,(x,y) in POS.items() if sample.getpixel((x+9,y))==color(s)]
            nodes=[{"id":s,"appearance":s,"roots":[root+":"+s]} for s in placed]
            visible_pairs=[{"ends":[a,b],"type":"floor_geometry" if "G" in (a,b) else "marked_example_relation",
                            "roots":[root+":"+a+b]} for a,b in combinations(placed,2) if sample.getpixel(mark(a,b))==LINK]
        evidence=graph(nodes,visible_pairs+visible_pairs if name=="AB_repeat" else visible_pairs)
        sanitized={"candidate":name,"regions":nodes,"spatial_reference_observations":[],"view_color_grid":[]}
        expected=len(set(edges))
        assert evidence["distinct_pairs"]==expected
        f0=figure1(sanitized)
        f1=figure1(sanitized,evidence)
        assert f0["I"]==f1["I"]
        results[name]={"source_png":ref(dest),"figure2":evidence,"F0":f0,"F1":f1}
        pixels.append(pic)
    def wave(ids):
        in_=[{"candidate_id":k,"pts":1000*i,"time_base":"1/1000"} for i,k in enumerate(ids)]
        return timeline(in_,[results[k]["F1"] for k in ids],[],{"start":0,"end":len(ids)})[0]
    shape=wave(["F1","F2","F3","F4"])
    hand=wave(["A_with_apple","B_with_apple"])
    assert [r["R"] for r in shape["points"]]==[0,1,3,3]
    assert "observed_R_plateau" in shape["points"][-1]["flags"]
    assert "relation_signature_change_at_equal_count" in hand["points"][-1]["flags"]
    sheet=Image.new("RGB",(320*3,258*3),(26,32,40))
    for i,pic in enumerate(pixels):
        sheet.paste(pic,(320*(i%3),258*(i//3)))
    sheet.save(folder/"relation_fixtures.png")
    doc={"cases":results,"F1_F4":shape,"same_height_participant_swap":hand,"all_assertions_passed":True,
         "rendered_sources_not_an_M9_trial":True,"heldFloorFailure":"floor has the same raw R=3 as all three marked pair interactions; no background discount"}
    checked_json(folder/"results.json",doc)
    return doc
