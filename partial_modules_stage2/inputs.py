"""Create an independent high-rate control and pin a real source and review labels."""
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from dacf_research.prepare_inputs import ROOT,digest,write
from evidenweft_phase1.source_review import timeline,WITHDRAWN
from partial_modules_stage1.fixtures import textile

H=Path(__file__).resolve().parent


def render(folder):
    src=folder/'source.mkv'; width,height,count=768,432,160
    rng=np.random.default_rng(28062)
    backdrop=textile(rng,[(54,62,66),(74,94,106),(109,130,120),(159,165,148),(191,175,149)],height,width)
    for _ in range(75):
        x,y=map(int,rng.integers([12,12],[width-12,height-12]))
        cv2.line(backdrop,(x-6,y-4),(x+6,y+4),(109,150,175),2)
    token=textile(rng,[(182,52,55),(249,210,131),(79,39,105),(240,120,104),(116,76,53)],92,118)
    marker=textile(rng,[(79,108,48),(159,190,77),(56,71,100)],69,74)
    dock=textile(rng,[(62,47,108),(105,122,145),(153,115,195)],73,86)
    rows=[]; frames=[]
    for k in range(count):
        render_src=backdrop.copy()
        # The target never covers a referent in native pixels. The vertical lane changes separately.
        bump_x=max(0,45-abs(37-k)*6)
        bump_y=max(0,24-abs(43-k)*4)
        slow=min(max(k-73,0),19)
        pan=max(0,min(k-108,24)); pan_y=min(max(k-111,0),12)
        x,y=179+bump_x+slow,159+bump_y
        active=k<56 or k>=72
        items=[('reference','rail',368,171,marker),('peer','dock',385,305,dock)]
        if active:
            active_token=token.copy()
            if 77<=k<=86:
                cv2.circle(active_token,(28,28),15,(251,238,91),-1)
            items.append(('candidate','g0' if k<56 else 'g1',x,y,active_token))
        else:
            cv2.rectangle(render_src,(155,149),(342,267),(69,77,84),-1)
        if 18<=k<47:
            # This is a copied projection in one simulated story. Pixels alone cannot tell its cause.
            cv2.rectangle(render_src,(508,81),(684,252),(28,33,44),5)
            items.append(('candidate','g2',533,113,token))
        if 103<=k<=108:
            items=[item for item in items if item[0]!='peer']
        obs=[]
        for role,name,xx,yy,tile in items:
            hh,ww=tile.shape[:2]; render_src[yy:yy+hh,xx:xx+ww]=tile
            obs.append({'role':role,'episode':name,'native_xywh':[xx+pan,yy+pan_y,ww,hh]})
        render_src=cv2.warpAffine(render_src,np.float32([[1,0,pan],[0,1,pan_y]]),(width,height),borderMode=cv2.BORDER_REFLECT)
        if 96<=k<122:
            render_src=np.uint8(np.clip(render_src.astype(np.int16)+26,0,255))
        frames.append(render_src.tobytes());rows.append(obs)
    subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','-','-an',
                    '-c:v','ffv1','-pix_fmt','bgr0',str(src)],input=b''.join(frames),check=True,timeout=50)
    stream,clock=timeline(src);assert len(clock)==count
    material={'source':src.name,'video_sha256':digest(src),'native_size':[width,height],'time_base':stream['time_base'],
              'observations':[o|r for olist,r in zip(rows,clock) for o in olist], 'sampled_pts':[v['pts'] for v in clock]}
    write(folder/'observations-dense.json',material)
    return {'single_source_sha256':material['video_sha256'], 'indistinguishable_worlds':{
      'a':'g0 and g1 are the same item behind an opaque occluder; g2 is a copied projection.',
      'b':'g1 is an identical replacement made behind the opaque occluder; g2 is a separate identical tile.'},
      'same_visible_source_in_both_worlds':True,'relation_roles':{
        'near_rail_x_apex':[35,39],'vertical_lane_apex':[41,45], 'local_state_marker':[77,86]},
      'independent_trajectory_reference':'native bbox coordinates from known renderer motions, visible endpoints only',
      'prediction_may_query_annotations_only_after_the_original_frame_was_fetched':True}


def real(folder):
    source=Path('C:/Users/Administrator/Desktop/视觉插件/素材/10.avi')
    sha=digest(source)
    assert sha!='' and sha!=WITHDRAWN
    known={digest(file) for d in ('dacf_round2','dacf_round3','dacf_round4','dacf_round5','dacf_round6','partial_modules_stage1','evidenweft_phase1')
           for file in (ROOT/d).rglob('*') if file.is_file() and file.suffix.lower() in ('.mp4','.mkv','.avi','.nut')}
    assert sha not in known
    path=folder/'source.avi'; shutil.copyfile(source,path); assert digest(path)==sha
    stream,clock=timeline(path); assert stream['time_base']=='1/10'
    # Rough observed boxes at sealed sample times. The camera and fixed sign are a 2D reference.
    boxes=[[715,286,48,113],[700,279,43,101],[676,269,47,98],[651,260,44,98],[630,249,43,104],[604,244,45,102],
           [588,235,43,95],[559,227,47,91],[531,226,43,88],[501,217,46,89],[480,214,42,85],[468,209,46,88]]
    peers={30:[666,246,41,98],34:[637,242,47,91],38:[612,235,43,93],42:[590,231,47,87],46:[572,226,39,85],
           50:[548,215,41,89],54:[522,216,42,83],58:[514,210,39,87]}
    samples=list(range(14,59,4)); observed=[]
    lookup={r['pts']:r for r in clock}
    for pts,box in zip(samples,boxes):
        data=[('candidate','g0',box),('reference','sign',[414,196,29,44])]
        if pts in peers:data.append(('peer','peer',peers[pts]))
        observed.extend(lookup[pts]|{'role':role,'episode':name,'native_xywh':roi} for role,name,roi in data)
    write(folder/'observations.json',{'source':path.name,'video_sha256':sha,'time_base':stream['time_base'],'observations':observed,
             'native_size':[stream['width'],stream['height']], 'sampled_pts':samples,
             'scope':'Approximate hand-visible projected boxes; source frame sequence and physical identity are different evidence.'})
    return {'sha256':sha,'new_sha_vs_dacf_stage1_collections':len(known),'observed_reference':'traffic sign',
      'visible_goal':'relative direction toward the sign and visible second moving region','physical_identity':'not assessed'}


if __name__=='__main__':
    home=H/'inputs';home.mkdir(); control=home/'control';street=home/'street'
    control.mkdir();street.mkdir();truth=render(control); provenance=real(street)
    write(home/'sealed_roles.json',{'control':truth,'street':provenance})
