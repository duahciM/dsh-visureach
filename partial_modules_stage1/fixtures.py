"""Make two new source videos; public observations and independent role truth stay separate."""
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dacf_research.prepare_inputs import digest, write
from evidenweft_phase1.source_review import timeline

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 640, 376
FPS = 4


def textile(rng, palette, h=90, w=112):
    result = np.zeros((h, w, 3), np.uint8)
    result[:] = palette[0]
    for _ in range(150):
        x,y = rng.integers(3,w-3),rng.integers(3,h-3)
        cv2.circle(result, (int(x),int(y)), int(rng.integers(1,6)), palette[int(rng.integers(1,len(palette)))], -1)
    cv2.rectangle(result,(1,1),(w-2,h-2),palette[2],2)
    return result


def fixture(name, seed, episodes, world, end):
    dest=HERE/'inputs'/name
    dest.mkdir(parents=True)
    rng=np.random.default_rng(seed)
    backdrop=textile(rng,[(47,53,58),(67,78,92),(128,136,145),(54,96,123),(208,204,190)],HEIGHT,WIDTH)
    for i in range(60):
        x,y=map(int,rng.integers([15,15],[WIDTH-15,HEIGHT-15]))
        cv2.line(backdrop,(x-5,y-4),(x+5,y+4),(110,130,165),1)
    tiles={
        'analog':textile(rng,[(174,73,51),(255,223,142),(66,50,71),(236,150,89),(42,110,171)]),
        'other':textile(rng,[(46,110,139),(126,174,99),(210,211,175),(39,60,55),(175,80,185)]),
        'reference':textile(rng,[(88,110,44),(226,174,50),(52,88,103),(173,157,126)],54,71),
        'peer':textile(rng,[(66,48,104),(173,136,184),(247,220,100),(115,177,190)],66,83),
    }
    source=dest/'source.mkv'
    rows=[]; truth={}; frames=[]
    for k in range(end):
        pan, ref_x, peer_x, obscure = world(k)
        raw=backdrop.copy()
        places=[]
        for role, key, label,x,y,hidden in [('reference','reference','ref',ref_x,271,False),('peer','peer','peer',peer_x,261,k in obscure)]:
            if not hidden: places.append((role,key,label,x,y,False))
        for label, (identity, style, start, stop, x0, y0, velocity, weak) in episodes.items():
            if start<=k<stop:
                places.append(('candidate',style,label,x0+velocity*(k-start),y0,k in weak))
                truth[label]=identity
        obs=[]
        for role,key,label,x,y,low in places:
            tile=tiles[key]
            h,w=tile.shape[:2]
            assert 0 <= x+pan and x+pan+w<=WIDTH and 0<=y and y+h<=HEIGHT
            if low:
                tile = cv2.GaussianBlur(tile,(55,55),18)
            raw[y:y+h,x:x+w]=tile
            obs.append({'role':role,'episode':label,'native_xywh':[x+pan,y,w,h]})
        if not any(o['role']=='candidate' for o in obs):
            cv2.rectangle(raw,(170,77),(355,224),(87,99,110),-1)
        shifted=cv2.warpAffine(raw,np.array([[1,0,pan],[0,1,0]],np.float32),(WIDTH,HEIGHT),borderMode=cv2.BORDER_REFLECT)
        frames.append(shifted.tobytes())
        rows.append(obs)
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
             '-r',str(FPS),'-i','-','-an','-c:v','ffv1','-pix_fmt','bgr0',str(source)]
    subprocess.run(command,input=b''.join(frames),check=True,timeout=40)
    stream,clock=timeline(source)
    assert len(clock)==end
    observations=[]
    for row,items in zip(clock,rows):
        for obs in items:
            observations.append(obs | {k:row[k] for k in ('frame','pts','seconds')})
    manifest={'source':source.name,'video_sha256':digest(source),'native_size':[WIDTH,HEIGHT], 'time_base':stream['time_base'],
              'observations':observations,'sampled_pts':[x['pts'] for x in clock],
              'scope':'Renderer-provided native ROI and local appearances. Cross-appearance identity withheld.'}
    write(dest/'observations.json',manifest)
    return {'source_sha256':digest(source),'local_episode_instances':truth, 'renderer_clock':clock,
            'relationship_reference':'Analytic fixture motion except fully/partly unobserved intervals. Physical story is assessor-only.'}


def main():
    public = HERE/'inputs'; public.mkdir()
    sealed = HERE/'evaluation'; sealed.mkdir()
    # Three distinct appearances of the same printed surface, plus an independently placed copy.
    first={
        'g0':('i0','analog',0,7,61,102,10,()),
        'g1':('i1','analog',0,5,365,113,-7,()),
        'g2':('i0','analog',9,15,145,113,9,(11,)),
        'g3':('i2','analog',17,23,355,100,-12,()),
    }
    second={
        'g0':('i0','other',0,7,97,117,12,(4,)),
        'g1':('i1','analog',1,6,398,117,-7,()),
        'g2':('i0','other',10,16,212,126,-9,()),
        'g3':('i3','analog',17,23,115,104,10,()),
    }
    truth={
        'paired':fixture('paired',17001,first,lambda k:(0,62,320+(k//6)*8,(8,9,10,11)),24),
        'panning':fixture('panning',17057,second,lambda k:(k*3-20,78,336-2*k,(11,12,13)),24),
    }
    assert truth['paired']['source_sha256']!=truth['panning']['source_sha256']
    write(sealed/'renderer_truth.json',truth)


if __name__=='__main__':main()
