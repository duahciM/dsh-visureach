"""Generate the unchanged policy's actual catalogue and AMR plan for the new control."""
import subprocess
import sys
from time import perf_counter

import cv2
from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_research.run import raw_frames
from dacf_round4.offline import Experiment,clip_encoder
from dacf_round5.setup import nodes
from evidenweft_phase1.source_review import timeline
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    dest=H/'plan'; dest.mkdir()
    source=H/'inputs/control/source.mkv';index=dest/'index'
    started=perf_counter()
    subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),'--video',str(source),
                    '--out',str(index),'--fps','5','--keep-analysis'],check=True)
    manifest=read(index/'manifest.json');stream,clock=timeline(source)
    raw_frames(source,stream,clock,{r['source_frame'] for r in manifest['pool']},dest,digest(source),320)
    setup=dest/'diagnostics';setup.mkdir()
    row={k:v.relative_to(ROOT).as_posix() for k,v in {'source':source,'index':index,'pixels':dest}.items()}|{'cohort':'new'}
    cv2.setNumThreads(1)
    task=Experiment('scoped-control',row,clip_encoder(),read(ROOT/'dacf_research/protocol.json'),setup)
    tree,edges=nodes(task,read(ROOT/'dacf_round4/protocol.json'))
    write(dest/'tree.json',{'sha256':digest(source),'time_base':stream['time_base'],'width':stream['width'],'height':stream['height'],
                            'clock':clock,'nodes':tree,'edges':edges})
    write(dest/'plan_receipt.json',{'source_sha256':digest(source),'catalog_sha256':digest(index/'catalog.json'),
           'nomination_pts':sorted(task.initial),'pool_pts':task.coarse,'original_policy_files':{
            name:digest(ROOT/name) for name in ('dacf_round4/protocol.json','dacf_research/protocol.json','dacf_round5/setup.py')},
           'build_seconds':perf_counter()-started})
