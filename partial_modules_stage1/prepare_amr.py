"""Build the existing DACF-AMR plan on the unchanged new research source."""
import subprocess
import sys
from argparse import ArgumentParser
from time import perf_counter

import cv2
from dacf_research.freeze import PLUGIN
from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_research.run import raw_frames
from dacf_round4.offline import Experiment,clip_encoder
from dacf_round5.setup import nodes
from evidenweft_phase1.source_review import timeline
from partial_modules_stage1.fixtures import HERE


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--output',default='amr_plan'); args=parser.parse_args()
    scope=HERE/args.output
    if scope.parent!=HERE:
        raise ValueError('Choose an output directory name inside the study.')
    scope.mkdir()
    source=HERE/'inputs/panning/source.mkv';index=scope/'index'
    start=perf_counter();cv2.setNumThreads(1)
    subprocess.run([sys.executable,'-B','-X','utf8',str(PLUGIN/'python/index_video.py'),'--video',str(source),
                    '--out',str(index),'--fps','5','--keep-analysis'],check=True)
    meta=read(index/'manifest.json'); stream,clock=timeline(source)
    raw_frames(source,stream,clock,{row['source_frame'] for row in meta['pool']},scope,digest(source),320)
    row={k:v.relative_to(ROOT).as_posix() for k,v in {'source':source,'index':index,'pixels':scope}.items()}|{'cohort':'new'}
    stage=scope/'diagnostics';stage.mkdir()
    task=Experiment('module-pan',row,clip_encoder(),read(ROOT/'dacf_research/protocol.json'),stage)
    tree,edges=nodes(task,read(ROOT/'dacf_round4/protocol.json'))
    write(scope/'plan.json',{'sha256':digest(source),'time_base':stream['time_base'],'width':stream['width'],
                             'height':stream['height'],'clock':clock,'nodes':tree,'edges':edges})
    write(scope/'cost.json',{'seconds':perf_counter()-start,'frozen_first_nominees':len(task.initial),
                             'coarse_pool_count':len(task.coarse),'source_pts_count':len(clock),'node_count':len(tree)})
