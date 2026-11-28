"""Re-run the existing alternating perturbation on the original source, with separate ROI/fit diagnostics."""
from collections import Counter
from statistics import median
from pathlib import Path

import cv2
import numpy as np
from dacf_research.prepare_inputs import ROOT,read,write
from partial_modules_stage2.scoped import freeze_known, perturb, save_case
from partial_modules_stage2.evaluate import controlled_error
from partial_modules_stage3.material import H
from partial_modules_stage1.study import centroid

if __name__=='__main__':
    dest=H/'roi_replay';dest.mkdir()
    source=ROOT/'partial_modules_stage2/inputs/control/source.mkv'
    manifest=read(ROOT/'partial_modules_stage2/scope_run/control-input/observations.json')
    original=manifest['observations'];rank={p:i for i,p in enumerate(manifest['sampled_pts'])}
    varied=perturb([v|{'frame':rank[v['pts']]} for v in original],manifest['native_size'])
    diagnostics=[]
    for b,a in zip(varied,original):
        b['frame']=a['frame']
        if a['role']=='candidate':
            u=np.array(a['native_xywh']);v=np.array(b['native_xywh'])
            diagnostics.append({'pts':a['pts'],'role':a['role'],'before_bbox':a['native_xywh'],'after_bbox':b['native_xywh'],
                'center_displacement':(centroid(v)-centroid(u)).tolist(),
                'boundary_displacement':(np.r_[v[:2],v[:2]+v[2:]]-np.r_[u[:2],u[:2]+u[2:]]).tolist()})
    freeze_known(manifest,dest/'inputs',source,varied)
    cv2.setNumThreads(1);cv2.setRNGSeed(628)
    save_case(dest/'inputs',dest/'run')
    output=read(dest/'run/result.json')
    fixed=read(ROOT/'partial_modules_stage2/roi_jitter_retry/comparison.json')
    score=controlled_error(output,original)
    assert score['axes']==fixed['reference_axis_error']['axes']
    write(dest/'proof.json',{'reproduces_frozen_phase2_axis_errors':True,'source_sha256':output['source_sha256'],
          'box_diagnostics':diagnostics,'scope_association_count':len(output['scoped']['relations']),
          'local_fit_states':dict(Counter(r['local']['state'] for r in output['computed_pairs'])),
          'background_holdout_states':dict(Counter(c['state'] for c in output['context_calculations'])),
          'relationship_axis_errors':score['axes']})
