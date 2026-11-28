"""Replay the prespecified alternating ROI error by sampled ordinal rather than native frame parity."""
import cv2
from dacf_research.prepare_inputs import read,write
from partial_modules_stage2.scoped import H,freeze_known,perturb,save_case
from partial_modules_stage2.evaluate import controlled_error,source_audit

if __name__=='__main__':
    original=read(H/'scope_run/control-input/observations.json')
    source=H/'inputs/control/source.mkv'
    stage=H/'roi_jitter_retry'; stage.mkdir()
    rank={p:i for i,p in enumerate(original['sampled_pts'])}
    actuals=original['observations']
    # The first run used native-index parity; every fourth source frame has the same parity.
    alternated=perturb([r|{'frame':rank[r['pts']]} for r in actuals],original['native_size'])
    for measured,actual in zip(alternated,actuals):measured['frame']=actual['frame']
    freeze_known(original,stage/'inputs',source,alternated)
    cv2.setNumThreads(1);cv2.setRNGSeed(628)
    save_case(stage/'inputs',stage/'prediction')
    result=read(stage/'prediction/result.json')
    proof=source_audit([result['receipts']],[source])
    write(stage/'comparison.json',{'correction':'Parity over the scheduled sample ordinal, not native frame index. First run preserved.',
          'expected_displacements':'Same locked ±6 px / ∓4 px, inward 4 px per edge.',
          'associated_local_edges':len(result['scoped']['relations']),
          'reference_axis_error':controlled_error(result,actuals),'source_proof':proof})
