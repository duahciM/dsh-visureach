"""Separate input/implementation hashes from the withheld role evaluation."""
from pathlib import Path
import subprocess

from dacf_research.prepare_inputs import HERE, ROOT, digest, read, write
from evidenweft_phase1.prepare import verify


PLUGIN = ROOT/'evidenweft_phase1/frozen/visureach'


def main():
    frozen = verify()
    repo=ROOT/'plugin_offline/github/EvidenWeft'
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],encoding='utf-8').strip()
    assert head == frozen['baselines']['evidenweft_main']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
    assert digest(ROOT/'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz') == frozen['baselines']['visureach_archive_sha256']
    model=ROOT/'p215_discovery_20261121/assets'
    install=ROOT/'if13_research_20261120/runtime_onnx'
    inputs = [*sorted((HERE/'inputs').iterdir()), *(HERE/f for f in ('protocol.json','prepare_inputs.py','audit_phase1.py','freeze.py',
                'field.py','run.py','phase1_attribution.json')), model/'vision.onnx']
    files={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    eval_paths=sorted(p for p in (HERE/'evaluation_only').rglob('*') if p.is_file())
    write(HERE/'lock.json',{
       'baseline':frozen['baselines'],'reference_study_lock':digest(ROOT/'evidenweft_phase1/protocol_lock.json'),
       'prediction_dependencies':files,'manual_anchor_input_sha256':digest(HERE/'manual_anchors.json'),
       'evaluation_hashes': {p.relative_to(HERE).as_posix():digest(p) for p in eval_paths},
       'embedding_runtime':str(install),'note':'Lock before DACF prediction. Auto process never reads roles, construction, review images, or manual anchors.'
    })


def validate(arm):
    l=read(HERE/'lock.json')
    assert digest(ROOT/'evidenweft_phase1/protocol_lock.json')==l['reference_study_lock']
    assert all(digest(ROOT/rel)==sig for rel,sig in l['prediction_dependencies'].items())
    if arm=='manual':
        assert digest(HERE/'manual_anchors.json')==l['manual_anchor_input_sha256']
    return l


if __name__=='__main__':
    main()
