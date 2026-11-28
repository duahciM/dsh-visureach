"""Prepare two source-diverse inputs outside the prior DACF/PHASE1 cohorts."""
from pathlib import Path
import shutil

from dacf_research.prepare_inputs import ROOT, digest, sheet, write
from evidenweft_phase1.source_review import WITHDRAWN

HERE=Path(__file__).resolve().parent

def main():
    inputs=HERE/'independent'
    inputs.mkdir()
    seen={digest(p) for root in (ROOT/'dacf_research/inputs',ROOT/'evidenweft_phase1/prediction_inputs')
          for p in root.rglob('*') if p.is_file() and p.suffix in {'.mp4','.mkv'}}
    sources={'rolling':Path('C:/Users/Administrator/Desktop/视觉插件/素材/9.mp4'),
             'separate_fixture':ROOT.parent/'real-observation-v2/materials/rendered/r02_v2.mp4'}
    inventory={}
    for name, source in sources.items():
        sig=digest(source)
        assert sig not in seen and sig != WITHDRAWN
        path=inputs/f'{name}.mp4'
        shutil.copyfile(source,path)
        reviewed=HERE/'references'
        reviewed.mkdir(exist_ok=True)
        view=sheet(path,reviewed/f'{name}.png')
        assert view['sha256']==sig
        write(reviewed/f'{name}-source-timeline.json',view)
        inventory[name]={'source_path':str(source),'source_sha256':sig,'new_to_dacf_and_phase1':True,'frame_count':len(view['frames'])}
    write(HERE/'independent_sources.json',inventory)

if __name__=='__main__':
    main()
