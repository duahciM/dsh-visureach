"""Read-only SHA256 freeze for the prediction inputs."""
from pathlib import Path
import argparse
from p1.io import dump
from if07_research.run import sources,sha
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
MEDIA=Path('C:/Users/Administrator/Desktop/视觉插件/素材')
CARD=MEDIA.parent/'video-keyframes-lightweight-checkpoint-IF09-next-IF10.md'


def history(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and
            not p.is_relative_to(HERE) and not p.is_relative_to(out) and not {'.venv','__pycache__'}.intersection(p.parts)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media-dir',type=Path,default=MEDIA)
    a=p.parse_args();o=a.out.resolve();o.mkdir(parents=True,exist_ok=True);f=o/'history_lock.json'
    if f.exists():raise SystemExit('Refusing to replace existing lock')
    new,old=sources(a.media_dir); files=history(o)
    dump(f,{'history':files,'historical_sources':old,'media':new,'card':{'path':str(CARD),'sha256':sha(CARD)}})
    print('Frozen',len(files),'files and',len(new),'directory sources',flush=True)

if __name__=='__main__':main()


