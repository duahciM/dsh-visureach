"""Confirm the installed research bundle carries the exact released baseline bytes."""
from hashlib import sha256
import argparse
import tarfile

from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_round5.setup import HERE,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixed',action='store_true');args=parser.parse_args()
    lock=read(HERE/'lock.json');verify(lock)
    old=ROOT/'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz'
    expected=read(ROOT/'plugin_offline/release/verification.json')
    assert digest(old)==expected['sha256']
    release=HERE/('release-fixed' if args.fixed else 'release'); archive=next(release.glob('*.tgz')); path=release/'installed'/'package'
    with tarfile.open(old) as baseline,tarfile.open(archive) as bundle:
        frozen={m.name.removeprefix('package/'):sha256(baseline.extractfile(m).read()).hexdigest() for m in baseline if m.isfile()}
        bundled={m.name.removeprefix('package/baseline/'):sha256(bundle.extractfile(m).read()).hexdigest() for m in bundle if m.isfile() and m.name.startswith('package/baseline/')}
        assert len(frozen)==expected['file_count'] and frozen==bundled
        assert all(digest(path/'baseline'/f)==sig for f,sig in frozen.items())
    assert digest(path/'entry.mjs')==digest(HERE/'entry.mjs')
    if args.fixed: assert digest(path/'transport.mjs')==digest(HERE/'transport.mjs')
    entry='transport.mjs' if args.fixed else 'entry.mjs'
    write(release/'verification.json',{'baseline_release_sha256':digest(old),'baseline_files_unchanged':len(frozen),
          'standalone_archive':archive.name,'archive_sha256':digest(archive),'installed_entry_sha256':digest(path/entry),
          'official_default_configuration_changed':False,'scope':'Offline study configuration with local source/plan binding'})


if __name__=='__main__':main()
