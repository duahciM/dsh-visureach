"""Independently re-decode original RGB and join every output root to source PTS."""
from hashlib import sha256
import gzip
import json

from PIL import Image

from dacf_research.freeze import validate
from dacf_research.prepare_inputs import HERE, ROOT, digest, read, write
from evidenweft_phase1.source_review import native_stream, timeline


def unpack(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return json.load(f)


def main():
    lock=validate('manual')
    validate('automatic')
    assert digest(ROOT/'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz')==lock['baseline']['visureach_archive_sha256']
    report=[]
    for clip in sorted((HERE/'inputs').iterdir()):
        case=clip.stem
        stream, ref=timeline(clip)
        source_sha=digest(clip)
        decoded=[]
        arms={}
        for arm in ('manual','automatic'):
            folder=HERE/f'run_{arm}'/case
            ledger=read(folder/'frame_receipts.json')
            rows={r['frame']:r for r in ledger}
            all_ids={r['id']:r for r in ledger}
            pair_data=unpack(folder/'pair_evidence.json.gz')
            local=unpack(folder/'local_holdout.json.gz')
            direct={r['root'] for r in pair_data}
            scatter={r['root'] for r in local}
            found=set(rows)
            for r in pair_data:
                assert r['pts']==[all_ids[id]['pts'] for id in r['frames']]
                assert r['root']=='>'.join(r['frames'])
            for entry in read(folder/'pair_results.json'):
                assert set(entry['ab_group_pair_roots']+entry['adjacent_geometry_roots']) <= direct
                for row in entry['rows']:
                    assert all_ids[row['frame_root']]['pts']==row['pts']
                    assert set(row['a']['evidence_roots']+row['b']['evidence_roots']) <= direct
                    assert set(row['local']['roots']) <= scatter
            decoded.append((folder,rows))
            arms[arm]={'source_receipts':len(found),'pair_roots':len(direct),'point_holdout_roots':len(scatter)}
        needed=set.union(*(set(r) for _,r in decoded))
        checked=set()
        for ref_row,bits in native_stream(clip,stream,ref):
            rank=ref_row['frame']
            if rank not in needed:
                continue
            rgb=sha256(bits).hexdigest()
            for folder, rows in decoded:
                if rank not in rows:
                    continue
                r=rows[rank]
                name=f'{source_sha}:{rank}:{ref_row["pts"]}:{stream["time_base"]}'
                assert r['id']==name and r['pts']==ref_row['pts'] and r['rgb_sha256']==rgb
                p=folder/r['png']
                assert digest(p)==r['png_sha256']
                with Image.open(p) as image:
                    assert image.size==(stream['width'],stream['height'])
                    assert sha256(image.convert('RGB').tobytes()).hexdigest()==rgb
            checked.add(rank)
        assert checked==needed
        report.append({'case':case,'source_sha256':source_sha,'independently_redecoded_selected_originals':len(checked),
                       'native_frame_count':len(ref),'arms':arms})
    write(HERE/'source_verification.json',{'cases':report,'all_checks':'passed',
       'total_unique_native_pts_by_case':sum(r['independently_redecoded_selected_originals'] for r in report),
       'total_output_original_receipts':sum(a['source_receipts'] for r in report for a in r['arms'].values())})
    print('PASS',len(report),'sources',sum(r['independently_redecoded_selected_originals'] for r in report),'unique original PTS')


if __name__=='__main__':
    main()
