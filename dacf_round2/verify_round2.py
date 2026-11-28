"""Verify preservation, raw source joins, region signals and retained catalogue."""
import gzip
import json

from dacf_research.freeze import validate
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.frozen_inputs import HERE


def unpack(path):
    with gzip.open(path,'rt',encoding='utf-8') as file:
        return json.load(file)


def main():
    lock=read(HERE/'lock.json')
    round1=validate('manual')
    assert round1['baseline']==lock['previous_baseline']
    assert digest(ROOT/'plugin_offline/release/local-dsh-video-observer-0.6.0.tgz')==round1['baseline']['visureach_archive_sha256']
    for group in ('inputs','preserved_round1_outputs','reference_hashes_scoring_only'):
        assert all(digest(ROOT/p)==v for p,v in lock[group].items()),group
    assert digest(HERE/'lock.json')==digest(HERE/'independent_execution/lock.json')
    report=[]
    for name,c in lock['cases'].items():
        work=HERE/('replay' if c['cohort']=='replay' else 'independent_execution/independent')/name
        source=read(ROOT/c['index']/'manifest.json')
        original=read(work/'regions.json')
        regions={r['id']:r for r in original['regions']}
        assert original['nomination_pts']==[r['pts'] for r in source['pool'] if r['in_catalog']]
        proof=unpack(work/'region_signals.json.gz')
        pair={r['root'] for r in unpack(work/'pair_evidence.json.gz')}
        local={r['root'] for r in unpack(work/'local_holdout.json.gz')}
        assert set(regions)=={r['region_id'] for r in proof}
        for p in proof:
            assert set(p['direct_group_pair_roots'])<=pair
            for row in p['rows']:
                for a in (row['left'],row['right']):
                    if a: assert set(a['evidence_roots'])<=pair
                if row['local']:
                    assert set(row['local']['roots'])<=local
            assert all(regions[p['region_id']]['bounds_pts'][0]<=x['pts']<=regions[p['region_id']]['bounds_pts'][1] for x in p['rows'])
        case=HERE/'evaluation'/name
        score=read(case/'score.json')
        reread=read(case/'actual_original_receipts.json')
        kept=[r for r in reread if r['from_paired_region']]
        assert len(kept)==score['actual_unique_native_reread']
        assert all(r['source_sha256']==source['video_sha256'] and digest(case/r['source_png'])==r['png_sha256'] for r in reread)
        report.append({'case':name,'frozen_catalogue_kept':len(original['nomination_pts']),'pair_roots':len(pair),
                       'held_out_roots':len(local),'region_records':len(proof),'reread_originals':len(kept),
                       'separate_reference_or_control_images':len(reread)-len(kept)})
    write(HERE/'verification.json',{'status':'passed','cases':report,'published_implementations_changed':False})
    print('PASS',len(report),'cases',sum(r['reread_originals'] for r in report),'reread native receipts',flush=True)

if __name__=='__main__':
    main()
