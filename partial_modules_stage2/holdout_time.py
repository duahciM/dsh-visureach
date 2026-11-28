"""Check untouched source timepoints, with source-level counters for shared roots."""
from pathlib import Path
from dacf_research.prepare_inputs import digest,read,write
from partial_modules_stage1.study import pair,prepare
from partial_modules_stage2.inputs import H

if __name__=='__main__':
    cases={'control':('g0','g1'),'paired':('g2','g3'),'panning':('g0','g2')}
    rows=[]
    for name,(lo,hi) in cases.items():
        data=read(H/'scope_run'/name/'result.json')
        ref={r['pts']:r for r in data['receipts']}
        input_views=[row for rs in data['view_inventory'].values() for row in rs]
        left,right=([v for v in input_views if v['episode']==ep] for ep in (lo,hi))
        left.sort(key=lambda v:v['pts']);right.sort(key=lambda v:v['pts'])
        earlier={v['source_root'] for v in left[:2]+right[:2]}
        later=[left[-1],right[-1]]
        holdout={v['source_root'] for v in later}
        assert len(earlier)==4 and len(holdout)==2 and not (earlier&holdout)
        rasters=[]
        for item in later:
            f=ref[item['pts']];p=Path(f['path']);assert digest(p)==f['png_sha256']
            rasters.append(prepare(p))
        measure=pair(*rasters, *(v['native_xywh'] for v in later))
        rows.append({'case':name,'source_sha256':data['source_sha256'],'episodes':[lo,hi],'first_multi_prototype_roots':sorted(earlier),
                    'source_holdout_roots':sorted(holdout),'holdout_pts':[v['pts'] for v in later],
                    'target_and_context':measure,'physical_identity':'unknown from appearance validation'})
    write(H/'evaluation/independent_time_holdout.json',{'cases':rows,'scope':'Later independent frame roots; measurements never promote an identity or change the locked predictions.'})
