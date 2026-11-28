"""Verify tree lineage, cross-region graph roots, standalone query pixels and controls."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import Image

from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.verify_round2 import unpack
from dacf_round3.sampling import cumulative, ranked_gaps
from dacf_round3.setup import HERE, R2, verify
from evidenweft_phase1.source_review import native_stream, timeline


def numeric_controls():
    p = list(range(9))
    t = dict(zip(p,p))
    flat_then_steep = [1,1,1,1,2,2,0,0]
    values = dict(zip(zip(p,p[1:]),flat_then_steep))
    coord,_,_ = cumulative(p,t,lambda a,b:values.get((a,b)),1)
    pick,_ = ranked_gaps(coord,p,{0,1,2,3,4,8},t,p,[('source',0,8)],[1,2,3],False)
    assert pick['pts']==5
    coord,edges,_ = cumulative([0,1,2,3,4],t,lambda a,b:None if a==1 else 0.,1)
    assert coord[2][0]==1 and edges[1]['state']=='missing_or_inapplicable' and edges[0]['state']=='zero'
    pick,_ = ranked_gaps(coord,[0,1,2,3,4],{0,4},t,p,[('source',0,4)],[1,2,3],False)
    assert pick is None
    return ['sparser of two equal cumulative-change sections is selected', 'static and missing remain separate; no false path interpolation']


def links(run, case, cfg, nodes):
    stamps = defaultdict(set)
    for node in nodes.values():
        if node['kind']=='frozen_r2_region':
            for s in node['source_points']:
                stamps[s['pts']].add(node['id'])
    actual = read(run/case/'pair_roots.json')
    if cfg['cohort']!='new':
        folder=R2/('replay' if cfg['cohort']=='replay' else 'independent_execution/independent')/case
        actual += unpack(folder/'pair_evidence.json.gz')
    output = {}
    for row in actual:
        if not row['sift']['verified']:
            continue
        a,b = row['pts']
        for first in sorted(stamps[a]):
            for second in sorted(stamps[b]-{first}):
                output[(first,second,row['root'])] = {'nodes':[first,second],'type':'saved_direct_geometry_comparison',
                                                      'pts':[a,b],'root':row['root'],'sift_inliers':row['sift']['inlier_count']}
    for p,ids in stamps.items():
        ids=sorted(ids)
        for i,first in enumerate(ids):
            for second in ids[i+1:]:
                output[(first,second,p)]={'nodes':[first,second],'type':'shared_actual_source_sample','pts':[p],
                                          'root':next(s['root'] for s in nodes[first]['source_points'] if s['pts']==p)}
    return list(output.values())


def extra_originals(lock, case, observations):
    source = ROOT/lock['cases'][case]['source']
    info, clock = timeline(source)
    seen = set()
    for row,bits in native_stream(source,info,clock):
        for ref in observations.get(row['pts'],[]):
            assert row['frame']==ref['source_frame'] and sha256(bits).hexdigest()==ref['rgb_sha256']
            assert digest(Path(ref['path']))==ref['png_sha256'] and info['time_base']==ref['time_base']
            seen.add(ref['path'])
    assert seen=={r['path'] for items in observations.values() for r in items}
    return len(seen)


def main():
    lock=read(HERE/'lock.json')
    verify(lock)
    run=HERE/'run'
    status=[]
    for case,cfg in lock['cases'].items():
        folder=run/case
        tree=read(folder/'tree-v2.json')
        nodes=tree['nodes']
        originals=read(folder/'reference_originals.json')|read(folder/'receipts.json')
        for key,n in nodes.items():
            a,b=n['bounds_pts']
            assert key==n['id'] and all(a<=s['pts']<=b and originals[str(s['pts'])]['id']==s['root'] for s in n['source_points'])
            for child in n['children']:
                c=nodes[child]
                assert c['parent_id']==key and c['level']==n['level']+1 and a<=c['bounds_pts'][0]<=c['bounds_pts'][1]<=b
        graph=links(run,case,cfg,nodes)
        write(folder/'region_graph.json',{'edges':graph,'scope':'nonhierarchical evidence joins, not identity or independent votes'})
        status.append({'case':case,'nodes':len(nodes),'partitions':sum(len(r.get('refinements',[])) for r in nodes.values()),
                       'nonhierarchical_graph_links':len(graph)})
    queries = []
    for case,name in [('doorway','doorway-query'),('new_laptop','laptop-roi-query')]:
        data=read(HERE/f'{name}.json')
        receipts=data['originals']
        assert all(digest(Path(r['path']))==r['png_sha256'] for r in receipts)
        if 'derived_crop' in data:
            crop=data['derived_crop']
            r=next(row for row in receipts if row['pts']==crop['pts'])
            with Image.open(r['path']) as source, Image.open(crop['path']) as derived:
                x,y,w,h=crop['native_xywh']
                assert derived.tobytes()==source.crop((x,y,x+w,y+h)).tobytes()
        queries.append({'case':case,'node':data['node']['id'],'originals_resolved':len(receipts),'without_new_decode':True,
                        'elapsed_after_loading_index_seconds':data['lookup_seconds_excluding_tree_load']})
    manual={}
    for case,files in [('doorway',['doorway-temporal/temporal.json','doorway-spatial/spatial.json']),('new_laptop',['laptop-spatial/spatial.json'])]:
        observations=defaultdict(list)
        for name in files:
            datum=read(HERE/name)
            for r in datum['source_originals']:
                observations[r['pts']].append(r)
        manual[case]=extra_originals(lock,case,observations)
    verify(lock)
    write(HERE/'validation.json',{'study_lock_unchanged':True,'tree_checks':status,'query_checks':queries,'manual_rgb_validation':manual,
                                 'numerical_controls':numeric_controls()})
    print('PASS',len(status),'trees,',sum(c['partitions'] for c in status),'original-backed temporal partitions; manual source and crop joins')


if __name__ == '__main__':
    main()
