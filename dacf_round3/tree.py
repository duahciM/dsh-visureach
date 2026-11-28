"""Queryable immutable lineage and optional exact-source, spatial ROI refinement."""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank, METRICS, clip_encoder
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.investigate import load_rasters
from dacf_round3.acquire import Reader
from dacf_round3.sampling import cumulative
from dacf_round3.setup import HERE, verify
from evidenweft_phase1.source_review import timeline
from local_observation import pair, prepare
from research_bridge import incremental_figure


def compile_tree(folder, tree_name):
    base = read(folder/'initial.json')
    originals = read(folder/'reference_originals.json') | read(folder/'receipts.json')
    times = dict(zip(base['source_pts'],base['source_seconds'],strict=True))
    nominations = set(base['initial_pts'])
    regions = {r['id']:r for r in base['r2_plan']}
    parents = {child:r['id'] for r in regions.values() for child in r['children']}
    nodes = {}
    links = []
    coarse = set(base['coarse_pts'])

    def node(key, parent, bounds, level, samples, roi=None, kind='time', evidence=None):
        lo,hi = bounds
        in_scope = sorted(p for p in samples if lo<=p<=hi)
        refs = [{'pts':p,'root':originals[str(p)]['id']} for p in in_scope if str(p) in originals]
        size = originals[str(base['coarse_pts'][0])]['native_size']
        nodes[key] = {'id':key,'parent_id':parent,'children':[],'kind':kind,'bounds_pts':bounds,
                      'bounds_seconds':[times[lo],times[hi]],'time_base':base['time_base'],'level':level,
                      'time_resolution_max_sample_gap_seconds':max((times[b]-times[a] for a,b in zip(in_scope,in_scope[1:])),default=None),
                      'space_native_xywh':roi or [0,0,*size],'measurement_raster_width':320,
                      'source_points':refs,'nominated_pts':sorted(nominations & set(in_scope)),
                      'applicability':'sampled_only' if len(in_scope)>1 else 'insufficient_comparison',
                      'unexplained_time_ranges_pts':[[a,b] for a,b in zip(in_scope,in_scope[1:]) if bisect_left(base['source_pts'],b)-bisect_left(base['source_pts'],a)>1],
                      'evidence':evidence or {}}
        if parent:
            nodes[parent]['children'].append(key)
        return nodes[key]

    source = [base['source_pts'][0],base['source_pts'][-1]]
    node('source',None,source,0,coarse,kind='frozen_source')
    for key,r in regions.items():
        parent = parents.get(key,'source')
        node(key,parent,r['bounds_pts'],nodes[parent]['level']+1,coarse,kind='frozen_r2_region',
             evidence={k:r.get(k) for k in ('kind','left','right','distinction','weak_cues','continuity')})
    touched = set()
    for route in read(folder/'routes.json'):
        prefix = route['arm']+'-'+(route['channel'] or 'time')
        samples = coarse.copy()
        active = {}
        for step in route['steps']:
            p = step['pts']
            origin = step['root']
            if origin not in active:
                identifier = f'{prefix}/{origin}'
                node(identifier,origin,nodes[origin]['bounds_pts'],nodes[origin]['level']+1,samples,kind='route_view')
                active[origin] = [identifier]
            key = next((k for k in active[origin] if nodes[k]['bounds_pts'][0] < p < nodes[k]['bounds_pts'][1]),None)
            if key is None:
                key = active[origin][0]
                links.append({'type':'boundary_probe','node':key,'pts':p,'root':step['receipt']})
                continue
            parent = nodes[key]
            lo,hi = parent['bounds_pts']
            samples.add(p)
            pivot = f'{prefix}/{origin}/{step["ordinal"]}'
            parent.setdefault('refinements',[]).append({'actual_pts':p,'source_root':step['receipt'],'trace':step,
                                                       'state':'sampling_partition; distinguishable_state_unconfirmed'})
            children = [node(pivot+side,key,bounds,parent['level']+1,samples,evidence={'acquisition_root':step['receipt']})['id']
                        for side,bounds in [('L',[lo,p]),('R',[p,hi])]]
            active[origin].remove(key)
            active[origin].extend(children)
            touched.add(origin)
            links.append({'type':'sample_split_shared_source','nodes':children,'pts':p,'root':step['receipt']})
    write(folder/tree_name,{'nodes':nodes,'nonhierarchical_links':links,'touched_frozen_regions':sorted(touched-{'source'})})


def spatial(experiment, node, pts, box, result):
    started = perf_counter()
    source = ROOT/experiment['source']
    stream, clock = timeline(source)
    real = {r['pts']:r for r in clock}
    lo,hi = node['bounds_pts']
    if len(set(pts))<2 or not all(lo<=p<=hi and p in real for p in pts):
        raise ValueError('Choose at least two different real PTS within the node.')
    x,y,w,h = box
    if min(x,y)<0 or min(w,h)<2 or x+w>stream['width'] or y+h>stream['height']:
        raise ValueError('ROI must be at least two pixels wide/high and inside the original.')
    result.mkdir()
    manifest = read(ROOT/experiment['index']/'manifest.json')
    assert digest(source)==manifest['video_sha256']
    decoder = Reader(source,manifest['video_sha256'],stream,clock,result/'source_originals')
    images = [decoder.read(p)[0] for p in sorted(set(pts))]
    prepared = [prepare(image['path']) for image in images]
    cells = [[x,y,w//2,h//2],[x+w//2,y,w-w//2,h//2],[x,y+h//2,w//2,h-h//2],
             [x+w//2,y+h//2,w-w//2,h-h//2]]
    ranges = []
    for i,roi in enumerate(cells):
        start = images[0]
        relations = []
        for j,image in enumerate(images[1:],1):
            comparison = pair(prepared[0],prepared[j],roi,roi)
            affine = comparison['excluded_context']
            root = f'{start["id"]}>{image["id"]}:roi:{i}'
            context = None
            if affine['state']=='heldout_supported':
                raster = comparison['coordinate_space']
                scale = np.diag([raster['width']/stream['width'],raster['height']/stream['height'],1])
                matrix = np.vstack((affine['affine'],[0,0,1]))
                native = (np.linalg.inv(scale)@matrix@scale)[:2]
                with Image.open(start['path']) as raw:
                    pixels = np.asarray(raw)
                width,height = stream['width'],stream['height']
                # Align only the requested destination region; retain a strict overlap mask.
                rx,ry,rw,rh = roi
                adjusted = native - np.array([[0,0,rx],[0,0,ry]])
                warped = cv2.warpAffine(pixels,adjusted,(rw,rh))
                mask = cv2.warpAffine(np.ones((height,width),np.float32),adjusted,(rw,rh)) > .999
                path = result/f'roi{i+1}-{image["pts"]}-mapped.png'
                mask_path = result/f'roi{i+1}-{image["pts"]}-overlap.png'
                Image.fromarray(warped).save(path)
                Image.fromarray(mask.astype('uint8')*255).save(mask_path)
                context = {'native_affine':native.tolist(),'destination_roi_native_xywh':roi,
                           'derived_warped_parent':{'file':path.name,'sha256':digest(path),'parent':start['id']},
                           'derived_overlap_mask':{'file':mask_path.name,'sha256':digest(mask_path),'valid_pixels':int(mask.sum())},
                           'scope':'heldout 2D background coordinates; target may disagree; no physical persistence'}
            relations.append({'root':root,'originals':[start['id'],image['id']], 'pts':[start['pts'],image['pts']],
                              'common_coordinates':context,'comparison':comparison})
        anchored = {images[0]['pts']:0.}
        projected = {images[0]['pts']:0.}
        for row in relations:
            residual = row['comparison']['residual']
            anchored[row['pts'][1]] = residual.get('target_mad_8bit') if row['common_coordinates'] else None
            projected[row['pts'][1]] = residual.get('target_projection_median_px') if row['common_coordinates'] else None
        times = {r['pts']:r['seconds'] for r in images}
        def coords(data):
            positions,edges,total = cumulative(list(times),times,lambda a,b:abs(data[a]-data[b]) if None not in (data[a],data[b]) else None,float('inf'))
            return {'coordinates':positions,'edges':edges,'total':total,'signed_to_common_reference':data,
                    'scope':'sampled ROI scalar relative to same held-out context; unobserved interior remains unknown'}
        ranges.append({'id':f'{node["id"]}/roi{i+1}','parent_id':node['id'],'children':[],'kind':'space','level':node['level']+1,
                       'bounds_pts':[lo,hi],'bounds_seconds':node['bounds_seconds'],'measurement_resolution':prepared[0][0].shape[::-1],
                       'time_resolution_max_sample_gap_seconds':max(b['seconds']-a['seconds'] for a,b in zip(images,images[1:])),
                       'time_base':manifest['time_base'],'space_native_xywh':roi,
                       'source_points':[{'pts':r['pts'],'root':r['id']} for r in images],
                       'unexplained_time_ranges_pts':[[a['pts'],b['pts']] for a,b in zip(images,images[1:]) if b['frame']-a['frame']>1],
                       'comparisons':relations,'state_roi_coordinate_mad_luma':coords(anchored),
                       'projected_relation_coordinate_px':coords(projected),
                       'applicability':'heldout_supported_in_some_samples' if any(r['common_coordinates'] for r in relations) else 'unknown_no_heldout_context'})
    graph = incremental_figure({'source':{'sha256':manifest['video_sha256'],'probe':stream},'frames':images,'pts':images[-1]['pts'],
                                'roi_roots':[r['id'] for r in ranges],'regions_native':cells,'evidence_id':images[-1]['id']})
    write(result/'spatial.json',{'parent':node['id'],'children':ranges,'source_originals':images,'figure1_2':graph,
                                'cross_node_links':[{'type':'projected_figure2','relation':r} for r in graph['figure2']['relations']],
                                'wall_seconds':round(perf_counter()-started,6)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--tree-name',default='tree.json')
    parser.add_argument('--case')
    parser.add_argument('--node')
    parser.add_argument('--spatial',nargs=4,type=int)
    parser.add_argument('--pts',nargs='+',type=int)
    parser.add_argument('--temporal',action='store_true')
    parser.add_argument('--overlay',action='append',type=Path,default=[])
    parser.add_argument('--crop-pts',type=int)
    parser.add_argument('--include-relations',action='store_true')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if not args.case:
        for folder in sorted(args.run.iterdir()):
            if (folder/'routes.json').is_file():
                compile_tree(folder,args.tree_name)
        return
    folder = args.run/args.case
    tree = read(folder/args.tree_name)
    receipts = read(folder/'reference_originals.json') | read(folder/'receipts.json')
    overlay_relations = {}
    for overlay in args.overlay:
        addition = read(overlay)
        parent = tree['nodes'][addition['parent']]
        for row in addition['source_originals']:
            assert row['source_sha256'] == read(folder/'initial.json')['source_sha256']
            receipts[str(row['pts'])] = row
        for node in addition['children']:
            if node['id'] in tree['nodes']:
                raise ValueError(f'Duplicate node ID: {node["id"]}')
            assert node['parent_id'] == parent['id']
            node.setdefault('children',[])
            parent['children'].append(node['id'])
            tree['nodes'][node['id']] = node
            overlay_relations[node['id']] = addition.get('cross_node_links',[])
    node = tree['nodes'][args.node]
    if args.spatial or args.temporal:
        lock = read(HERE/'lock.json')
        verify(lock)
        cv2.setNumThreads(1)
        if args.spatial:
            spatial(lock['cases'][args.case],node,args.pts,args.spatial,args.out)
        else:
            temporal(lock['cases'][args.case],node,args.out)
        return
    started = perf_counter()
    response = {'node':node,'children':[tree['nodes'][key]['id'] for key in node['children']],
                'originals':[receipts[str(r['pts'])] for r in node['source_points']],
                'query_mode':'metadata and exact original paths without sweep decode'}
    if args.include_relations:
        roots = {row['root'] for row in node['source_points']}
        edges = read(folder/'region_graph.json')['edges']
        response['existing_nonhierarchical_relations'] = [e for e in edges if args.node in e['nodes'] or e['root'] in roots]
        response['overlay_projected_relations'] = overlay_relations.get(args.node,[])
    if args.crop_pts is not None:
        if not args.out or args.crop_pts not in {r['pts'] for r in node['source_points']}:
            raise ValueError('Crop requires an output receipt and a source PTS already indexed by the node.')
        source = receipts[str(args.crop_pts)]
        path = args.out.with_name(args.out.stem+'-crop.png')
        if path.exists():
            raise FileExistsError(path)
        x,y,w,h = node['space_native_xywh']
        with Image.open(source['path']) as im:
            im.crop((x,y,x+w,y+h)).save(path)
        response['derived_crop'] = {'source_root':source['id'],'native_xywh':[x,y,w,h],
                                    'path':str(path.resolve()),'sha256':digest(path),'pts':args.crop_pts}
    response['lookup_seconds_excluding_tree_load'] = round(perf_counter()-started,6)
    if args.out:
        write(args.out,response)
    else:
        print(json.dumps(response,ensure_ascii=False))


def temporal(experiment, node, result):
    source = ROOT/experiment['source']
    stream, clock = timeline(source)
    manifest = read(ROOT/experiment['index']/'manifest.json')
    lo,hi = node['bounds_pts']
    inner = [r for r in clock if lo<r['pts']<hi]
    if not inner:
        raise ValueError('Node has no interior original at the source cadence.')
    result.mkdir()
    decoder = Reader(source,manifest['video_sha256'],stream,clock,result/'source_originals')
    points = [lo,inner[len(inner)//2]['pts'],hi]
    images = [decoder.read(p)[0] for p in points]
    raw = {}
    for r in images:
        with Image.open(r['path']) as image:
            raw[r['pts']] = r | {'rgb':np.asarray(image.resize((320,max(8,round(image.height*320/image.width))),Image.Resampling.BILINEAR))}
    bank = Bank(raw,read(ROOT/'dacf_research/protocol.json'),clip_encoder())
    old = bank.metrics_on_pair(lo,hi)
    children = []
    for i,(a,b) in enumerate(zip(points,points[1:]),1):
        item = {'id':f'{node["id"]}/manualT{i}','parent_id':node['id'],'level':node['level']+1,'kind':'time',
                'bounds_pts':[a,b],'bounds_seconds':[raw[a]['seconds'],raw[b]['seconds']],
                'time_base':manifest['time_base'],'space_native_xywh':node['space_native_xywh'],
                'source_points':[{'pts':p,'root':raw[p]['id']} for p in (a,b)],'analysis_raster':list(raw[a]['rgb'].shape[:2][::-1]),
                'time_resolution_max_sample_gap_seconds':raw[b]['seconds']-raw[a]['seconds'],
                'comparison':bank.metrics_on_pair(a,b),'applicability':'sampled_endpoint_measurement',
                'unexplained_time_ranges_pts':[[a,b]] if raw[b]['frame']-raw[a]['frame']>1 else []}
        children.append(item)
    write(result/'temporal.json',{'parent':node['id'],'children':children,'source_originals':images,'old_chord':old,
                                 'revised_totals':{ch:sum(c['comparison']['metrics'][ch] for c in children)
                                                   if all(c['comparison']['metrics'][ch] is not None for c in children) else None for ch in METRICS},
                                 'scope':'Original time probe. Spatial grid was not partitioned, intermediate physical state remains unknown.'})


if __name__ == '__main__':
    main()
