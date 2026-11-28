import hashlib
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from p1.io import command,dump
from .common import PROJECT,V06,OUT,read,sha,historical_paths


def audit_images(source,items):
    by_frame=defaultdict(list)
    for f in items:
        if f['native_image'] not in [x['native_image'] for x in by_frame[f['source_frame']]]:by_frame[f['source_frame']].append(f)
    order=sorted(by_frame);expr='+'.join(f'eq(n\\,{i})' for i in order)
    raw=command(['ffmpeg','-v','error','-noautorotate','-i',source['path'],'-vf',f"select='{expr}',format=rgb24",'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    h,w=source['probe']['height'],source['probe']['width'];rgb=np.frombuffer(raw,np.uint8).reshape(-1,h,w,3)
    if len(rgb)!=len(order):raise ValueError('Independent native source frame count mismatch')
    checks=[]
    for i,pixels in zip(order,rgb):
        for f in by_frame[i]:
            ok=bool(np.array_equal(pixels,np.asarray(Image.open(f['native_image']).convert('RGB'))))
            checks.append({'source_frame':i,'pts':f['pts'],'t':f['t'],'path':f['native_image'],'exact_source_rgb_equal':ok,
                'file_hash_equal':sha(f['native_image'])==f['native_image_sha256'],'rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
    return checks


def main():
    manifest=read(OUT/'run_manifest.json');freeze=read(OUT/'history_freeze.json');cases=[];coordinate_stats={'needed_pairs':0,'recomputed':0,'reused':0,'nonempty_complete_pairs':0,'full_inliers':0,'mismatches':0}
    for cid in manifest['completed']:
        root=OUT/cid;meta=read(root/'B/frames.json');native=read(root/'native_sources.json');k=read(root/'B/keyframes.json')
        imgs=native+k['frames']+k['controls']['v']+k['controls']['uniform'];checks=audit_images(meta['source'],imgs)
        region=read(root/'regional_evidence.json');dims_ok=True;sites_ok=True
        for r in region['frames']:
            shape=r['shape'];dims_ok=dims_ok and Image.open(root/'overlays'/f'{r["sample"]:04d}.png').size==(shape[1],shape[0])
        for n in native:
            dims_ok=dims_ok and Image.open(root/'native_overlays'/f'{n["sample"]:04d}.png').size==Image.open(n['native_image']).size
        pair_count=0
        with (root/'full_pair_evidence.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                p=json.loads(line);pair_count+=1;coordinate_stats['needed_pairs']+=1;coordinate_stats['recomputed' if p['coordinate_source'].startswith('recomputed') else 'reused']+=1
                coordinate_stats['mismatches']+=int(not p['metric_agreement']);coordinate_stats['full_inliers']+=p['full_coordinate_count']
                coordinate_stats['nonempty_complete_pairs']+=int(p['full_coordinate_count']>0 and p['metric_agreement'])
                if p['metric_agreement']:
                    if len(p['matches'])!=p['full_coordinate_count']:raise ValueError('Full coordinate/feature-ID count mismatch')
                    qa,qb=p['samples'];h,w=region['frames'][qa]['shape'];rh,rw=region['frames'][qb]['shape']
                    for m in p['matches']:
                        x,y=m['query_xy'];u,v=m['reference_xy'];sites_ok=sites_ok and 0<=x<w and 0<=y<h and 0<=u<rw and 0<=v<rh
        a=V06/cid/'full';equal=all(sha(a/name)==sha(root/'B'/name) for name in ('scene_ids.json','segments_coarse.json','segments_refined.json','keyframes.json','frames.json'))
        cases.append({'case':cid,'A_B_prediction_byte_equal':equal,'source_sha_equal':sha(meta['source']['path'])==meta['source']['sha256'],
             'native_image_checks':checks,'overlay_dimensions_match_original':dims_ok,'coordinates_in_bounds':bool(sites_ok),'pairs_checked':pair_count,
             'pass':equal and dims_ok and sites_ok and all(c['exact_source_rgb_equal'] and c['file_hash_equal'] for c in checks)})
    local=read(OUT/'B_resample/frames.json');k=read(OUT/'B_resample/keyframes.json');localchecks=audit_images(local['source'],k['frames']+k['controls']['v']+k['controls']['uniform'])
    current={str(p.relative_to(PROJECT)):sha(p) for p in historical_paths()};changed=[p for p,h in freeze['files'].items() if current.get(p)!=h];added=sorted(set(current)-set(freeze['files']))
    source_changed=[p for p,h in freeze['sources'].items() if sha(p)!=h]
    history={'historical_files':len(freeze['files']),'changed':changed,'added':added,'source_changed':source_changed,'pass':not(changed or added or source_changed)}
    dump(OUT/'history_verification.json',history)
    dump(OUT/'source_audit.json',{'cases':cases,'resample_checks':localchecks,'coordinate_stats':coordinate_stats,
         'native_image_checks':sum(len(c['native_image_checks']) for c in cases)+len(localchecks),'history':history,
         'pass':history['pass'] and all(c['pass'] for c in cases) and all(c['exact_source_rgb_equal'] and c['file_hash_equal'] for c in localchecks),
         'overlay_scope':'native source pixels audited before drawing; overlays intentionally annotated and checked for source dimensions, coordinates and provenance; not claimed equal to unannotated images'})
    print('AUDIT',len(cases),'cases',coordinate_stats,'history',history['pass'])


if __name__=='__main__':main()
