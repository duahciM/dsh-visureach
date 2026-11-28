"""Compare discovery, frame probes, actual native readback, and retained catalogue."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.prepare import select_time
from evidenweft_phase1.source_review import native_stream, timeline
from dacf_round2.frozen_inputs import HERE


def union(spans):
    values=[]
    for a,b in sorted(spans):
        assert a<=b
        if values and a<=values[-1][1]:
            values[-1][1]=max(b,values[-1][1])
        else:
            values.append([a,b])
    return values


def inside(p, spans):
    return any(lo<=p<=hi for lo,hi in spans)


def profile(spans, source):
    u=union(spans)
    return {'union_pts':u,'time_fraction':sum(b-a for a,b in u)/(source[-1]['pts']-source[0]['pts']),
            'expected_unique_source_pts':sum(inside(r['pts'],u) for r in source)}


def role_eval(references, src, accepted, uncertain, gaps, old, evaluated, probe, coarse_uniform):
    rows=[]
    orig=[r['pts'] for r in src]
    for role in references:
        lo,hi=role['pts']
        expected=[p for p in orig if lo<=p<=hi]
        assert expected, role
        a=[p for p in expected if inside(p,accepted)]
        w=[p for p in expected if inside(p,uncertain)]
        g=[p for p in expected if inside(p,gaps)]
        first=[p for p in expected if p in old]
        pics=[p for p in expected if p in evaluated]
        rows.append({'id':role['id'],'reference_pts':[lo,hi],'actual_source_positives':len(expected),'initial_catalogue_pts':first,
            'paired_region_cover_pts':a,'including_unknown_cover_pts':w,'all_nomination_gap_baseline_pts':g,
            'actual_reread_pts':pics,'is_new_reread_role':bool(pics and not first),
            'region_midpoint_probe_pts':[p for p in expected if p in probe],
            'same_count_whole_video_coarse_uniform_pts':[p for p in expected if p in coarse_uniform],
            'uncovered_actual_reference_pts':[p for p in expected if p not in a]})
    return rows


def decisive_read(source, video, clock, accepted, folder, index, all_pts, witnesses):
    t=monotonic()
    destination=folder/'originals';destination.mkdir()
    manifest={}
    src_dims=(video['width'],video['height'])
    coarse={r['pts'] for r in index['pool']}
    old={r['pts'] for r in index['pool'] if r['in_catalog']}
    for row,bits in native_stream(source,video,clock):
        if not inside(row['pts'],accepted) and row['pts'] not in witnesses:
            continue
        item=row|{'time_base':video['time_base'],'native_size':src_dims,'source_sha256':index['video_sha256'],
            'raw_rgb_sha256':sha256(bits).hexdigest(),'in_frozen_catalogue':row['pts'] in old,'in_coarse':row['pts'] in coarse,
            'from_paired_region':inside(row['pts'],accepted),'presentation':'native_rgb_png'}
        path=destination/f'pts-{row["pts"]}.png'
        Image.frombytes('RGB',src_dims,bits).save(path)
        item['source_png']=path.relative_to(folder).as_posix()
        item['png_sha256']=digest(path)
        with Image.open(path) as image:
            assert image.size==src_dims and sha256(image.convert('RGB').tobytes()).hexdigest()==item['raw_rgb_sha256']
        if row['pts'] in all_pts:
            assert item['raw_rgb_sha256']==all_pts[row['pts']]['rgb_sha256']
        manifest[row['pts']]=item
    return manifest,round(monotonic()-t,3)


def region_sheet(case, root, region, role_rows, evidence, clock):
    pset=sorted(set(role_pts for row in role_rows for role_pts in
                  [*row['actual_reread_pts'][:1],*row['initial_catalogue_pts'][:1],row['reference_pts'][0]]
                  if role_pts in evidence))
    if not pset:
        return
    img=Image.new('RGB',(1000,((len(pset)+3)//4)*180+55),'#172131')
    draw=ImageDraw.Draw(img)
    draw.text((8,10),f'{case}: Reviewable originals (region reread or frozen catalogue / reference)',fill='white')
    for i,p in enumerate(pset):
        x,y=i%4*250,(i//4)*180+43
        r=evidence[p]
        with Image.open(root/r['source_png']) as pix:
            pix.thumbnail((242,136));img.paste(pix,(x+4,y))
        labels=', '.join(v['id'] for v in role_rows if v['reference_pts'][0]<=p<=v['reference_pts'][1])
        note='paired reread' if r['from_paired_region'] else 'reference/context'
        draw.text((x+4,y+140),f'{p} / {r["seconds"]:.3f}s',fill='white')
        draw.text((x+4,y+155),f'{labels}  |  {note}',fill='#a9c5e6')
    img.save(root/'review_originals.png')


def round1_intervals(name):
    if name not in {'bench','doorway','reverse_doorway'}: return {}
    old=ROOT/'dacf_research'
    return {arm:[p['boundary_pts'] for p in read(old/f'run_{arm}'/name/'pair_results.json')] for arm in ('manual','automatic')}


def main():
    lock=read(HERE/'lock.json')
    for group in ('inputs','preserved_round1_outputs','reference_hashes_scoring_only'):
        assert all(digest(ROOT/p)==v for p,v in lock[group].items()),group
    independent=HERE/'independent_execution/independent'
    assert (independent/'summary.json').is_file() and (HERE/'replay/summary.json').is_file()
    reference=read(HERE/'references/roles.json')
    replay_roles=read(ROOT/'dacf_research/evaluation_only/roles.json')['roles']
    legacy=read(ROOT/'evidenweft_phase1/evaluation_only/rubric.json')['cases']
    root=HERE/'evaluation';root.mkdir()
    summary=[]
    for name,c in lock['cases'].items():
        src=ROOT/c['source']; cohort=c['cohort']
        m=read(ROOT/c['index']/'manifest.json')
        predictions_root=independent if cohort=='independent' else HERE/'replay'
        clips=read(predictions_root/name/'regions.json')
        catalogue=clips['nomination_pts']
        assert catalogue==[r['pts'] for r in m['pool'] if r['in_catalog']]
        addresses={r['pts']:r for r in read(ROOT/c['pixels']/'frame_receipts.json')}
        st,clock=timeline(src)
        native={r['pts']:r for r in clock}
        eligible=[r for r in clips['regions'] if r['kind']=='paired_investigation']
        unknown=[r for r in clips['regions'] if 'unknown' in r['kind']]
        paired=profile([r['bounds_pts'] for r in eligible],clock)
        with_unknown=profile([r['bounds_pts'] for r in eligible+unknown],clock)
        cum={p:i for i,p in enumerate(r['pts'] for r in m['pool'])}
        gaps=profile([(a,b) for a,b in zip(catalogue,catalogue[1:]) if cum[b]-cum[a]>1],clock)
        predictions=[]
        for r in eligible:
            a,b=r['bounds_pts'];center=(a+b)/2
            inside_coarse=[p for p in cum if a<=p<=b]
            if inside_coarse:
                predictions.append(min(inside_coarse,key=lambda p:(abs(p-center),p)))
        probes=sorted(set(predictions))
        coarse_control=[r['pts'] for r in select_time(m['pool'],len(probes))] if probes else []
        anchors=Counter(ch for r in eligible for ch in r['distinction']['supported_by'])
        roles=reference['independent'].get(name,replay_roles.get(name,[]))
        extras=[{'id':k,'pts':legacy[name]['slots'][k]['pts'],'witness_pts':legacy[name]['slots'][k]['witness']}
                for k in reference['frozen_catalogue_context'].get(name,[])]
        context={x for ref in roles+extras for x in [ref.get('witness_pts',ref['pts'][0])] if x in native}
        context.update(coarse_control)
        if extras:
            for ref in extras:
                context.update(first for first in [next((p for p in catalogue if ref['pts'][0]<=p<=ref['pts'][1]),None)] if first is not None)
        dest=root/name;dest.mkdir()
        receipts,seconds=decisive_read(src,st,clock,paired['union_pts'],dest,m,addresses,context)
        actual={p for p,r in receipts.items() if r['from_paired_region']}
        assert len(actual)==paired['expected_unique_source_pts']
        cols=role_eval(roles+extras,clock,paired['union_pts'],with_unknown['union_pts'],gaps['union_pts'],set(catalogue),actual,probes,coarse_control)
        groups=[]
        for items in reference['relation_sets'].get(name,[]):
            def flags(item):
                return next(r for r in cols if r['id']==item)
            initial=all(flags(q)['initial_catalogue_pts'] for q in items)
            after=all(flags(q)['initial_catalogue_pts'] or flags(q)['actual_reread_pts'] for q in items)
            groups.append({'roles':items,'all_roles_available_before':initial,'all_roles_available_after_actual_reread':after,
                           'newly_complete':after and not initial,'interpretation':'availability of distinct observed image roles'})
        full = dict(case=name,cohort=cohort,total_source_pts=len(clock),initial_catalogue_count=len(catalogue),
             unchanged_catalogue_count=len(catalogue),paired=paired,with_unknown=with_unknown,simple_gap_baseline=gaps,
             accepted_leaf_count=len(eligible),unknown_range_count=len(unknown),strong_anchor_channels=dict(anchors),
             actual_unique_native_reread=len(actual),actual_new_outside_catalogue=len(actual-set(catalogue)),native_reread_seconds=seconds,
             region_midpoint_auxiliary_pts=probes,equal_count_uniform_coarse_auxiliary_pts=coarse_control,
             role_checks=cols,relation_availability=groups,round1_comparable_spans=round1_intervals(name))
        write(dest/'score.json',full)
        write(dest/'actual_original_receipts.json',list(receipts.values()))
        region_sheet(name,dest,eligible,cols,receipts,clock)
        summary.append(full)
        print(name,'region fraction',round(paired['time_fraction'],4),'read',len(actual),'roles',
              [(x['id'],len(x['initial_catalogue_pts']),len(x['actual_reread_pts'])) for x in cols],flush=True)
    write(root/'summary.json',summary)


if __name__=='__main__':
    main()
