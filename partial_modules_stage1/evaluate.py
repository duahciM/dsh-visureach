"""Compare frozen predictions against independent renderer identities and analytic trajectories."""
from collections import Counter
from math import isfinite

from dacf_research.prepare_inputs import read,write
from partial_modules_stage1.fixtures import HERE


# Analytic assessor-only trajectory in world-image pixels. Does not consume measured ROI boxes.
SCENES={
    'paired': {'candidate':[(61,102,10,0),(365,113,-7,0),(145,113,9,9),(355,100,-12,17)],
               'ref':lambda i:(62,271),'peer':lambda i:(320+(i//6)*8,261),'absent_peer':{8,9,10,11}},
    'panning': {'candidate':[(97,117,12,0),(398,117,-7,1),(212,126,-9,10),(115,104,10,17)],
                'ref':lambda i:(78,271),'peer':lambda i:(336-2*i,261),'absent_peer':{11,12,13}}
}


def analytic_measure(case, first, second, episode_a, episode_b, role):
    scene=SCENES[case]
    if role=='peer' and ((first in scene['absent_peer']) or (second in scene['absent_peer'])):return None
    def offset(frame, ep):
        x,y,v,t=scene['candidate'][int(ep[-1])]
        rx,ry=scene['ref' if role=='reference' else 'peer'](frame)
        return x+v*(frame-t)-rx,y-ry
    a=offset(first,episode_a);b=offset(second,episode_b)
    return b[0]-a[0],b[1]-a[1]


def check_edge(case, edge, ep0, ep1, clock):
    ticks=edge['pts'];i,j=(clock[x] for x in ticks)
    rows=[]
    for axis in ('reference','peer'):
        record=edge['channels'][axis];expected=analytic_measure(case,i,j,ep0,ep1,axis)
        measured=record.get('delta_offset_common_native_xy')
        error=None if measured is None or expected is None else max(abs(a-b) for a,b in zip(measured,expected))
        status=('missing_expected' if expected is None and record['state']=='unknown' else 'unexpected_measurement' if expected is None
              else 'not_qualified' if error is None else 'within_3_native_px' if error<=3 else 'outside_3_native_px')
        rows.append({'type':axis,'pts':ticks,'episodes':[ep0,ep1],'analytic_delta_native_xy':expected,
                     'registered_delta_native_xy':measured,'absolute_max_error_px':error,'assessment':status})
    return rows


def coords(episode):
    # Accumulate independent components on maximal runs of comparable sampled edges.
    channels={}
    for role in ('reference','peer','object_luma'):
        components=[]; start=True
        for edge in episode['edges']:
            m=edge['channels'][role]
            if role=='object_luma':
                val=[abs(m['delta_8bit'])] if isfinite(m['delta_8bit']) else None
            else:
                c=m.get('delta_offset_common_native_xy')
                val=None if c is None else [abs(v) for v in c]
            if val is None:
                components.append({'pts':edge['pts'],'state':'unknown_gap','increments':None,'cumulative':None})
                start=True
                continue
            if start:
                values=[0.]*len(val); start=False
                components.append({'pts':[edge['pts'][0]],'state':'component_start','cumulative':list(values)})
            values=[a+b for a,b in zip(values,val)]
            components.append({'pts':edge['pts'],'state':'sampled_edge','increments':val,'cumulative':values})
        channels[role]=components
    return {'episode':episode['episode'],'separate_channels':channels,'scope':'Exploratory cumulative magnitude by axis, not continuous motion.'}


def evaluate(case, truth):
    folder=HERE/'first_run'/case
    a0,a1,a2,a3=(read(folder/f'A{i}.json') for i in range(4))
    assert a1['m2']==a3['m2'] and a2['m5']==a3['m5']
    names=truth['local_episode_instances']; clock={row['pts']:row['frame'] for row in truth['renderer_clock']}
    source=read(folder/'evidence.json'); all_roots={v['id'] for v in source['source_receipts']}
    refs={v['comparison_ref']:v['source_roots'] for v in source['local_calculations']}
    context={v['calculation_root']:v['source_roots'] for v in source['context_calculations']}
    associations=[];local=[];bridges=[]
    for row in a3['m2']:
        same=names[row['episodes'][0]]==names[row['episodes'][1]]
        used=[t for ref in row['independent_comparison_refs'] for t in refs[ref]]
        if used:
            assert len(used)==len(set(used))==4 and set(used)<=all_roots
        assert row['physical_identity']=='unknown'
        associations.append({'episodes':row['episodes'],'renderer_same_instance':same,'outcome':row['outcome'],
              'single_prototype_support':row['single_prototype_support'],'heldout_local_support':row['holdout_support'],
              'coappearance_evidence_count':len(row['counterevidence'])})
    for series in a3['m5']:
        for edge in series['edges']:
            assert set(context[edge['context_root']])==set(edge['source_roots'])<=all_roots
            local.extend(check_edge(case,edge,series['episode'],series['episode'],clock))
    for row in a3['bridges']:
        if 'measure' not in row:continue
        a,b=row['episodes']; measure=row['measure']
        valid=check_edge(case,measure,a,b,clock)
        same=names[a]==names[b]
        bridges.append({'episodes':[a,b],'requested_link_was_same_fixture_instance':same,'association_error_if_reused_as_identity':not same,
                        'unknown_interval':measure['gap_open_interval_pts'],'measurement':valid})
    counts=Counter(r['assessment'] for r in local)
    cross=Counter('correct_association' if u['requested_link_was_same_fixture_instance'] else 'wrong_association' for u in bridges)
    base_usable=sum(r['assessment']=='within_3_native_px' for r in local)
    useful_cross=sum(r['requested_link_was_same_fixture_instance'] and any(m['assessment']=='within_3_native_px' for m in r['measurement']) for r in bridges)
    return {'source':case,'short_local_projected_edges':local,'local_counts':dict(counts),'association_pairs':associations,'bridge_details':bridges,
            'bridge_counts':dict(cross),'change_coordinates':[coords(ep) for ep in a3['m5']], 'frozen_baseline_snapshot_pts':len(a0['graph_snapshots']),
            'shared_metric_validated_relation_endpoints':{'A0':0,'A1':0,'A2':base_usable,'A3':base_usable+sum(
              m['assessment']=='within_3_native_px' for r in bridges if r['requested_link_was_same_fixture_instance'] for m in r['measurement'])},
            'useful_cross_episode_pairs':useful_cross}


if __name__=='__main__':
    truth=read(HERE/'evaluation/renderer_truth.json')
    rows=[evaluate(k,v) for k,v in truth.items()]
    write(HERE/'evaluation/results.json',{'rows':rows,'scope':'Controlled same ROI data, analytic 2D error separate from hidden fixture identity. No video-understanding score.'})
