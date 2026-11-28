"""Development refinement of M1/M3 local comparability, using the original held-out fits."""

from collections import Counter
from itertools import combinations

import numpy as np

from dacf_research.field import correspond, fit
from partial_modules_stage1.study import root


def grid_cells(pts,width,height):
    return [f'3x2:{min(2,max(0,int(x*3/width)))},{min(1,max(0,int(y*2/height)))}' for x,y in pts]


def heldout_cells(measurement,bank_image,axis):
    if measurement['state']!='heldout_supported':
        return []
    valid=[p[axis:axis+2] for p,e in zip(measurement['held_points'],measurement['held_errors_px'])
           if e<=measurement['p90_px']]
    return sorted(set(grid_cells(valid,bank_image.shape[1],bank_image.shape[0])))


class Layout:
    def __init__(self,bank):
        self.bank=bank
        self.pairs={}

    def compare(self,a,b):
        assert a!=b
        key=tuple(sorted((a,b)))
        if key not in self.pairs:
            self.pairs[key]=self.measure(*key)
        return self.pairs[key]

    def measure(self,a,b):
        bank=self.bank
        left,right=(bank.rows[x] for x in (a,b))
        matches=correspond(left['descriptor'][1],right['descriptor'][1])
        ia=np.array([i for i,_ in matches],dtype=int)
        ib=np.array([j for _,j in matches],dtype=int)
        x,y=left['xy'][ia],right['xy'][ib]
        allowed=np.ones(len(x),bool)
        forward,_=fit(x,y,allowed)
        reverse,_=fit(y,x,allowed)
        left_cells=heldout_cells(forward,left['gray'],0)
        right_cells=heldout_cells(reverse,right['gray'],0)
        # Existing fixed-grid, independently fitted projections remain usable.
        pooled_support= len(left_cells)>=2 and len(right_cells)>=2
        fitted={}
        for first,last,side in ((a,b,'left'),(b,a,'right')):
            measured=bank.measure_local(first,last)
            grid_support=sorted([r['id'] for r in measured['cells'] if r['state']=='heldout_supported'])
            fitted[side]={'root':measured['root'],'supported_cells':grid_support}
        local_support=all(item['supported_cells'] for item in fitted.values())
        return {'id':root('local-context-fit-v7',left['id'],right['id']),
            'ranks':[a,b],'pts':[left['pts'],right['pts']],'source_roots':[left['id'],right['id']],
            'forward':forward,'reverse':reverse,'pooled_left_cells':left_cells if pooled_support else [],
            'pooled_right_cells':right_cells if pooled_support else [],
            'fixed_local':fitted,'matched_count':len(matches),
            'supported':pooled_support or local_support,
            'support_kind':'two_way_pooled' if pooled_support else 'two_way_locally_fitted' if local_support else 'unknown',
            'scope':'direct planar image correspondence, restricted to verified projection pixels'}

    def cells(self,query,reference):
        pair=self.compare(query,reference)
        side='left' if query==pair['ranks'][0] else 'right'
        return set(pair[f'pooled_{side}_cells']) | set(pair['fixed_local'][side]['supported_cells'])

    def group_profile(self,query,group):
        measures=[(r,self.compare(query,r)) for r in group if r!=query]
        counts=Counter(c for r,_ in measures for c in self.cells(query,r))
        return {'cells_by_support_count':dict(counts),'roots':[q['id'] for _,q in measures],
            'positive_cells':sorted(c for c,v in counts.items() if v>=2),
            'possible_direct_comparisons':len(measures)}


def local_groups(layout,pool,maximum_gap,dwell,radius):
    bank=layout.bank
    groups=[];rejected=[];offset=0
    def usable(pts):
        duration=bank.rows[pts[-1]]['seconds']-bank.rows[pts[0]]['seconds']
        gaps=[bank.rows[b]['seconds']-bank.rows[a]['seconds'] for a,b in zip(pts,pts[1:])]
        return dwell<=duration<=radius and all(g<=maximum_gap for g in gaps) and all(
            layout.compare(a,b)['supported'] for a,b in combinations(pts,2))
    while offset+2<len(pool):
        start=pool[offset:offset+3]
        if not usable(start):
            rejected.append({'pts':[bank.rows[r]['pts'] for r in start],'state':'no_verified_direct_triads'})
            offset+=1
            continue
        votes=[]
        end=offset+3
        while end<len(pool):
            a,b=pool[end-1:end+1]
            if bank.rows[b]['seconds']-bank.rows[a]['seconds']>maximum_gap:
                break
            support=[layout.compare(x,b) for x in start]
            positive=[x['id'] for x in support if x['supported']]
            if len(positive)<2:
                break
            votes.append({'pts':bank.rows[b]['pts'],'direct_seed_support':positive})
            end+=1
        accepted=pool[offset:end]
        tails=[accepted[x:x+3] for x in range(len(accepted)-3,-1,-1)]
        finish=next(q for q in tails if usable(q))
        groups.append({'id':f'local-{len(groups)+1}','ranks':accepted,'entry_prototypes':start,
            'exit_prototypes':finish,'bounds_pts':[bank.rows[accepted[0]]['pts'],bank.rows[accepted[-1]]['pts']],
            'entry_root_ids':[layout.compare(a,b)['id'] for a,b in combinations(start,2)],
            'exit_root_ids':[layout.compare(a,b)['id'] for a,b in combinations(finish,2)],
            'direct_extension_votes':votes})
        offset=end
    return {'groups':groups,'rejected':rejected}
