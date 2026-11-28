"""Select exact source candidates without inferring coordinates for unseen pixels."""
from bisect import bisect_left, bisect_right
from math import exp

import numpy as np


def cumulative(samples, times, measure, max_gap):
    positions = {samples[0]: (0, 0.)}
    edges = []
    for a, b in zip(samples, samples[1:]):
        component, coordinate = positions[a]
        value = measure(a, b)
        valid = value is not None and 0 < times[b]-times[a] <= max_gap
        positions[b] = (component, coordinate + value) if valid else (component+1, 0.)
        edges.append({'pts':[a,b], 'value':value, 'applicable':valid,
                      'state':'missing_or_inapplicable' if not valid else 'zero' if value == 0 else 'measured'})
    total = sum(edge['value'] for edge in edges if edge['applicable'])
    return positions, edges, total


def delta(positions, a, b):
    ca, x = positions[a]
    cb, y = positions[b]
    return y-x if ca==cb else None


def gaussian_profile(location, reference, widths):
    return [float(np.exp(-0.5*((location-reference)/width)**2).sum()) for width in widths]


def ranked_gaps(positions, samples, observed, times, native, bounds, sigma, kde):
    refs = {}
    coordinates = positions
    for pts in sorted(observed & positions.keys()):
        component, x = coordinates[pts]
        refs.setdefault(component, []).append(x)
    refs = {key:np.asarray(values) for key,values in refs.items()}
    candidates = []
    for key, lo, hi in bounds:
        window = samples[bisect_left(samples,lo):bisect_right(samples,hi)]
        boundaries = {lo,hi}
        for a,b in zip(window,window[1:]):
            if coordinates[a][0] != coordinates[b][0]:
                boundaries.update((a,b))
        anchors = sorted((boundaries | observed) & set(window))
        for a,b in zip(anchors, anchors[1:]):
            gap = delta(coordinates,a,b)
            if gap is None or gap <= 1e-12:
                continue
            component, x = coordinates[a]
            support = refs.get(component, np.array([]))
            if not len(support):
                continue
            midpoint = (times[a]+times[b])/2
            idx = bisect_left(samples,a,0)
            inner = samples[idx+1:bisect_left(samples,b)]
            if inner:
                probes = [(p, abs(coordinates[p][1]-support).min(),
                           gaussian_profile(coordinates[p][1],support,sigma) if kde else None) for p in inner]
                pts, coverage, density = max(probes,key=lambda m:((-m[2][1] if kde else m[1]), m[1],-abs(times[m[0]]-midpoint),-m[0]))
                measured = True
            else:
                interior = native[bisect_right(native,a):bisect_left(native,b)]
                if not interior:
                    continue
                pts = min(interior,key=lambda p:(abs(times[p]-midpoint),p))
                coverage = gap/2
                density = [max(u,v) for u,v in zip(gaussian_profile(x,support,sigma),gaussian_profile(coordinates[b][1],support,sigma))] if kde else None
                measured = False
            candidates.append({'root':key,'bounds':[a,b],'pts':pts,'gap':gap,'coverage':float(coverage),'density':density,
                               'measured_coordinate':measured,'state':'measured_candidate' if measured else 'midpoint_probe_coordinate_unknown'})
    if not candidates:
        return None, []
    selected = max(candidates,key=lambda m:((-m['density'][1] if kde else m['coverage']),m['gap'],-m['pts']))
    return selected, candidates


def time_pick(samples, observed, times, sigma, kde):
    if not observed or not (set(samples)-observed):
        return None, 0.
    x = np.asarray([times[p] for p in samples])
    anchors = np.asarray([times[p] for p in sorted(observed)])
    dist = abs(x[:, None]-anchors[None, :])
    separation = dist.min(axis=1)
    densities = [np.exp(-.5*(dist/width)**2).sum(axis=1) for width in sigma] if kde else None
    possible = [i for i,p in enumerate(samples) if p not in observed]
    index = max(possible,key=lambda i:(-densities[1][i] if kde else separation[i],separation[i],-i))
    p = samples[index]
    at = bisect_left(anchors,times[p])
    left = anchors[max(0,at-1)] if at else x[0]
    right = anchors[at] if at < len(anchors) else x[-1]
    bounds = [samples[np.searchsorted(x,left)],samples[np.searchsorted(x,right)]]
    return {'root':'source','pts':p,'bounds':bounds,'coverage':float(separation[index]),'gap':float(right-left),
            'density':[float(d[index]) for d in densities] if kde else None,'measured_coordinate':True,'state':'exact_source_time'},float(max(separation))


def conv_samples(samples, positions, edges, times, selected, scales):
    """Discrete, separately supported rate smoothing: no artificial samples."""
    comp, at = positions[selected]
    valid = [edge for edge in edges if edge['applicable'] and positions[edge['pts'][0]][0] == comp]
    length = max((positions[p][1] for p in samples if positions[p][0] == comp),default=0.)
    result = {'source_time_seconds':{},'path_fraction':{}}
    for axis, span, widths in [('source_time_seconds',1.,(.2,.5,1.)),('path_fraction',length,scales)]:
        for width in widths:
            pairs = []
            if span > 0:
                for edge in valid:
                    a,b = edge['pts']
                    distance = abs((times[a]+times[b])/2-times[selected]) if axis=='source_time_seconds' else abs((positions[a][1]+positions[b][1])/2-at)/span
                    if distance <= 3*width:
                        w = exp(-.5*(distance/width)**2)
                        pairs.append((w,edge['value']/(times[b]-times[a]),[a,b]))
            total = sum(w for w,_,_ in pairs)
            result[axis][str(width)] = {'smoothed_change_per_second':sum(w*v for w,v,_ in pairs)/total if total else None,
                                         'weight_sum':total,'support_pairs':[p for _,_,p in pairs]}
    return result
