"""Execute frozen equal-original and natural-stop routes, without scoring inputs."""
import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from dacf_research.field import Bank, clip_encoder, direct_group
from dacf_research.prepare_inputs import ROOT, digest, read, write
from dacf_round2.investigate import evidence, load_rasters
from dacf_round2.regions import discover
from dacf_round2.verify_round2 import unpack
from dacf_round3.acquire import Reader
from dacf_round3.sampling import conv_samples, cumulative, ranked_gaps, time_pick
from dacf_round3.setup import HERE, R2, verify
from evidenweft_phase1.source_review import timeline


def nearest(entries, times, aim):
    return min(entries,key=lambda p:(abs(times[p]-aim),p))


def round2_pick(paired, native, times, seen, visited):
    pending = []
    for key,a,b in paired:
        choices = [p for p in native[bisect_right(native,a):bisect_left(native,b)] if p not in seen]
        if choices:
            pending.append((key,a,b,choices))
    if not pending:
        return None, 0
    key,a,b,choices = min(pending,key=lambda item:(item[0] in visited,item[0]))
    p = nearest(choices,times,(times[a]+times[b])/2)
    return {'root':key,'bounds':[a,b],'pts':p,'state':'frozen_r2_bounds_temporal_midpoint',
            'gap':times[b]-times[a]}, sum(r[0] not in visited for r in pending)


def signals_override(option, signal, candidates, native, observed, times, channel, order):
    a,b = option['bounds']
    interior = [pts for pts in native[bisect_right(native,a):bisect_left(native,b)] if pts not in observed]
    aim = None
    if order=='anchor_crossing':
        for event in signal.get('crossings',{}).get(channel,{}).get('events',[]):
            if event['kind'] != 'sampled_equality_only' and max(event['from_pts'],a)<min(event['to_pts'],b):
                available = [p for p in interior if event['from_pts'] <= p <= event['to_pts']]
                if available:
                    aim = (event['from_seconds']+event['to_seconds'])/2
                    interior = available
                    break
    elif order=='multiscale_extremum':
        for scale in ('0.2','0.5','1.0'):
            for extremum in signal.get('multi_scale_extrema',{}).get(channel,{}).get(scale,[]):
                if a < extremum['pts'] < b:
                    aim = times[extremum['pts']]
                    break
            if aim is not None:
                break
    option = option | {'requested_operator':order,'available_coverage_brackets':len(candidates)}
    if aim is None or not interior:
        return option | {'used_operator':'coverage','secondary_available':False}
    return option | {'coverage_candidate_pts':option['pts'],'pts':nearest(interior,times,aim),
                     'used_operator':order,'secondary_available':True, 'measured_coordinate':False}


class Experiment:
    def __init__(self, case, registration, encoder, config, out):
        self.name = case
        self.out = out
        manifest = read(ROOT/registration['index']/'manifest.json')
        source = ROOT/registration['source']
        self.catalogue = {row['pts']: row for row in read(ROOT/registration['index']/'catalog.json')['frames']}
        stream, clock = timeline(source)
        self.times = {r['pts']:r['seconds'] for r in clock}
        self.ranks = {r['pts']:r['frame'] for r in clock}
        self.native = sorted(self.times)
        self.coarse = [row['pts'] for row in manifest['pool']]
        assert set(self.catalogue) <= set(self.coarse)
        self.reader = Reader(source,manifest['video_sha256'],stream,clock,out/'originals')
        self.initial_rasters = load_rasters(ROOT/registration['pixels'],manifest)
        self.bank = Bank(self.initial_rasters,config,encoder)
        self.encoder = encoder
        self.coarse_receipts = {r['pts']: {k:v for k,v in r.items() if k!='rgb'} | {'path':str(ROOT/registration['pixels']/r['png']), 'source_frame':r['frame']}
                                for r in read(ROOT/registration['pixels']/'frame_receipts.json')}
        self.initial = set(self.catalogue)
        if registration['cohort']=='new':
            plan = discover(self.bank,[self.ranks[p] for p in self.coarse],self.catalogue,read(R2/'protocol.json'))
            write(out/'new_round2_plan.json',plan)
            signals = [evidence(self.bank,r,[self.ranks[p] for p in self.coarse],self.catalogue) for r in plan['regions']]
            write(out/'new_round2_signals.json',signals)
        else:
            root = R2/('replay' if registration['cohort']=='replay' else 'independent_execution/independent')/case
            plan = read(root/'regions.json')
            signals = unpack(root/'region_signals.json.gz')
        assert set(plan['nomination_pts']) == self.initial
        self.plan = {r['id']:r for r in plan['regions']}
        self.signals = {r['region_id']:r for r in signals}
        self.paired = [(r['id'],*r['bounds_pts']) for r in plan['regions'] if r['kind']=='paired_investigation']
        self.anchor_cache = {}

    def receipt(self, pts):
        return self.reader.receipts.get(pts) or self.coarse_receipts[pts]

    def add(self, pts):
        receipt, decoded = self.reader.read(pts)
        if self.ranks[pts] not in self.bank.rows:
            with Image.open(receipt['path']) as image:
                rgb = np.asarray(image.resize((320,max(8,round(image.height*320/image.width))),Image.Resampling.BILINEAR))
            new = Bank({self.ranks[pts]:receipt | {'rgb':rgb}},self.bank.config,self.encoder)
            self.bank.rows.update(new.rows)
        return receipt, decoded

    def axis(self, samples, observed, channel):
        positions,edges,total = cumulative(samples,self.times,lambda a,b:self.bank.metric(self.ranks[a],self.ranks[b])[channel],
                                           self.bank.config['maximum_gap_seconds'])
        return {'positions':positions,'edges':edges,'total':total,'state':'zero' if total==0 and all(e['applicable'] for e in edges) else
                'partial' if any(not e['applicable'] for e in edges) else 'measured',
                'maximum_observed_gap':max_gap(positions,observed)}

    def anchors(self, region, point, channel):
        key = (region,point)
        if key not in self.anchor_cache:
            row = self.plan[region]
            self.anchor_cache[key] = [direct_group(self.bank,self.ranks[point],row[side]['prototypes']) for side in ('left','right')]
        groups = self.anchor_cache[key]
        a,b = [g['channels'][channel]['distance'] for g in groups]
        return {'a':a,'b':b,'relative':a/(a+b) if None not in (a,b) and a+b>1e-7 else None,
                'absolute_cap':self.bank.config['support_caps'][channel], 'parents':groups}

    def run(self, arm, channel, policy):
        start = perf_counter()
        samples = self.coarse.copy()
        observed = self.initial.copy()
        visits = set()
        scale = self.times[self.native[-1]]-self.times[self.native[0]]
        steps = []
        natural = None
        ceiling = policy['safety_stop_originals']
        equal = policy['equal_new_originals']
        while len(steps)<ceiling:
            t0 = perf_counter()
            axis = self.axis(samples,observed,channel) if channel else None
            if arm=='E0':
                option, remaining = round2_pick(self.paired,self.native,self.times,observed,visits)
                reached = remaining==0
            elif arm in {'E1','E2'}:
                option, largest = time_pick(self.native,observed,self.times,[scale*x for x in policy['kde_relative_scales']],arm=='E2')
                reached = largest <= scale*policy['natural_gap_fraction']/2
            else:
                total = axis['total']
                bounds = self.paired if arm in {'E5','E6'} else [('source',samples[0],samples[-1])]
                option,candidates = ranked_gaps(axis['positions'],samples,observed,self.times,self.native,bounds,
                                                [max(total*x,1e-12) for x in policy['kde_relative_scales']],arm=='E4')
                reached = max((c['gap'] for c in candidates),default=0.) <= total*policy['natural_gap_fraction']
                if arm=='E6' and option:
                    operator = ('anchor_crossing','multiscale_extremum','coverage')[len(steps)%3]
                    option = signals_override(option,self.signals[option['root']],candidates,self.native,observed,self.times,channel,operator)
            if natural is None and (not option or reached):
                natural = {'after':len(steps),'reason':'no_eligible_candidate' if not option else 'natural_rule_reached'}
            if not option or natural is not None and len(steps)>=equal:
                break
            p = option['pts']
            assert p not in observed
            plan_seconds = perf_counter()-t0
            try:
                rec,decoded = self.add(p)
            except (ValueError, TimeoutError) as error:
                natural = {'after':len(steps),'reason':'decode_failed','error':str(error)}
                break
            if p not in samples:
                samples.insert(bisect_left(samples,p),p)
            observed.add(p)
            visits.add(option['root'])
            revised = self.axis(samples,observed,channel) if channel else None
            a,b = option['bounds']
            record = option | {'ordinal':len(steps)+1, 'seconds':self.times[p], 'receipt':rec['id'],
                'reused_research_cache':not decoded,'bytes':rec['png_bytes'],'selection_seconds':round(plan_seconds,7),
                'after_natural_stop_for_equal_comparison':natural is not None}
            if axis:
                record.update(before_axis={**{q:axis['positions'][q] for q in (a,p,b) if q in axis['positions']}},
                    after_axis={q:revised['positions'][q] for q in (a,p,b)},
                    original_edges=window_edges(axis['edges'],a,b), revised_edges=window_edges(revised['edges'],a,b),
                    signal_convolution=conv_samples(samples,revised['positions'],revised['edges'],self.times,p,policy['kde_relative_scales']))
                if option['root'] in self.plan:
                    record['dual_anchor'] = self.anchors(option['root'],p,channel)
            steps.append(record)
        if natural is None:
            natural = {'after':len(steps),'reason':'observation_ceiling_censored'}
        return {'arm':arm,'channel':channel,'initial_preserved':self.initial<=observed,'steps':steps,'natural_stop':natural,'wall_seconds':round(perf_counter()-start,6)}


def window_edges(edges,a,b):
    return [e for e in edges if a<=e['pts'][0] and e['pts'][1]<=b]


def max_gap(positions, observed):
    groups = {}
    for p, (part, value) in positions.items():
        entry = groups.setdefault(part, {'all':[],'evidence':[]})
        entry['all'].append(value)
        if p in observed:
            entry['evidence'].append(value)
    largest = 0.
    for group in groups.values():
        ordered = [min(group['all']),*sorted(group['evidence']),max(group['all'])]
        largest = max(largest,*(b-a for a,b in zip(ordered,ordered[1:])))
    return largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case',action='append')
    args = parser.parse_args()
    lock = read(HERE/'lock.json')
    verify(lock)
    cfg = read(ROOT/'dacf_research/protocol.json')
    policy = read(HERE/'protocol.json')
    args.out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    encoder = clip_encoder()
    for name,row in lock['cases'].items():
        if args.case and name not in args.case:
            continue
        started = perf_counter()
        out = args.out/name
        out.mkdir()
        task = Experiment(name,row,encoder,cfg,out)
        arms = [task.run(key,channel,policy) for key in policy['arms'] for channel in (policy['axes'] if key in {'E3','E4','E5','E6'} else [None])]
        initial = {ch:task.axis(task.coarse,task.initial,ch) for ch in policy['axes']}
        footprints = {'r2_plan':list(task.plan.values()),'initial_pts':sorted(task.initial),'coarse_pts':task.coarse,'axes':initial,
                      'native_source_count':len(task.native),'source_sha256':task.reader.signature,'time_base':task.reader.stream['time_base'],
                      'source_seconds':list(task.times.values()),'source_pts':task.native}
        write(out/'initial.json',footprints)
        write(out/'receipts.json',task.reader.receipts)
        write(out/'failed_reads.json',task.reader.failed)
        old = {p:task.coarse_receipts[p] for p in task.coarse}
        write(out/'reference_originals.json',old)
        write(out/'routes.json',arms)
        write(out/'pair_roots.json',list(task.bank.evidence.values()))
        print(name,len(arms),len(task.reader.receipts),'unique actual rereads',round(perf_counter()-started,2),flush=True)
    verify(lock)


if __name__ == '__main__':
    main()
