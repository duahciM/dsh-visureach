"""Post-prediction source verification, delivery scoring and blinded review packets."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw

from evidenweft_phase1.prepare import HERE, FROZEN, PUBLIC, read, select_time, verify, write
from evidenweft_phase1.source_review import native_stream


def covered(pts, rubric):
    picked = sorted(set(pts))
    slots = {key: [p for p in picked if v['pts'][0] <= p <= v['pts'][1]] for key, v in rubric['slots'].items()}
    compounds = {}
    for key, v in rubric['compounds'].items():
        last = -float('inf')
        chain = []
        for slot in v['slots']:
            witness = next((p for p in slots[slot] if p > last), None)
            if witness is None:
                break
            chain.append(witness)
            last = witness
        if len(chain) == len(v['slots']):
            compounds[key] = chain
    exact = {k: v for k, v in rubric.get('exact_endpoints', {}).items() if set(v) <= set(picked)}
    return {'slots': {k: v for k, v in slots.items() if v}, 'complete_compounds': compounds, 'exact_brackets': exact,
            'unclassified_pts': [p for p in picked if not any(p in v for v in slots.values())]}


def csv_rows(path, rows):
    with path.open('x', newline='', encoding='utf-8-sig') as file:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(file, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def assessed(r):
    return {k: r[k] for k in ('slots', 'complete_compounds', 'exact_brackets')}


def offline(out):
    lock = verify()
    rubric = read(HERE / 'evaluation_only/rubric.json')['cases']
    protocol = read(HERE / 'public_protocol.json')
    out.mkdir(parents=True, exist_ok=False)
    flat = []
    vendor = FROZEN / 'visureach/vendor/P2_24'
    repo = HERE.parent / 'plugin_offline/github/EvidenWeft'
    files = ['if16_research/nominate.py','if16_research/common.py','if16_research/protocol.json','if17_research/catalog.py',
             'if17_research/protocol.json','p224_research/offline.py']
    write(out / 'algorithm_linkage.json', {'public_main_at_freeze': lock['baselines']['evidenweft_main'],
         'selection': 'same nomination and catalogue implementation as standalone. Uses plugin generic 5 FPS import path, not IF30 historical selection/delivery policy.',
         'identical': {x: (vendor / x).read_bytes() == (repo / x).read_bytes() for x in files}})
    for case, role in rubric.items():
        child = out / case
        child.mkdir()
        source_ref = read(HERE / 'evaluation_only/source_review' / case / 'native_timeline.json')
        rows = source_ref['frames']
        source = PUBLIC / case / protocol['sources'][case]['public_name']
        inventory = read(HERE / 'inventory' / case / 'inventory.json')
        desired = {p for s in inventory['method_proposed_deliveries'] for p in s['pts']}
        trace = {}
        native = child / 'originals'
        native.mkdir()
        dimensions = source_ref['format']
        for frame, rgb in native_stream(source, dimensions, rows):
            if frame['pts'] not in desired:
                continue
            rgb_sig = sha256(rgb).hexdigest()
            assert rgb_sig == frame['decoded_rgb_sha256']
            image = native / f'{frame["pts"]}.png'
            Image.frombytes('RGB', (dimensions['width'], dimensions['height']), rgb).save(image)
            with Image.open(image) as packaged:
                assert sha256(packaged.convert('RGB').tobytes()).hexdigest() == rgb_sig
            trace[frame['pts']] = frame | {'png_sha256': sha256(image.read_bytes()).hexdigest(), 'path': image.relative_to(out).as_posix()}
        assert set(trace) == desired
        deliveries = []
        groups = defaultdict(dict)
        for proposal in inventory['method_proposed_deliveries']:
            pts = proposal['pts']
            result = covered(pts, role)
            membership = Counter(tuple(k for k, hits in result['slots'].items() if p in hits) for p in pts)
            count = len(pts)
            distinct_rgb = len({trace[p]['decoded_rgb_sha256'] for p in pts})
            assert count == proposal['count']
            record = proposal | result | {'actual_delivered': count, 'exact_pixel_repeats': count - distinct_rgb,
                'same_assessed_slot_signature_after_first': sum(n - 1 for keys, n in membership.items() if keys),
                'png_bytes': sum((out / trace[p]['path']).stat().st_size for p in pts)}
            groups[proposal['requested']][proposal['mode']] = record
            deliveries.append(record)
        for item in deliveries:
            others = [v for k,v in groups[item['requested']].items() if k != item['mode']]
            item['roles_unique_to_this_method_at_equal_count'] = sorted(set(item['slots']) - {s for r in others for s in r['slots']})
            item['compounds_unique_to_this_method_at_equal_count'] = sorted(set(item['complete_compounds']) - {s for r in others for s in r['complete_compounds']})
            flat.append({'case':case,'n':item['count'],'mode':item['mode'],'slots':len(item['slots']),
                'complete_compounds':len(item['complete_compounds']), 'exact_brackets': len(item['exact_brackets']),
                'unclassified_frames': len(item['unclassified_pts']), 'rgb_duplicate_frames':item['exact_pixel_repeats'],
                'repeated_role_signature': item['same_assessed_slot_signature_after_first'], 'native_png_bytes':item['png_bytes'],
                'missed_roles':sorted(set(role['slots']) - set(item['slots'])), 'unique_roles':item['roles_unique_to_this_method_at_equal_count']})
        upper = covered(inventory['source_pts'], role)
        coarse = covered(inventory['coarse_pts'], role)
        nom = covered(inventory['evidenweft_pts'], role)
        write(child / 'delivery_score.json', {'deliveries':deliveries,'source_assessed_upper_bound': assessed(upper),
            'coarse_pool_potential':assessed(coarse),'evidenweft_inventory_potential':assessed(nom),
            'original_delivery_audit': list(trace.values()), 'compute_seconds_catalogue': inventory['compute_seconds']})
    csv_rows(out/'equal_count.csv', flat)
    print(f'{len(flat)} source-native local delivery comparisons in {out}', flush=True)


def tool_candidates(text):
    """Exact metadata occurrences in a tool result; not delivered originals."""
    pts = {int(p) for p in re.findall(r'\bpts=(-?\d+)', text)}
    try:
        obj = json.loads(text)
    except ValueError:
        return pts
    def visit(row):
        if isinstance(row, dict):
            if type(row.get('pts')) is int:
                pts.add(row['pts'])
            for k,v in row.items():
                if k in {'comparison_pts', 'source_pts'} and isinstance(v, list):
                    pts.update(p for p in v if type(p) is int)
                elif isinstance(v, (dict,list)):
                    visit(v)
        elif isinstance(row, list):
            for v in row:
                visit(v)
    visit(obj)
    return pts


def group_question_ids(answer):
    ids = [v.get('id') for v in (answer or {}).get('open_questions', []) if isinstance(v, dict)]
    return {k for k in ids if k}


def inspect_run(folder, protocol, rule, inventory, pack, rounds_csv):
    run = read(folder / 'result.json')
    case, arm = run['case'], run['arm']
    full = read(HERE / 'evaluation_only/source_review' / case / 'native_timeline.json')
    source = {r['pts']:r for r in full['frames']}
    allowed = read(PUBLIC / case / ('uniform.json' if arm == 'A0' else 'multi.json'))
    candidates = {r['pts'] for r in allowed['frames']}
    full_catalogue = set(inventory['evidenweft_pts'])
    pool = set(inventory['coarse_pts'])
    catalog = {r['pts']:r['frame_id'] for r in allowed['frames']}
    control = [json.loads(line) for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'calls.jsonl').exists() else []
    tool_surfaces = [r['tools'] for r in control if r.get('event') == 'tool_surface']
    if tool_surfaces and arm != 'A2':
        assert all(set(s) == set(protocol['common_tools']) for s in tool_surfaces)
    if arm == 'A2':
        assert all(set(protocol['common_tools']) < set(s) for s in tool_surfaces)
    for call in control:
        assert set(call.get('returned_catalogue_pts', [])) <= candidates
    history = [json.loads(l) for p in (folder/'cache/sessions').glob('*.jsonl') for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    ledger = defaultdict(list)
    derivatives = {}
    for row in history:
        if row.get('event') in {'source_resample','delivery_request'}:
            assert row['video_sha256'] == run['source_sha256']
            for f in row['frames']:
                ledger[f['evidence_id']].append((row['event'], f))
        if row.get('event') == 'derived_image':
            derivatives[row['derived_id']] = row
    natives = list((folder/'cache/observer-index-v5').glob('*/native'))
    cache = natives[0] if natives else None
    seen = {}
    checked_rgb = {}
    stage = {}
    review_turns = []
    for number, rnd in enumerate(run['rounds'], 1):
        read_events = []
        failures = []
        results = {t['event_index']:t for t in rnd['tool_results']}
        by_call = {c['callId']:c for c in rnd['calls']}
        by_type = Counter()
        for image in rnd['images']:
            attachment = image['attachment']
            aid = attachment['attachmentId'].removeprefix('sha256:')
            evidence = image['evidence_id']
            t = results[image['event_index']]
            ct = by_call.get(t.get('call'), {}).get('name')
            if evidence is None:
                crop = next((v for v in derivatives.values() if v['attachment_sha256'] == aid), None)
                if crop:
                    read_events.append({'kind':'derived','evidence_id':crop['derived_id'],'parent':crop['parent_evidence_id'],
                                        'pts':crop['pts'],'pixels':[attachment['width'],attachment['height']],'via':ct,
                                        'followed_by_model_reply':image['followed_by_model_reply']})
                    by_type['derived'] += 1
                    continue
                failures.append({'type':'unmatched_attachment', 'sha':aid})
                continue
            match = next(((via,f) for via,f in ledger[evidence] if f['attachment_sha256'] == aid and (ct is None or via == ct)), None)
            if not match:
                failures.append({'type':'unknown_original', 'evidence':evidence, 'sha':aid})
                continue
            via, f = match
            pts = f['pts']
            p = cache / f'frame_{f["source_frame"]}.png'
            original_sha = source[pts]['decoded_rgb_sha256']
            assert source[pts]['frame'] == f['source_frame'] and f['time_base'] == full['format']['time_base']
            if pts not in checked_rgb:
                assert sha256(p.read_bytes()).hexdigest() == f['png_sha256']
                with Image.open(p) as original:
                    assert original.size == (full['format']['width'], full['format']['height'])
                    assert original_sha == sha256(original.convert('RGB').tobytes()).hexdigest()
                checked_rgb[pts] = f['png_sha256']
            assert checked_rgb[pts] == f['png_sha256']
            assert [attachment['width'],attachment['height']] == [full['format']['width'],full['format']['height']]
            record = {'kind':'original','evidence_id':evidence,'pts':pts,'seconds':source[pts]['seconds'],'via':via,
                'catalogue':pts in candidates,'in_coarse':pts in pool,'in_evidenweft':pts in full_catalogue,
                'original_rgb_sha256':original_sha,'attachment_sha256':aid,'followed_by_model_reply':image['followed_by_model_reply'],
                'image_bytes':attachment.get('bytes',0),'event_index':image['event_index'],'path':str(p.relative_to(HERE))}
            read_events.append(record)
            by_type['original'] += 1
        metadata = set()
        mentions = set()
        for t in rnd['tool_results']:
            called = by_call.get(t.get('call'),{})
            name = called.get('name', '')
            if name and not t['images'] and name not in {'source_resample','delivery_request','evidence_list','claim_verify','evidence_crop','video_open'}:
                metadata |= tool_candidates(t['text']) & source.keys()
        for c in rnd['calls']:
            raw = c['arguments']
            if not isinstance(raw,str):
                continue
            try:
                args = json.loads(raw)
            except ValueError:
                continue
            mentions |= tool_candidates(raw) & source.keys()
            if c['name'] == 'delivery_request':
                mentions |= {k for k, v in catalog.items() if v in args.get('frame_ids',[])}
                mentions |= set(args.get('pts', [])) & source.keys()
        first = []
        for r in read_events:
            if r['kind'] == 'original' and r['followed_by_model_reply'] and r['pts'] not in seen:
                seen[r['pts']] = r
                first.append(r)
        answer = rnd['answer'] or {}
        using = {r['evidence_id']:r for r in seen.values()}
        trace = []
        for i, fact in enumerate(answer.get('facts',[]),1):
            if not isinstance(fact,dict):
                continue
            ids = fact.get('evidence_ids',[])
            refs = [using[j] for j in ids if j in using]
            ref_pts = [r['pts'] for r in refs]
            trace.append({'id': i,'claim':fact.get('claim'),'kind':fact.get('kind'),
                'cited_pts':ref_pts,'cited_seconds':[round(source[p]['seconds'],6) for p in ref_pts], 'cited_ids':ids,
                'unresolved_ids':[j for j in ids if j not in using], 'candidate_roles_from_images':assessed(covered(ref_pts, rule))})
        stage[number] = {'metadata_pts': sorted(metadata), 'explicit_requested_pts': sorted(mentions), 'original_reads': read_events,
            'attachment_failures':failures,'distinct_originals_through_round':len(seen), 'first_new_originals':first,
            'image_evidence_through_round':assessed(covered(seen.keys(), rule)),
            'first_read_outside_catalogue': [r for r in first if not r['catalogue']], 'read_costs': dict(by_type)}
        usage = rnd['usage']
        rounds_csv.append({'blind_id':run['blind_id'],'case':case,'arm':arm,'repeat':run['repetition'],'attempt':run['attempt'],
            'round':number,'status':run['status'],'stop':rnd['stop']['kind'],'valid_original_receipts':by_type['original'],
            'derived_receipts':by_type['derived'],'unmatched_attachments':len(failures), 'new_distinct_originals':len(first),
            'distinct_originals_cumulative':len(seen), 'new_outside_catalogue':len(stage[number]['first_read_outside_catalogue']),
            'new_outside_coarse':sum(not v['in_coarse'] for v in first),
            'tool_calls': len(rnd['calls']),'tool_errors':sum(bool(t['error']) for t in rnd['tool_results']),
            'text_utf8_bytes':sum(r['metadata_bytes'] for r in rnd['tool_results']),
            'picture_bytes': sum(i['attachment'].get('bytes',0) for i in rnd['images']),
            'noncache_input':usage.get('inputTokens'), 'output_tokens':usage.get('outputTokens'),
            'cached_read':usage.get('cacheReadTokens'), 'cache_write':usage.get('cacheWriteTokens'), 'seconds':rnd['seconds']})
        review_turns.append({'turn':number, 'answer_available':isinstance(rnd['answer'],dict) and rnd['stop']['kind']=='completed',
            'stop':rnd['stop']['kind'], 'facts':trace, 'summary':answer.get('summary'), 'questions':answer.get('open_questions'),
            'prior_questions':answer.get('prior_questions'), 'explicit_pt_sources':{
                q:{k:r[k] for k in ('path','via','seconds')} for q,r in seen.items() if any(q in t['cited_pts'] for t in trace)}})
    retention = {}
    if len(run['rounds']) == 2:
        prior = group_question_ids(run['rounds'][0]['answer'])
        final = group_question_ids(run['rounds'][1]['answer'])
        retention = {'prior_question_ids':sorted(prior), 'new_question_ids':sorted(final - prior),
            'only_reported_as_still_open':sorted(final & prior)}
    pack[case].append({'blind_id': f'{run["blind_id"]}-t{run["attempt"]}', 'rounds':review_turns})
    return {'blind_id':run['blind_id'], 'case':case,'arm':arm,'repeat':run['repetition'],'attempt':run['attempt'],'status':run['status'],
        'stages':stage,'question_tracking':retention,'runtime_source':run['source'],'tool_surface':tool_surfaces,
        'algorithm_inventory':assessed(covered(full_catalogue,rule)), 'input_catalogue':assessed(covered(candidates,rule)),
        'coarse':assessed(covered(pool,rule)), 'source_upper':assessed(covered(source,rule))}


def native(out):
    verify()
    rubric = read(HERE / 'evaluation_only/rubric.json')['cases']
    protocol = read(HERE / 'public_protocol.json')
    out.mkdir(parents=True, exist_ok=False)
    pack = defaultdict(list)
    run_rows = []
    cost_rows = []
    invalid = {f'{r["blind_id"]}-t1' for r in read(HERE/'transport_amendment.json')['first_attempts']
               if r['effective_status'] == 'technical_failure'} if (HERE/'transport_amendment.json').is_file() else set()
    inventory = {k:read(HERE/'inventory'/k/'inventory.json') for k in rubric}
    for result in sorted((HERE/'sessions').glob('*/attempt-*/result.json')):
        run = read(result)
        row = inspect_run(result.parent,protocol,rubric[run['case']],inventory[run['case']],pack,cost_rows)
        if f'{run["blind_id"]}-t{run["attempt"]}' in invalid:
            row['recorded_status'] = row['status']
            row['status'] = 'technical_failure'
        run_rows.append(row)
    effective = {(r['blind_id'],r['attempt']):r['status'] for r in run_rows}
    for r in cost_rows:
        r['recorded_status'] = r['status']
        r['status'] = effective[r['blind_id'],r['attempt']]
    write(out/'attribution_inputs.json', run_rows)
    csv_rows(out/'costs_and_receipts.csv',cost_rows)
    for case, records in pack.items():
        eligible = [r for r in records if r['blind_id'] not in invalid]
        # Recode before standalone review. Keep the identity key separately.
        from random import Random
        Random(40517).shuffle(eligible)
        codes = {f'R{i:02d}':r['blind_id'] for i,r in enumerate(eligible,1)}
        write(out/f'{case}-blind-review.json',[{'review_id':key,'rounds':r['rounds']} for key,r in zip(codes,eligible)])
        write(out/f'{case}-review-key.json',codes)
    print(f'{len(run_rows)} native attempted sessions, {len(cost_rows)} recorded turns', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('offline','native'))
    parser.add_argument('out', type=Path)
    a = parser.parse_args()
    (offline if a.kind == 'offline' else native)(a.out)
