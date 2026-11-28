"""Check what an actual Stage 3 measurement retains under a local scope revision."""

from argparse import ArgumentParser

from dacf_research.prepare_inputs import read, write
from partial_modules_stage2.ledger import Ledger
from partial_modules_stage1.study import box_distance
from partial_modules_stage3.material import H


def register_raw(ledger, row):
    ledger.add(row['measurement_ref'], 'raw_relation', row['source_roots'], measurement=row)


def review_controlled(out):
    data = read(H / 'diagnostic_manual/roles/A4.json')
    ledger = Ledger(out / 'controlled.journal.jsonl', data['source_sha256'])
    relation = next(row for row in data['relation']['raw_cross_time']
                    if row['identity_dependency'] and row['native_projection_delta_xy'] != [0, 0])
    link = next(row for row in data['identity']['links']
                if row['calculation_root'] == relation['identity_dependency'])
    coappearance = next(row for row in data['identity']['coappearance_projections']
                        if row['pts'] in link['pts'] and set(row['views']) & set(link['views']))
    views = {region['id']: region for region in data['regions']}
    projections = [views[view] for view in coappearance['views']]
    assert box_distance(*(r['native_xywh'] for r in projections)) > 0

    register_raw(ledger, relation)
    local = link['calculation_root']
    ledger.add('actual-local-appearance', 'appearance', link['source_roots'], calculation=link)
    ledger.add(local, 'identity_hypothesis', link['source_roots'], ['actual-local-appearance'],
               scope='Only the paired views in this controlled case', view_ids=link['views'])
    ledger.add('bounded-interpretation', 'identity_dependent_relation', relation['source_roots'],
               [relation['measurement_ref'], local])

    # Exercise a deliberately overbroad projection claim, using an actual observed competing projection.
    ledger.add('trial-single-projection', 'identity_hypothesis', link['source_roots'], [local],
               scope='Controlled test claim that both disjoint observations are one visible projection')
    ledger.add('trial-dependent-summary', 'identity_dependent_relation', relation['source_roots'],
               [relation['measurement_ref'], 'trial-single-projection'])
    ledger.add('coappearance-two-views', 'coappearance_projection', [coappearance['source_root']],
               projections=projections, physical_exclusion='unresolved')
    try:
        ledger.withdraw('trial-single-projection', 'coappearance-two-views', 'Unassessed physical exclusion.')
    except ValueError as exc:
        unknown_rejected = str(exc)
    else:
        raise AssertionError('An unassessed projection must not be interpreted as counterevidence.')
    ledger.add('disjoint-visible-projections', 'coappearance_projection', [coappearance['source_root']],
               ['coappearance-two-views'], status='counter_evidence_in_scope',
               scope='Two image projections only. No physical instance exclusion.')
    before = {row['id']: row for row in ledger.query()}
    ledger.withdraw('trial-single-projection', 'disjoint-visible-projections',
                    'Two distinct image regions are visible at that exact PTS.')
    recovered = Ledger(out / 'controlled.journal.jsonl', data['source_sha256'])
    after = {row['id']: row for row in recovered.query()}
    assert after['trial-single-projection']['effective_status'] == 'withdrawn'
    assert after['trial-dependent-summary']['effective_status'] == 'dependency_withdrawn'
    for key in (relation['measurement_ref'], local, 'bounded-interpretation'):
        assert after[key] == before[key]
    return {'before': before, 'after_from_reloaded_ledger': after, 'unknown_rejected': unknown_rejected}


def review_street(out):
    data = read(H / 'diagnostic_manual/street-repeat/A4.json')
    ledger = Ledger(out / 'street.journal.jsonl', data['source_sha256'])
    relations = data['relation']['raw_cross_time']
    assert relations and not data['identity']['links']
    for row in relations:
        assert row['identity_dependency'] is None
        register_raw(ledger, row)
    restored = Ledger(out / 'street.journal.jsonl', data['source_sha256'])
    assert len(list(restored.query(kind='raw_relation'))) == len(relations)
    return {'raw_rows_recovered_without_link': len(relations)}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out', default='ledger_check')
    out = H / parser.parse_args().out
    if out.parent != H:
        raise ValueError('Use a study-local result directory.')
    out.mkdir()
    write(out / 'result.json', {'controlled': review_controlled(out), 'street': review_street(out)})
