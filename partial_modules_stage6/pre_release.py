"""Reproducible, write-isolated dependency tests for the offline release lane."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import traceback

import cv2

from dacf_research.prepare_inputs import ROOT, digest, read, write
from partial_modules_stage5.isolation import boundary_contract, influence_graph, raw_measurements
from partial_modules_stage5.verify import verify_six
from partial_modules_stage6.material import H


def first_undeclared_change(baseline, variant, declared):
    for dependency,key in baseline:
        if dependency not in declared and baseline[dependency,key]!=variant.get((dependency,key)):
            return {'stage':dependency,'field':key,'before':baseline[dependency,key],
                    'after':variant.get((dependency,key)),'error':'undeclared downstream change'}
    for dependency,key in variant.keys()-baseline.keys():
        if dependency not in declared:
            return {'stage':dependency,'field':key,'after':variant[dependency,key],'error':'undeclared new output'}
    return None


def amr_source(out):
    original=ROOT/'partial_modules_stage5/host_regression.mjs'
    code=original.read_text(encoding='utf8')
    root="const study=resolve('partial_modules_stage5')"
    folder="assert.equal(resolve(out,'..'),study)"
    assert code.count(root)==1 and code.count(folder)==1
    # Use the known test byte-for-byte except for the isolated destination.
    adapted=H/f'{out.name}-host.mjs'
    with adapted.open('x',encoding='utf8') as f:
        f.write(code.replace(root,"const study=resolve('partial_modules_stage6')"))
    target=f'{out.name}-host'
    with (out/'amr.log').open('x',encoding='utf8') as log:
        subprocess.run(['node',str(adapted),target],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100)
    result=read(H/target/'result.json')
    expected=read(ROOT/'partial_modules_stage5/host-verified/result.json')
    assert result['original_attachment_without_scene']==expected['original_attachment_without_scene']
    assert result['refinement']['host_status']=='computed' and len(result['native_attachment_digests'])==3
    return {'status':'passed','receipt':f'{target}/result.json','stop_after_which_refined':result['checked_stopped_node']['policy_hint'],
           'same_exact_probe_as_stage5':result['probe']}


def scoped_outputs():
    auto=H/'observations/automatic'
    ref=H/'observations/reference'
    a=read(auto/'run_complete.json'); b=read(ref/'run_complete.json')
    assert a['cases']==b['cases'] and not a['read_assessor'] and b['read_assessor']
    locks=read(H/'downstream_lock.json')
    assert digest(H/'downstream.py')==locks['code_sha256']
    evidence=0;relations=0
    for name in a['cases']:
        original=H/'automatic'/name/'auto.json'
        assert digest(original)==locks['automatic_outputs'][name],('envelope_version',name)
        result=read(original)
        initial=ROOT/result['initial_index']
        assert all(digest(initial/key)==sha for key,sha in result['index_sha'].items()),('frozen_catalogue',name)
        for location,arms in ((auto,['GLOBAL','AUTO']),(ref,['MANUAL','WRONG'])):
            for arm in arms:
                report=read(location/name/f'{arm}.json')
                assert report['input_hashes']==result['index_sha'] and report['source_sha256']==result['sha256'],('input',name,arm)
                assert report['additional_source_decodes']==0 and report['raw_source_access_remains_available'],('original_access',name,arm)
                lookup={r['id']:r for r in report['scope']['closed']} if report['scope'] else {}
                for group in report['nomination_groups']:
                    if report['scope']:
                        lo,hi=lookup[group['scope_id']]['pts']
                        assert all(lo<p<hi for p in group['positions']),('outside_strict_scope',name,arm)
                    refs=set(group.get('exact_context_input_frame_ids',[]))
                    for row in group['nominations']:
                        assert row['source_root'] in refs and set(row['comparison_frame_ids'])<=refs,('premature_dependency_escape',name,arm,row['pts'])
                        evidence+=1
                local=location/name/f'{arm}-local.json'
                if local.is_file():
                    for group in read(local)['partitions']:
                        x,y=(group[k] for k in ('m5_with_identity','raw_m5_without_identity'))
                        assert x['same_tick_projected']==y['same_tick_projected'],('single-time_raw',name,arm)
                        assert raw_measurements(x)==raw_measurements(y),('cross-time_raw',name,arm)
                        relations+=len(y['raw_cross_time'])
    return {'status':'passed','actual_supplementary_nominations_with_scoped_parents':evidence,
            'raw_cross_time_confirmed_equal_on_identity_withdrawal':relations,'sources':len(a['cases'])}


def run(out,include_scopes=False):
    out.mkdir()
    checks={};failures={}
    def check(name,fn,pipeline):
        try:
            checks[name]=fn()
        except (AssertionError,ValueError,KeyError,OSError,subprocess.SubprocessError):
            failures[name]={'first_scope':name,'impact_path':pipeline,'error':traceback.format_exc()}
    lock=read(H/'freeze.json')
    def frozen():
        mismatches=[[k,p] for k in ('history','predictors') for p,sha in lock[k].items()
                    if not (ROOT/p).is_file() or digest(ROOT/p)!=sha]
        assert not mismatches, mismatches[:1]
        assert digest(ROOT/lock['assessor']['file'])==lock['assessor']['sha256']
        return {'status':'passed','files':len(lock['history'])+len(lock['predictors'])}
    check('immutable_sources_and_releases',frozen,'source/cache/config → first nomination → old study')
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    cases=read(ROOT/'partial_modules_stage4/inputs/source_inventory.json')
    for name in ('roles','camera','street-repeat'):
        check('old_'+name,lambda:influence_graph(name,cases[name],out),
              'region evidence → role view → schedule → raw geometry / conditional interpretation')
    check('six_dimension_switches',verify_six,'original timeline → independent D1–D6 → published catalogue')
    check('scene_scope_edges',boundary_contract,'typed brackets → strict interior → no change to source access')
    check('amr_free_reread_after_stop',lambda:amr_source(out),'automatic guidance → explicit exact reader → new node only')
    if include_scopes:
        check('readonly_scope_overlay',scoped_outputs,'boundary input → exact derived scope → contextual roots and raw relations')
    original={('evidence','root'):'verified',('measurement','pixel_delta'):4,('interpretation','support_scope'):'identity_linked'}
    allowed=original|{('interpretation','support_scope'):'conditional'}
    broken=allowed|{('measurement','pixel_delta'):3}
    assert first_undeclared_change(original,allowed,{'interpretation'}) is None
    check('first_changed_field_diagnostic',lambda:first_undeclared_change(original,broken,{'interpretation'}),
          'simulated raw measurement change → recorded first failure; allowed interpretation change ignored')
    assert checks['first_changed_field_diagnostic']['stage']=='measurement'
    write(out/'result.json',{'passed':not failures,'checks':checks,'failures':failures,
        'scope':'physical input and original algorithms, one isolated native worker host; no model session'})
    return not failures


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--out',default='release-check')
    parser.add_argument('--include-scopes',action='store_true')
    args=parser.parse_args()
    path=H/args.out
    if path.parent!=H:
        raise ValueError('Use a distinct stage-6 output directory name.')
    raise SystemExit(0 if run(path,args.include_scopes) else 1)
