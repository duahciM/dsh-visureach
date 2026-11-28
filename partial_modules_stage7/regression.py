"""Check the affected boundaries plus the prior declared independent release paths."""

from argparse import ArgumentParser
import subprocess
import traceback

import cv2

from dacf_research.prepare_inputs import ROOT,digest,read,write
from partial_modules_stage5.isolation import boundary_contract,influence_graph
from partial_modules_stage5.verify import verify_six
from partial_modules_stage6.pre_release import first_undeclared_change,scoped_outputs
from partial_modules_stage7.inputs import H


def check_source_history():
    lock=read(H/'freeze.json')
    for name in ('history','prediction','assessor_files'):
        for path,sha in lock[name].items():
            assert digest(ROOT/path)==sha,(name,path)
    windows=read(H/'window_lock.json')
    assert all(digest(H/'automatic'/case/'results.json')==sig for case,sig in windows['automatic'].items())
    assert all(digest(H/key)==sig for key,sig in windows['source_code'].items())
    return {key:len(lock[key]) for key in ('history','prediction','assessor_files')}


def window_comparability():
    count=0;structures=0;relations=0;reproduced={}
    def details(data):
        return [(x['pts'],x['applicable_region'],x['dimension_id'],x['calculation_id'])
                for x in data['nominations'] if x['dimension_id'] in ('D1','D3')]
    for name in ('reel','real_return','new_recording'):
        before=H/'windows/blind'/name
        after=H/'windows/reference'/name
        old=read(before/'A.json')
        b=read(before/'B.json')
        matched=read(before/'D0.json')
        assert details(b)==details(matched),('independent additional reasons deleted or changed raw D1/D3',name)
        assert b['pair_channel']==matched['pair_channel']==old['pair_channel']=='FULL'
        check=read(before/'summary.json')['checks']
        assert all(row['executed_calls']==row['A_equivalent_execution_for_those_targets'] for row in check.values())
        if name!='new_recording':
            prev=read(ROOT/f'partial_modules_stage6/observations/automatic/{name}/GLOBAL.json')
            reference={(r['pts'],tuple(r['applicable_region']),r['dimension_id']) for g in prev['nomination_groups']
                       for r in g['nominations'] if r['dimension_id'] in ('D1','D3')}
            observed={(q[0],tuple(q[1]),q[2]) for q in details(old)}
            assert reference==observed,('old-window semantic regression',name)
            reproduced[name]=len(observed)
        for folder,condition in ((before,'E'),(after,'D')):
            data=read(folder/f'{condition}.json')
            bounds={s['id']:s['pts'] for s in data['scope']['closed']}
            roots={r['rank_independent_root'] for r in data['measurements']}
            stable={r['calculation_id'] for r in data['nominations'] if r['dimension_id']=='structure'}
            structures+=len(stable)
            for r in data['measurements']:
                lo,hi=bounds[r['context_id']]
                assert all(lo<p<hi for p in [r['pts']]+r['comparison_pts_requested']),('scope parents',name)
                count+=1
            for f in data['nominations']:
                if f['dimension_id']=='relation':
                    a,b=f['requires']
                    assert a in roots and b in stable and 'identity' not in f,('unsupported relation dependency',name)
                    relations+=1
    return {'old_D1_and_D3_keys_reproduced':reproduced,'scoped_measurements_verified':count,
        'independent_structure_proofs':structures,'independent_relations_without_identity_dependency':relations}


def actual_host(out):
    original=ROOT/'partial_modules_stage5/host_regression.mjs'
    code=original.read_text(encoding='utf8')
    key="const study=resolve('partial_modules_stage5')"
    assert code.count(key)==1
    script=H/f'{out.name}-host.mjs'
    with script.open('x',encoding='utf-8') as f:f.write(code.replace(key,"const study=resolve('partial_modules_stage7')"))
    stamped=f'{out.name}-host'
    with (out/'host.log').open('x',encoding='utf-8') as f:
        subprocess.run(['node',str(script),stamped],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=105,check=True)
    result=read(H/stamped/'result.json')
    assert all(x['state']=='stopped' for x in result['checked_stopped_node']['policy_hint']['statuses'].values())
    assert result['original_attachment_without_scene']==read(
        ROOT/'partial_modules_stage5/host-verified/result.json')['original_attachment_without_scene']
    return {'original_same_pixels_as_stage5':True,'PTS':result['probe']['pts'],'receipt':f'{stamped}/result.json'}


def run(out):
    out.mkdir()
    tests={};fails={}
    def check(name,value,path):
        try:tests[name]=value()
        except (AssertionError,KeyError,ValueError,OSError,subprocess.SubprocessError):
            fails[name]={'first_dependency':name,'impact_path':path,'details':traceback.format_exc()}
    cv2.setNumThreads(1);cv2.setRNGSeed(730)
    check('historical_and_prediction_freeze',check_source_history,'original → cached visual evidence → frozen FIRST and reference')
    check('dimensions',verify_six,'original → pool → independent dimension nominations')
    check('strict_context_contract',boundary_contract,'qualified scope → local decisions; independent source access')
    check('stage6_scoped_invariants_rechecked',scoped_outputs,'previous stage actual scopes → raw measurements and derived candidate roots')
    check('new_comparison_and_scope_invariants',window_comparability,'global/shift/scope → D1,D3 parents → independent per-frame relations')
    prior=read(ROOT/'partial_modules_stage4/inputs/source_inventory.json')
    for key in ('roles','camera','street-repeat'):
        check(f'raw_measurement_independent_of_M2_{key}',lambda:influence_graph(key,prior[key],out),
              'independent nominations → proposed identity → interpretation only')
    check('original_reread_after_S0',lambda:actual_host(out),'stopped research route → independent exact decoder/frozen original')
    baseline={('measurement','pixels'):8,('interpretation','mode'):'identity'}
    changed=baseline|{('measurement','pixels'):4,('interpretation','mode'):'conditional'}
    tests['injected_fault']=first_undeclared_change(baseline,changed,{'interpretation'})
    assert tests['injected_fault']['stage']=='measurement'
    write(out/'result.json',{'passed':not fails,'checks':tests,'failures':fails,'live_default_revision':'unchanged'})
    return not fails


if __name__=='__main__':
    args=ArgumentParser();args.add_argument('--out',default='regression');p=args.parse_args()
    out=H/p.out
    assert out.parent==H
    raise SystemExit(0 if run(out) else 1)
