"""Summarize request injection, source-backed capabilities and raw cognition outcomes."""
from collections import Counter
import json
from pathlib import Path
import re

from dacf_research.prepare_inputs import ROOT,digest,read,write
from dacf_round7.run import H,PROTOCOL
from dacf_round7.compact_review import selected,patches
from dacf_round7.score import call_checks
from dacf_round7.receipts import lines


def label_count(rows, field):
    return dict(Counter(v[field] for v in rows))


def scrub_descriptions(value):
    if isinstance(value,dict):
        return {k:scrub_descriptions(v) for k,v in value.items() if k!='description'}
    if isinstance(value,list):
        return [scrub_descriptions(x) for x in value]
    return value


def request_audit(data):
    out={}
    descriptions={}
    for (phase, key),v in data.items():
        p=H/v['path']; requests=lines(p.with_name('provider.jsonl.effective')); first=requests[0]
        descriptions[phase]=first['tools']
        assert scrub_descriptions(descriptions[phase])==scrub_descriptions(first['tools'])
        out[f'{phase}/{key}']={'wire':p.with_name('provider.jsonl.effective').relative_to(H).as_posix(),
            'first_tool_list_hash':first['tool_hash'],'requests':len(requests),'first_request_system':first['system'],
            'tools':len(first['tools']), 'stable_tool_hash':len({a['tool_hash'] for a in requests})==1,
            'later_help':any('example_parameters' in json.dumps(r['tool_results']) or 'map before you' in json.dumps(r['tool_results']) for r in requests[1:]),
            'revised_overview_in_system':'map before you' in str(first['system']),
            'revised_overview_in_results':any('map before you' in json.dumps(r['tool_results']) for r in requests[1:])}
    a,b=descriptions['original'],descriptions['revised']
    assert scrub_descriptions(a)==scrub_descriptions(b)
    audit_path=next((H/'audit').glob('*/provider.jsonl.effective'))
    actual=lines(audit_path)
    guide_count=sum('example_parameters' in json.dumps(r['tool_results']) for r in actual)
    return {'same_tool_validation_contract_without_docs':True,'tool_names':[t['name'] for t in a],
            'changed_top_level_tool_descriptions':sum(x.get('description')!=y.get('description') for x,y in zip(a,b)),
            'representative_original_live_audit':{'path':audit_path.relative_to(H).as_posix(),'requests':len(actual),
              'guide_request_steps':guide_count,'visible_hints':any('Optional related evidence:' in json.dumps(r['tool_results']) for r in actual)},
            'conditions':out}


def accepted_scores():
    scores={}; grading=[]; arch={}
    base=H/'grading-compact'; mapping=read(base/'index.json')
    for batch,parts in mapping.items():
        path=base/batch/'result.json'
        if not path.is_file():
            continue
        r=read(path); a=r.get('answer') or {}
        grade=a.get('scores')
        grading.append({'file':path.relative_to(H).as_posix(),'stop':r['stop'],'usable':isinstance(grade,dict),'mapped_rows':len(parts)})
        if not isinstance(grade,dict):
            continue
        for idx,marks in grade.items():
            if idx not in parts:
                continue
            phase,task,src,*tool=parts[idx]
            if tool:
                if isinstance(marks,list) and len(marks)==6 and all(isinstance(m,str) and re.fullmatch(r'[01?]+', m) and len(m)==sz for m,sz in zip(marks,[4,6,6,6,1,1])):
                    scores[phase,tool[0]]={'masks':marks,'remark':a.get('notes',{}).get(idx),'file':str(path.relative_to(H))}
            elif isinstance(marks,dict):
                arch[phase]=marks
    return scores,grading,arch


def measure(data, graded):
    summary={}; rows=[]
    required={'phenomenon','known','question','fit','result','supports','unknown'}
    for cond in ('original','revised'):
        s=Counter(); match=[]; check=[]
        for (phase,task),v in data.items():
            if cond!=phase:
                continue
            a=v['answer'];s['valid_sessions']+=1
            s['explicit_sdk_help_calls']+=sum(c['name'] in ('observer_guide','amr_query') for c in read(H/v['path'])['calls'])
            for raw in a.get('tools',[]):
                name=raw.get('tool');s['tools']+=1
                judge=graded.get((phase,name),{}); marks=judge.get('masks',[])
                if marks:
                    s['independent_reviewed_tools']+=1
                for i,(kind,case) in enumerate([('valid',sc) for sc in raw.get('scenarios',[])]+[('misuse',raw.get('misuse')),('skip',raw.get('skip'))],1):
                    if not isinstance(case,dict):
                        continue
                    required_gaps=sorted(k for k in required if not isinstance(case.get(k),str) or not case[k].strip())
                    s[kind+'_cases']+=1;s[kind+'_structure_complete']+=not required_gaps
                    label=marks[i] if marks and i<len(marks) else None
                    rows.append({'phase':phase,'tool':name,'case_type':kind,'case_index':i,'scenario':case,'missing_required_fields':required_gaps,
                          'independent_marks':label,'independent_explanation_marks':marks[0] if marks else None,
                          'rating_source':judge.get('file'),'source':v['path']})
            if task.startswith('match'):
                for item in a.get('answers',[]):
                    checks=call_checks(item.get('calls',[]));check.extend(checks)
                    match.append({'id':item.get('id'),'checks':checks,'source':v['path'],'parse':v['parse']})
            if task=='architecture':
                s['architecture_response_slots']=len(a.get('answers',[]))
                s['proposed_combinations']=len(a.get('combinations',[]))
                combinations=[{'question':x.get('question'),'steps':call_checks(x.get('steps',[]))} for x in a.get('combinations',[])]
        summary[cond]={'counts':dict(s),'matching':match,'schema_candidate_call_counts':label_count(check,'verdict'),
                       'combination_checks':combinations}
    return summary,rows


def reference_document():
    original=read(H/'original_surface.json')['tools']; actual=read(H/'engineering-retry/result.json')['tool_results']; basis=patches()
    files=[ROOT/'dacf_round5/release-fixed/installed/package/baseline'/name for name in ('guide.js','index.js','integration.js')]+[ROOT/'dacf_round6/persistent.mjs']
    code={p:p.read_text('utf8').splitlines() for p in files}
    material=['# 离线工具能力参考','',
      '冻结的 34 个插件工具。评价方使用的事实勘误仅纠正已核验职责。DSH 自带的 `exit_plan_mode` 和独立实时工具不属于本表。',
      '每项列出工具存在理由、实际输入和返回、来源与邻近工具边界。完整签名和初始说明见 [原始 surface](original_surface.json)。执行例与依赖状态来自 [本轮实际宿主验收](engineering-retry/result.json)。','',
      'Round 6 预先生成双锚点计划。打开视频时，通用粗目录可能重用缓存，也可能重新构建。研究计划内具备计算根的部分是保存证据，是否已解码/交付的记录仍按会话区分。',
      '与早期架构表的差异包括语义和文字依赖是可选的、自动化物理身份未实现、图中关系是投影、变化触发与多尺度变化树是两条已有路径。','']
    for name,v in original.items():
        spec=v['input']; out=actual[name]
        snippet=(rf"(?:name:\s*['\"]{name}['\"]|tool\(['\"]{name}['\"]|^\s+{name}:|['\"]{name}['\"]:)")
        loc=[]
        for source,contents in code.items():
            found=next((i for i,q in enumerate(contents,1) if re.search(snippet,q)),None)
            if found:
                path='../../p1-transition/'+source.relative_to(ROOT).as_posix()
                loc.append(f'[{source.name} L{found}]({path}#L{found})')
        if name in ('amr_query','amr_refine'):
            loc.append('[研究执行入口](../dacf_round6/persistent.mjs)')
        params=', '.join(f'`{k}`' for k in spec.get('properties',{})) or '无字段'
        required=', '.join(f'`{k}`' for k in spec.get('required',[])) or '视模式而定，详见契约'
        material.extend([f'## `{name}`','',basis[name],'',f'输入字段 {params}。必需字段 {required}。',
          f'本轮实际返回 `{out["status"]}`，附件 {out["attachments"]}，字段 '+', '.join(f'`{k}`' for k in out['keys'])+'。',
          ((' · '.join(loc)) or '[冻结索引](freeze.json)'), ''])
    with (H/'TOOL_CAPABILITY_REFERENCE.md').open('x',encoding='utf8') as f:f.write('\n'.join(material))


def main():
    data=selected()
    assert len(data)==18
    assert all(digest(ROOT/p)==h for p,h in read(H/'freeze.json')['files'].items())
    score,attempts,architectures=accepted_scores()
    metrics,scenarios=measure(data,score)
    audit=request_audit(data)
    write(H/'prompt_audit.json',audit)
    write(H/'cognition_summary.json',{'conditions':metrics,'independent_architecture_review':architectures,
          'independent_reviews':attempts,'condition_selected_responses':{f'{p}/{k}':v['path'] for (p,k),v in data.items()}})
    write(H/'scenario_evaluations.json',{'rows':scenarios,'marks_order':PROTOCOL['rubric']['scenario'],
      'interpretation':'Field completeness is mechanical. A null independent mark is unscored; this is not a rejection. Misuse and skip are single marks.'})
    reference_document()
    print({k:v['counts'] for k,v in metrics.items()})


if __name__=='__main__':main()
