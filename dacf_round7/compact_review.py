"""Condition-blind short review. Raw model answers remain the primary record."""
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
import json
from random import Random

from dacf_research.prepare_inputs import read,write
from dacf_round7.run import H,PROTOCOL,launch
from dacf_round7.score import call_checks,recover


def selected():
    data={}
    for group in ('baseline','transfer','transfer-retry'):
        for ident,item in read(H/group/'assignment.json').items():
            path=H/group/ident/'result.json'
            if not path.is_file():
                continue
            r=read(path)
            answer,parse=recover(r['answer'],r['answer_text'])
            if r['stop']['kind']=='completed' and answer is not None:
                phase='original' if group=='baseline' else 'revised'
                data[phase,item['task']]={'path':path.relative_to(H).as_posix(),'answer':answer,'parse':parse,
                                          'seconds':r['seconds'],'usage':r['usage']}
    return data


def patches():
    basis={r.split('\t')[0]:' '.join(r.split('\t')[1:]) for r in (H/'capability_basis.tsv').read_text('utf8').splitlines()[1:]}
    return basis | read(H/'reference_errata.json')['replacements']


def compact_scopes(tool):
    return [tool.get(k,'') for k in ('purpose','contrast','supports','limit')]


def scenarios(tool):
    return [[sc.get(k,'') for k in ('phenomenon','known','question','fit','result','supports','unknown')]
            for sc in tool.get('scenarios',[])+[tool.get('misuse',{}),tool.get('skip',{})] if isinstance(sc,dict)]


def main():
    phases=selected(); target=H/'grading-compact';target.mkdir()
    original=read(H/'original_surface.json')['tools']; revised=read(H/'revised_surface.json')['tools']
    mapping={}; jobs=[]; rng=Random(PROTOCOL['seed']+71); basis=patches()
    for start in range(0,len(original),3):
        subset=list(original)[start:start+3];rows=[];ref={}
        for key in subset:
            ref[key]=[basis[key], (original[key].get('guide') or [None,None])[1],revised[key]['example']]
        for (phase,k),v in phases.items():
            for item in v['answer'].get('tools',[]) if k.startswith('tools-') else []:
                if item.get('tool') in subset:
                    rows.append((phase,k,v['path'],item['tool'],compact_scopes(item),scenarios(item)))
        rng.shuffle(rows)
        review={str(i):row[:4] for i,row in enumerate(rows,1)}
        body=[[i,row[3],row[4],row[5]] for i,row in enumerate(rows,1)]
        prompt=('Fast evidence grading. The input contains short observer answers with random IDs. Mark each criterion 1 correct, 0 incorrect/absent, ? unclear. '
            'Return compact JSON only: {"scores":{"1":["1111","111111","111111","111111","1","1"]},"notes":{"1":"one critical issue or omit"}}. '
            'Four explanation marks: problem/purpose, correct neighboring difference, valid support, honest boundary. '
            'Each of 3 normal cases has six marks: original problem/evidence path versus given examples (name swaps fail), tool fit, executable input/result, distinction/why use, honest support/unknown, useful evidence. '
            'Then give one mark each for effectively describing misuse and when to skip. Use other tools as valid alternatives. No lengthy explanation. '
            'Reference entries contain capability and sample arguments. Assessed rows are [random_id, tool, explanation_fields, cases]. Case fields = phenomenon, known, question, fit, result, supports, unknown.\n'
            +json.dumps({'reference':ref,'rows':body},ensure_ascii=False,separators=(',',':')))
        if rows: jobs.append((review,prompt))
    overview=[]
    for (phase,k), v in phases.items():
        if k=='architecture':overview.append((phase,k,v['path'],v['answer']))
    rng.shuffle(overview)
    prompt=('Grade two anonymous explanations against actual implementation. Return exactly JSON {"scores":{"1":{"architecture":"11111111",'
        '"combinations":["1111","1111","1111"],"issues":["short issue"]}}}. In masks, 1 explicit and accurate, 0 missing or wrong, ? unclear. '
        'Architecture dimensions in order: prepared vs on-demand; independent first pass vs second phase; measured path vs seconds; correct relationships among actual tools; '
        'direct original and citation access; reuse vs compute; scoped stop; mechanical vs visual limits. Each combination mask: fit, executable data flow, evidence scope, valid recovery/stopping. '
        'Give a short concrete issue for each answer. Architectural frozen facts: '+json.dumps([basis[k] for k in ('catalog_search','change_recheck','amr_query','amr_refine','relation_figure2','evidence_detail')])
        +'\nAnonymous answers '+json.dumps({str(i):r[3] for i,r in enumerate(overview,1)},ensure_ascii=False))
    jobs.append(({str(i):r[:3] for i,r in enumerate(overview,1)},prompt))
    rng.shuffle(jobs)
    for i,(key,p) in enumerate(jobs,1):mapping[f'g{i:02}']=key
    write(target/'index.json',mapping)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda job:launch(target/f'g{job[0]:02}',job[1][1],plain=True,watchdog=240),enumerate(jobs,1)))


if __name__=='__main__':main()
