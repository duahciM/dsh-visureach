"""Append evidence and revisable assessments without invalidating independent measurements."""
import json
from pathlib import Path


KINDS={'appearance','background_geometry','coappearance_projection','identity_hypothesis','raw_relation','identity_dependent_relation'}


class Ledger:
    def __init__(self,path,source_sha):
        self.path=Path(path);self.source_sha=source_sha
        self.claims={};self.revisions=[]
        if self.path.exists():
            for line in self.path.read_text('utf8').splitlines():
                self._apply(json.loads(line))

    def _apply(self,row):
        if row['source_sha256']!=self.source_sha:
            raise ValueError('Evidence source differs from ledger source.')
        if row['event']=='assertion':
            if row['id'] in self.claims or not set(row['requires'])<=self.claims.keys() or row['kind'] not in KINDS:
                raise ValueError('Unknown dependency, unsupported kind or duplicate assertion.')
            self.claims[row['id']]=row
        elif row['event']=='revision':
            claim=self.claims.get(row['target'])
            valid=claim and claim['kind']=='identity_hypothesis' and row['counterevidence'] in self.claims
            if not valid:
                raise ValueError('A same-source registered counterevidence claim is required.')
            if self.claims[row['counterevidence']]['status']!='counter_evidence_in_scope':
                raise ValueError('Unknown or unassessable is not counterevidence.')
            self.revisions.append(row)
        else:
            raise ValueError('Unknown journal event.')

    def _append(self,row):
        row={'source_sha256':self.source_sha,**row}
        self._apply(row)
        with self.path.open('a',encoding='utf8') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        return row

    def add(self,id_,kind,roots,requires=(),status='recorded',**data):
        if not roots:
            raise ValueError('A claim must have at least one real source root.')
        return self._append({'event':'assertion','id':id_,'kind':kind,'source_roots':sorted(set(roots)),
                            'requires':list(dict.fromkeys(requires)),'status':status,**data})

    def withdraw(self,id_,counterevidence,reason):
        if not reason:
            raise ValueError('Record why the scope is revised.')
        return self._append({'event':'revision','target':id_,'counterevidence':counterevidence,'reason':reason,
                             'action':'withdraw_scoped_interpretation'})

    def query(self,kind=None,root=None):
        withdrawn={r['target'] for r in self.revisions}; status={}
        for id_,v in self.claims.items():
            bad=[parent for parent in v['requires'] if status[parent] in ('withdrawn','dependency_withdrawn')]
            status[id_]='withdrawn' if id_ in withdrawn else 'dependency_withdrawn' if bad else v['status']
            if (not kind or kind==v['kind']) and (not root or root in v['source_roots']):
                yield {**v,'effective_status':status[id_],'affected_by':bad}
