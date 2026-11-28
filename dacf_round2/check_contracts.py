"""Structural controls independent of any labelled video."""
from pathlib import Path
from types import SimpleNamespace

from dacf_research.prepare_inputs import read
from dacf_round2.regions import discover


class ControlledBank:
    def __init__(self, signal):
        self.signal=signal
        self.rows={i:{'pts':i*200,'seconds':i*.2,'id':f'f{i}', 'rgb_sha256':f'v{v}',
                      'sharpness':20} for i,v in enumerate(signal)}
        self.params=SimpleNamespace(activity_mad=.003,min_sharpness=8)

    def metric(self,a,b):
        dist=abs(self.signal[a]-self.signal[b])
        return {'gray':dist,'hsv':dist,'clip':dist}

    def metrics_on_pair(self,a,b):
        return {'metrics':self.metric(a,b),'phase':{'registration_valid':True},'pts':[a*200,b*200],
                'root':f'f{a}>f{b}'}


def main():
    cfg=read(Path(__file__).with_name('protocol.json'))
    first=ControlledBank([0]*4+[.2]+[.4]*4+[.6]*2+[.8]*4)
    result=discover(first,list(first.rows),{600:{},2200:{}},cfg)
    parent=next(r for r in result['regions'] if r['kind']=='nested_parent')
    assert parent['middle']['pts']==1200
    children=[next(r for r in result['regions'] if r['id']==key) for key in parent['children']]
    assert [c['bounds_pts'] for c in children]==[[600,1200],[1200,2200]]
    assert len([r for r in result['regions'] if r['kind']=='one_sided_unknown'])==2
    repeated=ControlledBank([0]*4+[.5]*6+[0]*6)
    b=discover(repeated,list(repeated.rows),{600:{},1400:{},2400:{}},cfg)
    assert any(r['middle_pts']==1400 for r in b['region_chains'])
    assert any(r['a_pts']==600 and r['return_pts']==2400 for r in b['recurrence_candidates'])
    assert all(r['continuity']=='unknown' for r in b['regions'])
    quiet=ControlledBank([0]*15)
    q=discover(quiet,list(quiet.rows),{0:{},2800:{}},cfg)
    assert len(q['regions'])==0 and len(q['omitted'])==1
    assert q['nomination_pts']==[0,2800]
    degraded=ControlledBank([0]*7+[.010]*8)
    degraded.rows[7]['sharpness']=4
    weak=discover(degraded,list(degraded.rows),{0:{},2800:{}},cfg)
    assert len(weak['regions'])==1 and weak['regions'][0]['kind']=='unknown_investigation'
    assert not weak['regions'][0]['distinction']['supported_by']
    print('PASS recursive split, intermediate nominee, return, two open flanks, weak low-quality area and zero-change abstention')


if __name__=='__main__':
    main()
