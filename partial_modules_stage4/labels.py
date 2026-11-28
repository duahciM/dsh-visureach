"""Evaluator-only targets authored from frozen references and new exact source sheets."""

from dacf_research.prepare_inputs import ROOT, read, write
from partial_modules_stage3.observation import ENV, NEIGHBOR
from partial_modules_stage4.prepare import H


def interior(box, percent):
    x,y,w,h = box
    a,b = round(w*percent),round(h*percent)
    return [x+a,y+b,w-2*a,h-2*b]


if __name__ == '__main__':
    old = read(ROOT / 'partial_modules_stage3/references/sealed.json')['targets']
    assessment = {}
    for case, rows in old.items():
        anchors = [{'pts':r['pts'],'key':r['episode'],'roi':interior(r['box'],.10),
                    'participant':r['role']==NEIGHBOR,'edge_roi':r['box']}
                   for r in rows if r['role'] in (ENV, NEIGHBOR)]
        assessment[case] = {'legacy_targets':rows,'local_anchors':anchors,'state_regions':[]}
    assessment['roles']['state_regions'] = [{'pts':t,'roi':[526,273,57,45]} for t in (600,1400)]
    assessment['new-pan'] = {'legacy_targets':[],'state_regions':[], 'local_anchors':[
        {'pts':t,'key':'gray-door-panel','roi':box,'participant':False}
        for t,box in ((3186,[460,112,350,474]),(4262,[78,112,355,474]),(5317,[560,112,340,474]))]}
    assessment['new-local'] = {'legacy_targets':[],'state_regions':[], 'local_anchors':[
        {'pts':t,'key':'cup-print','roi':[330,105,420,425],'participant':False}
        for t in (400,820,1260,1660,2060,2480,2900)]}
    write(H / 'inputs/sealed_assessment.json', assessment)
