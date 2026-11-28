"""Directed AMR engineering smoke, excluded from autonomous rates."""
from dacf_round6.native import H,trial
from dacf_research.prepare_inputs import ROOT,read,write

if __name__=='__main__':
    d=H/'capability';d.mkdir()
    src=(ROOT/'dacf_round5/inputs/fixture/source.mp4').as_posix()
    p=f'Native tool engineering verification for {src}. Open it, record a verification goal, then call amr_query across an initial range. Use AMR to refine source in time. Refine source separately in space using original PTS 0 and 5120 and native rectangle [50,56,180,100], requesting a crop at 5120. Query one returned child by ID with detail proof and use evidence_routes with amr_node_id=source. Return brief JSON with facts, open_questions and summary. These specified calls are for engineering, not an observation evaluation.'
    write(d/'design.json',{'jobs':[['fixture','P0','A2']],'scope':'Explicitly directed engineering, no autonomous success credit'})
    trial(d/'fixture-P0-A2','fixture','P0','A2',p,H/'discovery.mjs')
