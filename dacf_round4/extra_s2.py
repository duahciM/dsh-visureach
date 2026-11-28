"""Small exploratory A2 repeat exposing the original global stop only."""
from unittest.mock import patch

from dacf_round4 import agent
from dacf_research.prepare_inputs import digest, read, write


def main():
    base=agent.HERE/'agent'
    public=base/'public'
    settings=read(base/'protocol_lock.json')
    out=base/'global_stop_only'
    out.mkdir()
    original=public/'regions.json'
    projected=read(original)
    for node in projected['nodes']:
        node['stop']={ch:{'S0_initial_source_stop':data['S0_initial_source_stop']} for ch,data in node['stop'].items()}
    path=out/'regions.json'
    write(path,projected)
    write(out/'amendment.json',{'same_public_source_sha256':settings['public_hashes']['source.mp4'],
                               'public_regions_before':digest(original),'after':digest(path),
                               'scope':'Exploratory S2: remove S1 local stop annotations only. Exact tools, source path, prompt, model and rights match. Single independent session.'})
    assert all(digest(agent.HERE/f)==h for f,h in settings['tool_and_driver_hashes'].items())
    configure=agent.configuration
    with patch.object(agent,'configuration',lambda folder,arm,_:configure(folder,arm,public)|{'regionPlan':str(path)}):
        agent.trial('A2',settings,out)


if __name__=='__main__':main()
