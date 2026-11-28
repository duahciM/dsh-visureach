"""Original A2 contract retry: only normalize lossless JSON at the SDK boundary."""
from unittest.mock import patch

from dacf_round4 import agent
from dacf_research.prepare_inputs import digest, read, write


def main():
    root=agent.HERE/'agent'
    prior=read(root/'A2/result.json')
    assert any('not lossless JSON' in row['text'] for row in prior['tool_results'])
    settings=read(root/'protocol_lock.json')
    assert all(digest(agent.HERE/f)==h for f,h in settings['tool_and_driver_hashes'].items())
    public=root/'public'
    retry=root/'transport_retry'
    retry.mkdir()
    transport=agent.HERE/'transport_repair'
    configure=agent.configuration
    write(retry/'amendment.json',{'cause':'Native SDK rejected research object containing JavaScript undefined fields.',
                                 'scope':'Strip undefined from research output and declare explicit summary/image response shape. Original thresholds, prompts, inputs, model and permissions unchanged.',
                                 'wrapper_sha256':digest(transport/'probe.mjs')})
    with patch.object(agent,'HERE',transport), patch.object(agent,'configuration',lambda folder,arm,_:configure(folder,arm,public)):
        agent.trial('A2',settings,retry)


if __name__=='__main__':main()
