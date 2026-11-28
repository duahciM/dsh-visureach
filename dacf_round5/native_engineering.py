"""Native format regression only; intentionally directs tool calls and stays out of Agent scoring."""
from dacf_research.prepare_inputs import read,write
from dacf_round5.native import HERE,trial


def main():
    initial=read(HERE/'native/lock.json')
    ref=read(HERE/'engineering/result.json')
    plan=read(HERE/'inputs/fixture/plan.json')
    node=ref['time']['node']; pts=ref['time']['pts'][1]
    root=HERE/'native-engineering'; root.mkdir()
    prompt=(f'Run a native integration regression on the assigned source {HERE.as_posix()}/inputs/fixture/source.mp4. '
            'This is a tool-host check, not an observation task. Open the video and record the goal as source evidence verification. '
            f'Call amr_query for node {node}, call amr_refine on that node with mode=time and pts=[{pts}] for exact verification. '
            f'Call amr_refine on source with mode=space, pts=[0,{pts}], native_xywh=[48,56,180,135], crop_pts={pts}. '
            'Then use evidence_routes with amr_node_id=source and channel=computed. '
            'Report briefly which calls succeeded, using JSON with facts, open_questions and summary. No need to describe the video.')
    config=read(HERE/'protocol.json')
    known=initial|{'prompts':{'fixture':prompt}}
    write(root/'engineering_protocol.json',{'excluded_from_agent_comparison':True,'prompt':prompt,'role':'Native SDK format regression'})
    trial({'source':'fixture','repeat':1,'arm':'A2'},config,known,root,HERE/'release/installed/package/entry.mjs')


if __name__=='__main__':main()
