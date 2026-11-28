"""Regression of common tool JSON transport on the repair bundle only."""
from dacf_research.prepare_inputs import digest,read,write
from dacf_round5.native import HERE,trial


def main():
    current=read(HERE/'engineering-fixed/result.json'); assert current['status']=='passed'
    first=read(HERE/'native/lock.json')
    entry=HERE/'release-fixed/installed/package/transport.mjs'
    out=HERE/'native-transport';out.mkdir()
    prompt=(f'Native SDK transport regression only. Open {HERE.as_posix()}/inputs/fixture/source.mp4 and record a source-verification goal. '
            'Use delivery_request at exact PTS 0 and 5120. Call visual_compare with pts_a=0 and pts_b=5120. '
            'Create one direct observation with the received original image ID, then call observation_review on its observation ID. '
            'Finally call video_text_search with start=0,end=1 (a valid result may report no audio or subtitles). '
            'Return a short JSON object describing which calls completed; this is not the video understanding task.')
    write(out/'engineering_protocol.json',{'transport_only':True,'not_part_of_native_comparison':True,'entry_sha256':digest(entry),'prompt':prompt})
    trial({'source':'fixture','repeat':1,'arm':'A2'},read(HERE/'protocol.json'),first|{'prompts':{'fixture':prompt}},out,entry)


if __name__=='__main__':main()
