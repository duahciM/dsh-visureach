"""Directed real DSH check using Stage 4 source/ROI with published offline tools."""

import json
from argparse import ArgumentParser
from time import monotonic

from dacf_research.prepare_inputs import ROOT, read, write
from dacf_round7.run import PROTOCOL, SYSTEM_WIDE_TOOLS, TrackedHarness, answer_of, native_artifacts, parse_events, strip_private
from partial_modules_stage4.prepare import H


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--direct-only',action='store_true');args=parser.parse_args()
    output=H/('native-recovery' if args.direct_only else 'native');output.mkdir()
    workspace=output/'workspace';workspace.mkdir()
    profile=output/'profile.yml'
    config={'insert':[{'id':'visureach-0.6.0-stage4-directed',
        'name':(ROOT/'video_observer_offline060/index.js').as_uri(),
        'config':{'cacheDir':str(output/'cache'),'pythonPath':str(ROOT/'.venv/Scripts/python.exe'),'integration':True}}]}
    profile.write_text(SYSTEM_WIDE_TOOLS+'\n- '+json.dumps(config)+'\n',encoding='utf-8')
    spec=read(H/'inputs/source_inventory.json')['new-local']
    grade=read(H/'evaluation.json')['cohorts']['new-local']['arms']['R1P1']['reference_scores']
    witness=grade['feature_support_detail'][0]
    patch=next(r for r in read(H/'predictions/new-local/region_features.json')['extra_geometry']['feature_patches']
               if r['calculation_root']==witness['calculation_root'])
    ticks=patch['pts']; box=patch['rois'][1]
    # Give only actual addresses; the assessment and analysis result are unavailable to the agent.
    source=(ROOT/spec['path']).as_posix()
    prompt=(f'Make a short real visual check in this video only with the video tools: {source}. '
            f'Read the original at both exact source PTS {ticks[0]} and {ticks[1]}. Use the small native XYWH region {box} '
            'of the second image as a possible local reference. Retrieve the crop and verify its source parent if applicable. '
            'Report what you can see in the two originals and what you can check from this support. Explain the purpose of a local reference, '
            'give one novel use, one misuse, and one case where the crop or local-reference investigation is unnecessary. '
            'Could this patch alone support a complete box gap or a physical identity conclusion? Cite the evidence and return concise JSON. '
            'If a tool is inapplicable report the result and use the actual source evidence you have.')
    if args.direct_only:
        args_to_test={'mode':'direct','prototypes':[{'pts':ticks[0],'native_xywh':patch['rois'][0]}],
                      'samples':[{'pts':ticks[1],'native_xywh':box}]}
        prompt=(f'Open only this video with the video tools: {source}. Call compare_object_views once using these exact '
                f'native pixel parameters: {json.dumps(args_to_test)}. Explain which 2D results were supported and which '
                'were insufficient. If the tool output fails, report that outcome exactly. Return short JSON.')
    write(output/'test_lock.json',{'scope':'directed alternative tool after runtime format error' if args.direct_only else
          'directed tool comprehension and source delivery; not autonomous region discovery',
          'source_sha256':spec['sha256'],'pts':ticks,'roi':box,'actual_study_calculation':patch['calculation_root'],'route':PROTOCOL['route']})
    started=monotonic();h=None;events=[]
    try:
        h=TrackedHarness(output,profile)
        h.result(h.send('initialize',PROTOCOL['route']|{'cwd':str(workspace)}),50)
        stop=h.turn(prompt,235,events)
    except (OSError,RuntimeError,ValueError,TimeoutError) as exc:
        stop={'kind':'technical_failure','message':str(exc)}
    finally:
        if h:
            h.close()
            (output/'raw_runtime_private.jsonl').unlink(missing_ok=True)
    raw,answer=answer_of(events)
    write(output/'events.json',strip_private(events))
    write(output/'result.json',native_artifacts(events)|{'stop':stop,'scope':patch['scope'],'prompt':prompt,'source_sha256':spec['sha256'],
        'answer':answer,'answer_text':raw,'route':PROTOCOL['route'],'seconds':round(monotonic()-started,3),
        'usage':parse_events(events,[])['provider_usage_tokens']})
    print(stop,flush=True)
