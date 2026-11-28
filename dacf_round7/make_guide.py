"""Build observer-oriented descriptions and existing on-demand guide replies."""
from dacf_research.prepare_inputs import read,write
from dacf_round7.run import H

OVERVIEW = ('The video has a map before you start watching. The prepared investigation plan and the coarse source index provide sample addresses and saved measurements. '
  'Opening binds the source and reuses the coarse cache, or builds it if needed. One sample may be nominated independently for several visual reasons. '
  'A separate later pass compares distinguishable earlier and later anchors and looks for gaps in measured-change coverage. Change-space spacing is per measurement channel and can differ from seconds. '
  'It gives possible time brackets and a tree of saved time and local-region work. The P2-17 trigger recheck is a separate way to investigate a comparison bracket. '
  'The catalogue finds source addresses, the wave and unviewed axis help navigate, Figure 1 describes regions, and Figure 2 measures projected relations. '
  'Saved result IDs let you inspect full fields and evidence roots. Node queries locate saved evidence, and optional node refinement returns real original samples and can add a derived crop and saved measurements. '
  'Automatic stopping never closes original access or a legal refinement. Read the exact source directly when you already know where to look. '
  'Local comparisons test image correspondence with stated applicability. Object candidates and state chains organize evidence across source times. '
  'Observation records, disputes and review preserve corrections, while citation verification checks the media. Images, derived crops, calculations and text leads have different scope. '
  'Unseen intervals, physical identity, contact and cause can remain unresolved. Choose tools, precision and stopping as the question warrants.')

COMMON = {'start':'Inclusive source seconds, not integer source ticks. Use with end when narrowing an interval.',
 'end':'End of the interval in source seconds, not integer source ticks. Use the actual file bounds.',
 'pts':'Exact integer source tick from this video. Use the source time_base to convert seconds to ticks only if the result is an actual sample.',
 'cursor':'Use the returned next_cursor with unchanged filters and other sampling arguments.', 'page_size':'Number of records or attachments in this response page; continue the cursor if useful.',
 'reason':'State what this retrieval or update is meant to check.', 'native_xywh':'Rectangle [x,y,width,height] in original pixels, with positive dimensions inside the original.',
 'result_ref':'Complete saved vo- reference from this source and session. A visual receipt ID is not a calculation reference.',
 'evidence_id':'ID of an actually delivered original from this source (ev-...), not a catalogue frame ID.',
 'evidence_ids':'Citations from this source. Check the required types for this operation, since original, crop and computation have different scope.',
 'node_id':'Exact ID from a range list or previous refinement. The root is source.'}

LOCAL = {
 'observer_guide':{'tool':'Exact existing tool name for an example. For AMR help call amr_query with detail=guide.','topic':'One listed optional workflow topic. Call with no fields for the overview.'},
 'observation_goal':{'goal':'The visual question to investigate. Omit to read the active purpose.'},
 'catalog_search':{'mode':'Select result ordering or listing mode. This returns metadata only.','dimension':'Optional nomination dimension from the tool options. Several nominations may apply to one sample.'},
 'delivery_request':{'pts':'List of actual exact integer source ticks; use frame_ids instead when known.','frame_ids':'List of exact full frame IDs returned by the catalogue. Use pts for exact raw times.'},
 'source_resample':{'fps':'Requested samples per source second. Supply fps or count. Returned times are actual source frames.','count':'Requested number of samples across the interval, as an alternative to fps.'},
 'change_search':{'dimension':'Search D2–D5 measurement categories. Use catalog_search for other independent nomination roles.'},
 'change_recheck':{'trigger_result_ref':'Saved result_ref from change_triggers; pair it with a trigger_id from that same result.',
                    'trigger_id':'Exact item ID from that result, not an AMR node ID. Pass a positive source-seconds span and fps.'},
 'amr_query':{'detail':'guide works before opening. With node_id, proof shows recorded measurements, wave joins a within-node waveform, links joins evidence. A time-range request without node_id lists nodes.'},
 'amr_refine':{'pts':'Time mode: omit for an actual nearest-to-midpoint sample or pass one exact interior tick. Space mode: at least two actual ticks within the node.',
               'mode':'time adds temporal children; space compares a selected original-pixel region and adds ROI children.',
               'crop_pts':'Optional in space mode. Must be one of the supplied source ticks. Returned crop has its own derived receipt.'},
 'evidence_search':{'xywh_analysis':'Rectangle in the returned analysis raster coordinates, not native original pixels.'},
 'evidence_routes':{'amr_node_id':'Optional exact tree node ID to intersect the evidence range and recover existing calculation refs.',
                    'compare_to_pts':'Use with a different anchor_pts for a nomination comparison. Omit for graph hops and ordinary evidence lookup.'},
 'relation_figure2':{'incremental':'For a non-catalogue or user-region measurement get an original at pts first.',
                     'context_pts':'List of other exact source ticks for context, not seconds.',
                     'regions_native':'List of native [x,y,width,height] rectangles within the original. These are marked as requested ROIs.'},
 'visual_compare':{'pts_a':'First exact integer sample tick at this source. Use distinct pts_b. May decode for measurement, does not attach the pictures.',
                   'pts_b':'Second exact integer sample tick at this source.'},
 'evidence_detail':{'path':'Nested field names or indices returned by the stored computation. Use [] to inspect the available top-level fields.'},
 'compare_object_views':{'mode':'direct compares prototypes and sample boxes; ab compares samples against two separate anchor views.',
                         'reference_a':'In ab mode, one known A anchor view at a real PTS and native ROI, distinct from reference_b and separate from samples.',
                         'reference_b':'In ab mode, the other known anchor view, distinct from reference_a and separate from samples.',
                         'samples':'The local views being examined. Each uses actual integer pts and an original-pixel ROI.'},
 'frame_ocr':{'language':'Requested installed OCR language. The tool can return an unavailable status.'},
 'video_text_search':{'channel':'subtitle: embedded subtitle track. asr: optional speech transcription. Neither describes visual pixels.'},
 'claim_verify':{'revision_id':'Optional stored observation revision. Verifies eligible media citations; semantic claims need a visual review.'},
 'evidence_crop':{'scale':'Display magnification for the derived view. It changes presentation, not native detail.'},
 'state_chain':{'identity_id':'A saved ID for the exact recorded candidate view when available. Shared appearance does not prove identity.'},
 'observation_dispute':{'revision_id':'Identify the immutable revision you are challenging. To add a dispute also pass same-source evidence and reason.'},
 'observation_record':{'supersedes_revision':'The current revision for this observation when action=revise; history remains in the ledger.'},
 'observation_review':{'final_claims':'Draft claim objects with actual evidence and revision IDs plus any details to check literally. Mechanical checks do not adjudicate meaning.'},
 'external_source':{'supports':'Narrow scope of the external statement. Registering this does not run or verify the external command.'}
}

SALIENT = {
 'catalog_search':'Choose addresses from the coarse first-pass catalogue. Each of the independent nomination measurements can nominate this same sample; no label is a confirmed event. For actual pixels pass the selected frame ID or exact tick to delivery_request.',
 'observer_guide':'Look up the short observer map, a tool example or an optional topic. Useful if you need to distinguish prior analysis, new measurements and direct picture access. No frame or calculation is delivered.',
 'change_recheck':'Run a new P2-17 calculation on actual interior samples tied to a saved trigger. Pass the original trigger reference and its item ID, range and fps. Returns measured outputs and eligible sample counts without image attachments. Use source_resample or delivery_request to see originals.',
 'amr_query':'Reuse the separate dual-anchor investigation tree without rereading video when the branch or local comparison already exists. Query by time to get node IDs, then use one ID with detail=proof, wave, or links. For example amr_query({node_id:"source",detail:"links"}) joins saved source evidence. No image attachment. Scoped automatic stop still permits legal read or refinement.',
 'amr_refine':'Take real samples to investigate an unresolved gap in an existing branch, including branches where the automatic policy stopped. Time delivers native endpoint and interior samples and records two children. Space uses actual node PTS and a native box, delivers originals and can return a separately cited crop. For a known time and only a picture use delivery_request or source_resample. Measurements, hierarchy and delivery do not prove physical continuity.',
 'relation_figure2':'Get measured image-plane regions and projected relations, or add a local measurement at a received original PTS. Catalogue rasters can be compared without delivering original images. Use the incremental path for a received native ROI; the output cannot establish contact or distance in physical space.',
 'evidence_detail':'Read the full saved calculation when the summary omits coordinates or diagnostics. Use its vo- result_ref and a field path. Start with an empty path if unknown. Reusing fields does not rerun the computation or produce an original image.'
}

AMR_EXAMPLES = {'amr_query': {'node_id':'source','detail':'links'},
 'amr_refine': {'node_id':'source','mode':'time','reason':'Determine whether measured divergence is concentrated in one half before requesting more samples.'}}

SKIP = {'amr_query':'When the exact PTS and question are already known, request the original or use the existing receipt without a tree lookup.',
 'amr_refine':'A direct original or derived crop is sufficient when the only unresolved question is a tiny printed mark at one already known PTS.',
 'change_recheck':'If you only need to see an interval and already know its seconds, source_resample avoids trigger-specific computation.',
 'observation_coverage':'If a sharp delivered original and its ID already answer the question, finish with that citation.'}


def main():
    old=read(H/'original_surface.json')['tools']
    outcomes=read(H.parent/'dacf_round6/guide-acceptance-Evn26L/result.json')['tools']
    corrections=read(H/'reference_errata.json')['replacements']
    rows=[r.split('\t') for r in (H/'capability_basis.tsv').read_text('utf8').splitlines()[1:]]
    docs={}
    page=['# VisuReach 实验说明 · Round 7','',OVERVIEW,'',
          '默认请求使用工具自身说明和参数解释。现有 `observer_guide` 可按需显示系统关系、例子、返回线索、恢复方法。工具名称、原参数校验、预测算法和原图权限沿用 Round 6。','']
    for name,purpose,mechanism,neighbor in rows:
        tool=old[name]; guide=tool.get('guide'); verified=outcomes.get(name,{})
        example=guide[1] if guide else AMR_EXAMPLES[name]
        desc=SALIENT.get(name, purpose+'. '+mechanism+'. '+neighbor+'.')
        if guide:
            desc+=' Optional example: observer_guide with tool="'+name+'".'
        result={'observed_status':verified.get('status','completed'), 'fields':verified.get('keys',['summary','images','envelope'] if name=='amr_refine' else ['id','bounds_pts','links','refs']),
                'attachment_count_in_reference_call':verified.get('images',3 if name=='amr_refine' else 0)}
        params={k:v for k,v in (COMMON|LOCAL.get(name,{})).items() if k in tool['input']['properties']}
        skip=SKIP.get(name, 'Skip when the current source evidence already answers the question or use the narrower related operation.')
        recovery=guide[2] if guide else ('Query the legal range for a node ID before asking about a missing node. Omit node_id for a range list.' if name=='amr_query'
                                          else 'Use exact node ticks, not seconds. For invalid ROI check native size. For a node without a valid interior, use a wider parent or direct source access.')
        docs[name]={'description':desc,'parameters':params,'example':example,
                    'result_example':result,'skip_when':skip}
        page.extend([f'## `{name}`','',desc,'', f'适用与证据范围。{corrections.get(name,mechanism)}','',
                     f'可跳过的情况。{skip}','', f'恢复方式。{recovery}','',
                     f'调用示例 `{name}({__import__("json").dumps(example, ensure_ascii=False)})`。示例中的 `$` 值用同一来源上游返回值替换。','',
                     '验证回执字段为 '+', '.join(f'`{x}`' for x in result['fields'])+'。',''])
    write(H/'revised_surface.json',{'overview':OVERVIEW,'tools':docs,'basis':'Original observations and independent mechanical reference; no evaluation form content.'})
    with (H/'REVISED_GUIDE.md').open('x',encoding='utf8') as handle:
        handle.write('\n'.join(page))


if __name__=='__main__':
    main()
