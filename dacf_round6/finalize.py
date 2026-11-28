"""Merge usable discovery conditions, keeping all original and replacement costs."""
from collections import Counter
from pathlib import Path

from dacf_research.prepare_inputs import read,write
from dacf_round6.summarize import H


def main():
    batches=('prompt','prompt-retry','discovery','discovery-corrected','capability')
    arrangements={b:read(H/b/('summary-with-costs.json' if b=='prompt-retry' else 'summary.json'))['attempts'] for b in batches}
    results=[]
    for b in batches:
        shape=Counter()
        for a in arrangements[b]:shape.update(a['usage'])
        results.append({'batch':b,'sessions':len(arrangements[b]),'usage':dict(shape),
            'seconds_sum':sum(v['seconds'] for v in arrangements[b]),'images_bytes':sum(a['bytes_images'] for a in arrangements[b]),
            'native_tool_time_ms':sum(a['host_tool_ms'] for a in arrangements[b]),'source_png_cache_count':sum(a['frames_in_native_cache'] for a in arrangements[b])})
    wire=read(H/'wire/P3-wire/result.json')
    results.append({'batch':'wire','sessions':1,'usage':wire['usage'],'seconds_sum':wire['seconds']})
    combined=[]
    for b in ('discovery','discovery-corrected'):
        ingested=read(H/b/'window_scores.json')
        for a in arrangements[b]:
            run=read(H/b/f"{a['case']}-{a['condition']}-{a['arm']}"/'result.json')
            visible=[r for r in run['tool_results'] if not r['error'] and ('Optional related evidence:' in r['text'] or 'optional_related_evidence' in r['text'])]
            sliced=next(s for s in ingested['attempts'] if all(s[k]==a[k] for k in ('case','arm','condition')))
            relevant={key:next((r for r in a['delivery_progress'] if r['later_message'] and value['window'][0]<=r['pts']<=value['window'][1]),None)
                      for key,value in ingested['roles'][a['case']].items()}
            combined.append({'batch':b,'case':a['case'],'arm':a['arm'],'label':a['condition'],
                'variant':'preserved_before_renderer_fix' if b=='discovery' and a['arm']=='A3' else 'comparison',
                'stop':a['stop'],'answer_parseable':a['answer_parseable'],'seconds':a['seconds'],
                'hint_tool_results':len(visible),'hint_results_followed_by_model_message':sum(t['followed_by_model_reply'] for t in visible),
                'tool_calls_total':sum(a['all_tool_counts'].values()),'additional_calls':a['extra_calls'],
                'images':a['actual_original_attachment_count'],'crops':a['derived_attachment_count'],
                'usage':a['usage'],'first_original_in_role_window':relevant,'role_windows':sliced})
    write(H/'comparison.json',{'all_attempt_costs':results,'discovery':combined,
          'cost_limits':'Provider USD charge unavailable. PNG file count is a lower bound on work; decoder internal frames not exposed. First role-window receipt is not model understanding.'})

if __name__=='__main__':main()
