"""One transport-only retry with a longer external watchdog."""
from pathlib import Path

from dacf_round2.agent_probe import trial
from dacf_research.prepare_inputs import ROOT, digest, read, write

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'agent_probe'
    public=root/'public'
    lock=read(root/'lock.json')
    assert all(digest(public/k)==v for k,v in lock['sha256'].items())
    prior=read(root/'region_metadata/result.json')
    assert prior['stop']['kind']=='technical_failure' and 'timed out' in prior['stop']['message']
    out=root/'retry_longer_watchdog'
    out.mkdir()
    write(out/'amendment.json',{'prior_lock_hash':digest(root/'lock.json'),'old_watchdog_seconds':lock['watchdog_seconds'],
        'new_watchdog_seconds':720,'changed_only':'external transport watchdog; same provider, prompt, source, initial catalogue and metadata tool'})
    base=read(ROOT/'evidenweft_phase1/public_protocol.json')
    trial('region_metadata',public/'video.mp4',base['sources']['doorway'],public,base,lock|{'watchdog_seconds':720},out)

if __name__=='__main__':
    main()
