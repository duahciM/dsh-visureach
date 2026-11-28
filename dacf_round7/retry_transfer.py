"""Retry only failed native transports. Inputs and guide are exactly the locked ones."""
from concurrent.futures import ThreadPoolExecutor
from dacf_research.prepare_inputs import read,write,digest
from dacf_round7.run import H,launch

if __name__=='__main__':
    old=H/'transfer'; new=H/'transfer-retry'; new.mkdir()
    docs=H/'revised_surface.json'
    assert read(old/'guide_lock.json')['hash']==digest(docs)
    assignment={k:v for k,v in read(old/'assignment.json').items()
                if read(old/k/'result.json')['stop']['kind']!='completed'}
    write(new/'assignment.json',assignment)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: launch(new/item[0],item[1]['prompt'],docs,overview=False),assignment.items()))
