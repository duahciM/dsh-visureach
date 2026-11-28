"""Attach non-blind manual audit for rows missing an independent grade."""
import csv
from dacf_research.prepare_inputs import read,write
from dacf_round7.run import H

if __name__=='__main__':
    file=H/'manual_scenario_review.tsv'
    with file.open(encoding='utf8',newline='') as f:
        audit={(r['phase'],r['tool']):r for r in csv.DictReader(f,delimiter='\t')}
    base=read(H/'scenario_evaluations.json')
    filled=0
    for item in base['rows']:
        if item['independent_marks'] is not None:
            continue
        row=audit[item['phase'],item['tool']]
        column='case'+str(item['case_index']) if item['case_type']=='valid' else item['case_type']
        mark=row[column]
        assert len(mark)==(6 if item['case_type']=='valid' else 1)
        item.update(manual_marks=mark,manual_scope_reason=row['note'],manual_source=file.name)
        filled+=1
    write(H/'scenario_review_complete.json',base|{'manual_fills':filled,
          'method':'Independent masked marks retained as is. Missing marks covered by investigator manual review, not condition-blind. Keep label sources separate. The near-ceiling independent rubric and investigator audit do not support a cross-condition headline score.'})
