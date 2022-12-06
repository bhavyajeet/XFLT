# this script merges the xalign dataset into paragraph wise dataset

from collections import defaultdict
import glob
import json 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


all_datasets = glob.glob('../datasets_v2.7/*/*.jsonl')


print (all_datasets)

# {lang: {entity: {sents: [], ...}, entity: {...}}
merged_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


for dataset in all_datasets:
    if 'test.jsonl' in dataset:
        continue
    print(dataset)
    lang = dataset.split('/')[-2]
    source = dataset.split('/')[-1][:-6]
    with open(dataset) as f:
      for line in f:
          data = json.loads(line)
          qid = data['qid']
          if(qid not in merged_dict[lang]):
              merged_dict[lang][qid]['entity_name'] = data['entity_name']
              merged_dict[lang][qid]['qid'] = qid
          for key in data.keys():
              if(key not in ['qid', 'entity_name', 'lang']):
                  merged_dict[lang][qid][f'{key}_list'].append(data[key])
          merged_dict[lang][qid]['source_list'].append(source)





"""
for lang in merged_dict:
  for qid in merged_dict[lang]:
    section_list = [(i, val) for (i, val) in enumerate(zip(merged_dict[lang][qid]['native_sentence_section_list'], merged_dict[lang][qid]['sent_index_list']))]
    sorted_section_list = [i[0] for i in sorted(section_list, key = lambda x: (x[1][0], x[1][1]))]
    merged_dict[lang][qid]['native_section_sort_index'] = sorted_section_list
"""


for lang in merged_dict:
  with open(f'merged_dataset/{lang}_merged.jsonl', 'w', encoding='utf-8') as f:
    for qid in merged_dict[lang]:
      json_record = json.dumps(dict(merged_dict[lang][qid]), ensure_ascii=False)
      f.write(json_record+'\n')


keys = list(merged_dict.keys())


print (keys)



sents_per_lang = []
sent_counts = []
for lang in keys:
    count = 0
    for qid in merged_dict[lang]:
        sent_count = len(merged_dict[lang][qid]['sentence_list'])
        count+=sent_count if sent_count<=10 else 10
        sent_counts.append(sent_count if sent_count<=10 else 10)
    sents_per_lang.append(count)




sent_plt_data = sorted(list(zip(sents_per_lang, keys)), key=lambda x: x[0])


## Sentences per language (truncated to 10)
sns.barplot([i[1] for i in sent_plt_data], [i[0] for i in sent_plt_data])
plt.show()










