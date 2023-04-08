import json
import glob as glob 
import os 

langlist = {
    'assamese': 'as', 
    'bengali': 'bn',
    'english': 'en',
    'gujarati': 'gu', 
    'hindi': 'hi',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'odia': 'or', 
    'punjabi': 'pa',
    'tamil': 'ta',
    'telugu': 'te'
}

for lang in langlist.values():
    print(lang)
    os.makedirs(f'merged/cosine/{lang}', exist_ok=True)
    with open(f'/home2/aditya_hari/multisent/pls/lin_data/{lang}/train.jsonl', 'r') as f, open(f'new_pruned/{lang}-train-new-file', 'r') as n, open(f'merged/cosine/{lang}/train.jsonl', 'w') as fp:
        for line_1, line_2 in zip(f, n):
            obj = json.loads(line_1)
            obj['original_sent'] = obj['sentence']
            obj['sentence'] = line_2[:-1]
            fp.write(json.dumps(obj, ensure_ascii=False)+'\n')
    
    with open(f'/home2/aditya_hari/multisent/pls/lin_data/{lang}/val.jsonl', 'r') as f, open(f'new_pruned/{lang}-val-new-file', 'r') as n, open(f'merged/cosine/{lang}/val.jsonl', 'w') as fp:
        for line_1, line_2 in zip(f, n):
            obj = json.loads(line_1)
            obj['original_sent'] = obj['sentence']
            obj['sentence'] = line_2[:-1]
            fp.write(json.dumps(obj, ensure_ascii=False)+'\n')

    with open(f'/home2/aditya_hari/multisent/pls/lin_data/{lang}/test.jsonl', 'r') as f, open(f'merged/cosine/{lang}/test.jsonl', 'w') as fp:
        for line_1 in f:
            obj = json.loads(line_1)
            fp.write(json.dumps(obj, ensure_ascii=False)+'\n')
