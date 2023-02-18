import json 
from transformers import MT5Tokenizer

from utils import get_text_in_unified_script, get_language_normalizer

normalizer = get_language_normalizer()

langlist  = {'as':[],'bn':[],'en':[],'gu':[] ,'hi':[],'kn':[],'ml':[],'mr':[],'or':[],'pa':[],'ta':[],'te':[]}
langlist  = {'as':[],'bn':[],'gu':[] ,'hi':[],'kn':[],'ml':[],'mr':[],'or':[],'pa':[],'ta':[],'te':[]}

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small', cache_dir="/tmp/hugginface")



for lang in langlist:
    preover = []
    read_file = open('/scratch/XAlign/datasets/split_data/'+lang + '/train.jsonl').readlines()
    for line in read_file:

        line = json.loads(line.strip())
        number = len(line['facts_list'])
        sentence = line['sentence']
        lang_this = line['lang']
        
        tokenzier_args = {'text': sentence, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': 1024, 'return_attention_mask': True}
        tokenized_data = tokenizer.encode_plus(**tokenzier_args)
        preover += list(set(tokenized_data['input_ids']))

    preover = set(preover)
    langlist[lang] = preover

for target_lang in langlist:
    print ('============',target_lang,'============')
    gnum = 0
    for lang in langlist:
        postover = []
        read_file = open('/scratch/XAlign/datasets/split_data/'+lang + '/train.jsonl').readlines()
        for line in read_file:

            line = json.loads(line.strip())
            number = len(line['facts_list'])
            sentence = line['sentence']
            lang_this = line['lang']
            uni_sentence = get_text_in_unified_script(sentence,normalizer[lang_this],lang_this,target_lang)
            

            uni_tokenzier_args = {'text': uni_sentence, 'truncation': True, 'pad_to_max_length': False, 
                                        'max_length': 1024, 'return_attention_mask': True}
            uni_tokenized_data = tokenizer.encode_plus(**uni_tokenzier_args)
            postover += list(set(uni_tokenized_data['input_ids']))


        postover = set(postover)

        gnum += len(postover.intersection(langlist[target_lang])) /len(postover.union(langlist[target_lang])) 
        print (lang, "overlap:", len(postover.intersection(langlist[target_lang])) /len(postover.union(langlist[target_lang])), "overlap:", len(postover.intersection(langlist[target_lang])), "after uni:",len (postover), "preuni:", len(langlist[lang]) )

    print (gnum/len(langlist))



