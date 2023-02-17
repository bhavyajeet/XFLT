import json 
from transformers import MT5Tokenizer

from utils import get_text_in_unified_script, get_language_normalizer

normalizer = get_language_normalizer()

langlist  = {'as':[],'bn':[],'en':[],'gu':[] ,'hi':[],'kn':[],'ma':[],'mr':[],'or':[],'pa':[],'ta':[],'te':[]}

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small', cache_dir="/tmp/hugginface")

target_lang = 'en'

for lang in langlist:
    tot_len = 0
    fin_number = 0

    print ("========", lang ,"========")
   
    pretok = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
    preover = []
    postover = []
    postok = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
    read_file = open('/scratch/XAlign/datasets/split_data/'+lang + '/train.jsonl').readlines()
    for line in read_file:

        line = json.loads(line.strip())
        number = len(line['facts_list'])
        sentence = line['sentence']
        lang_this = line['lang']
        uni_sentence = get_text_in_unified_script(sentence,normalizer[lang_this],lang_this,target_lang)
        
        tokenzier_args = {'text': sentence, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': 1024, 'return_attention_mask': True}
        tokenized_data = tokenizer.encode_plus(**tokenzier_args)
        print (tokenized_data)
        prenum = len(tokenized_data['input_ids'])
        pretok[number].append(prenum)
        preover.append(prenum)
        
    for i in range(1,11):
        print ("pre ",i, " ", int(sum(pretok[i])/len(pretok[i])), end = ' ')
        print ( int(sum(postok[i])/len(postok[i])) )
    print ("pre ", int(sum(preover)/len(preover) ) )
    print ("post ", int(sum(postover)/len(postover) )) 
