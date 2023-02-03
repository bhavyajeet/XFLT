## Use mT5 generated output and prepare dataset in training format
import string
import os 
import json

INPUT_PATH = '/scratch/model_outputs/towork/'
OUTPUT_REF_PATH = '/scratch/model_outputs/towork/ref/'

GENERATED_FILE = 'test-predicted-epoch-22.txt'
REF_FILE = 'test-ref.txt'
FACT_FILE = 'test-src.txt'


# with open(INPUT_PATH + FACT_FILE) as f, open(INPUT_PATH + GENERATED_FILE) as g:
#     for line,g_line in zip(f,g):
#         generated_sent = g_line.replace("\n","").replace("।","")
#         src_list = line.replace("\n","").split(" ") 
#         lang = src_list[1][0:2]
#         fact = src_list[5:]

#         # fact_list = 
#         # for word in fact:

#         fact_removal = ['<H>','<R>','<T>','<QR>','<QT>']
#         new_fact = []
#         for word in fact:
#             if word not in fact_removal:
#                 new_fact.append(word)
#         #print(new_fact)

#         d = {}
#         d["facts"] = new_fact
#         d["generated_sentence"] = generated_sent
#         #print(d)

#         if not os.path.exists(INPUT_PATH + lang):
#             os.makedirs(INPUT_PATH + lang)

#         with open(INPUT_PATH + lang + '/test.jsonl','a') as fp:
#             json.dump(d,fp,ensure_ascii=False)
#             fp.write('\n')

langlist = {'marathi':'mr','malayalam':'ml'}

with open(INPUT_PATH + FACT_FILE) as f, open(INPUT_PATH + REF_FILE) as r, open (INPUT_PATH+GENERATED_FILE) as g:       
    for line,r_line,g_line in zip(f,r,g):
        #print(line)
        generated_sent = g_line.replace("\n","").replace("।"," ")
        src_list = line.replace("\n","").split(" ") 
        lang = src_list[1][0:2]
        if src_list[1].lower().strip() in langlist:
            lang = langlist[src_list[1].lower().strip()]
        fact = src_list[4:]

        ref_sent  = r_line.replace("\n","").replace("।"," ")
        # fact_list = 
        # for word in fact:

        fact_removal = ['<H>','<R>','<T>','<QR>','<QT>']
        new_fact = []
        for word in fact:
            if word not in fact_removal:
                new_fact.append(word)
        #print(new_fact)

        d = {}
        ref_sent = ref_sent.translate(str.maketrans('', '', string.punctuation))
        d["facts"] = new_fact
        d["generated_sentence"] = generated_sent
        d["ref_sentence"] = ref_sent
        #print(d)

        if not os.path.exists(OUTPUT_REF_PATH  + lang):
            os.makedirs(OUTPUT_REF_PATH  + lang)

        with open(OUTPUT_REF_PATH  + lang + '/test.jsonl','a') as fp:
            json.dump(d,fp,ensure_ascii=False)
            fp.write('\n')

        
        
