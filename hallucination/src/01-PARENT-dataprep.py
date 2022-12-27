## Use mT5 generated output and prepare dataset in training format
import os 
import json

INPUT_PATH = '/Users/rahulmehta/Desktop/MultiSent/datasets/17Dec2022/model_outputs/inference-as-bn-en-gu-hi-kn-ml-mr-or-pa-ta-te-google-mt5-small-30-0.001-unified-script/'
GENERATED_FILE = 'test-predicted-epoch-22.txt'
FACT_FILE = 'test-src.txt'


with open(INPUT_PATH + FACT_FILE) as f, open(INPUT_PATH + GENERATED_FILE) as g:
    for line,g_line in zip(f,g):
        generated_sent = g_line.replace("\n","").replace("।","")
        src_list = line.replace("\n","").split(" ") 
        lang = src_list[1][0:2]
        fact = src_list[5:]

        # fact_list = 
        # for word in fact:

        fact_removal = ['<H>','<R>','<T>','<QR>','<QT>']
        new_fact = []
        for word in fact:
            if word not in fact_removal:
                new_fact.append(word)
        #print(new_fact)

        d = {}
        d["facts"] = new_fact
        d["generated_sentence"] = generated_sent
        #print(d)

        if not os.path.exists(INPUT_PATH + lang):
            os.makedirs(INPUT_PATH + lang)

        with open(INPUT_PATH + lang + '/test.jsonl','a') as fp:
            json.dump(d,fp,ensure_ascii=False)
            fp.write('\n')

        

