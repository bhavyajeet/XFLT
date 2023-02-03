"""
 Tail only cross lingual hallucination evaluation
"""

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial import distance
import numpy as np 
import torch
import scipy
from scipy.spatial import distance
import numpy as np
from numpy.linalg import norm
import glob 
import json 
import os
import string
from collections import Counter
from statistics import median 
import time 



#FACT_TYPE =    'tailandfact'
DISTANCE = 'euclidean'
TOKENS_TO_EXCLUDE = ['[CLS]','[SEP]','।']

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)

punct_list = ['a','b']


def generate_embeddings(text):

    """
    text : list of words  ['a','b']
    """

    #print({x : tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in text.split()})
    input_encoded = tokenizer(text,is_split_into_words=False, return_tensors="pt", truncation=True, max_length=512)
    #print(input_encoded)

    tokens = tokenizer.convert_ids_to_tokens(input_encoded["input_ids"].tolist()[0])
    #print(tokens)

    with torch.no_grad():
            states = model(**input_encoded).hidden_states
    output = torch.stack([states[i] for i in range(len(states))])
    output = output.squeeze()

    #print("Output shape is {}".format(output.shape))


    final_hidden_state = output[-4, :, ...]

    return final_hidden_state,tokens,input_encoded.word_ids(),text.split(" ")


def compute_distances(emb1,emb2,dis):

    euc = distance.euclidean(np.array(emb1), np.array(emb2))
    cosine = 1 - np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))

    if distance=='euclidean':
        return euc
    else:
        return cosine

if __name__ == "__main__":
    start = time.time()

    # Generate reference scores
    root = "/scratch/model_outputs/towork/ref/"
    output_path = f'/scratch/model_outputs/towork/results-all-' + DISTANCE + '/'


    for subdir, dirs, files in os.walk(root):
        lang = subdir[subdir.rfind('/')+1:]
        print ('=========')
        print(subdir)
        print(lang)
        print ('=========')

        j=0

        results_all = []
        if lang in ['7/','']:    
            continue
        else:
        #if lang in ['hi']:
            with open(root + '/' + lang + '/test.jsonl') as f,open(output_path  + lang + '-coverage-temp.jsonl','a') as fp:
                for line in tqdm(f,total=4292):
                    j+=1
                    #print(j)
                    # if j>100:
                    #     break

                    c = 0
                    data = json.loads(line)
                    f = []
                    facts_list = data['facts']
                    ref_sent = data['ref_sentence']


                    text2 = data['generated_sentence']
                    #entity = data['entity_name']
                    #print(facts_list)

                    results = {}    
                    temp_top = []
                    temp_topdist = []

                    # fact = []
                    # for f in facts_list:
                    #     fact.append(f[0])
                    #     fact.append(f[1])
                    fact = " ".join(facts_list)
                    fact = fact.replace("_"," ")
                    #print(fact)
                    
                    #print(text2)
                    text2 = text2.translate(str.maketrans('', '', string.punctuation))
                    #print(text2)
                  
                    # fact = "Indian National Congress"
                    # text2 = "(भारतीय राष्ट्रीय कांग्रेस) के नेता है"


                    #start2 = time.time()

                    #print (fact)
                    #print (text2)
                    #print ("--"*10)


                    embf,tokensf,subword_ids,word_list = generate_embeddings(fact)

                    #print (word_list)
                    #print (len(word_list))
                    #print (len(subword_ids))
                    #print (tokensf)
                    embf = embf[1:-1]
                    #print (embf.shape)
                    #print (subword_ids)
                    #print (subword_ids)

                    embt,tokenst,_,_= generate_embeddings(text2)
                    #end2 = time.time()
                    #print(end2 - start2)

                    distances = scipy.spatial.distance_matrix(embt,embf)
                    #print(distances)

                    # ids
                    fact_idx =  np.argmin(distances,axis=1)
                    #print(fact_idx)

                    # words 
                    words = [tokensf[i+1] for i in fact_idx]
                    #print(words)

                    # distances 
                    dist = np.min(distances,axis=1)
                    #print(dist)

                    min_pairs = list(zip(tokenst,words,dist))
                    #print(list(zip(tokenst,words,dist)))
              
                    #results['entity_name'] = data['entity_name']
                    results['facts'] = data['facts']
                    results['sentence'] = data['generated_sentence'].translate(str.maketrans('', '', string.punctuation))
                    results['ref_sentence'] = data['ref_sentence']

                    results['scores'] = min_pairs

                    #print (results)

                    results_all.append(results)

                    # with open(output_path  + lang + '-coverage.jsonl','a') as fp: 
                    #     json.dump(results,fp,ensure_ascii=False)
                    #     fp.write('\n')

        with open(output_path  + lang + '-coverage.jsonl','a') as fp: 
            for i in results_all:
                json.dump(i,fp,ensure_ascii=False)
                fp.write('\n')
        
                    
        end = time.time()
        print(end - start)
