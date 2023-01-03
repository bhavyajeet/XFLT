"""
 Tail only cross lingual hallucination evaluation
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial import distance
import numpy as np 
import torch
from scipy.spatial import distance
import numpy as np
from numpy.linalg import norm
import glob 
import json 
import os
import string
from collections import Counter
from statistics import mean


#FACT_TYPE = 'tailandfact'
DISTANCE = 'cosine'
TOKENS_TO_EXCLUDE = ['[CLS]','[SEP]','।']

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)

punct_list = ['a','b']

def generate_embeddings(text):

    """
    text : list of words  ['a','b']
    """
    #print(text)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text,is_split_into_words=False))
    #print(tokens)
    


    #print({x : tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in text.split()})
    input_encoded = tokenizer(text,is_split_into_words=False, return_tensors="pt")
    #print(input_encoded)
    #print(tokenizer.is_fast)

    #input_encoded = tokenizer.encode_plus(text,is_split_into_words=False, return_tensors="pt",return_offsets_mapping=True)
    #print(input_encoded)

    #print(input_encoded.word_ids())

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
    #root = "/Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/"
    # root = "/Users/rahulmehta/Desktop/MultiSent/datasets/17Dec2022/model_outputs/inference-as-bn-en-gu-hi-kn-ml-mr-or-pa-ta-te-google-mt5-small-30-0.001-unified-script"
    # output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-all-' + DISTANCE + '/'

    # Generate reference scores
    root = "/Users/rahulmehta/Desktop/MultiSent/datasets/17Dec2022/model_outputs/inference-as-bn-en-gu-hi-kn-ml-mr-or-pa-ta-te-google-mt5-small-30-0.001-unified-script/ref/"
    output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-new/results-all-' + DISTANCE + '/'



    for subdir, dirs, files in os.walk(root):
        lang = subdir[subdir.rfind('/')+1:]
        #print(subdir)
        print(lang)



        #lang = subdir[-2:]
        # print(lang)
    
        # if len(lang)>2:
        #     continue
        if lang in ['7/','','hi']:    
            continue
        else:
        #if lang in ['hi']:
            with open(root + '/' + lang + '/test.jsonl') as f:
                #j=0
                for line in f:
                    #j+=1
                    #print(j)
                    # if j>3:
                    #     break

                    c = 0
                    data = json.loads(line)
                    f = []
                    facts_list = data['facts']


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

                      # fact = "Indian Congress"
                    # text2 = "भारतीय राष्ट्रीय कांग्रेस"

                    emb1,tokens1,subword_ids,word_list = generate_embeddings(fact)

                    emb2,tokens2,_,_= generate_embeddings(text2)


                    # Match each word of the sentence with fact words
                    min_pairs = []
                    #print(emb1)
                    #print(tokens2)
                    #print(emb2)
                    #print(tokens2)
                    for index,word in enumerate(tokens2):
                        #print(word)

                        #print(emb2[index:index+1][0])
                        word_emb = emb2[index:index+1,]
                        #print(word_emb[0][0:10])
                        dist = []
                        

                        #print(len(emb1))
                        for i in range(1,len(emb1)-1):
                            #print(emb1)
                            #print(emb1[i:i+1].shape)
                            dist.append(compute_distances(word_emb[0],emb1[i:i+1][0],DISTANCE))

                        #print(len(dist))
                        #print(dist)

                        dist_updated = []
                        cnt_subwords = dict(Counter(subword_ids))
                        #print(cnt_subwords)
                        prev = 0 
                        new = 0 
                        for i in range(0,len(word_list)):
                            #print(word_list[i])
                            # get count of  
                            c = cnt_subwords[i]
                            sub_dist = dist[new:new+c]
                            #sub_dist = dist[new:i+c]
                            new = new + c
                            #print(i,new,c)
                            #print(sub_dist)
                            dist_updated.append(mean(sub_dist))    


                        #print(dist_updated)
                        min_index = np.argmin(dist_updated)
                        
                        #token_top = tokens1[min_index+1]
                        token_top = word_list[min_index]


                        token_score = float(round(min(dist_updated),2))

                        #print(min_index,token_top)
                        pair = tuple((word,token_top,token_score))
                        #print(pair)

                        if token_top in TOKENS_TO_EXCLUDE:
                            continue
                        else:
                            min_pairs.append(pair)

                    
                    #results['entity_name'] = data['entity_name']
                    results['facts'] = data['facts']
                    results['sentence'] = data['generated_sentence']

                    results['scores'] = min_pairs

                    #print(word_list)
                    #print(results)

                    # r.write(str(results))
                    # r.write("\n")


                    with open(output_path  + lang + '-coverage.jsonl','a') as fp: 
                        json.dump(results,fp,ensure_ascii=False)
                        fp.write('\n')