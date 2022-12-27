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


FACT_TYPE = 'tailandfact'
DISTANCE = 'cosine'
TOKENS_TO_EXCLUDE = ['[CLS]','[SEP]','।']

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)

punct_list = ['a','b']

def generate_embeddings(text):

    """
    text : list of words  ['a','b']
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text,is_split_into_words=False))

    input_encoded = tokenizer.encode_plus(text,is_split_into_words=False, return_tensors="pt")
    #print(input_encoded)

    with torch.no_grad():
            states = model(**input_encoded).hidden_states
    output = torch.stack([states[i] for i in range(len(states))])
    output = output.squeeze()

    #print("Output shape is {}".format(output.shape))

    final_hidden_state = output[-4, :, ...]

    return final_hidden_state,tokens


sentence = ["वह देवास निर्वाचन क्षेत्र के प्रतिनिधित्व (भारतीय राष्ट्रीय कांग्रेस) के नेता है ।"]
sentence = ["शिवपुरी बाबा हिन्दू धर्म के एक सन्त थे।"]

facts_list = ["member of political party", "Indian National Congress"]

# 1.Tail level comparision
# Pass all phrases with window length of the fact
# Average embeddings - done 
# computer distance between all pairs and identify minimum distance phrase 

# , Fact level comparison

def compute_distances(emb1,emb2,dis):

    euc = distance.euclidean(np.array(emb1), np.array(emb2))
    cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))


    if distance=='euclidean':
        return euc
    else:
        return cosine

if __name__ == "__main__":
    #root = "/Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/"
    root = "/Users/rahulmehta/Desktop/MultiSent/datasets/17Dec2022/model_outputs/inference-as-bn-en-gu-hi-kn-ml-mr-or-pa-ta-te-google-mt5-small-30-0.001-unified-script"
    output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-all-' + DISTANCE + '/'
    for subdir, dirs, files in os.walk(root):
        lang = subdir[subdir.rfind('/')+1:]
        print(lang)
        # lang = subdir[-2:]
        # print(lang)
    
        if len(lang)>2:
            continue
        # if lang in ['7/','']:    
        #     continue
        else:
        # #if lang in ['hi']:
            with open(root + '/' + lang + '/test.jsonl') as f:
                #j=0
                for line in f:
                    #j+=1
                    #print(j)
                    # if j>10:
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
                    #print(fact)
                    
                    #print(text2)
                    text2 = text2.translate(str.maketrans('', '', string.punctuation))
                    #print(text2)
                    # for f in facts_list:

                    #     if FACT_TYPE == 'tail':
                    #         fact = f[1]
                    #     elif FACT_TYPE == 'tailandfact':
                    #         fact = f[0] + ' ' + f[1]
                    #     #print(fact)

                    #fact = "Indian National Congress"
                    #text2 = "वह देवास निर्वाचन क्षेत्र के प्रतिनिधित्व (भारतीय राष्ट्रीय कांग्रेस) के नेता है"

                    # fact = "Hinduism"
                    # text2 = "शिवपुरी बाबा हिन्दू धर्म के एक सन्त थे"

                    #text2 = ["भारतीय","राष्ट्रीय","कांग्रेस"]  # # Separate pairwise 0.06131437420845032,Cosine Similarity: 0.99526656
                    #text2 = ["वह देवास निर्वाचन क्षेत्र के प्रतिनिधित्व (भारतीय राष्ट्रीय कांग्रेस) के नेता है"]


                    # fact = "Indian Congress"
                    # text2 = "भारतीय राष्ट्रीय कांग्रेस"

                    emb1,tokens1 = generate_embeddings(fact)

                    #print(emb1.shape)
                    #print(tokens1)


                    

                    #avg_emb1 = torch.mean(emb1[1:emb1.shape[0]-1],dim=0)  #Exclude CLS and SEP embeddings
                    #print(avg_emb1)

                    emb2,tokens2 = generate_embeddings(text2)


                    # Match each word of the sentence with fact words
                    max_pairs = []
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
                        
                        for i in range(1,len(emb1)):
                            #print(emb1)
                            #print(emb1[i:i+1].shape)
                            dist.append(compute_distances(word_emb[0],emb1[i:i+1][0],DISTANCE))
                        #print(dist)
                        max_index = np.argmax(dist)
                        
                        token_top = tokens1[max_index+1]

                        token_score = float(round(max(dist),2))

                        #print(max_index,token_top)
                        pair = tuple((word,token_top,token_score))

                        if token_top in TOKENS_TO_EXCLUDE:
                            continue
                        else:
                            max_pairs.append(pair)

                    
                    #results['entity_name'] = data['entity_name']
                    results['facts'] = data['facts']
                    results['sentence'] = data['generated_sentence']

                    results['scores'] = max_pairs

                    #print(results)

                    # r.write(str(results))
                    # r.write("\n")


                    with open(output_path  + lang + '-coverage.jsonl','a') as fp: 
                        json.dump(results,fp,ensure_ascii=False)
                        fp.write('\n')