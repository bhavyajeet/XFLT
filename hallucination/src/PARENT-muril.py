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


tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)

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

def compute_distances(emb1,emb2):

    euc = distance.euclidean(np.array(emb1), np.array(emb2))
    cosine = np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))

    return cosine 

if __name__ == "__main__":
    root = "/Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/"
   
    for subdir, dirs, files in os.walk(root):

        lang = subdir[-2:]
        print(lang)
    
        if lang in ['gu','hi','7/','']:    
            continue
        else:
            with open(subdir + '/test.jsonl') as f,open(f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results/' + lang + '-coverage.txt','a') as r:
                i=0
                for line in f:
                    # i+=1
                    # if i>10:
                    #     break

                    c = 0
                    data = json.loads(line)
                    f = []
                    facts_list = data['facts']
                    text2 = data['sentence']
                    entity = data['entity_name']
                    #print(facts_list)

                    results = {}    
                    temp_top = []
                    temp_topdist = []

                    for f in facts_list:

                        fact = f[1]
                        #print(fact)

                    #fact = "Indian National Congress"
                    #text2 = "वह देवास निर्वाचन क्षेत्र के प्रतिनिधित्व (भारतीय राष्ट्रीय कांग्रेस) के नेता है"

                    # fact = "Hinduism"
                    # text2 = "शिवपुरी बाबा हिन्दू धर्म के एक सन्त थे"

                    #text2 = ["भारतीय","राष्ट्रीय","कांग्रेस"]  # # Separate pairwise 0.06131437420845032,Cosine Similarity: 0.99526656
                    #text2 = ["वह देवास निर्वाचन क्षेत्र के प्रतिनिधित्व (भारतीय राष्ट्रीय कांग्रेस) के नेता है"]

                        emb1,_ = generate_embeddings(fact)
                        #print(emb1.shape[0])
                        avg_emb1 = torch.mean(emb1[1:emb1.shape[0]-1],dim=0)  #Exclude CLS and SEP embeddings
                        #print(avg_emb1)

                        emb2,tokens2 = generate_embeddings(text2)

                    # How to average
                    # avg_emb2 = torch.mean(emb2,dim=0) # or # Take rolling based on fact length

                        fact_len = emb1.shape[0]-2
                        # print(fact_len)
                        # print(tokens2)
                        # print(emb2.shape)
                        phrase_dict = {}

                        for i in range(1,emb2.shape[0]-1):
                            #print(i)
                            phrase = ' '.join(tokens2[i:i+fact_len])
                            #print(phrase)

                            avg_emb2 = torch.mean(emb2[i:i+fact_len],dim=0)
                            phrase_dict[phrase] = avg_emb2

                        #print(phrase_dict.keys())

                
                        phrase_scores = {}
                        for k,v in phrase_dict.items():
                            #print(v)
                            phrase_scores[k] = compute_distances(avg_emb1,v)

                        #print(phrase_scores)
                        
                        # Maximum phrase scores
                        max_key = max(phrase_scores, key=phrase_scores.get)
                        #max_value = max(phrase_scores)
                        print(max_key,phrase_scores[max_key])

                        temp_top.append(max_key)
                        temp_topdist.append(phrase_scores[max_key])



                    results['entity_name'] = data['entity_name']
                    results['facts'] = data['facts']
                    results['sentence'] = data['sentence']

                    results['phrase_top'] = temp_top
                    results['phrase_topdist'] = temp_topdist
        
                    #print(results)

                    r.write(str(results))
                    r.write("\n")