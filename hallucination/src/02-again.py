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
import sklearn.metrics
import numpy as np
from numpy.linalg import norm
import glob 
import json 
import os
import string
from collections import Counter
from statistics import median 
import time 
import sys 

from functools import lru_cache
import functorch
import regex as re 
import torch 
import time 
import torch.nn.functional as F


#FACT_TYPE =    'tailandfact'
DISTANCE = 'cosine'

TOKENS_TO_EXCLUDE = ['[CLS]','[SEP]','।']

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)

punct_list = ['a','b']

@lru_cache
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


    final_hidden_state = torch.mean(output[:, :, ...], dim=0)

    return final_hidden_state,tokens,input_encoded.word_ids(),text.split(" ")


def compute_distances(emb1,emb2,dis):

    euc = distance.euclidean(np.array(emb1), np.array(emb2))
    cosine = 1 - np.dot(emb1,emb2)/(norm(emb1)*norm(emb2))

    if distance=='euclidean':
        return euc
    else:
        return cosine

@lru_cache
def get_embedding(tokens):
    #print(tokens)
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=False, truncation=False, return_tensors="pt")
        #print(tokenizer.convert_ids_to_tokens(tokenized_facts['input_ids'][0]))
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        final_hidden_state = torch.mean(output[:, :, ...], dim=0)
        #print(final_hidden_state.shape)
        return final_hidden_state
        #return embeddings['last_hidden_state'] #tokenized_facts['attention_mask']
        #return torch.mean(embeddings['last_hidden_state'], dim=1)

def get_similarity(token, input_sent, metric):
    words = input_sent.split(" ")
    language = words[1]
    facts = re.split(r'<.>', " ".join(words[3:]))
    facts = [re.sub(r'_', ' ', f) for i in range(len(facts))  for f in facts[i].split()]
    fact_embedding = get_embedding(" ".join(facts))[1:-1]
    #print(fact_embedding.shape)
    #fact_embedding = fact_embedding.reshape(fact_embedding.shape[0]*fact_embedding.shape[1], -1)
    #print(fact_embedding.shape)
    token_embedding = get_embedding(tuple([token,]))[1:-1]
    #print(token_embedding.shape, fact_embedding.shape)
    token_embedding = token_embedding.repeat(fact_embedding.shape[0], 1, 1)
    #print(token_embedding.shape)
    return functorch.vmap(lambda x, y: metric(x, y))(token_embedding, fact_embedding)

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


if __name__ == "__main__":
    start = time.time()

    # Generate reference scores
    root = "/home2/aditya_hari/multisent/pls/lin_data"
    output_path = f'/scratch/aditya_hari/vectors/' + DISTANCE + '/'
    
    for lang in sorted(list(langlist.values())):
    #for lang in ['en']:
        for thing in ['train', 'val']:
            print ('=========')
            print(lang, thing)
            print ('=========')

            j=0

            if lang in ['7/','', 'en', 'hi']:    
                continue
            results_all = []
            count = 0 
            lines = open(f'{root}/{lang}/{thing}.jsonl', 'r').readlines()
            pb = tqdm(range(len(lines)))
            for line in lines:
                pb.update(1)
                j+=1
                c = 0
                data = json.loads(line)
                f = []
                facts_list_all = data['facts_list']
                entity_name = data['entity_name']
                
                facts_list = []
                facts_list.extend(entity_name.split(" "))
                #print(facts_list)
                for fact_ in facts_list_all:
                    for fact in fact_:
                        facts_list.append(fact[0])
                        facts_list.append(fact[1])
                        for q in fact[2]:
                            facts_list.extend(q)
                ref_sent = data['sentence_list'][0]
                results = {}    
                temp_top = []
                temp_topdist = []
                fact = " ".join(facts_list)
                fact = fact.replace("_"," ")

                embf,tokensf,subword_ids,word_list = generate_embeddings(fact)
                embf = embf[1:-1]

                embt,tokenst,_,_= generate_embeddings(ref_sent)

                if distance == 'cosine':
                    distances = scipy.spatial.distance_matrix(embt,embf)
                else:
                    distances = sklearn.metrics.pairwise.cosine_distances(embt, embf)
                fact_idx =  np.argmin(distances,axis=1)
                words = [tokensf[i+1] for i in fact_idx]

                dist = np.min(distances,axis=1)
                dist = [i.item() for i in dist]
                min_pairs = list(zip(tokenst,words,dist))
                # print(facts_list)
                # print(min_pairs)
                # input()
                results['facts'] = data['facts_list']
                results['ref_sentence'] = ref_sent
                results['scores'] = min_pairs
                results_all.append(results)

            with open(output_path + lang + f'-{thing}-coverage.jsonl','w') as fp: 
                for i in results_all:
                    json.dump(i,fp,ensure_ascii=False)
                    fp.write('\n')
                            
        end = time.time()
        print(end - start)
