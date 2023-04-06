
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



TOKENS_TO_EXCLUDE = ['[CLS]','[SEP]','।']

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased",output_hidden_states=True)


def generate_embeddings(text):

    """
    text : list of words  ['a','b']
    """

    input_encoded = tokenizer(text,is_split_into_words=False, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(input_encoded["input_ids"].tolist()[0])

    with torch.no_grad():
            states = model(**input_encoded).hidden_states
    output = torch.stack([states[i] for i in range(len(states))])
    output = output.squeeze()

    final_hidden_state = torch.mean(output[:, :, ...], dim=0)


    return final_hidden_state,tokens,input_encoded.word_ids(),text.split(" ")


count = 0 


enlist = open('nouns.txt').readlines()
otherlang = open('gu_nouns.txt')


for x in len(enlist):
    lol = generate_embeddings(x.strip())
    print (lol)
    count += 1 

    if count >= 5 :
        break 






