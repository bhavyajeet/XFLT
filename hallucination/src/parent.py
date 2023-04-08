from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from nltk.util import ngrams 
from indicnlp.tokenize import indic_tokenize
from collections import Counter 
import json
from collections import defaultdict 

import functorch
import regex as re 
import torch 
import time 
import torch.nn.functional as F
import numpy as np
from parent import parent
from pygoogletranslation import Translator
import tqdm

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModel.from_pretrained("google/muril-base-cased", output_hidden_states=True)

lang_code_map = {
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

@lru_cache
def get_embedding(tokens):
    #print(tokens)
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=False, truncation=False, return_tensors="pt")
        #print(tokenizer.convert_ids_to_tokens(tokenized_facts['input_ids'][0]))
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        #print(output.shape)
        final_hidden_state = torch.mean(output[:, :, ...], dim=0)
        #final_hidden_state = output[-2, :, ...]
        #print(final_hidden_state.shape)
        return final_hidden_state
        #return embeddings['last_hidden_state'] #tokenized_facts['attention_mask']
        #return torch.mean(embeddings['last_hidden_state'], dim=1)

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None]*(n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    return L[m][n]

def get_similarity(token, facts, metric=F.cosine_similarity):
    fact_embedding = get_embedding(facts)[1:-1]
    #print(fact_embedding.shape)
    #fact_embedding = fact_embedding.reshape(fact_embedding.shape[0]*fact_embedding.shape[1], -1)
    #print(fact_embedding.shape)
    token_embedding = get_embedding(tuple([token,]))[1:-1]
    # print("heyo", metric(token_embedding, fact_embedding))
    #print(token_embedding.shape, fact_embedding.shape)
    token_embedding = token_embedding.repeat(fact_embedding.shape[0], 1, 1)
    #print(token_embedding.shape)
    return functorch.vmap(lambda x, y: metric(x, y))(token_embedding, fact_embedding)

def get_facts(source, token_split=True, names=False):
    words = source.split(" ")
    language = words[1]
    facts = re.split(r'<.>', " ".join(words[3:]))
    if(names):
        facts = [facts[i].strip() for i in range(len(facts))]
    else:
        facts = [facts[i].strip() for i in range(1, len(facts), 2)]
    #print(facts)
    if(token_split):
        return [re.sub(r'_', ' ', f) for i in range(len(facts))  for f in facts[i].split()]    
    return [re.sub(r'_', ' ', f).strip() for f in facts if f!='']


def entailment_prob(source, generated_ngram, threshold=None): 
    facts = get_facts(source)
    fact_string = " ".join(facts)
    if(threshold):
        return np.mean([torch.max(get_similarity(ng, fact_string)).item() > threshold for ng in generated_ngram])
    return np.mean([torch.max(get_similarity(ng, fact_string)).item() for ng in generated_ngram])

def parent_precision(source, reference_tokens, generated_tokens, threshold=None, n_max=4):
    # language = lang_code_map[source.split(" ")[1]]
    # generated = re.sub(r"[',।]", '', generated)
    # reference = re.sub(r"[',।]", '', reference)
    # if(language!='en'):
    #     generated_tokens = indic_tokenize.trivial_tokenize(generated, lang=language)
    #     reference_tokens = indic_tokenize.trivial_tokenize(reference, lang=language)
    # else:
    #     generated_tokens = generated.split(" ")
    #     reference_tokens = reference.split(" ")
    entailed_precisions = [] 
    for n in range(1, n_max+1):
        generated_ngrams = Counter(list(ngrams(generated_tokens, n)))
        reference_ngrams = Counter(list(ngrams(reference_tokens, n)))
        ngram_intersection = generated_ngrams & reference_ngrams
        entailed_precision = 0
        for ngram in generated_ngrams:
            ep = entailment_prob(source, ngram, threshold)
            entailed_precision+=(generated_ngrams[ngram]*ep)
            #entailed_precision+=1
            if(ngram in ngram_intersection):
                entailed_precision+=(ngram_intersection[ngram]*(1-ep))
                #entailed_precision+=ep
        entailed_precisions.append(entailed_precision/len(generated_ngrams))
    final = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_precisions))/n_max)
    return final

def parent_recall(source, reference_tokens, generated_tokens, generated, trade_off=0.3, threshold=None, n_max=4):
    # language = lang_code_map[source.split(" ")[1]]
    # generated = re.sub(r"[',।]", '', generated)
    # reference = re.sub(r"[',।]", '', reference)
    
    # if(language!='en'):
    #     generated_tokens = indic_tokenize.trivial_tokenize(generated, lang=language)
    #     reference_tokens = indic_tokenize.trivial_tokenize(reference, lang=language)
    # else:
    #     generated_tokens = generated.split(" ")
    #     reference_tokens = reference.split(" ")
        
    entailed_recall_reference = [] 
    for n in range(1, n_max+1):
        generated_ngrams = Counter(list(ngrams(generated_tokens, n)))
        reference_ngrams = Counter(list(ngrams(reference_tokens, n)))
        ngram_intersection = generated_ngrams & reference_ngrams
        entailed_recall = 0
        denominator = 0 
        for ngram in reference_ngrams:
            ep = entailment_prob(source, ngram, threshold)
            denominator+=(reference_ngrams[ngram]*ep)
            if(ngram in ngram_intersection):
                entailed_recall+=(ngram_intersection[ngram]*ep)
        entailed_recall_reference.append(entailed_recall/denominator)
    final_entailed_recall_reference = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_recall_reference))/n_max)
    #print("recall-ref", final_entailed_recall_reference)

    facts = get_facts(source, token_split=False)
    # for fact in facts:
    #     print(fact)
    #     print(generated)
    #     print(get_similarity(fact, generated))
    tokenwise_max_matches = [torch.max(torch.max(get_similarity(fact, generated), dim=1).values).item() for fact in facts]
    #tokenwise_max_matches = [torch.mean((torch.max(get_similarity(fact, generated), dim=1).values > 0.5).float()).item() for fact in facts]
    #tokenwise_max_matches = [torch.mean((torch.max(get_similarity(fact, generated), dim=0).values > 0.5).float()).item() for fact in facts]
    #print(tokenwise_max_matches)
    entailed_recall_source = np.mean(tokenwise_max_matches)
    #print("recall-source", entailed_recall_source)
    # fact_token_matches = []
    # for fact in facts:
    #     similarity = get_similarity(fact, generated)
    #     match_idx = torch.max(similarity, dim=0).indices 
    #     fact_token_match = lcs(match_idx, range(len(match_idx)))
    #     fact_token_matches.append(fact_token_match/len(match_idx))
    # print(facts)
    # print(fact_token_matches)
    #print(tokenwise_max_matches)

    #entailed_recall_source = np.mean(fact_token_matches)
    #print(entailed_recall_source)
    #print(final_entailed_recall_reference, entailed_recall_source)
    recall_ref = np.power(final_entailed_recall_reference, trade_off)
    recall_src = np.power(entailed_recall_source, 1-trade_off)
    #print(recall_ref, recall_src)
    return recall_ref*recall_src

def xparent_f1(source, reference, generated, trade_off=0.5):
    language = lang_code_map[source.split(" ")[1]]
    generated = re.sub(r"[',।]", '', generated)
    reference = re.sub(r"[',।]", '', reference)
    if(language!='en'):
        generated_tokens = indic_tokenize.trivial_tokenize(generated, lang=language)
        reference_tokens = indic_tokenize.trivial_tokenize(reference, lang=language)
    else:
        generated_tokens = generated.split(" ")
        reference_tokens = reference.split(" ")
    precision = parent_precision(source, reference_tokens, generated_tokens, threshold=0.5, n_max=4)    
    #print("precisio", precision)
    recall = parent_recall(source, reference_tokens, generated_tokens, generated, trade_off, n_max=2)
    #print("recall", recall)
    return (2*precision*recall)/(precision+recall)
