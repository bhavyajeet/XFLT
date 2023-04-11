from transformers import AutoTokenizer, AutoModel
import functorch
import regex as re 
import torch 
import torch.nn.functional as F
import numpy as np
from functools import lru_cache
from nltk.util import ngrams 
from collections import Counter




def group_duplicates(embeddings, lst, mean=True):
    output = [None for _ in range(len(set(lst)))]
    i = 0
    for idx, i in enumerate(lst):
        if(i!=None):
            if(output[i] == None):
                output[i] = embeddings[idx, :].reshape(1, -1)
            else:
                output[i] = torch.cat((output[i], embeddings[idx, :].reshape(1, -1)), dim=0)
    if(mean):
        for idx, val in enumerate(output):
            output[idx] = torch.mean(output[idx], dim=0).reshape(1, -1)
    return output

def get_facts(source, token_split=True):
    words = " ".join(source.split(":")[1]).split(" ")
    facts = re.split(r'<[^>]*>', " ".join(words))
    facts = [re.sub(r'_', ' ', facts[i].strip()) for i in range(len(facts))]
    if(token_split):
        return [re.sub(r'_', ' ', f) for i in range(len(facts)) for f in facts[i].split()], [i-1 for i, word in enumerate(facts) for _ in range(len(word.split()))]
    return facts, []

@lru_cache(maxsize=10000)
def get_embedding(tokens, model, tokenizer, split_into_words=False, parent_device = 'cuda:4'):
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=True, truncation=True, max_length=512, is_split_into_words=split_into_words, return_tensors="pt").to(parent_device)
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        final_hidden_state = torch.mean(output[:, :, ...], dim=0)
        return final_hidden_state[1:-1], tokenized_facts.word_ids()[1:-1]

def entailment_prob(fact_embeddings, generated_ngram, threshold=None): 
    if(generated_ngram.dim() == 1):
        generated_ngram = generated_ngram.reshape(1, -1)
    similarities = functorch.vmap(lambda row_a: F.cosine_similarity(row_a, fact_embeddings))(generated_ngram)
    if(threshold):
        #print(torch.max(similarities, dim=1).values.cpu())
        return np.mean((torch.max(similarities, dim=1).values > threshold).int().cpu().numpy())
    return np.mean(torch.max(similarities, dim=1).values.cpu().numpy())

def get_coverage_reward(generated, source,  model, parentTokenizer, parent_device):
    generated = re.sub(r"[',.।()]", '', generated)
    generated_tokens = re.sub(r'[ ]{2,}', ' ', generated.strip()).split()
    print(generated_tokens)
    generated_embeddings, g_idx = get_embedding(tuple(generated_tokens), model, parentTokenizer, True, parent_device)
    gen_emb = group_duplicates(generated_embeddings, g_idx)

    facts, fact_pos = get_facts(source, token_split=True)
    fact_embeddings, f_idx = get_embedding(tuple(facts), model, parentTokenizer, split_into_words=True, parent_device=parent_device)
    fact_embeddings = group_duplicates(fact_embeddings, f_idx)
    fact_emb = torch.cat(fact_embeddings).squeeze()
    
    g_dict = {t: emb for t, emb in zip(generated_tokens, gen_emb)}

    generated_ngrams = Counter(list(ngrams(generated_tokens, 1)))
    eps = [] 
    for ngram, count in generated_ngrams.items():
        ngram_embedding = torch.cat([g_dict[i] for i in ngram if i in g_dict]).squeeze()
        ep = entailment_prob(fact_emb, ngram_embedding, 0.4)
        eps.append(ep)
    print(generated_ngrams)
    return eps 