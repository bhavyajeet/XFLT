from transformers import AutoTokenizer, AutoModel
import functorch
import regex as re 
import torch 
import torch.nn.functional as F
import numpy as np
from functools import lru_cache
from nltk.util import ngrams 
from collections import Counter
import os 
from icecream import ic 

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def group_duplicates(embeddings, lst, mean=True):
    embeddings = torch.cat([e.reshape(1, -1) for i, e in enumerate(embeddings) if lst[i] is not None], dim=0)
    lst = [i for i in lst if i is not None]
    output = {key: None for key in (set(lst))}
    if(list(set(lst)) != list(range(max(lst)+1))):
        ic(f"Group duplicates problem {lst}")
    output_list = [] 
    i = 0
    for idx, i in enumerate(lst):
        if(i is not None):
            if(output[i] == None):
                output[i] = embeddings[idx, :].reshape(1, -1)
            else:
                output[i] = torch.cat((output[i], embeddings[idx, :].reshape(1, -1)), dim=0)
    if(mean):
        for key, val in output.items():
            output_list.append(torch.mean(val, dim=0).reshape(1, -1))
    return output_list

def get_facts(source, token_split=True):
    words = source.split(":")[1].split(" ")
    facts = re.split(r'<[^>]*>', " ".join(words))
    facts = [re.sub(r'_', ' ', facts[i].strip()) for i in range(len(facts))]
    if(token_split):
        return tuple([re.sub(r'_', ' ', f) for i in range(len(facts)) for f in facts[i].split()]), [i-1 for i, word in enumerate(facts) for _ in range(len(word.split()))]
    return tuple(facts), []

@lru_cache(maxsize=10000)
def get_embedding(tokens, model, tokenizer, split_into_words=False, parent_device = 'cuda:4'):
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=True, truncation=True, max_length=512, is_split_into_words=split_into_words, return_tensors="pt").to(parent_device)
        #print(tokenizer.decode(tokenized_facts[1].ids))
        batch_states = model(**tokenized_facts).hidden_states
        batch_output = torch.stack([batch_states[i] for i in range(len(batch_states))])
        batch_output = batch_output.squeeze()
        batch_final_hidden_state = torch.mean(batch_output[:, :, ...], dim=0)
        return batch_final_hidden_state[:, 1:-1, :], list(map(lambda i: tokenized_facts.word_ids(i)[1:-1], range(len(tokens))))

def entailment_prob(fact_embeddings, generated_ngram, threshold=None): 
    if(generated_ngram.dim() == 1):
        generated_ngram = generated_ngram.reshape(1, -1)
    similarities = functorch.vmap(lambda row_a: F.cosine_similarity(row_a, fact_embeddings))(generated_ngram)
    if(threshold):
        #print(torch.max(similarities, dim=1).values.cpu())
        return np.mean((torch.max(similarities, dim=1).values > threshold).int().cpu().numpy())
    return np.mean(torch.max(similarities, dim=1).values.cpu().numpy())

def get_coverage_reward(generated, source,  model, parentTokenizer, parent_device):
    try:
        generated = list(map(lambda gen: re.sub(r"[',.।()]", '', gen), generated))
        generated_tokens = tuple(map(lambda gen: tuple(re.sub(r'[ ]{2,}', ' ', gen.strip()).split()),  generated))
        generated_embeddings, g_idx = get_embedding(generated_tokens, model, parentTokenizer, split_into_words=True, parent_device=parent_device)
        gen_emb = list(map(lambda ge, gi: group_duplicates(ge, gi), generated_embeddings, g_idx))
        # gen_emb = [] 
        # for ge, gi in zip(generated_embeddings, g_idx):
        #     gen_emb.append(group_duplicates(ge, gi))

        fact_pos_pairs = tuple(map(lambda src: get_facts(src, token_split=True), source))
        facts = tuple(fact_pos_pairs[i][0] for i in range(len(fact_pos_pairs)))
        #print(facts)
        fact_pos = tuple(fact_pos_pairs[i][1] for i in range(len(fact_pos_pairs)))
        fact_embeddings, f_idx = get_embedding(facts, model, parentTokenizer, split_into_words=True, parent_device=parent_device)
        #print(fact_embeddings[0].shape)
        #print(f_idx[0])
        fact_embeddings = list(map(lambda fe, fi: group_duplicates(fe, fi), fact_embeddings, f_idx))
        fact_emb = [torch.cat(fe).squeeze() for fe in fact_embeddings]
        batch_eps = []
        for generated_tokens_, gen_emb_, fact_emb_ in zip(generated_tokens, gen_emb, fact_emb):
            g_dict = {t: emb for t, emb in zip(generated_tokens_, gen_emb_)}
            generated_ngrams = Counter(list(ngrams(generated_tokens_, 1)))
            eps = [] 
            for ngram, count in generated_ngrams.items():
                ngram_embs = [g_dict[i] for i in ngram if i in g_dict]
                if(ngram_embs != []):
                    ngram_embedding = torch.cat([g_dict[i] for i in ngram if i in g_dict]).squeeze()
                    ep = entailment_prob(fact_emb_, ngram_embedding, 0.4)
                    eps.append(ep)
                else:
                    eps.append(0)
            batch_eps.append(np.mean(eps))
        return torch.FloatTensor(batch_eps)
    except Exception:
        ic(f"Coverage reward problem {generated} {source}")
        return torch.FloatToensor([0.3 for _ in range(len(generated))])

# def get_coverage_reward(generated, source, model, parentTokenizer, parent_device):
