from transformers import AutoTokenizer, AutoModel
import functorch
import regex as re 
import torch 
import torch.nn.functional as F
import numpy as np
from functools import lru_cache
from nltk.util import ngrams 
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", padding='max_length', truncation='max_length', max_length=512)
model = AutoModel.from_pretrained("google/muril-base-cased", output_hidden_states=True).to(device)

def group_duplicates(embeddings, lst, mean=True):
    embeddings = torch.cat([e.reshape(1, -1) for i, e in enumerate(embeddings) if lst[i] is not None], dim=0)
    lst = [i for i in lst if i is not None]
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

@lru_cache(maxsize=10000)
def get_embedding(tokens, split_into_words=False):
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=True, truncation=True, max_length=512, is_split_into_words=split_into_words, return_tensors="pt").to(device)
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        final_hidden_state = torch.mean(output[:, :, ...], dim=0)

        return final_hidden_state, tokenized_facts.word_ids()


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

def get_similarity(embedding_a, embedding_b, metric=F.cosine_similarity):
    embedding_a = embedding_a.repeat(embedding_b.shape[0], 1, 1)
    return functorch.vmap(lambda x, y: metric(x, y))(embedding_a, embedding_b)

def get_facts(source, token_split=True):
    words = source.split(":")[1].split(" ")
    facts = re.split(r'<[^>]*>', " ".join(words))
    facts = [re.sub(r'_', ' ', facts[i].strip()) for i in range(len(facts))]
    if(token_split):
        return tuple([re.sub(r'_', ' ', f) for i in range(len(facts)) for f in facts[i].split()]), [i-1 for i, word in enumerate(facts) for _ in range(len(word.split()))]
    return tuple(facts), []

def entailment_prob(fact_embeddings, generated_ngram, threshold=None): 
    if(generated_ngram.dim() == 1):
        generated_ngram = generated_ngram.reshape(1, -1)
    similarities = functorch.vmap(lambda row_a: F.cosine_similarity(row_a, fact_embeddings))(generated_ngram)
    if(threshold):
        return np.mean((torch.max(similarities, dim=1).values > threshold).int().cpu().numpy())
    return np.exp(np.mean(np.log(torch.max(similarities, dim=1).values.cpu().numpy())))

def parent_precision(fact_embeddings, reference_tokens, generated_tokens, reference_embeddings, generated_embeddings, threshold=None, n_max=1):
    entailed_precisions = [] 
    for n in range(1, n_max+1):
        generated_ngrams = Counter(list(ngrams(generated_tokens, n)))
        reference_ngrams = Counter(list(ngrams(reference_tokens, n)))
        ngram_intersection = generated_ngrams & reference_ngrams
        entailed_precision = 0
        numerator = 0
        denominator = 0
        for ngram, count in generated_ngrams.items():
            denominator+=count 
            ngram_embedding = torch.cat([generated_embeddings[i] for i in ngram if i in generated_embeddings]).squeeze()
            ep = entailment_prob(fact_embeddings, ngram_embedding, threshold)
            prob_ngram_in_ref = min(1, reference_ngrams.get(ngram, 0)/count)
            numerator += count*(prob_ngram_in_ref + (1-prob_ngram_in_ref)*ep)

        if(denominator == 0):
            entailed_precisions.append(0)
        else:
            entailed_precisions.append(numerator/denominator)
    final = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_precisions))/n_max)
    return min(1, final)

def parent_recall(fact_embeddings, fact_embeddings_wo, fact_tokens, reference_tokens, generated_tokens, reference_embeddings, generated_embeddings, trade_off=0.3, threshold=None, n_max=4):

    entailed_recall_reference = [] 
    for n in range(1, n_max+1):
        generated_ngrams = Counter(list(ngrams(generated_tokens, n)))
        reference_ngrams = Counter(list(ngrams(reference_tokens, n)))
        numerator = 0
        denominator = 0
        for ngram, count in reference_ngrams.items():
            ngram_embedding = torch.cat([reference_embeddings[i] for i in ngram if i in reference_embeddings]).squeeze()
            ep = entailment_prob(fact_embeddings, ngram_embedding, threshold)
            prob_ngram_in_pred = min(1, generated_ngrams.get(ngram, 0)/count)
            denominator+=count*ep
            numerator+=count*prob_ngram_in_pred*ep 
        if(denominator == 0):
            entailed_recall_reference.append(1)
        else:
            entailed_recall_reference.append(numerator/denominator)
    
    final_entailed_recall_reference_1 = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_recall_reference))/n_max)
    final_entailed_recall_reference_2 = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_recall_reference[:-1]))/n_max)
    final_entailed_recall_reference_3 = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_recall_reference[:-2]))/n_max)
    final_entailed_recall_reference_4 = np.exp(sum(map(lambda x: np.log(x+1e-10), entailed_recall_reference[:-3]))/n_max)

    gen_emb = torch.cat(generated_embeddings).squeeze()
    if(gen_emb.dim() == 1):
        gen_emb = gen_emb.reshape(1, -1)
    similarities_gen = functorch.vmap(lambda row_a: F.cosine_similarity(row_a, fact_embeddings_wo))(gen_emb).T
    tokenwise_max_matches = torch.max(similarities_gen, dim=1).values
    if(threshold):
        entailed_recall_source_max = np.mean((tokenwise_max_matches>threshold).int().cpu().numpy())
    else:
        entailed_recall_source_max = np.exp(np.mean(np.log(tokenwise_max_matches, dim=1).values.cpu().numpy()))


    return final_entailed_recall_reference_1, final_entailed_recall_reference_2, final_entailed_recall_reference_3, final_entailed_recall_reference_4, entailed_recall_source_max

def xparent_f1(source, reference, generated, tokenizer, threshold = 0.5, trade_off=0.5):
    generated = re.sub(r"[',.।()-]", '', generated)
    reference = re.sub(r"[',.।()-]", '', reference)

    generated_tokens = tokenizer(re.sub(r'[ ]{2,}', ' ', generated.strip()))
    reference_tokens = tokenizer(re.sub(r'[ ]{2,}', ' ', reference.strip()))
    generated_embeddings, g_idx = get_embedding(tuple(generated_tokens), True)
    reference_embeddings, r_idx = get_embedding(tuple(reference_tokens), True)

    gen_emb = group_duplicates(generated_embeddings, g_idx)
    ref_emb = group_duplicates(reference_embeddings, r_idx)

    g_dict = {t: emb for t, emb in zip(generated_tokens, gen_emb)}
    r_dict = {t: emb for t, emb in zip(reference_tokens, ref_emb)}

    facts, fact_pos = get_facts(source, token_split=True)
    fact_embeddings, f_idx = get_embedding(tuple(facts), split_into_words=True)
    fact_embeddings = group_duplicates(fact_embeddings, f_idx)
    fact_emb = torch.cat(fact_embeddings).squeeze()
    fact_emb_wo_names = torch.cat([fact_embeddings[idx] for idx, i in enumerate(fact_pos) if i%2==0]).squeeze()
    precision = parent_precision(fact_emb, reference_tokens, generated_tokens, r_dict, g_dict, threshold=threshold, n_max=4)    
    recalls = parent_recall(fact_emb, fact_emb_wo_names, facts, reference_tokens, generated_tokens, r_dict, gen_emb, threshold=threshold, n_max=4)

    return precision, recalls[0], recalls[1], recalls[2], recalls[3], recalls[4]
