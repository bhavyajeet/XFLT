from evaluate import load
import evaluate
import sys
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from sacrebleu.metrics import BLEU
from indicnlp.tokenize import indic_tokenize
from rouge_score import rouge_scorer
import csv
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
from pygoogletranslation import Translator
import tqdm

f = open('mt5-resfile.csv', 'w')
writer = csv.writer(f)

space_split = lambda x: x.split() 

varlol = sys.argv[1]
stratList = [varlol]

# tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
# model = AutoModel.from_pretrained("google/muril-base-cased", output_hidden_states=True)

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

def calc_meteor(srclines,dstlines):
    results = meteor.compute(predictions=srclines, references=dstlines)
    # print (results)
    return results

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
        return np.mean([torch.max(torch.clamp(get_similarity(ng, fact_string)), min=0, max=1).item() > threshold for ng in generated_ngram])
    return np.mean([torch.max(torch.clamp(get_similarity(ng, fact_string)), min=0, max=1).item() for ng in generated_ngram])

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
    # entailed_recall_source = np.mean(tokenwise_max_matches)
    # print("recall-source", entailed_recall_source)
    fact_token_matches = []
    for fact in facts:
        similarity = get_similarity(fact, generated)
        match_idx = torch.max(similarity, dim=0).indices 
        fact_token_match = lcs(match_idx, range(len(match_idx)))
        fact_token_matches.append(fact_token_match/len(match_idx))
    # print(facts)
    # print(fact_token_matches)
    #print(tokenwise_max_matches)

    entailed_recall_source = np.mean(fact_token_matches)
    #print(entailed_recall_source)
    #print(final_entailed_recall_reference, entailed_recall_source)
    recall_ref = np.power(final_entailed_recall_reference, trade_off)
    recall_src = np.power(entailed_recall_source, 1-trade_off)
    #print(recall_ref, recall_src)
    return recall_ref*recall_src

def xparent_f1(source, reference, generated, tokenizer, trade_off=0.5):
    language = lang_code_map[source.split(" ")[1]]
    generated = re.sub(r"[',।]", '', generated)
    reference = re.sub(r"[',।]", '', reference)
    generated_tokens = tokenizer(generated, lang=language)
    reference_tokens = tokenizer(reference, lang=language)
    precision = parent_precision(source, reference_tokens, generated_tokens, threshold=0.5, n_max=4)    
    #print("precisio", precision)
    recall = parent_recall(source, reference_tokens, generated_tokens, generated, trade_off, n_max=4)
    #print("recall", recall)
    return (2*precision*recall)/(precision+recall)

# def ter(ref, gen):
#         '''
#         Args:
#             ref - reference sentences - in a list
#             gen - generated sentences - in a list
#         Returns:
#             averaged TER score over all sentence pairs
#         '''
#         if len(ref) == 1:
#             total_score =  pyter.ter(gen[0].split(), ref[0].split())
#         else:
#             total_score = 0
#             for i in range(len(gen)):
#                 total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
#             total_score = total_score/len(gen)
#         return total_score

def bleu(ref, gen, tokenizer=None):
    ''' 
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(tokenizer(l))
    for i,l in enumerate(ref):
        ref_bleu.append(tokenizer(l))
    cc = SmoothingFunction()
    sent_bleus = []
    for ref, gen in zip(ref_bleu, gen_bleu):
        score = sentence_bleu([ref], gen, smoothing_function=cc.method4)
        # print(ref)
        # print(gen)
        # print(score)
        # input()
        sent_bleus.append(score)

    score_bleu = corpus_bleu([[r] for r in ref_bleu], gen_bleu, smoothing_function=cc.method4)
    mean_sent_bleu = sum(sent_bleus)/len(sent_bleus)
    return score_bleu, mean_sent_bleu

# print (bleu(reflines,predlines))
def char_bleu(ref, gen):
    ''' 
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''

class DummyStemmer(object):
    def __call__(self, token):
        return token

# example with custom segmenter/tokenizer
def dummy_tokenize(text):
    tokens = text.split()
    # your tokenizer implementation
    return tokens

def calc_rouge(dstlines,srclines):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    zeroCount = 0 
    r1_sum = 0
    r2_sum =0

    for i in range(len(dstlines)):
        scores = scorer.score(dstlines[i],srclines[i])
        if scores['rouge1'].fmeasure == 0 :
            # print (scores, dstlines[i],srclines[i])
            zeroCount += 1 
        r1_sum += scores['rouge1'].fmeasure
        r2_sum += scores['rougeL'].fmeasure

    print ('Rouge1' , r1_sum/len(dstlines),'   RougeL' , r2_sum/len(dstlines))
    return r1_sum/len(dstlines)
    print ('zeros', zeroCount)

def calc_chrf(dstlines,srclines):
    listolist = []

    for i in range(len(dstlines)):
        listolist.append([dstlines[i]])

    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=srclines, references=listolist)
    return results

def calc_parent(src, ref, gen, tokenizer=space_split):
    scores = []
    for s, r, g in zip(src, ref, gen):
        score = xparent_f1(s, r, g, tokenizer)
        scores.append(score)
    return sum(scores)/len(scores)

if __name__=='__main__':
    for strat in stratList :
        languages_map = {
                'hi': {"label": "Hindi", 'id': 1},
                'mr': {"label": "Marathi", 'id': 9},
                'te': {"label": "Telugu", 'id': 2}, 
                'ta': {"label": "Tamil", 'id': 11},
                'en': {"label": "English", 'id': 0},
                'gu': {"label": "Gujarati", 'id': 8},
                'bn': {"label": "Bengali", 'id': 3},
                'kn': {"label": "Kannada", 'id': 10},
                'ml': {"label": "Malayalam", 'id': 12}, 
                'pa': {"label": "Punjabi", 'id': 4},
                'or': {"label": "Odia", 'id': 6}, 
                'as': {"label": "Assamese", 'id': 7},
                }

        lang_dict = {
                'hi': {'ref':[],'pred':[]},
                'mr': {'ref':[],'pred':[]},
                'te': {'ref':[],'pred':[]}, 
                'ta': {'ref':[],'pred':[]},
                'en': {'ref':[],'pred':[]},
                'gu': {'ref':[],'pred':[]},
                'bn': {'ref':[],'pred':[]},
                'kn': {'ref':[],'pred':[]},
                'ml': {'ref':[],'pred':[]},
                'pa': {'ref':[],'pred':[]},
                'or': {'ref':[],'pred':[]},
                'as': {'ref':[],'pred':[]},
                }

        def get_lang_key(word):
            for i in languages_map:
                if languages_map[i]['label'].lower() == word.lower().strip():
                    return i

            return -1 

        meteor = evaluate.load('meteor')

        srcfile = './' + '/calc_gen.txt'
        dstfile = './' + '/calc_ref.txt'
        tripfile = './' + '/test-src.txt'
        langlines = open('langs.txt').readlines()

        srcf = open(srcfile)
        dstf = open(dstfile)
        tripf = open(tripfile)

        predlines = srcf.readlines()
        reflines = dstf.readlines()
        triplines = tripf.readlines()

        for i in range(len(reflines)):
            
            if('<R>' not in triplines[i]):
                continue 
            # lang = triplines[i].split()[1].lower()
            # lang_key = get_lang_key(lang)
            lang_key = langlines[i][:-1]

            if('src' not in lang_dict[lang_key]):
                lang_dict[lang_key]['src'] = []

            # if('Bashirul Haq' in reflines[i]):
            #     print (lang)
            #     print (lang_key)
            #     print (reflines[i])
            #     print (predlines[i])
            #     print (triplines[i])
            #     input()
            '''
            print (lang)
            print (lang_key)
            print (reflines[i])
            print (predlines[i])
            print (triplines[i])
            '''
            lang_dict[lang_key]['ref'].append(reflines[i])
            lang_dict[lang_key]['pred'].append(predlines[i])
            lang_dict[lang_key]['src'].append(triplines[i])

        mtrlist = []
        chrlist = []
        bleulist = []
        bleulist2 = []
        parentlist = [] 
        slist = []
        for i in sorted(list(lang_dict.keys())):    
        #for i in ['en']:
            print (i)
            print ('meteor')
            #mtrsc = calc_meteor(lang_dict[i]['pred'],lang_dict[i]['ref'])
            #print (mtrsc)
            #mtrlist.append(mtrsc['meteor'])
            # r1sc = calc_rouge(lang_dict[i]['ref'],lang_dict[i]['pred'])
            # slist.append(r1sc)
            # print ('chrf')
            # chrfsc = calc_chrf(lang_dict[i]['ref'],lang_dict[i]['pred'])
            # print (chrfsc)
            # #parent = calc_parent(lang_dict[i]['src'], lang_dict[i]['ref'],lang_dict[i]['pred'])
            # #print (parent)
            # chrlist.append(chrfsc['score'])
            #print('bleu')
            bleu_score, bleu_score_2 = bleu(lang_dict[i]['ref'],lang_dict[i]['pred'], tokenizer=lambda word: indic_tokenize.trivial_tokenize(word, lang=i))
            print(bleu_score, bleu_score_2)
            bleulist.append(bleu_score)
            bleulist2.append(bleu_score_2)
            print ('--'*20)
        #writer.writerow(mtrlist)
        #writer.writerow(chrlist)
        writer.writerow(bleulist)
        writer.writerow(bleulist2)


# lambda x: indic_tokenize.trivial_tokenize(x, lang='hi)
# lambda x: x.split()
