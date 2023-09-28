import csv 
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from collections import defaultdict
import evaluate
import glob 
from parent import xparent_f1 
import tqdm 
import numpy as np
import json 
import os 
import regex as re 


meteor = evaluate.load('meteor')
chrf = evaluate.load("chrf")
bleu = BLEU()

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

thresholds = {"as":	0.34,
            "bn":	0.34,
            "en":	0.75, 
            "gu":	0.31,
            "hi":	0.33,
            "kn":	0.33,
            "ml":	0.30,
            "mr":	0.33,
            "or":	0.29,
            "pa":	0.32,
            "ta":   0.33, 
            "te":	0.30
            }


def get_lang_key(word):
    for i in languages_map:
        if languages_map[i]['label'].lower() == word.lower().strip():
            return i
    return -1 

class MetricCalculator:
    def __init__(self, src_file, ref_file, gen_file, test_jsonl, tokenizer_obj, root_file, single=False, exclusion=False, name=None):
        self.tokenizer_obj = tokenizer_obj 
        self.src_file = src_file 
        self.ref_file = ref_file 
        self.gen_file = gen_file 
        self.bigscore = []
        self.name = name
        self.root_file = root_file
        os.makedirs(f'{root_file}/bleu', exist_ok=True)
        self.src_lines = open(src_file).readlines()
        self.ref_lines = open(ref_file).readlines()
        self.gen_lines = open(gen_file).readlines() 

        self.lang_dict = {
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
        
        if(single):          
            print("concat")    
            test_json = open(test_jsonl, 'r').readlines()
            entity_lang_gen = defaultdict(lambda: defaultdict(str))
            entity_lang_ref = defaultdict(lambda: defaultdict(str))
            entity_lang_src = defaultdict(lambda: defaultdict(str))       
            entity_lang_num = defaultdict(lambda: defaultdict(int))       
            for i, line in enumerate(test_json):
                d = json.loads(line)
                l = d['lang']
                if(exclusion):
                    if(len(d['sentence_list']) < exclusion):
                        continue 
                r = " ".join(d['sentence_list'])
                e = d['entity_name']
                g = self.gen_lines[i]
                fl = d['qid']
                entity_lang_num[l][e+str(fl)] = len(d['sentence_list'])
                entity_lang_gen[l][e+str(fl)]+=" ".join(tokenizer_obj.tokenize(g.strip('\n')+" ", lang=l))
                entity_lang_ref[l][e+str(fl)]=" ".join(tokenizer_obj.tokenize(r, lang=l))
                if(entity_lang_src[l][e+str(fl)])=='':
                    entity_lang_src[l][e+str(fl)] = self.src_lines[i].strip()+" "
                else:
                    entity_lang_src[l][e+str(fl)] += '<R>'+("<R>".join(self.src_lines[i].strip().split('<R>')[1:]))

            for lang in entity_lang_gen:
                for entity in entity_lang_gen[lang]:
                    if('src' not in self.lang_dict[lang]):
                        self.lang_dict[lang]['src'] = []
                    if('num' not in self.lang_dict[lang]):
                        self.lang_dict[lang]['num'] = []
                    self.lang_dict[lang]['src'].append(entity_lang_src[lang][entity])
                    self.lang_dict[lang]['ref'].append(entity_lang_ref[lang][entity])
                    self.lang_dict[lang]['pred'].append(entity_lang_gen[lang][entity])
                    self.lang_dict[lang]['num'].append(entity_lang_num[lang][entity])


            print(sum([len(self.lang_dict[lang]['ref']) for lang in self.lang_dict]))

        else:
            test_json = open(test_jsonl, 'r').readlines()
            for i in range(len(self.ref_lines)):
                d_obj = json.loads(test_json[i])
                if(exclusion):
                    if(len(d_obj['sentence_list']) < exclusion):
                        continue
                if('<R>' not in self.src_lines[i]):
                    continue 
                lang = self.src_lines[i].split()[1].lower()
                lang_key = get_lang_key(lang)
                if('src' not in self.lang_dict[lang_key]):
                    self.lang_dict[lang_key]['src'] = []
                if('num' not in self.lang_dict[lang_key]):
                    self.lang_dict[lang_key]['num'] = []
                self.lang_dict[lang_key]['ref'].append(self.ref_lines[i])
                self.lang_dict[lang_key]['pred'].append(self.gen_lines[i])
                self.lang_dict[lang_key]['src'].append(self.src_lines[i])
                self.lang_dict[lang_key]['num'].append(len(d_obj['sentence_list']))
            
    def get_all_scores(self, out_dir):
        meteor_scores = []
        bleu_scores_sent = []
        bleu_scores_corp = [] 
        chrf_scores = []
        rogue1_scores = []
        roguel_scores = []

        for lang in sorted(list(self.lang_dict.keys())):
            print(lang)
            meteor_score = self.compute_meteor(self.lang_dict[lang]['pred'], self.lang_dict[lang]['ref'], lambda x: self.tokenizer_obj.tokenize(x, lang=lang))
            print("meteor", meteor_score)
            bleu_score_corp, bleu_score_sent = self.compute_bleu(self.lang_dict[lang]['pred'], self.lang_dict[lang]['ref'], lambda x: self.tokenizer_obj.tokenize(x, lang=lang), lang)
            print("bleu", bleu_score_corp, bleu_score_sent)
            rouge1_score, rougel_score = self.compute_rouge(self.lang_dict[lang]['pred'], self.lang_dict[lang]['ref'])
            print("rouge", rouge1_score, rougel_score)
            chrf_score = self.compute_chrf(self.lang_dict[lang]['pred'], self.lang_dict[lang]['ref'])
            print("chrf", chrf_score)

            meteor_scores.append(meteor_score['meteor'])
            bleu_scores_sent.append(bleu_score_sent)
            bleu_scores_corp.append(bleu_score_corp)
            chrf_scores.append(chrf_score['score'])
            rogue1_scores.append(rouge1_score)
            roguel_scores.append(rougel_score)

        save_name = 'all_'+self.name if self.name else 'all_scores'
        f = open(f'{out_dir}/{save_name}', 'w')
        f.write('BLEU_SENT, CHRF, METEOR, BLEU_CORP, ROUGE1, ROUGEL\n')
        for meteor, bleu_sent, bleu_corp, chrf, rouge1, rougel in zip(meteor_scores, bleu_scores_sent, bleu_scores_corp, chrf_scores, rogue1_scores, roguel_scores):
            f.write(f'{bleu_sent}, {chrf}, {meteor*100}, {bleu_corp}, {rouge1}, {rougel}\n')

    def get_tokenized(self, ref_lines, gen_lines, tokenizer):
        ref_tokenized = []
        gen_tokenized = []
        for l in gen_lines:
            gen_tokenized.append(tokenizer(l))
        for l in ref_lines:
            ref_tokenized.append(tokenizer(l))
        
        return ref_tokenized, gen_tokenized
    def compute_bleu(self, ref_lines, gen_lines, tokenizer, lang):
        sent_bleus = []
        with open(f'{self.root_file}/bleu/{lang}.txt', 'w') as lang_file:
            for ref_, gen_ in zip(ref_lines, gen_lines):
                score = bleu.corpus_score([gen_], [[ref_]]).score
                lang_file.write(f'{score}\n')
                sent_bleus.append(score)
        corpus_score = bleu.corpus_score(gen_lines, [ref_lines]).score
        mean_sent_score = sum(sent_bleus)/len(sent_bleus)
        self.bigscore += sent_bleus
        return corpus_score, mean_sent_score

    def compute_meteor(self, ref_lines, gen_lines, tokenizer):
        ref, gen = self.get_tokenized(ref_lines, gen_lines, tokenizer)
        ref_new, gen_new = [], [] 
        for l in ref:
            ref_new.append(" ".join(l))
        for l in gen:
            gen_new.append(" ".join(l))
        results = meteor.compute(predictions=gen_new, references=ref_new)
        return results 

    def compute_rouge(self, ref_lines, gen_lines):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
        zeroCount = 0 
        r1_sum = 0
        r2_sum =0
        for i in range(len(ref_lines)):
            scores = scorer.score(ref_lines[i], gen_lines[i])
            if scores['rouge1'].fmeasure == 0 :
                zeroCount += 1 
            r1_sum += scores['rouge1'].fmeasure
            r2_sum += scores['rougeL'].fmeasure
        return r1_sum/len(ref_lines), r2_sum/len(ref_lines)
    
    def compute_chrf(self, ref_lines, gen_lines):
        listolist = []
        for i in range(len(ref_lines)):
            listolist.append([ref_lines[i]])

        results = chrf.compute(predictions=gen_lines, references=listolist)
        return results

class ParentCalculator(MetricCalculator):
    def __init__(self, src_file, ref_file, gen_file, test_jsonl,  tokenizer_obj, root_file, name=None, single=False, exclusion=False, threshold_type='lang'):
        super().__init__(src_file, ref_file, gen_file, test_jsonl, tokenizer_obj, root_file, single, exclusion, name)
        os.makedirs(f'{root_file}/{self.name}', exist_ok=True)
        self.threshold_type = threshold_type
        self.exclusion = exclusion

    def compute_xparent(self, out_dir):
        lang_scores = defaultdict(lambda: {'prec':[], 'recall_ref_4':[], 'recall_ref_3':[], 'recall_ref_2':[], 'recall_ref_1':[], 'recall_src_max':[], 'recall_src_lcs':[], 'num': []})
        for lang in sorted(list(self.lang_dict.keys())):
            print(lang+'\n')
            if(self.threshold_type == 'none'):
                threshold = None 
            if(self.threshold_type == 'lang'):
                threshold = thresholds[lang]
            else:
                threshold = 0.50
            pb = tqdm.tqdm(range(len(self.lang_dict[lang]['ref'])))
        #for lang in ['en']:
            for ref, src, gen, num in zip(self.lang_dict[lang]['ref'], self.lang_dict[lang]['src'], self.lang_dict[lang]['pred'], self.lang_dict[lang]['num']):
                ref = ref.rstrip('\n')
                src = src.rstrip('\n')
                gen = gen.rstrip('\n')
                pb.update(1)
                if(ref == '' or src == '' or gen == ''):
                    lang_scores[lang]['prec'].append(-1)
                    lang_scores[lang]['recall_ref_4'].append(-1)
                    lang_scores[lang]['recall_ref_3'].append(-1)
                    lang_scores[lang]['recall_ref_2'].append(-1)
                    lang_scores[lang]['recall_ref_1'].append(-1)
                    lang_scores[lang]['recall_src_max'].append(-1)
                    lang_scores[lang]['num'].append(-1)
                    continue
                try:
                    precision, recall_ref_4, recall_ref_3, recall_ref_2, recall_ref_1, recall_src_max = xparent_f1(src, ref, gen, lambda x: self.tokenizer_obj.tokenize(x, lang=lang), threshold=threshold)
                    lang_scores[lang]['prec'].append(precision)
                    lang_scores[lang]['recall_ref_4'].append(recall_ref_4)
                    lang_scores[lang]['recall_ref_3'].append(recall_ref_3)
                    lang_scores[lang]['recall_ref_2'].append(recall_ref_2)
                    lang_scores[lang]['recall_ref_1'].append(recall_ref_1)
                    lang_scores[lang]['recall_src_max'].append(recall_src_max)
                    lang_scores[lang]['num'].append(num)
                except Exception as e:
                    lang_scores[lang]['prec'].append(-1)
                    lang_scores[lang]['recall_ref_4'].append(-1)
                    lang_scores[lang]['recall_ref_3'].append(-1)
                    lang_scores[lang]['recall_ref_2'].append(-1)
                    lang_scores[lang]['recall_ref_1'].append(-1)
                    lang_scores[lang]['recall_src_max'].append(-1)
                    lang_scores[lang]['num'].append(-1)
                    print(e)
                    print("failed", ref, src, gen, lang)

            with open(f'{out_dir}/{self.name}/{lang}-parent.csv', 'w') as parent_file:
                for i in range(len(lang_scores[lang]['prec'])):
                    parent_file.write(f"{lang_scores[lang]['prec'][i]}, {lang_scores[lang]['recall_ref_4'][i]}, {lang_scores[lang]['recall_ref_3'][i]}, {lang_scores[lang]['recall_ref_2'][i]}, {lang_scores[lang]['recall_ref_1'][i]}, {lang_scores[lang]['recall_src_max'][i]}, {lang_scores[lang]['num'][i]}\n")

    def combine_xparent(self, out_dir, trade_off=0.5):
        lang_files = sorted(glob.glob(f'{out_dir}/{self.name}/*.csv'))
        out_precs = []
        out_recalls_ref_4 = []
        out_recalls_ref_3 = []
        out_recalls_ref_2 = []
        out_recalls_ref_1 = []
        out_recalls_max = []
        out_scores_max_4 = []
        out_scores_max_3 = []
        out_scores_max_2 = []
        out_scores_max_1 = []
        for lang in lang_files:
            print(lang)
            parent_maxs_4 = []
            parent_maxs_3 = []
            parent_maxs_2 = []
            parent_maxs_1 = []
            precisions = []
            recalls_ref_4 = []
            recalls_ref_3 = []
            recalls_ref_2 = []
            recalls_ref_1 = []
            recalls_max = [] 
            scores = open(lang, 'r').readlines()
            for score in scores:
                if not (score.strip()):
                    continue
                prec, recall_ref_4, recall_ref_3, recall_ref_2, recall_ref_1, recall_src_max, num = list(map(lambda x: float(x), score.strip().split(',')))
                if(self.exclusion):
                    if(num < self.exclusion):
                        continue 
                if -1 in [prec, recall_ref_4, recall_ref_3, recall_ref_2, recall_ref_1, recall_src_max]:
                    continue
                recall_ref_final_4 = np.power(recall_ref_4, trade_off)
                recall_ref_final_3 = np.power(recall_ref_3, trade_off)
                recall_ref_final_2 = np.power(recall_ref_2, trade_off)
                recall_ref_final_1 = np.power(recall_ref_1, trade_off)
                recall_src_max_final = np.power(recall_src_max, 1-trade_off)
                recall_final_max_4 = recall_ref_final_4*recall_src_max_final
                recall_final_max_3 = recall_ref_final_3*recall_src_max_final
                recall_final_max_2 = recall_ref_final_2*recall_src_max_final
                recall_final_max_1 = recall_ref_final_1*recall_src_max_final

                parent_max_4 = (2*prec*recall_final_max_4)/(prec+recall_final_max_4)
                parent_max_3 = (2*prec*recall_final_max_3)/(prec+recall_final_max_3)
                parent_max_2 = (2*prec*recall_final_max_2)/(prec+recall_final_max_2)
                parent_max_1 = (2*prec*recall_final_max_1)/(prec+recall_final_max_1)
                
                precisions.append(prec)
                recalls_ref_4.append(recall_ref_final_4)
                recalls_ref_3.append(recall_ref_final_3)
                recalls_ref_2.append(recall_ref_final_2)
                recalls_ref_1.append(recall_ref_final_1)
                recalls_max.append(recall_src_max_final)
                parent_maxs_4.append(parent_max_4)
                parent_maxs_3.append(parent_max_3)
                parent_maxs_2.append(parent_max_2)
                parent_maxs_1.append(parent_max_1)

            out_precs.append(np.mean(precisions))
            out_recalls_ref_4.append(np.mean(recalls_ref_4))
            out_recalls_ref_3.append(np.mean(recalls_ref_3))
            out_recalls_ref_2.append(np.mean(recalls_ref_2))
            out_recalls_ref_1.append(np.mean(recalls_ref_1))
            out_recalls_max.append(np.mean(recalls_max))

            out_scores_max_4.append(np.mean(parent_maxs_4))
            out_scores_max_3.append(np.mean(parent_maxs_3))
            out_scores_max_2.append(np.mean(parent_maxs_2))
            out_scores_max_1.append(np.mean(parent_maxs_1))

        f = open(f'{out_dir}/{self.name}/parent', 'w')
        f.write('parent_2, precisions, recall_src, recall_ref_2, parent_3, parent_2, parent_1, recall_ref_4, recall_ref_3, recall_ref_1, \n')
        for score0, score1, score2, score3, score4, score5, score6, score7, score8, score9 in zip(out_scores_max_2, out_precs, out_recalls_max, out_recalls_ref_2, out_scores_max_3, out_scores_max_4, out_scores_max_1, out_recalls_ref_4, out_recalls_ref_3, out_recalls_ref_1):
            f.write(f'{score0*100}, {score1*100}, {score2*100}, {score3*100}, {score4*100}, {score5*100}, {score6*100}, {score7*100}, {score8*100}, {score9*100}\n')
