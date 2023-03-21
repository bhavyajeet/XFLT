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


def get_lang_key(word):
    for i in languages_map:
        if languages_map[i]['label'].lower() == word.lower().strip():
            return i
    return -1 

class MetricCalculator:
    def __init__(self, src_file, ref_file, gen_file, test_jsonl, tokenizer_obj, single=False):
        self.tokenizer_obj = tokenizer_obj 
        self.src_file = src_file 
        self.ref_file = ref_file 
        self.gen_file = gen_file 
        
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
            for i, line in enumerate(test_json):
                d = json.loads(line)
                l = d['lang']
                r = " ".join(d['sentence_list'])
                e = d['entity_name']
                g = self.gen_lines[i]
                fl = d['qid']
                #entity_lang_gen[l][e]+=g.strip('\n')+" "
                #entity_lang_ref[l][e]=r
                entity_lang_gen[l][e+str(fl)]+=" ".join(tokenizer_obj.tokenize(g.strip('\n')+" ", lang=l))
                entity_lang_ref[l][e+str(fl)]=" ".join(tokenizer_obj.tokenize(r, lang=l))
                if(entity_lang_src[l][e+str(fl)])=='':
                    entity_lang_src[l][e+str(fl)] = self.src_lines[i]+" "
                else:
                    entity_lang_src[l][e+str(fl)] += '<R>'+self.src_lines[i].split('<R>')[1]
            print(sum([len(entity_lang_gen[l]) for l in entity_lang_ref]))
            for lang in entity_lang_gen:
                for entity in entity_lang_gen[lang]:
                    if('src' not in self.lang_dict[lang]):
                        self.lang_dict[lang]['src'] = []
                    self.lang_dict[lang]['src'].append(entity_lang_src[lang][entity])
                    self.lang_dict[lang]['ref'].append(entity_lang_ref[lang][entity])
                    self.lang_dict[lang]['pred'].append(entity_lang_gen[lang][entity])

            # langdict = {
            #     'as':'assamese',
            #     'bn':'bengali',
            #     'en':'english',
            #     'gu':'gujarati',
            #     'hi':'hindi',
            #     'kn':'kannada',
            #     'ml':'malayalam',
            #     'mr':'marathi',
            #     'or':'odia',
            #     'pa':'punjabi',
            #     'ta':'tamil',
            #     'te':'telugu'
            #     }
            # test_json = open('eval_module/complete_test.jsonl', 'r').readlines()
            # json_dict = {} 

            # for line in test_json:
            #     obj = json.loads(line)
            #     json_dict[f"{obj['entity_name'].lower()}_{obj['lang']}"] = obj 

            # rev_langdict = {value: key for key, value in langdict.items()}
            # merged_dict = defaultdict(lambda: defaultdict(list))
            # for line_src, line_ref, line_gen in zip(self.src_lines, self.ref_lines, self.gen_lines):
            #     lang = rev_langdict[line_src.split(" ")[1].strip()]
            #     entity_name = line_src.split(':')[1].split('<R>')[0][4:].strip()
            #     merged_dict[f'{entity_name}_{lang}']['src'].append(line_src.strip().strip('\n'))
            #     merged_dict[f'{entity_name}_{lang}']['ref'].append(line_ref.strip().strip('\n'))
            #     merged_dict[f'{entity_name}_{lang}']['gen'].append(line_gen.strip().strip('\n'))

            # for key in merged_dict:
            #     combined_src = ""
            #     combined_ref = ""
            #     combined_gen = ""
            #     if key in json_dict:
            #         lang_key = key.split('_')[1]
            #         for sent in json_dict[key]['sentence_list']:
            #             for i, ref_sent in enumerate(merged_dict[key]['ref']):
            #                 if re.sub(" ", "", sent) == re.sub(" ", "", ref_sent):
            #                     combined_ref+=ref_sent+" "
            #                     combined_gen+=merged_dict[key]['gen'][i]+" "
            #                     if(combined_src == ""):
            #                         combined_src+=merged_dict[key]['src'][i]+" "
            #                     else:
            #                         combined_src+= '<R>'+merged_dict[key]['src'][i].split('<R>')[1]
            #                     break
            #         if('src' not in self.lang_dict[lang_key]):
            #             self.lang_dict[lang_key]['src'] = []
            #         self.lang_dict[lang_key]['src'].append(combined_src)
            #         self.lang_dict[lang_key]['ref'].append(combined_ref)
            #         self.lang_dict[lang_key]['pred'].append(combined_gen)
            print(sum([len(self.lang_dict[lang]['ref']) for lang in self.lang_dict]))

        else:
            for i in range(len(self.ref_lines)):
                if('<R>' not in self.src_lines[i]):
                    continue 
                lang = self.src_lines[i].split()[1].lower()
                lang_key = get_lang_key(lang)

                if('src' not in self.lang_dict[lang_key]):
                    self.lang_dict[lang_key]['src'] = []
                self.lang_dict[lang_key]['ref'].append(self.ref_lines[i])
                self.lang_dict[lang_key]['pred'].append(self.gen_lines[i])
                self.lang_dict[lang_key]['src'].append(self.src_lines[i])
            
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
            bleu_score_corp, bleu_score_sent = self.compute_bleu(self.lang_dict[lang]['pred'], self.lang_dict[lang]['ref'], lambda x: self.tokenizer_obj.tokenize(x, lang=lang))
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

        
        f = open(f'{out_dir}/all_scores', 'w')
        for meteor, bleu_sent, bleu_corp, chrf, rouge1, rougel in zip(meteor_scores, bleu_scores_sent, bleu_scores_corp, chrf_scores, rogue1_scores, roguel_scores):
            f.write(f'{str(meteor*100)}, {str(bleu_sent)}, {str(bleu_corp)}, {str(chrf)}, {str(rouge1*100)}, {str(rougel*100)}\n')

    def get_tokenized(self, ref_lines, gen_lines, tokenizer):
        ref_tokenized = []
        gen_tokenized = []
        for l in gen_lines:
            gen_tokenized.append(tokenizer(l))
        for l in ref_lines:
            ref_tokenized.append(tokenizer(l))
        
        return ref_tokenized, gen_tokenized

    # def compute_bleu(self, ref_lines, gen_lines, tokenizer):
    #     ref, gen = self.get_tokenized(ref_lines, gen_lines, tokenizer) 
    #     cc = SmoothingFunction()
    #     sent_bleus = []
    #     for ref_, gen_ in zip(ref, gen):
    #         score = sentence_bleu([ref_], gen_, smoothing_function=cc.method4)
    #         sent_bleus.append(score)
    #     score_bleu = corpus_bleu([[r] for r in ref], gen, smoothing_function=cc.method4)
    #     mean_sent_bleu = sum(sent_bleus)/len(sent_bleus)
    #     return score_bleu, mean_sent_bleu
    
    def compute_bleu(self, ref_lines, gen_lines, tokenizer):
        sent_bleus = []
        for ref_, gen_ in zip(ref_lines, gen_lines):
            score = bleu.corpus_score([gen_], [[ref_]]).score
            sent_bleus.append(score)
        corpus_score = bleu.corpus_score(gen_lines, [ref_lines]).score
        mean_sent_score = sum(sent_bleus)/len(sent_bleus)
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
    def __init__(self, src_file, ref_file, gen_file, test_jsonl,  tokenizer_obj, single=False):
        super().__init__(src_file, ref_file, gen_file, test_jsonl, tokenizer_obj, single)

    def compute_xparent(self, out_dir):
        lang_scores = defaultdict(lambda: {'prec':[], 'recall_ref':[], 'recall_src_max':[], 'recall_src_lcs':[]})
        for lang in sorted(list(self.lang_dict.keys())):
            print(lang+'\n')
            threshold = 0.7 if lang =='en' else 0.35
            pb = tqdm.tqdm(range(len(self.lang_dict[lang]['ref'])))
        #for lang in ['en']:
            for ref, src, gen in zip(self.lang_dict[lang]['ref'], self.lang_dict[lang]['src'], self.lang_dict[lang]['pred']):
                ref = ref.rstrip('\n')
                src = src.rstrip('\n')
                gen = gen.rstrip('\n')
                pb.update(1)
                if(ref == '' or src == '' or gen == ''):
                    continue
                try:
                    precision, recall_ref, recall_src_max = xparent_f1(src, ref, gen, lambda x: self.tokenizer_obj.tokenize(x, lang=lang))
                    lang_scores[lang]['prec'].append(precision)
                    lang_scores[lang]['recall_ref'].append(recall_ref)
                    lang_scores[lang]['recall_src_max'].append(recall_src_max)
                except:
                    continue 
                #lang_scores[lang]['recall_src_lcs'].append(recall_src_lcs)

            with open(f'{out_dir}/{lang}-parent.csv', 'w') as parent_file:
                for i in range(len(lang_scores[lang]['prec'])):
                    parent_file.write(f"{lang_scores[lang]['prec'][i]}, {lang_scores[lang]['recall_ref'][i]}, {lang_scores[lang]['recall_src_max'][i]}\n")#, {lang_scores[lang]['recall_src_lcs'][i]}\n")

    def combine_xparent(self, out_dir, trade_off=0.5):
        lang_files = sorted(glob.glob(f'{out_dir}/*.csv'))
        out_precs = []
        out_recalls_ref = [] 
        out_recalls_max = []
        #out_recalls_lcs = []
        #out_scores_lcs = []
        out_scores_max = []
        for lang in lang_files:
            print(lang)
            parent_lcs = []
            parent_max = []
            precisions = []
            recalls_ref = []
            #recalls_lcs = []
            recalls_max = [] 
            scores = open(lang, 'r').readlines()
            for score in scores:
                if not (score.strip()):
                    continue
                prec, recall_ref, recall_src_max = list(map(lambda x: min(1, float(x)), score[:-1].split(',')))
                if -1 in [prec, recall_ref, recall_src_max]:
                    continue
                recall_ref_final = np.power(recall_ref, trade_off)
                #recall_src_lcs_final = np.power(recall_src_lcs, 1-trade_off)
                recall_src_max_final = np.power(recall_src_max, 1-trade_off)
                recall_final_max = recall_ref_final*recall_src_max_final
                #recall_final_lcs = recall_ref_final*recall_src_lcs_final

                #parent_lcs_ = (2*prec*recall_final_lcs)/(prec+recall_final_lcs)
                parent_max_ = (2*prec*recall_final_max)/(prec+recall_final_max)
                
                precisions.append(prec)
                recalls_ref.append(recall_ref_final)
                #recalls_lcs.append(recall_src_lcs_final)
                recalls_max.append(recall_src_max_final)
                #parent_lcs.append(parent_lcs_)
                parent_max.append(parent_max_)

            out_precs.append(np.mean(precisions))
            out_recalls_ref.append(np.mean(recalls_ref))
            out_recalls_max.append(np.mean(recalls_max))
            #out_recalls_lcs.append(np.mean(recalls_lcs))
            #out_scores_lcs.append(np.mean(parent_lcs))
            out_scores_max.append(np.mean(parent_max))

        f = open(f'{out_dir}/parent-45-thresh-2-recall', 'w')
        for score0, score1, score2, score3 in zip(out_scores_max, out_precs, out_recalls_ref, out_recalls_max):# out_recalls_lcs):
            f.write(f'{str(score0*100)}, {str(score1*100)}, {str(score2*100)}, {str(score3*100)}\n')
