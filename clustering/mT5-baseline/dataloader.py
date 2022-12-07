import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import load_jsonl, linear_fact_str, languages_map, get_language_normalizer, get_text_in_unified_script, get_relation
import random
from indicnlp.tokenize import indic_tokenize
from sacremoses import MosesTokenizer
import torch
en_tok = MosesTokenizer(lang="en")


class TextDataset(Dataset):
    def __init__(self, prefix, tokenizer, filename, dataset_count, src_max_seq_len, tgt_max_seq_len, script_unification, logger, complete_coverage, sorted_order=True, add_label = 1):
        self.tokenizer = tokenizer
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len
        self.dataset = load_jsonl(filename)
        if complete_coverage:
            self.dataset = [x for x in self.dataset if x['complete_coverage']==1]
        self.logger = logger
        self.prefix = prefix
        self.sorted_order = sorted_order
        self.script_unification = script_unification
        self.add_label = add_label
        self.lang_normalizer = get_language_normalizer()
        if dataset_count>0:
            # retain selected dataset count
            self.dataset = self.dataset[:dataset_count]
        data_type = os.path.basename(filename).split('.')[0]
        if self.script_unification:
            logger.critical("%s : script unification to Devanagari is enabled." % data_type)
        logger.info("%s dataset count : %d" % (data_type, len(self.dataset)))
    
    def process_facts(self, facts):
        """ linearizes the facts on the encoder side """
        if self.sorted_order:
            facts = sorted(facts, key=lambda x: get_relation(x[0]).lower())
        linearized_facts = []
        for i in range(len(facts)):
            linearized_facts += linear_fact_str(facts[i], enable_qualifiers=True)
        processed_facts_str = ' '.join(linearized_facts)
        return processed_facts_str

    def process_text(self, text, lang):
        """ normalize and tokenize and then space join the text """
        if lang == 'en':
            return " ".join(en_tok.tokenize(self.lang_normalizer[lang].normalize(text.strip()), escape=False)).strip()
        else:
            # return unified script text
            if self.script_unification:
                return get_text_in_unified_script(text, self.lang_normalizer[lang], lang)
            
            # return original text
            return " ".join(
                indic_tokenize.trivial_tokenize(self.lang_normalizer[lang].normalize(text.strip()), lang)
            ).strip()

    def preprocess(self, text, max_seq_len):
        tokenzier_args = {'text': text, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': max_seq_len, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']

    def getlbstr(self,score,lang):
        langcutoff = {
                'as':[0.5437,0.5766],
                'bn':[0.5708,0.6166],
                'en':[0.5993,0.6205],
                'gu':[0.5441,0.5793],
                'hi':[0.5662,0.6192],
                'kn':[0.554,0.5903],
                'ml':[0.5644,0.6062],
                'mr':[0.5767,0.629],
                'or':[0.5602,0.606],
                'pa':[0.5355,0.5771],
                'ta':[0.5606,0.6038],
                'te':[0.5575,0.5912]
                }

        retstr = ""
        comp = langcutoff[lang]

        if score < comp[0]:
            retstr = ' low'
        elif score >= comp[0] and score < comp[1]:
            restr = ' medium'
        else :
            retstr = ' high'

        return retstr


    def __getitem__(self, idx):
        prefix_str = ''
        data_instance = self.dataset[idx]
        lang_iso = data_instance['lang'].strip().lower()
        lang_id = languages_map[lang_iso]['id']
        if self.prefix:
            prefix_str = "generate  %s" % languages_map[lang_iso]['label'].lower()
            if self.add_label:
                if 'coverage_score' in data_instance:
                    labelstr = self.getlbstr(float(data_instance['coverage_score']),lang_iso)
                elif 'avg_coverage' in data_instance: 
                    labelstr = self.getlbstr(float(data_instance['avg_coverage']),lang_iso)
                else :
                    labelstr = self.getlbstr(0.99,lang_iso)

                prefix_str += labelstr
            prefix_str += ' : '
        # preparing the input
        #section_info = data_instance['native_sentence_section'] if lang_iso=='en' else data_instance['translated_sentence_section'] 
        input_str = "{prefix}<H> {entity} {triples} ".format(prefix=prefix_str, 
                                        entity=data_instance['entity_name'].lower().strip(), triples=self.process_facts(data_instance['facts']))

        src_ids, src_mask = self.preprocess(input_str, self.src_max_seq_len)
        outputstr = self.process_out_facts(data_instance['facts_list'])
        tgt_ids, tgt_mask = self.preprocess(outputstr, self.tgt_max_seq_len)
        return src_ids, src_mask, tgt_ids, tgt_mask, lang_id, idx

    def __len__(self):
        return len(self.dataset)

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_batch(batch, tokenizer):
    batch_src_inputs = []
    batch_src_masks = []
    batch_tgt_inputs = []
    batch_tgt_masks = []
    lang_id = []
    idx = []

    max_src_len = max([len(ex[0]) for ex in batch])
    max_tgt_len = max([len(ex[2]) for ex in batch])
    
    for item in batch:
        batch_src_inputs += [pad_seq(item[0], max_src_len, tokenizer.pad_token_id)]
        batch_src_masks += [pad_seq(item[1], max_src_len, 0)]
        batch_tgt_inputs += [pad_seq(item[2], max_tgt_len, tokenizer.pad_token_id)]
        batch_tgt_masks += [pad_seq(item[3], max_tgt_len, 0)]
        lang_id.append(item[4])
        idx.append(item[5])

    return torch.tensor(batch_src_inputs, dtype=torch.long), torch.tensor(batch_src_masks, dtype=torch.long), torch.tensor(batch_tgt_inputs, dtype=torch.long), torch.tensor(batch_tgt_masks, dtype=torch.long), torch.tensor(lang_id, dtype=torch.long), torch.tensor(idx, dtype=torch.long)

def get_dataset_loaders(tokenizer, filename, logger, prefix=True, dataset_count=0, batch_size=8, num_threads=1, src_max_seq_len=200, tgt_max_seq_len=200, script_unification=False, complete_coverage=False, add_label = 1):
    dataset = TextDataset(prefix, tokenizer, filename, dataset_count, src_max_seq_len, tgt_max_seq_len, script_unification, logger, complete_coverage, add_label)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads, collate_fn=lambda x : collate_batch(x, tokenizer))
    return input_dataloader
