import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import load_jsonl, linear_fact_str, linearize_webnlg_facts, languages_map, get_language_normalizer, get_text_in_unified_script, get_relation
import random
from indicnlp.tokenize import indic_tokenize
from sacremoses import MosesTokenizer
import torch


class TextDataset(Dataset):
    def __init__(self, tokenizer, filename, dataset_count, corpus_type, script_unification=False):
        self.tokenizer = tokenizer
        self.dataset = load_jsonl(filename)
        self.lang_normalizer = get_language_normalizer()
        self.en_tok = MosesTokenizer(lang="en")
        self.script_unification = script_unification
        self.corpus_type = corpus_type
        if dataset_count>0:
            # retain selected dataset count
            self.dataset = self.dataset[:dataset_count]
    
    def process_facts(self, facts):
        """ linearizes the facts on the encoder side """
        facts = sorted(facts, key=lambda x: get_relation(x[0]).lower())
        linearized_facts = []
        for i in range(len(facts)):
            linearized_facts += linear_fact_str(facts[i], enable_qualifiers=True)+['|']
        # linearized_facts += linear_fact_str(facts[len(facts)-1], enable_qualifiers=True)
        processed_facts_str = ' '.join(linearized_facts)
        return processed_facts_str

    def process_text(self, text, lang):
        """ normalize and tokenize and then space join the text """
        if lang == 'en':
            return " ".join(self.en_tok.tokenize(self.lang_normalizer[lang].normalize(text.strip()), escape=False)).strip()
        else:
            # return unified script text
            if self.script_unification:
                return get_text_in_unified_script(text, self.lang_normalizer[lang], lang)

            # return original text
            return " ".join(
                indic_tokenize.trivial_tokenize(self.lang_normalizer[lang].normalize(text.strip()), lang)
            ).strip()

    def preprocess(self, text):
        tokenzier_args = {'text': text, 'padding': False, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']
    
    def get_input_str(self, data_instance):
        if self.corpus_type=='webnlg':
            facts = [("%s:%s" % (x[0], x[1]) , x) for x in data_instance['facts']]
            facts = sorted(facts, key=lambda x: x[0])
            facts = [x[1] for x in facts]
            input_str = "{sentence} {sep} {triples}".format(sentence=self.process_text(data_instance['sentence'].lower(), 'en'), sep=self.tokenizer.sep_token,
                        triples=linearize_webnlg_facts(facts))
        else:
            input_str = "{sentence} {sep} {entity} {triples}".format(sentence=self.process_text(data_instance['sentence'], data_instance['language']), sep=self.tokenizer.sep_token,
                                        entity=data_instance['entity_name'].lower().strip(), triples=self.process_facts(data_instance['facts']))
        return input_str

    def __getitem__(self, idx):
        input_str = self.get_input_str(self.dataset[idx])
        input_ids, input_mask = self.preprocess(input_str)
        label = 0
        if(self.dataset[idx]['coverage']=="complete"):
            label = 1
        return input_ids, input_mask, label

    def __len__(self):
        return len(self.dataset)

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_batch(batch, tokenizer):
    batch_src_inputs = []
    batch_src_masks = []
    label = []
    
    max_src_len = max([len(ex[0]) for ex in batch])
    
    for item in batch:
        batch_src_inputs += [pad_seq(item[0], max_src_len, tokenizer.pad_token_id)]
        batch_src_masks += [pad_seq(item[1], max_src_len, 0)]
        label.append(item[2])

    return torch.tensor(batch_src_inputs, dtype=torch.long), torch.tensor(batch_src_masks, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
def get_dataset_loaders(tokenizer, file_name, dataset_count, corpus_type, script_unification, batch_size=8, num_threads=0):
    dataset = TextDataset(tokenizer, file_name, dataset_count, corpus_type, script_unification=script_unification)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads, collate_fn=lambda x : collate_batch(x, tokenizer))
    return input_dataloader