import os
import torch
import numpy as np 
from icecream import ic
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import get_native_text_from_unified_script, languages_map

languages_map_inv = {int(val['id']): key for key, val in languages_map.items()}

from transformers import (
    MT5ForConditionalGeneration,
    MBartForConditionalGeneration,
    AutoConfig,
)

class GenModel(torch.nn.Module):
    def __init__(self,
                learning_rate,
                model_name_or_path,
                config,
                is_mt5,
                eval_beams,
                tgt_max_seq_len,
                tokenizer,
                model_gpus,
                isTest,
                final_checkpoint,
                mt5_checkpoint,
                checkpoint=''
                ):
        super().__init__()

        self.learning_rate = learning_rate
        self.model_name_or_path = model_name_or_path
        self.is_mt5 = is_mt5
        self.eval_beams = eval_beams
        self.tgt_max_seq_len = tgt_max_seq_len
        self.tokenizer = tokenizer
        self.model_gpus = model_gpus
        self.checkpoint = checkpoint
        self.isTest = isTest
        self.final_checkpoint = final_checkpoint
        self.mt5_checkpoint = mt5_checkpoint

        print("Loading Main Model")
        self.config = AutoConfig.from_pretrained(config)
        if self.is_mt5:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
            if self.mt5_checkpoint != 'None' and self.mt5_checkpoint:
                ic(f"training from custom mt5 checkpoint {self.mt5_checkpoint}")
                ckpt_state_dict = torch.load(mt5_checkpoint)["state_dict"]
                ckpt_state_dict = {key[6:]:value for key, value in ckpt_state_dict.items()}
                self.model.load_state_dict(ckpt_state_dict)
            else :
                ic("training from pretrained mt5")
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_name_or_path)

        if self.isTest:
            ic(self.checkpoint)
            a = torch.load(self.checkpoint)
            b = OrderedDict([(k[6:], v) for (k, v) in a.items()])
            self.model.load_state_dict(b)
        else:
            if self.final_checkpoint is not None:
                a = torch.load(self.final_checkpoint)
                b = OrderedDict([(k[13:], v) for (k, v) in a.items()])
                self.model.load_state_dict(b)

        print("Main Model successfully loaded")

    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        return outputs

    def test(self, batch):
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=True,
            max_length=self.tgt_max_seq_len
        )

        # input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if self.is_mt5:
            batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id
        gold_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        return {'pred_text': pred_text, 'gold_text': gold_text}

    def middle(self, batch):
        # generated_ids = self.model.generate(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask'],
        #     use_cache=True,
        #     # num_beams = 1,
        #     do_sample=True,
        #     top_k = 1, 
        #     max_length=self.tgt_max_seq_len,
        #     # return_dict_in_generate=True,
        # )
        outputs = self(batch)
        loss, logits = outputs['loss'], outputs['logits']
        out = F.softmax(logits, dim=-1)
        lang = batch['lang_id']
        lang_codes = [languages_map_inv[l.item()] for l in lang]
        greedy_idx = torch.argmax(out, dim=-1)
        tgt_gre = []
        for g in greedy_idx:
            g_e = torch.arange(len(g))[g.eq(self.tokenizer.eos_token_id).cpu()]
            g_e = g_e[0] if len(g_e)>0  and 0<g_e[0]<self.tgt_max_seq_len else self.tgt_max_seq_len
            our_gen = ' '.join(self.tokenizer.batch_decode([g[:g_e]], skip_special_tokens=True))
            tgt_gre.append(our_gen)
        
        for i, txt in enumerate(tgt_gre):
            lang = lang_codes[i]
            if(lang not in ['en']):
                if lang in ['kn', 'te','ml', 'ta']:
                    tgt_gre[i] = get_native_text_from_unified_script(txt, lang_codes[i], 'ml')
                else:
                    tgt_gre[i] = get_native_text_from_unified_script(txt, lang_codes[i], 'hi')

        # ic(generated_ids.shape, greedy_idx.shape)
        # model_gen = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # ic(model_gen)
        input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        # print(tgt_gre)
        # pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {'main_loss': loss, 'logits': logits, 'input_text': input_text, 'pred_text': tgt_gre, 'ref_text':ref_text}



    # def genOutput(self, batch):
    #     generated_ids = self.model.generate(
    #         input_ids=batch['input_ids'],
    #         attention_mask=batch['attention_mask'],
    #         use_cache=True,
    #         num_beams=self.eval_beams,
    #         max_length=self.tgt_max_seq_len
    #     )

    #     input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    #     if self.is_mt5:
    #         batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id
    #     ref_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    #     pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     return pred_text, ref_text, input_text
