import os
import torch
import numpy as np 
from icecream import ic
from collections import OrderedDict

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

        print("Loading Main Model")
        self.config = AutoConfig.from_pretrained(config)
        if self.is_mt5:
            self.model = torch.nn.DataParallel(MT5ForConditionalGeneration.from_pretrained(self.model_name_or_path), device_ids=self.model_gpus)
        else:
            self.model = torch.nn.DataParallel(MBartForConditionalGeneration.from_pretrained(self.model_name_or_path), device_ids=self.model_gpus)

        if self.isTest:
            ic(self.checkpoint)
            a = torch.load(self.checkpoint)
            b = OrderedDict([(k[6:], v) for (k, v) in a.items()])
            self.model.load_state_dict(b)
        else:
            if self.final_checkpoint is not None:
                a = torch.load(self.final_checkpoint)
                b = OrderedDict([(k[6:], v) for (k, v) in a.items()])
                self.model.load_state_dict(b)

        print("Main Model successfully loaded")

    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        return outputs

    def test(self, batch):
        generated_ids = self.model.module.generate(
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
        generated_ids = self.model.module.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=True,
            max_length=self.tgt_max_seq_len
        )
        outputs = self(batch)
        #print(outputs['loss'])
        loss, logits = outputs['loss'], outputs['logits']
        input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {'main_loss': sum(loss)/len(loss), 'logits': logits, 'input_text': input_text, 'pred_text': pred_text, 'ref_text':ref_text}


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
