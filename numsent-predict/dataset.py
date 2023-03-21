import pytorch_lightning as pl
import pandas as pd 
from torch.utils.data import Dataset
import torch

class FactDataset(Dataset):
    def __init__(self,dataframe,tokenizer,max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len 
    
    def __len__(self):
        return self.len 

    def __getitem__(self,index):
        facts = str(self.data.FACTS[index])
        inputs = self.tokenizer.encode_plus(
            facts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        facts_ids = inputs['input_ids']
        fact_mask = inputs['attention_mask']

        return { 
            'fact_ids': torch.tensor(facts_ids,dtype=torch.long),
            'fact_mask':torch.tensor(fact_mask,dtype=torch.long),
            'targets':torch.tensor(self.data.label[index],dtype=torch.long)
        }

        



