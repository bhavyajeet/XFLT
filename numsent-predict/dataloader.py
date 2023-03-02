import pytorch_lightning as pl
import pandas as pd 
from torch.utils.data import Dataset,DataLoader
from dataset import FactDataset

class FactDataModule(pl.LightningDataModule):
    def __init__(self,train_df,test_df,tokenizer,max_token_len,train_batch_size,test_batch_size):
        super().__init__()
        #self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer=tokenizer
        self.max_token_len = max_token_len

    def setup(self,stage=None):

        self.train_dataset = FactDataset(self.train_df,self.tokenizer,self.max_token_len)
        print(self.train_dataset)
        self.test_dataset = FactDataset(self.test_df,self.tokenizer,self.max_token_len)

        # add split if required
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.test_batch_size)
        
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,batch_size=self.test_batch_size)
        



