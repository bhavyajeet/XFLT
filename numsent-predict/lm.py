import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from transformers import BertModel, BertTokenizerFast as BertTokenizer
import sys 
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
import torch 
from transformers import AutoTokenizer,AutoModel
from torch import cuda
from dataloader import FactDataModule
import pandas as pd
import numpy as np
import torchmetrics

#BERT_MODEL_NAME = 'bert-base-case' 
BERT_MODEL_NAME = sys.argv[1]
LEARNING_RATE = float(sys.argv[3])
MAX_LEN = 512
TRAIN_BATCH_SIZE = int(sys.argv[2])
VALID_BATCH_SIZE = int(sys.argv[2])
DROP_OUT = float(sys.argv[4])
EPOCHS =  float(sys.argv[5])

wandb_logger = WandbLogger(name='Adam-32-0.001',project='pytorchlightning')


## nn modules interaction
class BERTclassifierLightning(pl.LightningModule):
    def __init__(self,model,n_classes : int):
        super().__init__()
        #self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME,return_dict=True)
        self.bert = model
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(DROP_OUT)
        self.classifier = nn.Linear(self.bert.config.hidden_size,n_classes)
        #self.save_hyperparameters()
        
        #self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()


        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self,input_ids,attention_mask,labels=None):
        output_1 = self.bert(input_ids,attention_mask)
        hidden_state1 = output_1[0]

        pooler = hidden_state1[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        #print(input_ids)
        # output = self.bert(input_ids,attention_mask=attention_mask)
        # #print(output.shape)
        # output = self.classifier(output.pooler_output)
        # output = torch.sigmoid(output)
        #print(output)
        #print(labels)

        loss = self.criterion(output, labels)
        #print(loss)
        return loss,output 
        #return None 

    def training_step(self, batch, batch_idx):
        input_ids = batch["fact_ids"]
        attention_mask = batch['fact_mask']
        labels = batch['targets']
        #print(labels)  tensor object
        #print(input_ids.shape)
        loss,outputs = self(input_ids,attention_mask,labels)
        self.accuracy(outputs,labels)

        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.log("train_acc_step",self.accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        input_ids = batch["fact_ids"]
        attention_mask = batch['fact_mask']
        labels = batch['targets']
        loss, outputs = self(input_ids, attention_mask, labels)

        self.log("val_loss",loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.log("val_acc_step",self.accuracy,on_step=True,on_epoch=True,prog_bar=True,logger=True)
       
        return loss 

    # def training_epoch_end(self,outputs):
    #     # calculate accuracy

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer


if __name__ =="__main__":
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    model = AutoModel.from_pretrained(sys.argv[1])


    num_added_toks = tokenizer.add_tokens(['<r>'], special_tokens=True) 
    model.resize_token_embeddings(len(tokenizer))
    
    print(model.config.vocab_size)
    print(len(tokenizer))


    #device = 'cuda' if cuda.is_available() else 'cpu'

    
    tokenizer = tokenizer

    output_file_name = str(sys.argv[5]) + "_" + str(sys.argv[1]) + "_" + str(sys.argv[2]) + "_" + str(sys.argv[3]) + "_" + str(sys.argv[4]) + ".txt"
    file = open(output_file_name,'w')


    # DATA module 
    train_df = pd.read_csv('train_sample.csv', sep=',', names=['FACTS','label'])
    test_df =pd.read_csv('val_sample.csv', sep=',', names=['FACTS','label'])

    data_module = FactDataModule(
        train_df,
        test_df,
        tokenizer,
        max_token_len = MAX_LEN,
        train_batch_size= TRAIN_BATCH_SIZE,
        test_batch_size = VALID_BATCH_SIZE
    )

    # model

    
    n_classes = len(np.unique(train_df['label']))
    print(np.unique(train_df['label']))
    print(n_classes)

    classifier_model = BERTclassifierLightning(model,n_classes=n_classes)


    # train model
    #trainer = pl.Trainer(gpus=1)
    trainer = pl.Trainer(accelerator="cpu",logger= wandb_logger)

    trainer.fit(model=classifier_model,train_dataloaders=None,val_dataloaders=None,datamodule=data_module) # early_stop_callback=True

    # From checkpoint, load model
    #model = BERTclassifierLightning.load_from_checkpoint("/path/to/checkpoint.ckpt")
    
    # test/predict model 
    # test the model
    #trainer.test(model, dataloaders=DataLoader(test_set))



