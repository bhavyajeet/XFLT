import pandas as pd
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import sys
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel


LMTokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
LMModel = AutoModel.from_pretrained(sys.argv[1])

num_added_toks = LMTokenizer.add_tokens(['<r>'], special_tokens=True) 
LMModel.resize_token_embeddings(len(LMTokenizer))

device = 'cuda' if cuda.is_available() else 'cpu'

train_dataset = pd.read_csv('./train_all.csv', sep=',', names=['FACTS','label'])
testing_dataset = pd.read_csv('./val_all.csv', sep=',', names=['FACTS','label'])

MAX_LEN = 512
TRAIN_BATCH_SIZE = int(sys.argv[2])
VALID_BATCH_SIZE = int(sys.argv[2])
LEARNING_RATE = float(sys.argv[3])
drop_out = float(sys.argv[4])
EPOCHS = 12
tokenizer = LMTokenizer

output_file_name = str(sys.argv[5]) + "_" + str(sys.argv[1]) + "_" + str(sys.argv[2]) + "_" + str(sys.argv[3]) + "_" + str(sys.argv[4]) + ".txt"
file = open(output_file_name,'w')

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        FACTS = str(self.data.FACTS[index])
        #FACTS = " ".join(FACTS.split())
        inputs = self.tokenizer.encode_plus(
            FACTS,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        fact_ids = inputs['input_ids']
        fact_mask = inputs['attention_mask']

        """
        CDT = str(self.data.CDT[index])
        CDT = " ".join(CDT.split())
        inputs = self.tokenizer.encode_plus(
            CDT,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        CDT_ids = inputs['input_ids']
        CDT_mask = inputs['attention_mask']


        CC = str(self.data.CC[index])
        CC = " ".join(CC.split())
        inputs = self.tokenizer.encode_plus(
            CC,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        CC_ids = inputs['input_ids']
        CC_mask = inputs['attention_mask']
        """
    
        return {
            'fact_ids': torch.tensor(fact_ids, dtype=torch.long),
            'fact_mask': torch.tensor(fact_mask, dtype=torch.long),

            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class LMClass(torch.nn.Module):
    def __init__(self):
        super(LMClass, self).__init__()
        self.l1 = LMModel
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(768, 210)

    def forward(self, data):
        
        input_ids = data['fact_ids'].to(device, dtype = torch.long)
        attention_mask = data['fact_mask'].to(device, dtype = torch.long)
        
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state1 = output_1[0]

        pooler = hidden_state1[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

from torch.optim import lr_scheduler

model = LMClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    print ('epoch ',epoch)
    for _,data in enumerate(tqdm(training_loader), 0):
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(data)
        # print(outputs.shape)
        # print(targets.shape)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    file.write(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Training Loss Epoch: {epoch_loss}\n")
    file.write(f"Training Accuracy Epoch: {epoch_accu}\n")
    print(f"Training Accuracy Epoch: {epoch_accu}\n")
    file.write("\n")
    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; tr_loss = 0
    nb_tr_steps =0
    nb_tr_examples =0
    pred = []
    act = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(data).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            pred += big_idx.tolist()
            act += targets.tolist()
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Validation Loss Epoch: {epoch_loss}\n")
    file.write(f"Validation Accuracy Epoch: {epoch_accu}\n")
    print(f"Validation Accuracy Epoch: {epoch_accu}\n")
    mf1 = f1_score(act, pred, average='macro')
    file.write(f"Validation Macro F1: {mf1}\n")
    print(f"Validation Macro F1: {mf1}\n")
    return mf1,epoch_accu

best_mf1 = 0
best_epoch = 0
best_acc = 0

for epoch in range(EPOCHS):
    train(epoch)
    scheduler.step()
    mf1,acc = valid(model, testing_loader)
    if acc > best_acc:
        best_mf1 = mf1
        best_acc = acc
        best_epoch = epoch+1
        PATH = 'best_predict.ckpt'
        torch.save(model.state_dict(), PATH)
    if (epoch+1)%3 == 0 :
        PATH = str(epoch+1)+'_predict.ckpt'
        torch.save(model.state_dict(), PATH)


    file.write("\n")

file.write("Best \nAccuracy: {0} \nMacro F1 Score: {1}\nAt Epoch: {2}\n".format(best_acc,best_mf1,best_epoch))
file.close()
