import json

from icecream import ic
from torch.utils.data import Dataset, DataLoader
#from langdetect import detect, DetectorFactory

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)


class ModelDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_length, max_target_length, is_mt5):
        f = open(path, 'r')
        self.df = [json.loads(line, strict=False) for line in f.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.lang_map = {
        'bn' : 'bn_IN',
        'de' : 'de_DE',
        'en' : 'en_XX',
        'es' : 'es_XX',
        'fr' : 'fr_XX',
        'gu' : 'gu_IN',
        'hi' : 'hi_IN',
        'it' : 'it_IT',
        'kn' : 'kn_IN',
        'ml' : 'ml_IN',
        'mr' : 'mr_IN',
        'or' : 'or_IN',
        'pa' : 'pa_IN',
        'ta' : 'ta_IN',
        'te' : 'te_IN'
        }

        #DetectorFactory.seed = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        content = self.df[index]['references']
        content = ' '.join(content)
        target = self.df[index]['section_title']
        arttitle = self.df[index]['page_title']
        domain = self.df[index]['domain']
        tlang = self.df[index]['tgt_lang']
        # tlang = 'hi_IN'
        # try:
        #     clang = detect(content)
        #     clang = self.lang_map[clang]
        # except:
        #     clang = 'en_XX'

        # try:
        #     tlang = self.lang_map[tlang]
        # except:
        #     tlang = 'en_XX'
        clang = 'en'
        tlang = 'en'

        cencoding = self.tokenizer(f'{clang} {content} </s>', return_tensors='pt', max_length=self.max_source_length, padding='max_length', truncation=True)
        tencoding = self.tokenizer(f'{tlang} {target} </s>', return_tensors='pt', max_length=self.max_target_length, padding='max_length', truncation=True)

        input_ids, attention_mask = cencoding['input_ids'], cencoding['attention_mask']
        labels = tencoding['input_ids']

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'clang': clang, 'tlang': tlang, 'arttitle': arttitle, 'domain': domain}

# tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
# dataset = ModelDataset('/scratch/aditya.hari/data/val_extractive.json', tokenizer, 200, 200, 1)
# loader = DataLoader(
#     dataset,
#     batch_size=2,
#     num_workers=6,
#     shuffle=True
# )

# item = iter(loader).next()
# print(item.keys())
# print(item['input_ids'].shape)
