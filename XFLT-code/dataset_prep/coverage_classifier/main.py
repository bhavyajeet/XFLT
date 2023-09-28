import os,sys, re
import random
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report, f1_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import defaultdict
from utils import (load_jsonl, linear_fact_str, 
                languages_map, get_language_normalizer, 
                get_text_in_unified_script, get_relation,
                load_jsonl)

from transformers import (
    AdamW,
    Adafactor,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
import unidecode
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate
from indicnlp.tokenize import indic_tokenize
import urduhack
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from logger import MyLogger, LOG_LEVELS
from dataloader import get_dataset_loaders


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.dirname(os.path.realpath(__file__))

# allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TexClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.config_args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if args.use_pretrained:
            # using pretrained transformers
            self.model = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False, cache_dir='/tmp/huggingface')
        else:
            # training transformer from scratch
            self.model = AutoModel.from_config(AutoModel.from_pretrained(
                args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False))
        self.task_head = TexClassificationHead(self.model.config.hidden_size, 2, args.dropout_rate)
        #metrics
        self.train_metric = pl.metrics.classification.f_beta.FBeta(num_classes=2, average='weighted') #pl.metrics.Accuracy()
        self.val_metric = pl.metrics.classification.f_beta.FBeta(num_classes=2, average='weighted')

    def forward(self, input_ids, attention_mask):
        # loading to cuda devices
        # input_seq = input_seq.to(self.transformer.device)
        # attention_mask = attention_mask.to(self.transformer.device)
        # calculating the output logits
        doc_rep = self.model(input_ids, attention_mask=attention_mask)[0]
        output_logits = self.task_head(doc_rep)
        return output_logits
        
    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config_args.learning_rate)
        # optimizer = Adafactor(optimizer_grouped_parameters, lr=self.config_args.learning_rate, 
        #                                                   scale_parameter=False, relative_step=False, warmup_init=False)
        
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-6)

        if self.config_args.enable_scheduler:
            total_dataset_count = self.config_args.train_dataset_count
            total_steps = int(np.ceil((self.config_args.epochs * total_dataset_count) /
                              (self.config_args.batch_size*self.config_args.gpus)))

            scheduler = {
                # 'scheduler': get_constant_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps)
                'scheduler': get_linear_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps, total_steps),
                'interval': 'step',
            }
            return [optimizer], [scheduler]

        return optimizer

    def _step(self, batch, step_type):
        if step_type == 'train':
            step_metric = self.train_metric
        else:
            step_metric = self.val_metric
        
        input_ids, attention_mask = batch[:-1]
        model_output = self(input_ids, attention_mask)
        
        if step_type!='test':
            label_ids = batch[-1]

        return_map = {}
        online_logger_data = {}
        pbar = {}
        if step_type!='test':
            task_loss = F.cross_entropy(model_output, label_ids.long())
            acc = step_metric(model_output.softmax(dim=-1), label_ids.long())
            if step_type == 'val':
                return_map['val_loss'] = task_loss
                return_map['val_acc'] = acc 
            else:
                return_map['loss'] = task_loss
                pbar['acc'] = acc

            # updating the online logger
            online_logger_data.update(pbar)
            online_logger_data.update(return_map)
            self.logger.log_metrics(online_logger_data)

            if len(pbar):
                return_map['progress_bar'] = pbar
        return return_map

    def _epoch_end(self, step_outputs, end_type):
        if end_type == 'train':
            end_metric = self.train_metric
        else:
            end_metric = self.val_metric
        
        loss_label = 'loss'
        if end_type == 'val':
            loss_label = 'val_loss'

        if end_type!='test':
            avg_loss = torch.stack([x[loss_label] for x in step_outputs]).mean()
            overall_acc = end_metric.compute()
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f, overall_%s_acc : %f' % (self.current_epoch, end_type, avg_loss.item(),
                                                                                                            end_type, overall_acc.item()))
            # logging to weight and bias if online mode is enabled
            self.logger.log_metrics(
                {'avg_%s_loss' % end_type: avg_loss, 'overall_%s_acc' % end_type: overall_acc})
            self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True)
            self.log('overall_%s_acc' % end_type, overall_acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')

class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args

    def val_dataloader(self):
        dev_file_path = os.path.join(os.path.abspath(args.dataset_path), 'val.jsonl')
        val_dataset = get_dataset_loaders(self.tokenizer, dev_file_path, self.args.val_dataset_count, self.args.corpus, 
                                          self.args.enable_script_unification>0, batch_size=self.args.batch_size)
        return val_dataset

    def train_dataloader(self):
        train_file_path = os.path.join(os.path.abspath(args.dataset_path), 'train.jsonl')
        train_dataset = get_dataset_loaders(self.tokenizer, train_file_path, self.args.train_dataset_count, self.args.corpus,
                                          self.args.enable_script_unification>0, batch_size=self.args.batch_size)
        return train_dataset

def start_training(args):
    model_name = args.logger_exp_name

    args.logger.debug('initiating training process...')
    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(final_checkpoint_path, exist_ok=True)

    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_acc',
        'mode': 'max',
    }

    # Load datasets
    dm = TextDataModule(args)

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)

    # early stop callback
    early_stop_callback = EarlyStopping(
        monitor='overall_val_acc',
        patience=args.patience,
        verbose=True,
        mode='max',
    )

    model = ModelWrapper(args)

    if args.load_from_checkpoint:
        args.logger.debug("loading the checkpoint weights from : %s" % os.path.abspath(args.load_from_checkpoint))
        with open(os.path.abspath(args.load_from_checkpoint), 'rb') as tfile:
            checkpoint_weight = torch.load(tfile)
        
        model.load_state_dict(checkpoint_weight['state_dict'])
        args.logger.info('loaded weights successfully !!!')

        del checkpoint_weight

    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' %
                     count_parameters(model))

    callback_list = [checkpoint_callback, early_stop_callback]

    global_callback_params = {
        "callbacks": callback_list,
        "max_epochs": args.epochs,
        "min_epochs": 1,
        "gradient_clip_val": args.clip_grad_norm,
        "gpus": 1 if args.inference else args.gpus,
        "distributed_backend": "ddp",
        "logger": args.online_logger,
    }

    trainer = pl.Trainer(**global_callback_params)
    # finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, dm)
    args.logger.debug('training done.')

def process_facts(facts):
    """ linearizes the facts on the encoder side """
    facts = sorted(facts, key=lambda x: get_relation(x[0]).lower())
    linearized_facts = []
    for i in range(len(facts)):
        linearized_facts += linear_fact_str(facts[i], enable_qualifiers=True)+['|']
    # linearized_facts += linear_fact_str(facts[len(facts)-1], enable_qualifiers=True)
    processed_facts_str = ' '.join(linearized_facts)
    return processed_facts_str

def process_text(script_unification, en_tok, lang_normalizer, text, lang):
    """ normalize and tokenize and then space join the text """
    if lang == 'en':
        return " ".join(en_tok.tokenize(lang_normalizer[lang].normalize(text.strip()), escape=False)).strip()
    else:
        # return unified script text
        if script_unification:
            return get_text_in_unified_script(text, lang_normalizer[lang], lang)

        # return original text
        return " ".join(
            indic_tokenize.trivial_tokenize(lang_normalizer[lang].normalize(text.strip()), lang)
        ).strip()

def get_data_instance(script_unification, en_tok, sep_token, lang_normalizer, data_instance):
    input_str = "{sentence} {sep} {entity} {triples}".format(sentence=process_text(script_unification, en_tok, lang_normalizer, data_instance['sentence'], data_instance['language']), sep=sep_token,
                                        entity=data_instance['entity_name'].lower().strip(), triples=process_facts(data_instance['facts']))
    return input_str

def get_coverage(model, tokenizer, script_unification, seqs_list, eval_batch_size):
    lang_normalizer = get_language_normalizer()
    en_tok = MosesTokenizer(lang="en")
    with torch.no_grad():
        enc = tokenizer.batch_encode_plus(
                [get_data_instance(script_unification, en_tok, tokenizer.sep_token, lang_normalizer, x) for x in seqs_list], padding='longest', return_attention_mask=True, return_tensors='pt')
        dataset = TensorDataset(enc['input_ids'], enc['attention_mask'])
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=eval_batch_size)
        temp = []
        for batch in dataloader:
            temp_out = model(batch[0].to(
                device), batch[1].to(device))
            temp_out = temp_out.softmax(dim=-1).argmax(dim=-1).detach()
            temp.extend(temp_out.cpu().tolist())
    return temp

def evaluate_model(args):
    args.logger.debug("using device : %s" % device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = ModelWrapper(args)

    args.logger.debug("loading the checkpoint weights from : %s" % os.path.abspath(args.load_from_checkpoint))
    with open(os.path.abspath(args.load_from_checkpoint), 'rb') as tfile:
        checkpoint_weight = torch.load(tfile)
    
    model.load_state_dict(checkpoint_weight['state_dict'])
    args.logger.info('loaded weights successfully !!!')

    del checkpoint_weight

    model = model.to(device)
    model.eval()

    test = load_jsonl(os.path.join(os.path.abspath(args.dataset_path), 'test.jsonl'))

    actual_coverage = [1 if x['coverage']=='complete' else 0 for x in test]
    predicted_coverage = get_coverage(model, tokenizer, args.enable_script_unification > 0, test, args.eval_batch_size)

    args.logger.info('calculating accuracy and confusion matrix')
    args.logger.info("acc: %0.4f, f1-weighted: %0.2f" % (accuracy_score(actual_coverage, predicted_coverage), f1_score(actual_coverage, predicted_coverage, average='weighted')))
    args.logger.info(classification_report(actual_coverage, predicted_coverage))

    args.logger.info('working on language-wise confusion matrix')
    for lang in languages_map:
        args.logger.info('--'*30)
        args.logger.info('working on : %s' % lang)
        temp = [x for x in test if x['language']==lang]
        temp_actual_coverage = [1 if x['coverage']=='complete' else 0 for x in temp]
        temp_predicted_coverage = get_coverage(model, tokenizer, args.enable_script_unification > 0, temp, args.eval_batch_size)
        args.logger.info(classification_report(temp_actual_coverage, temp_predicted_coverage))

if __name__ == "__main__":
    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')

    # Global model configuration
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--dataset_path', required=True, type=str,
                        help='directory where dataset exits')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--eval_batch_size', default=16, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float,
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.001, type=float,
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    parser.add_argument('--patience', default=3, type=int,
                        help='specify patience for early stop algorithm. if its 0 then disable this feature.')
    parser.add_argument('--corpus', default='cross-lingual', choices=['webnlg', 'cross-lingual'],
                        help='specify which corpus to use in order to change the dataloader.')
    # parser.add_argument('--seed', default=42, type=int,
    # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=0.01, type=float,
                        help="percentage of total step used as linear warmup while training the model.")
    # below three arguments are for debugging purpose
    parser.add_argument("--train_dataset_count", type=int, default=0,
                        help="specify number of training data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--val_dataset_count", type=int, default=0,
                        help="specify number of validation data to use. (for debugging purpose). If zero then takes all the available dataset.")

    # logger configs
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    
    # architecture configs
    parser.add_argument('--model_name', type=str, default='google/muril-base-cased',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--use_pretrained', type=int, default=1,
                        help='loads pretrained transformer model.')
    
    # script unification
    parser.add_argument('--enable_script_unification', type=int, default=0,
                        help="specify value greater than 0 to enable script unification to Devanagri for Indic languages.")

    # GPU memory utilization optimizations
    parser.add_argument('--fp16', type=int, default=1,
                        help='enable the automatic mixed precision training')

    # loading from checkpoint
    parser.add_argument('--load_from_checkpoint', type=str,
                        help='specify the checkpoint path to load the model parameters')

    # inference
    parser.add_argument('--inference', action='store_true',
                        help="enables inference on stored checkpoint")
    args = parser.parse_args()

    args.logger_exp_name = "%s-%s-%s-%s" % (args.model_name, args.epochs, args.learning_rate, args.corpus)
    args.logger_exp_name = args.logger_exp_name.replace('/', '-')

    if args.enable_script_unification > 0:
        args.logger_exp_name = "%s-unified-script" % args.logger_exp_name

    if args.inference:
        args.logger_exp_name = "inference-%s" % args.logger_exp_name

    # offline logger
    args.logger = MyLogger('', os.path.join(base_dir, "%s.log" % args.logger_exp_name),
                           use_stdout=True, log_level=LOG_LEVELS.DEBUG, overwrite=True)


    # get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)

    logger_args = {
        'project': 'coverage-classifier',    # first create a project on weight & bias with local account
        'name': args.logger_exp_name,
        'config': params,
        'tags': ['pytorch-lightning'],
    }

    # turn off the online sync
    if args.online_mode == 0:
        logger_args.update({'offline': True}),

    # configure and add logger to arguments
    args.online_logger = WandbLogger(**logger_args)

    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)

    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)
    
    if args.inference:
        evaluate_model(args)
    else:
        start_training(args)
