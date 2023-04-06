import os
import wandb
import torch
import argparse
import sys
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from tqdm import tqdm
from icecream import ic
from torch.utils.data import DataLoader
#from model.logger import * 
from model.model_ddp import GenModel
from model.dataloader_ddp import get_dataset_loaders
from model.rewards import nerReward, sectitleReward
from model.utils import _intiate_dataset_merging

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)

SEED = 42

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


all_langs = ['as', 'bn', 'en', 'hi', 'gu', 'or', 'te', 'ta', 'kn', 'ml', 'pa', 'mr']
# def calcTestReward(
#         batch,
#         logits,
#         foreign,
#         input_text,
#         pred_text
# ):
#     nerlosses = []
#     seclosses = []

#     for idx in range(len(batch['input_ids'])):
#         input_lang = True if batch['clang'][idx] in foreign else False
#         pred_lang = True if batch['tlang'][idx] in foreign else False

#         # sec_reward = sectitleReward(input_text[idx], pred_text[idx], titletok, titlemodel, titledevice)
#         # ner_reward = nerReward(pred_text[idx], input_text[idx], nertok, nermodel, nerdevice, pred_lang, input_lang, nerf_pipeline)

#         sec_reward = 0.5
#         ner_reward = 0.5

#         probs = torch.nn.functional.softmax(logits, dim=-1)
#         argmax = torch.amax(probs, dim=2)
#         bestaction = torch.log(argmax)
#         nerloss = (-bestaction*ner_reward).mean()
#         secloss = (-bestaction*sec_reward).mean()
#         nerlosses.append(nerloss)
#         seclosses.append(secloss)

#     return {'nerloss': sum(nerlosses)/len(nerlosses), 'secloss': sum(seclosses)/len(seclosses)}

def add_special_tokens(tokenizer):
        new_tokens = ['<H>', '<R>', '<T>', '<QR>', '<QT>', '<S>', '<BR>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
        return tokenizer 

def calcReward(batch, logits, foreign, input_text, pred_text, titletok, titlemodel, titledevice, nertok, nermodel, nerdevice, nerf_pipeline):
    nerlosses = []
    seclosses = []

    for idx in range(len(batch['input_ids'])):
        input_lang = True if batch['clang'][idx] in foreign else False
        pred_lang = True if batch['tlang'][idx] in foreign else False

        sec_reward = sectitleReward(input_text[idx], pred_text[idx], titletok, titlemodel, titledevice)
        ner_reward = nerReward(pred_text[idx], input_text[idx], nertok, nermodel, nerdevice, pred_lang, input_lang, nerf_pipeline)

        # sec_reward = 0.5
        # ner_reward = 0.5

        probs = torch.nn.functional.softmax(logits, dim=-1)
        argmax = torch.amax(probs, dim=2)
        bestaction = torch.log(argmax)
        nerloss = (-bestaction*ner_reward).mean()
        secloss = (-bestaction*sec_reward).mean()
        nerlosses.append(nerloss)
        seclosses.append(secloss)

    return {'nerloss': sum(nerlosses)/len(nerlosses), 'secloss': sum(seclosses)/len(seclosses)}

def main():
    dist.init_process_group("nccl")
    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    #parser.add_argument('--train_path', help='path to input json file for a given domain in given language')
    #parser.add_argument('--val_path', help='path to intermediate output json file for a given domain in given language')
    #parser.add_argument('--test_path', help='path to output json file for a given domain in given language')
    #parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    parser.add_argument('--dataset_dir', help='dataset directory')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--tokenizer', default='google/mt5-small', help='which tokenizer to use')
    parser.add_argument('--model', default='google/mt5-small', help='which model to use')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--save_dir', default='/scratch/aditya_hari/checkpoints/', help='where to save the logs and checkpoints')
    parser.add_argument('--lr', default=2e-5, help='learning rate for main model')
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--train_batch_size', default=8, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=8, type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='test batch size')
    parser.add_argument('--max_source_length', default=1024, type=int, help='max source length')
    parser.add_argument('--max_target_length', default=1024, type=int, help='max target length')
    # parser.add_argument('--model_device', default='cuda:0', type=str, help='device to run the main generation model on')
    parser.add_argument('--model_gpus', default='0', type=str, help='multiple gpus on which main model will be loaded')
    parser.add_argument('--ner_device', default='cuda:1', type=str, help='device to load IndicNER on')
    parser.add_argument('--ner_model_path', default='ai4bharat/IndicNER', type=str, help='path to the NER model checkpoint')
    parser.add_argument('--ner_tok', default='ai4bharat/IndicNER', type=str, help='tokenizer for NER model')
    parser.add_argument('--ner_f_device', default=3, type=int, help='device to load Foreign NER on')
    parser.add_argument('--ner_f_model_path', default='Babelscape/wikineural-multilingual-ner', type=str, help='path to the NER model checkpoint')
    parser.add_argument('--ner_f_tok', default='Babelscape/wikineural-multilingual-ner', type=str, help='tokenizer for NER model')
    parser.add_argument('--sectitle_device', default='cuda:2', type=str, help='device to load section-title compatibility model on')
    parser.add_argument('--sectitle_model_path', default='xlm-roberta-base', type=str, help='path to the sectitle model checkpoint')
    parser.add_argument('--sectitle_tok', default='xlm-roberta-base', type=str, help='tokenizer for section-title model')
    parser.add_argument('--isTest', default=0, type=int, help='test run')
    parser.add_argument('--isTrial', default=0, type=int, help='toy run')
    parser.add_argument('--langs', default='all', type=str, help='languages to use')
    parser.add_argument('--world_size', default=3, type=int, help="world size")

    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    ic(local_rank)
    args.is_master = local_rank == 0
    # train_path = args.train_path
    # val_path = args.val_path
    dataset_path = args.dataset_dir

    lang = args.langs 
    if(lang == 'all'):
        lang = all_langs
    else:
        lang = lang.split(',')

    tokenizer_name_or_path = args.tokenizer
    model_name_or_path = args.model
    is_mt5 = args.is_mt5

    if args.config is not None:
        config = args.config
    else:
        config = model_name_or_path

    EXP_NAME = args.exp_name
    num_epochs = args.num_epochs
    lr = args.lr
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    save_dir = args.save_dir
    rank = local_rank
    # model_device = args.model_device
    #model_gpus = [int(c) for c in args.model_gpus.split(',')]
    model_gpus = [local_rank]
    device = torch.cuda.device(local_rank)
    ic(f"starting setup {local_rank}")
    #setup(rank, args.world_size)
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(SEED)
    ner_device = args.ner_device
    ner_model_path = args.ner_model_path
    ner_tok = args.ner_tok

    ner_f_device = args.ner_f_device
    ner_f_model_path = args.ner_f_model_path
    ner_f_tok = args.ner_f_tok

    sectitle_device = args.sectitle_device
    sectitle_model_path = args.sectitle_model_path
    sectitle_tok = args.sectitle_tok

    isTrial = args.isTrial
    isTest = args.isTest

    foreign = {'en', 'fr', 'es', 'de', 'it'}
    ic(f"getting tokenizer {local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    new_tokens = ['<H>', '<R>', '<T>', '<QR>', '<QT>', '<S>']
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = []
    for idx, t in enumerate(new_tokens):
        new_tokens_vocab['additional_special_tokens'].append(t)
    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

    ic(f"got tokenizer {local_rank}")
    # train_dataset = ModelDataset(
    #     train_path,
    #     tokenizer,
    #     max_source_length,
    #     max_target_length,
    #     is_mt5
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=train_batch_size,
    #     num_workers=6,
    #     shuffle=True
    # )

    # val_dataset = ModelDataset(
    #     val_path,
    #     tokenizer,
    #     max_source_length,
    #     max_target_length,
    #     is_mt5
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=val_batch_size,
    #     num_workers=6,
    #     shuffle=True
    # )
    ic(f"getting datasets {local_rank}")

    enable_prefix = False
    merged_directory = _intiate_dataset_merging('train', dataset_path, languages=lang, logger=None)
    train_file_path = os.path.join(os.path.abspath(merged_directory), 'train.jsonl')
    train_loader = get_dataset_loaders(tokenizer, train_file_path, None, dataset_count=0,
                                            batch_size=train_batch_size,  src_max_seq_len=200, 
                                            tgt_max_seq_len=200, script_unification=True, prefix=enable_prefix,
                                            complete_coverage=False)

    enable_prefix = False
    merged_directory = _intiate_dataset_merging('val', dataset_path, languages=lang, logger=None)
    val_file_path = os.path.join(os.path.abspath(merged_directory), 'val.jsonl')
    val_loader = get_dataset_loaders(tokenizer, val_file_path, None, dataset_count=0,
                                            batch_size=args.val_batch_size,  src_max_seq_len=200, 
                                            tgt_max_seq_len=200, script_unification=True, prefix=enable_prefix,
                                            complete_coverage=False)
    ic(f"got datasets {local_rank}")
    start_epoch = 0
    final_checkpoint = None


    if os.path.exists(save_dir):
        filenames = [f for f in os.listdir(save_dir) if "half" not in f]
        if len(filenames) > 0:
            filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
            final_checkpoint = filenames[-1]
            start_epoch = int(final_checkpoint.split('.')[0]) + 1
            final_checkpoint = f'{save_dir}{final_checkpoint}'
            
    ic(f"getting model {local_rank}")
    model_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    model = GenModel(
        learning_rate=lr,
        model_name_or_path=model_name_or_path,
        config = config,
        is_mt5=is_mt5,
        eval_beams=4,
        tgt_max_seq_len=max_target_length,
        tokenizer=tokenizer,
        model_gpus=local_rank,
        isTest=isTest,
        final_checkpoint=final_checkpoint
    )
    ic(f"got model {local_rank}")
    # print("Loading Section-Title Model")
    # titlemodel = AutoModelForSequenceClassification.from_pretrained(
    #     sectitle_model_path,
    #     num_labels=2
    # ).to(sectitle_device)
    # titletok = AutoTokenizer.from_pretrained(sectitle_tok)
    # titlemodel.eval()
    # print("Loaded Section-Title Model")

    # print("Loading IndicNER Model")
    # nermodel = AutoModelForTokenClassification.from_pretrained(
    #     ner_model_path
    # ).to(ner_device)
    # nertok = AutoTokenizer.from_pretrained(ner_tok)
    # nermodel.eval()
    # print("Loaded IndicNER Model")

    # print("Loading Foreign NER Model")
    # nerfmodel = AutoModelForTokenClassification.from_pretrained(
    #     ner_f_model_path
    # )
    # nerftok = AutoTokenizer.from_pretrained(ner_f_tok)
    # nerfmodel.eval()
    # nerf_pipeline = pipeline("ner", model=nerfmodel, tokenizer=nerftok, device=ner_f_device)
    # print("Loaded Foreign NER Model")

    wandb.init(
        project='msme-rl',
        config={
            'learning_rate': lr,
            'epochs': num_epochs,
            'batch_size': train_batch_size
        }
    )
    wandb.run.name = EXP_NAME

    model_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ic(model_gpus)
    model.to(model_device)
    model = DDP(model, device_ids=[local_rank])
    ic(next(model.parameters()).is_cuda)

    ic(start_epoch, final_checkpoint)
    pbar = tqdm(range(start_epoch, num_epochs))
    for epoch in pbar:
        train_loader.sampler.set_epoch(epoch)
        pbar.set_postfix(loss=local_rank)
        avg_train_loss = []
        avg_val_loss = []
        dist.barrier()  
        pbar2 = tqdm(train_loader)
        for batch in pbar2:
            pbar2.set_postfix(model_number=local_rank)
            src_ids, src_mask, tgt_ids, tgt_mask, lang_id, idx = batch
            batch = {} 
            # batch['input_ids'] = src_ids.to(model_device).squeeze()
            batch['input_ids'] = src_ids.to(model_device)
            #print(batch['input_ids'].shape)
            #print(src_ids.shape)
            # batch['attention_mask'] = src_mask.to(model_device).squeeze()
            batch['attention_mask'] = src_mask.to(model_device)
            # batch['labels'] = tgt_ids.to(model_device).squeeze()
            batch['labels'] = tgt_ids.to(model_device)
            #print(tgt_ids.shape)
            # outputs = model(batch)
            middle_output = model.module.middle(batch)
            main_loss, logits, input_text, pred_text = middle_output['main_loss'], middle_output['logits'], middle_output['input_text'], middle_output['pred_text']
            #ic(f'{input_text} in model {local_rank}')
            # reward_loss = calcTestReward(
            #     batch=batch,
            #     logits=logits,
            #     foreign=foreign,
            #     input_text=input_text,
            #     pred_text=pred_text
            # )

            # reward_loss = calcReward(
            #     batch=batch,
            #     logits=logits,
            #     foreign=foreign,
            #     input_text=input_text,
            #     pred_text=pred_text,
            #     titlemodel=titlemodel,
            #     titletok=titletok,
            #     titledevice=sectitle_device,
            #     nertok=nertok,
            #     nermodel=nermodel,
            #     nerdevice=ner_device,
            #     nerf_pipeline=nerf_pipeline
            # )

            reward_loss = 0

            # nerloss, secloss = reward_loss['nerloss'], reward_loss['secloss']
            total_loss = main_loss + reward_loss

            avg_train_loss.append(total_loss.item())
            total_loss.backward()

            if len(avg_train_loss) == 10 and isTrial:
                break

            if len(avg_train_loss) == len(train_loader)//2:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = f'{save_dir}/{epoch}_half.pt'
                torch.save(model.state_dict(), save_path)


            optimizer.step()
            optimizer.zero_grad()

        if(local_rank == 0):
            ic(f"Model number {local_rank} saving")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = f'{save_dir}/{epoch}.pt'
            torch.save(model.state_dict(), save_path)

        dist.barrier() 

        for batch in tqdm(val_loader):
            src_ids, src_mask, tgt_ids, tgt_mask, lang_id, idx = batch
            batch = {} 
            # batch['input_ids'] = src_ids.to(model_device).squeeze()
            batch['input_ids'] = src_ids.to(model_device)
            #print(batch['input_ids'].shape)
            #print(src_ids.shape)
            # batch['attention_mask'] = src_mask.to(model_device).squeeze()
            batch['attention_mask'] = src_mask.to(model_device)
            # batch['labels'] = tgt_ids.to(model_device).squeeze()
            batch['labels'] = tgt_ids.to(model_device)
            with torch.no_grad():
                outputs = model(batch)

            middle_output = model.module.middle(batch)
            loss = middle_output['main_loss']
            avg_val_loss.append(loss.item())

            if len(avg_val_loss) == 10 and isTrial:
                break

        valloss = sum(avg_val_loss)/len(avg_val_loss)
        trainloss = sum(avg_train_loss)/len(avg_train_loss)
        
        if(local_rank == 0):
            wandb.log({
                'val_loss': valloss,
                'train_loss': trainloss,
                'epoch': epoch
            })

    cleanup()

if __name__ == '__main__':
    main()

