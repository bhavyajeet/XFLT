# MT5
#python train_og.py --train_path /scratch/aditya.hari/data/train_extractive.json --val_path /scratch/aditya.hari/data/val_extractive.json --test_path /scratch/aditya.hari/data/test_extractive.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name multi-ling-multi-dom-mt5 --save_dir ./genmodels/ --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --max_source_length 512 --max_target_length 32 --ner_device cuda:1 --sectitle_device cuda:1 --sectitle_model_path ./secmodel/9/ --sectitle_tok xlm-roberta-base --ner_f_device 1 --isTrial 1 --model_gpus 0,2,3

# MBart
#python train.py --train_path data/train_extractive.json --val_path data/val_extractive.json --test_path data/test_extractive.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name multi-ling-multi-dom-mbart --save_dir ./genmodels/ --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --max_source_length 512 --max_target_length 32 --ner_device cuda:1 --sectitle_device cuda:1 --sectitle_model_path ./secmodel/9/ --sectitle_tok xlm-roberta-base --ner_f_device 1 --model_gpus 0,2,3


# python train_og.py --train_path /scratch/aditya.hari/data/train_extractive.json --val_path /scratch/aditya.hari/data/val_extractive.json --test_path /scratch/aditya.hari/data/test_extractive.json --tokenizer google/mt5-small --model google/mt5-small --is_mt5 1 --exp_name multi-ling-multi-dom-mt5 --save_dir ./genmodels/ --num_epochs 10 --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 --max_source_length 512 --max_target_length 32 --ner_device cuda:1 --sectitle_device cuda:1 --sectitle_model_path ./secmodel/9/ --sectitle_tok xlm-roberta-base --ner_f_device 1 --isTrial 1 --model_gpus 0,2,3

python train.py --dataset_dir /multi_sent_data/ --max_source_length 200 --max_target_length 200 --is_mt5 1 --exp_name multisent_mt5_rl

