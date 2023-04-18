# MT5

python test.py --test_path data/test_extractive.json --checkpoint ./genmodels/2.pt --tokenizer google/mt5-base --model google/mt5-base --save_dir ./inferences --test_batch_size 4 --max_source_length 512 --max_target_length 32 --model_device cuda:0 --model_gpus 0,1,2,3 --isTest 1 --isTrial 0 --is_mt5 1

# MBart
python test.py --test_path /scratch/shivansh.s/data/test_extractive.json --checkpoint /scratch/shivansh.s/genmodels/1.pt --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --save_dir /scratch/shivansh.s/outputs/ --test_batch_size 4 --max_source_length 128 --max_target_length 32 --model_gpus 0,1,2,3 --model_device cuda:0 --isTest 1 --isTrial 1 --is_mt5 0
