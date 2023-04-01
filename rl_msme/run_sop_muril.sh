#!/bin/bash
#SBATCH -c 38
#SBATCH -w gnode042
#SBATCH --mem-per-cpu 2G
#SBATCH --gres gpu:4
#SBATCH --time 3-00:00:00
#SBATCH --output eng_only.log
#SBATCH --mail-user aditya.hari@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name dataloader

python train.py --dataset_dir /home2/aditya_hari/multisent/data/lin_data --max_source_length 200 --max_target_length 200 --is_mt5 1 --exp_name dataloader_pls --model_gpus 0,2,3 --langs all --isTrial 1
