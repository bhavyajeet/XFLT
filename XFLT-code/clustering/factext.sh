#!/bin/bash

#SBATCH --job-name=sentence_number
#SBATCH -p long
#SBATCH -c 30
#SBATCH -t 04-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --output sent_num.log
#SBATCH --ntasks 1
#SBATCH -w gnode059


cd  /scratch/msme/datasets_v2.7/para_data


python3 first.py bert-base-uncased 8 0.00001 0.1 run1

scp *.ckpt ada:/share1/userid/msme/
scp *.txt ada:/share1/userid/msme/ 

echo "COMPLIETE"
sleep 1d
