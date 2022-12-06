#!/bin/bash

#SBATCH --job-name=sentence_number
#SBATCH -A irel
#SBATCH -p long
#SBATCH -c 10
#SBATCH -t 04-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output random.log
#SBATCH --ntasks 1


pip install --upgrade transformers

echo "COMPLIETE"
