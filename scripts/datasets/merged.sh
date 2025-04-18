#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@h100
#SBATCH -C h100
#SBATCH --time=20:00:00
#SBATCH --output=train/results/run.out


python ../source/prepare_data/merged_library.py 

python ../source/prepare_data/filter_errors.py
