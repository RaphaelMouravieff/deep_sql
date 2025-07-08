#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/results/get_wtq.out


python ../source/data_modules/down_wtq_small.py \
  --save_path ../data/fine_tuning/wikitablequestions 