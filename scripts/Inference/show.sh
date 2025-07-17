#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/results/baseline.out


python source/utils/show_results.py \
    --tapex_path models/tapex_step1/checkpoint-20000/wtq/tapex-preds.json \
    --nous_path models/bart_large_step1/checkpoint-12000/wtq/nous-preds.json \
    --save_path logs/plots/step1.png