#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/results/baseline.out


python Datasets/tapex_baseline_dataset.py \
    --dataset_stepx_path ../data/training_dataset/step0 \
    --save_baseline_dataset ../data/tapex_training_dataset/step0