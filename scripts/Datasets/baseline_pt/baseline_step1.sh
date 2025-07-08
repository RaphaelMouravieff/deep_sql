#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/results/baseline.out


python ../source/data_utils/get_baseline_dataset.py \
    --dataset_stepx_path ../data/training_dataset/step1 \
    --actual_baseline_path ../data/tapex_training_dataset/step1 \
    --previous_baseline_path ../data/tapex_training_dataset/step0


