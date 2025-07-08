#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=hard
#SBATCH --gpus-per-node=1
#SBATCH --job-name=step
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/prepate_pt/results/step1.out


python ../dataset.py \
    --library_path ../data/library/library_step1/library.json \
    --save_dataset_path ../data/training_dataset/step1 \
    --length_filter 1 \
    --max_target_length 128 
   

