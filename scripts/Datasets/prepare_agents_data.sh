#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=Datasets/results/prep.out


python ../dataset.py \
    --library_path ../data/library/library_step0/library.json \
    --save_dataset_path ../data/training_dataset/step_del \
    --length_filter 0 \
   

