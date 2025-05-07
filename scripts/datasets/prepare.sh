#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --time=20:00:00
#SBATCH --output=datasets/results/prep.out


python ../data_prep.py \
    --merged_library_folder ../data/library/library_step0 \
    --save_dataset_path ../data/training_dataset/step0 \
    --length_filter 0 \
    --model_name_or_path "" \
    --output_dir ""

