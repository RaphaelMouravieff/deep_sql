#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=step1_verify
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=Verify/step1/results/run.out

# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../.."

python ../source/step1/verification.py \
  --model_name_or_path ../models/bart_large_step0/checkpoint-10000 \
  --dataset_name ../data/training_dataset/step0 \
  --split_name validation \
  --max_source_length 1024 \
  --max_target_length 128 \
  --num_beams 5 \
  --use_fast_tokenizer True 