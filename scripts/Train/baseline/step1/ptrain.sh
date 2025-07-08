#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=Basel1
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=Train/baseline/step1/results/run.out

python ../train.py \
  --do_train \
  --do_eval \
  --do_predict \
  --dataset_name ../data/tapex_training_dataset/step1 \
  --output_dir ../models/tapex_step1 \
  --model_name_or_path ../models/tapex_step0/checkpoint-1074 \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --logging_steps 1 \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --num_train_epochs 1 \
  --logging_strategy epoch \
  --eval_strategy epoch \
  --save_strategy epoch


