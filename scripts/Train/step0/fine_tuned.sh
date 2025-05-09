#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=step0
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=Train/step0/results/fine_tuned.out

python ../train.py \
  --do_train \
  --do_eval \
  --do_predict \
  --dataset_name wikitablequestions \
  --output_dir ../models/bart_large_step0 \
  --model_name_or_path ../models/bart_large_step0/checkpoint-1074 \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 2000 \
  --save_steps 2000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000

