#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ftnous
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --constraint=A6000
#SBATCH --output=Train/nous/step1/results/fine_tuned.out

python ../train.py \
  --do_train \
  --do_eval \
  --do_predict \
  --dataset_name ../data/fine_tuning/wikitablequestions \
  --output_dir ../models/bart_large_step1 \
  --model_name_or_path ../models/bart_large_step1/checkpoint-1508 \
  --resume_from_checkpoint ../models/bart_large_step1/checkpoint-10000 \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 2000 \
  --save_steps 2000 \
  --warmup_steps 1000 \
  --eval_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000 

