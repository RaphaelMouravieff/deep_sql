#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=base1
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=scripts/Inference/baseline/results/step1.out

export PYTHONPATH=$(pwd)
dataset_name="yilunzhao/robut"
split_name="wtq"
model_name="models/tapex_step1/checkpoint-20000"


BASE_DIR="/home/raphael.gervillie/deep_sql"
echo "BASE_DIR: $BASE_DIR"

python $BASE_DIR/source/training/inference_model.py \
  --do_predict \
  --output_dir $BASE_DIR/${model_name}/${split_name} \
  --model_name_or_path $BASE_DIR/${model_name} \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 128 \
  --split_name ${split_name} \
  --dataset_name ${dataset_name} \
  --per_device_eval_batch_size 24 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5