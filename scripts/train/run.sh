#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v5
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --constraint=A6000
#SBATCH --output=train/results/run.out


export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"


export OLLAMA_DEBUG=0
export OLLAMA_LOG_LEVEL=ERROR
export OLLAMA_VERBOSE=0


python ../run.py \
  --output_dir ../models/T2_M5_CPE_B1_E0/ALL \
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \
  --num_iterations 102 \
  --library_path ../data/library1.json \
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \
  --table_limit 11 \
  --base_prompt_path ../data/prompts/base_prompt.yaml


