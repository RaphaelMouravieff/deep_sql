#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v21
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@h100
#SBATCH -C h100
#SBATCH --time=20:00:00
#SBATCH --output=train/generated_jobs/results/v21.out

ollama serve & 

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

export OLLAMA_DEBUG=0
export OLLAMA_LOG_LEVEL=ERROR
export OLLAMA_VERBOSE=0
export OLLAMA_LOG_LEVEL=error
export GIN_MODE=release

python ../run.py \
  --output_dir ../models/not-used \
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \
  --num_iterations 11 \
  --library_path ../data/library_step.json \
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \
  --table_limit 5 \
  --base_prompt_path ../data/prompts/base_prompt.yaml \
  --max_source_length 8192 \
  --chunk 21 \
  --Nchunk 30
