#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v4
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=Agents/jobs/results/v4.out


if ! pgrep -x "ollama" > /dev/null; then
  ollama serve &
fi

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export OLLAMA_DEBUG=0
export OLLAMA_LOG_LEVEL=ERROR
export OLLAMA_VERBOSE=0
export OLLAMA_LOG_LEVEL=error
export GIN_MODE=release

python ../agent.py \
  --model_name_or_path ../models/bart_large_step0/checkpoint-10000 \
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \
  --num_iterations 11 \
  --library_path ../data/library/library_step1/library.json \
  --vector_store_path ../data/library/vector_store_step \
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \
  --table_limit 5 \
  --base_prompt_path ../source/prompts/base_prompt_v2.yaml \
  --max_source_length_llm 8192 \
  --use_model_check 1 \
  --output_generation 1 \
  --max_source_length 1024 \
  --max_target_length 128 \
  --num_beams 5 \
  --chunk 4 \
  --Nchunk 5
