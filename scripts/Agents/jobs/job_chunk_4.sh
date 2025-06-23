#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@a100
#SBATCH -C a100
#SBATCH --time=10:00:00
#SBATCH --output=Agents/jobs/results/v4.out



export PATH=/lustre/fswork/projects/rech/kns/uxe25ug/ollama/bin:/home/raphael.gervillie/.local/bin:/data/raphael.gervillie/envs/ollama_env/bin:/usr/local/miniconda/condabin:/usr/local/miniconda/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/snap/bin:/data/raphael.gervillie/ollama/bin:/data/raphael.gervillie/ollama/bin
export OLLAMA_MODELS=/lustre/fswork/projects/rech/kns/uxe25ug/ollama/models

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
  --output_dir ../models/not-used \
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \
  --num_iterations 11 \
  --library_path ../data/library/library_step.json \
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \
  --table_limit 5 \
  --base_prompt_path ../data/prompts/base_prompt.yaml \
  --max_source_length_llm 8192 \
  --chunk 4 \
  --Nchunk 30
