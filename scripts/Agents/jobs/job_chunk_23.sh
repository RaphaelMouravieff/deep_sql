#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v23
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@a100
#SBATCH -C a100
#SBATCH --time=10:00:00
#SBATCH --output=Agents/jobs/results/v23.out



export PATH=/lustre/fswork/projects/rech/kns/uxe25ug/ollama/bin:/lustre/fswork/projects/rech/kns/uxe25ug/ollama/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Library/Apple/usr/bin:/Library/TeX/texbin:/Users/tanguyherserant/anaconda3/envs/deep/bin:/Users/tanguyherserant/anaconda3/condabin
export OLLAMA_MODELS=/lustre/fswork/projects/rech/kns/uxe25ug/ollama/models

if ! pgrep -x "ollama" > /dev/null; then
  ollama serve &
fi

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

export OLLAMA_API_BASE=http://127.0.0.1:32149 
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
  --chunk 23 \
  --Nchunk 30 \
  --model_name_or_path "not-used-but-must-be-set" 
