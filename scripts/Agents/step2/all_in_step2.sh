#!/bin/bash

# Total number of chunks
NCHUNKS=10


for (( i=0; i<$NCHUNKS; i++ ))
do
  JOB_NAME="step2_$i"
  OUTPUT_DIR="Agents/step2/results"
  SCRIPT_PATH="Agents/step2/job_chunk_${i}.sh"

  mkdir -p "$OUTPUT_DIR"

  cat <<EOF > "$SCRIPT_PATH"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@a100
#SBATCH -C a100
#SBATCH --time=20:00:00
#SBATCH --output=${OUTPUT_DIR}/$JOB_NAME.out


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

python ../agent.py \\
  --model_name_or_path ../models/bart_large_step1/checkpoint-12000 \\
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \\
  --num_iterations 11 \\
  --library_path ../data/library/library_step2/library.json \\
  --vector_store_path ../data/library/vector_store_step \\
  --likelihood_step ../logs/likelihood_step1.json \\
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \\
  --table_limit 5 \\
  --base_prompt_path ../source/prompts/base_prompt_v2.yaml \\
  --max_source_length_llm 8192 \\
  --use_model_check 1 \\
  --output_generation 1 \\
  --max_source_length 1024 \\
  --max_target_length 128 \\
  --num_beams 5 \\
  --chunk $i \\
  --Nchunk $NCHUNKS
EOF

  echo "Generated: $SCRIPT_PATH"
done

# Optionally submit all step2
if [ "$1" == "true" ]; then
  for (( i=0; i<$NCHUNKS; i++ ))
  do
    sbatch Agents/step2/job_chunk_${i}.sh
  done
fi


