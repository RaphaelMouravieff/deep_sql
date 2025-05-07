#!/bin/bash

# Total number of chunks
NCHUNKS=30

# Directory to save the generated scripts
mkdir -p Agents/dataset

for (( i=0; i<$NCHUNKS; i++ ))
do
  JOB_NAME="v$i"
  OUTPUT_DIR="Agents/jobs/results"
  SCRIPT_PATH="Agents/jobs/job_chunk_${i}.sh"

  mkdir -p "$OUTPUT_DIR"

  cat <<EOF > "$SCRIPT_PATH"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -A kns@a100
#SBATCH -C a100
#SBATCH --time=10:00:00
#SBATCH --output=${OUTPUT_DIR}/$JOB_NAME.out



export PATH=/lustre/fswork/projects/rech/kns/uxe25ug/ollama/bin:$PATH
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

python ../agent.py \\
  --output_dir ../models/not-used \\
  --ollama_model_name_or_path ollama_chat/qwen2.5:14b \\
  --num_iterations 11 \\
  --library_path ../data/library/library_step.json \\
  --embedding_model_name "Alibaba-NLP/gte-large-en-v1.5" \\
  --table_limit 5 \\
  --base_prompt_path ../data/prompts/base_prompt.yaml \\
  --max_source_length_llm 8192 \\
  --chunk $i \\
  --Nchunk $NCHUNKS
EOF

  echo "Generated: $SCRIPT_PATH"
done

# Optionally submit all jobs
if [ "$1" == "true" ]; then
  for (( i=0; i<$NCHUNKS; i++ ))
  do
    sbatch Agents/jobs/job_chunk_${i}.sh
  done
fi