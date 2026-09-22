#!/bin/bash -l
#SBATCH --job-name=step4a_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step4a_array_%A_%a.out
#SBATCH --error=logs/step4a_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8   # 9 taches (0-8)

dcsrsoft use 20241118

set -euo pipefail

# =============================================================================
# USER VARIABLES -- edit here, nowhere else
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
# Input = fichier fusionne produit par l'etape precedente
INPUT=${1:-"${WORKDIR}/data/output/step3_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4a
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 100 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
BATCH_SIZE=4
MAX_NEW_TOKENS=80     # justification (qui critique, 1 phrase) + SOURCED ou JOURNALIST
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# Pass 2 (verify) -- reads only the pass-1 justification, not the article,
# so it needs a much smaller batch/context than pass 1 above.
VERIFY_BATCH_SIZE=16
VERIFY_MAX_NEW_TOKENS=8
VERIFY_MAX_INPUT_TOKENS=512
VERIFY_TEMPERATURE=0.0

NUM_TASKS=9   # doit correspondre au nombre de taches dans --array (0-8 = 9 taches)

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/output

# -- Auto-chain : task 0 soumet le merge des le debut (SLURM attend que TOUTES les taches reussissent) --
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "0" ]]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        "${WORKDIR}/sbatch_merge_step4a.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_step4a.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
fi

echo "=== SLURM ARRAY ==="
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>} TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

echo "=== RUN CONFIG ==="
echo "INPUT=${INPUT} | OUTPUT_BASE=${OUTPUT_BASE}"
echo "TEXT_COL=${TEXT_COL} | NUM_TASKS=${NUM_TASKS} | TASK=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL_PATH} | DTYPE=${DTYPE} | BACKEND=${BACKEND}"
echo "BATCH=${BATCH_SIZE} | MAX_NEW_TOKENS=${MAX_NEW_TOKENS} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | TEMP=${TEMPERATURE}"
echo "VERIFY_BATCH=${VERIFY_BATCH_SIZE} | VERIFY_MAX_NEW_TOKENS=${VERIFY_MAX_NEW_TOKENS} | VERIFY_MAX_INPUT_TOKENS=${VERIFY_MAX_INPUT_TOKENS} | VERIFY_TEMP=${VERIFY_TEMPERATURE}"

python scripts/step4a_source_attribution.py \
  --input             "$INPUT" \
  --output_base       "$OUTPUT_BASE" \
  --text_col          "$TEXT_COL" \
  --n_rows            "$N_ROWS" \
  --model_path        "$MODEL_PATH" \
  --dtype             "$DTYPE" \
  --backend           "$BACKEND" \
  --trust_remote_code \
  --batch_size        "$BATCH_SIZE" \
  --max_new_tokens    "$MAX_NEW_TOKENS" \
  --max_input_tokens  "$MAX_INPUT_TOKENS" \
  --temperature       "$TEMPERATURE" \
  --verify_batch_size       "$VERIFY_BATCH_SIZE" \
  --verify_max_new_tokens   "$VERIFY_MAX_NEW_TOKENS" \
  --verify_max_input_tokens "$VERIFY_MAX_INPUT_TOKENS" \
  --verify_temperature      "$VERIFY_TEMPERATURE" \
  --num_tasks         "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
