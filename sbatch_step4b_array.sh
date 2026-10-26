#!/bin/bash -l
#SBATCH --job-name=step4b_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step4b_array_%A_%a.out
#SBATCH --error=logs/step4b_array_%A_%a.err
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
INPUT=${1:-"${WORKDIR}/data/output/step4a_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4b
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 100 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference -- Pass 1 (draft) : lit l'article complet
BATCH_SIZE=8
MAX_NEW_TOKENS=90     # justification (1 phrase) + categorie (la plus longue = 'Administrative Unit of the State')
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# Pass 2 (verify, AVEC l'article) -- 2026-10-26 redesign : relit l'article
# complet + la seule categorie brute du pass 1 (jamais la justification, ni
# l'acteur nomme), et repond OUI/NON. Meme gabarit que pass 1 puisqu'il relit
# l'article (mais sortie plus courte : explication + OUI/NON).
PASS2_BATCH_SIZE=8
PASS2_MAX_NEW_TOKENS=90
PASS2_MAX_INPUT_TOKENS=16384
PASS2_TEMPERATURE=0.0

# Pass 3 (reclassification, UNIQUEMENT pour les NON du pass 2) -- ne lit que
# la justification du rejet du pass 2 (jamais l'article), reclasse parmi les
# 6 categories. Gabarit court, identique a l'ancien pass 2 a justification.
PASS3_BATCH_SIZE=16
PASS3_MAX_NEW_TOKENS=12
PASS3_MAX_INPUT_TOKENS=512
PASS3_TEMPERATURE=0.0

# Pass 1 retries + last-resort force -- every eligible row is now supposed
# to come out of this step with a real answer (see src/verify_utils.py and
# the retry/force block in the script). Same mechanism as step4a/step3.
DRAFT_MAX_RETRIES=2
RETRY_TEMPERATURE=0.4

NUM_TASKS=${NUM_TASKS:-9}   # doit correspondre au nombre de taches dans --array (0-8 = 9 taches) -- surchargeable : NUM_TASKS=1 sbatch --array=0-0 sbatch_step4b_array.sh <input> pour un run rapide sur un petit echantillon (ex. les 400 lignes gold)

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
        "${WORKDIR}/sbatch_merge_step4b.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_step4b.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
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
echo "PASS2_BATCH=${PASS2_BATCH_SIZE} | PASS2_MAX_NEW_TOKENS=${PASS2_MAX_NEW_TOKENS} | PASS2_MAX_INPUT_TOKENS=${PASS2_MAX_INPUT_TOKENS} | PASS2_TEMP=${PASS2_TEMPERATURE}"
echo "PASS3_BATCH=${PASS3_BATCH_SIZE} | PASS3_MAX_NEW_TOKENS=${PASS3_MAX_NEW_TOKENS} | PASS3_MAX_INPUT_TOKENS=${PASS3_MAX_INPUT_TOKENS} | PASS3_TEMP=${PASS3_TEMPERATURE}"
echo "DRAFT_MAX_RETRIES=${DRAFT_MAX_RETRIES} | RETRY_TEMPERATURE=${RETRY_TEMPERATURE}"

python scripts/step4b_source_category.py \
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
  --pass2_batch_size       "$PASS2_BATCH_SIZE" \
  --pass2_max_new_tokens   "$PASS2_MAX_NEW_TOKENS" \
  --pass2_max_input_tokens "$PASS2_MAX_INPUT_TOKENS" \
  --pass2_temperature      "$PASS2_TEMPERATURE" \
  --pass3_batch_size       "$PASS3_BATCH_SIZE" \
  --pass3_max_new_tokens   "$PASS3_MAX_NEW_TOKENS" \
  --pass3_max_input_tokens "$PASS3_MAX_INPUT_TOKENS" \
  --pass3_temperature      "$PASS3_TEMPERATURE" \
  --draft_max_retries       "$DRAFT_MAX_RETRIES" \
  --retry_temperature       "$RETRY_TEMPERATURE" \
  --num_tasks         "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
