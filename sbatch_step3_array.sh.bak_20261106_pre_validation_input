#!/bin/bash -l
#SBATCH --job-name=step3_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step3_array_%A_%a.out
#SBATCH --error=logs/step3_array_%A_%a.err
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
INPUT=${1:-"${WORKDIR}/data/processed/step2_clean.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step3
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 100 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
BATCH_SIZE=4
MAX_NEW_TOKENS=140    # justification (1 phrase) + YES ou NO -- releve de 80 a 140 (meme
                      # correctif que step4a sur job65119601) : l'analyse du run job65125145
                      # montre que les NA de step3 sont concentres sur les articles longs, et
                      # avec seulement 80 tokens de budget de sortie une justification un peu
                      # plus longue peut deja pousser le label YES/NO hors du budget avant
                      # meme d'etre genere -- voir aussi le fix de troncature a gauche dans
                      # src/client.py (meme diagnostic)
MAX_INPUT_TOKENS=24576  # releve de 16384 : marge de securite supplementaire pour la queue
                        # d'articles tres longs (jusqu'a ~373k caracteres observes) ; au-dela
                        # de cette taille le fix de troncature a gauche (src/client.py) prend
                        # le relais pour au moins preserver la consigne finale du prompt
TEMPERATURE=0.0

# Pass 2 (verify) -- reads only the pass-1 justification, not the article,
# so it still needs a much smaller batch/context than pass 1 above.
# 2026-10-02: reverted to the job65205108 architecture (pass 1 writes the
# justification + YES/NO itself, pass 2 just verifies that justification).
# 2026-10-05: VERIFY_USER_TEMPLATE's worked examples (added 2026-10-02)
# measured net-harmful on job65304761 -- see src/step3_prompts.py's module
# docstring -- and were removed; the verify prompt is back to the bare
# original two-line question (~500 chars of fixed text, measured), so
# VERIFY_MAX_INPUT_TOKENS goes back to 512 (job65205108's own value) rather
# than the 768 the examples needed.
VERIFY_BATCH_SIZE=16
VERIFY_MAX_NEW_TOKENS=8
VERIFY_MAX_INPUT_TOKENS=512
VERIFY_TEMPERATURE=0.0

# Pass 1 retries + last-resort force -- every eligible row is now supposed
# to come out of this step with a real answer (see src/verify_utils.py and
# the retry/force block in the script). Same mechanism as step4a.
DRAFT_MAX_RETRIES=2
RETRY_TEMPERATURE=0.4

# Validation pass (2026-11-02 "validate-on-full-article", logit-probing) --
# re-reads the FULL ARTICLE for every row pass 1 said YES to (never NO),
# same calibrated-logit machinery as step4b's own validation pass (one
# forward pass, order-swap, context-free calibration), and can only ever
# flip YES->NO. Added after a deep manual read of job66790113's 148 step3
# errors found a heavily FP-skewed pattern (109 FP vs 39 FN) concentrated
# on named Federal Councillors, all tracing to the same root cause -- see
# src/step3_logit_prompts.py's module docstring for the full diagnosis and
# why this is NOT a repeat of the generation-based pass 2 removed on
# 2026-10-13. CANDIDATE_LETTERS reuses the pair already checked for
# step4b's tokenizer (scripts/check_step4b_logit_tokens.py) -- same model,
# same tokenizer, no need to re-check.
CANDIDATE_LETTERS=A,B
VALIDATION_BATCH_SIZE=32
VALIDATION_CHECKPOINT_EVERY=200
SKIP_VALIDATION=${SKIP_VALIDATION:-0}   # 0 = validation pass runs (first test of this new mechanism); set SKIP_VALIDATION=1 to disable without touching code if it measures net-negative like step4b's pass 3 did.

NUM_TASKS=${NUM_TASKS:-9}   # doit correspondre au nombre de taches dans --array (0-8 = 9 taches) -- surchargeable : NUM_TASKS=1 sbatch --array=0-0 sbatch_step3_array.sh <input> pour un run rapide sur un petit echantillon (ex. les 400 lignes gold)

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
        "${WORKDIR}/sbatch_merge_step3.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_step3.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
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
echo "DRAFT_MAX_RETRIES=${DRAFT_MAX_RETRIES} | RETRY_TEMPERATURE=${RETRY_TEMPERATURE}"
echo "CANDIDATE_LETTERS=${CANDIDATE_LETTERS} | VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE} | VALIDATION_CHECKPOINT_EVERY=${VALIDATION_CHECKPOINT_EVERY} | SKIP_VALIDATION=${SKIP_VALIDATION}"

SKIP_VALIDATION_FLAG=()
if [[ "${SKIP_VALIDATION}" == "1" ]]; then
    SKIP_VALIDATION_FLAG=(--skip_validation)
fi

python scripts/step3_criticism_detection.py \
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
  --draft_max_retries       "$DRAFT_MAX_RETRIES" \
  --retry_temperature       "$RETRY_TEMPERATURE" \
  --candidate_letters       "$CANDIDATE_LETTERS" \
  --validation_batch_size   "$VALIDATION_BATCH_SIZE" \
  --validation_checkpoint_every "$VALIDATION_CHECKPOINT_EVERY" \
  --num_tasks         "$NUM_TASKS" \
  "${SKIP_VALIDATION_FLAG[@]}"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
