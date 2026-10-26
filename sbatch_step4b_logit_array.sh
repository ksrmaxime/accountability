#!/bin/bash -l
#SBATCH --job-name=step4b_logit_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step4b_logit_array_%A_%a.out
#SBATCH --error=logs/step4b_logit_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8   # 9 taches (0-8) -- meme decoupage que le 3-pass

dcsrsoft use 20241118

set -euo pipefail

# =============================================================================
# step 4b -- REDESIGN "logit-probing" (2026-10-26+), en pause par rapport au
# design a 3 passes (sbatch_step4b_array.sh, toujours disponible tel quel).
#
# Au lieu de generer une reponse texte parmi 6 categories, ce script pose 6
# questions OUI/NON independantes (une par categorie) et lit directement les
# logits du prochain token (A/B) -- pas de generation, pas de parsing qui
# peut echouer, pas de retry/force. Voir scripts/step4b_source_category_logit.py
# et src/step4b_logit_prompts.py / src/step4b_logit_utils.py pour le detail.
#
# IMPORTANT -- a faire AVANT de lancer ce sbatch (une seule fois, rapide) :
#   srun --pty --gres=gpu:1 --mem=24G --time=00:15:00 bash -l
#   module load python/3.12.1 && source .venv/bin/activate
#   python scripts/check_step4b_logit_tokens.py \
#       --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
#       --dtype bf16 --backend transformers --candidate_letters A,B
# Confirme que "A"/"B" sont bien des tokens uniques et des reponses
# plausibles pour ce tokenizer avant d'investir du temps GPU sur l'array.
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
INPUT=${1:-"${WORKDIR}/data/output/step4a_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4b_logit
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference -- pas de temperature/max_new_tokens : une seule passe forward
# (lecture de logits), jamais de generation. batch_size compte des PROMPTS
# individuels (row x categorie x ordre), pas des lignes -- chaque ligne
# eligible produit 6 categories x 2 ordres = 12 prompts.
BATCH_SIZE=32
MAX_INPUT_TOKENS=16384
CANDIDATE_LETTERS=A,B   # changer ici si check_step4b_logit_tokens.py recommande une autre paire
CHECKPOINT_EVERY=200

NUM_TASKS=${NUM_TASKS:-9}   # doit correspondre au nombre de taches dans --array

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/output

# -- Auto-chain : task 0 soumet le merge des le debut --
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "0" ]]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        "${WORKDIR}/sbatch_merge_step4b_logit.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_step4b_logit.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
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
echo "BATCH_SIZE=${BATCH_SIZE} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | CANDIDATE_LETTERS=${CANDIDATE_LETTERS}"

python scripts/step4b_source_category_logit.py \
  --input             "$INPUT" \
  --output_base       "$OUTPUT_BASE" \
  --text_col          "$TEXT_COL" \
  --n_rows            "$N_ROWS" \
  --model_path        "$MODEL_PATH" \
  --dtype             "$DTYPE" \
  --backend           "$BACKEND" \
  --trust_remote_code \
  --batch_size        "$BATCH_SIZE" \
  --max_input_tokens  "$MAX_INPUT_TOKENS" \
  --candidate_letters "$CANDIDATE_LETTERS" \
  --checkpoint_every  "$CHECKPOINT_EVERY" \
  --num_tasks         "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
