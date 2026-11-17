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
MAX_NEW_TOKENS=140    # justification (qui critique, 1 phrase) + SOURCED ou JOURNALIST -- releve de 80 a 140 : sur job65119601, ~1 reponse finale sur 5 de ce step reposait sur une justification vide/tronquee (voir src/verify_utils.py), hypothese a verifier sur le prochain run
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# Pass 2 ("forced re-search", 2026-11-17 redesign) -- re-reads the FULL
# ARTICLE a second time (not just pass 1's justification, which was the
# removed 2026-10-13/2026-11-16 designs' blind spot/dead end -- see
# scripts/step4a_source_attribution.py's module docstring for the full
# history), so it needs a similar input budget to pass 1 above and a
# larger output budget (reasoning + enumeration + answer). Scoped to
# JOURNALIST-only rows; can only flip JOURNALIST->SOURCED.
VERIFY_BATCH_SIZE=8
VERIFY_MAX_NEW_TOKENS=250
VERIFY_MAX_INPUT_TOKENS=16384
VERIFY_TEMPERATURE=0.0
SKIP_PASS2=${SKIP_PASS2:-0}   # 0 = pass 2 runs (first test of this new mechanism); set SKIP_PASS2=1 to disable without touching code if it measures net-negative.

# Pass 3 ("pass2 coherence check", 2026-11-18, logit-probing) -- reads
# ONLY pass 2's own justification text (short, cheap), so needs none of
# pass 1/pass 2's large context budget; reuses VERIFY_MAX_INPUT_TOKENS in
# the python script purely as a generous ceiling, never actually
# approached by a short justification. See
# scripts/step4a_source_attribution.py's module docstring for why this
# pass exists (job20620441: ~half of pass 2's SOURCED flips were
# self-contradictory -- reasoning said JOURNALIST, final word said
# SOURCED anyway -- and deterministic Python rules were tried and found
# not robust enough to catch it).
COHERENCE_CANDIDATE_LETTERS=A,B
COHERENCE_BATCH_SIZE=32
COHERENCE_CHECKPOINT_EVERY=200
COHERENCE_THRESHOLD=${COHERENCE_THRESHOLD:-0.5}   # neutral logit midpoint, NOT yet empirically tuned -- see --coherence_threshold's own help text in the python script.
COHERENCE_MARGIN=${COHERENCE_MARGIN:-0.15}   # minimum confident-disagreement margin before overriding pass 2 -- NOT yet empirically tuned, see --coherence_margin's own help text.
SKIP_COHERENCE=${SKIP_COHERENCE:-0}   # 0 = pass 3 runs (first test of this new mechanism); set SKIP_COHERENCE=1 to disable without touching code if it measures net-negative.

# Pass 1 retries + last-resort force -- every row step 3 flagged as
# criticism is now supposed to come out of this step with a real answer
# (see src/verify_utils.py and the retry/force block in the script).
DRAFT_MAX_RETRIES=2
RETRY_TEMPERATURE=0.4

NUM_TASKS=${NUM_TASKS:-9}   # doit correspondre au nombre de taches dans --array (0-8 = 9 taches) -- surchargeable : NUM_TASKS=1 sbatch --array=0-0 sbatch_step4a_array.sh <input> pour un run rapide sur un petit echantillon (ex. les 400 lignes gold)

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
echo "DRAFT_MAX_RETRIES=${DRAFT_MAX_RETRIES} | RETRY_TEMPERATURE=${RETRY_TEMPERATURE}"
echo "SKIP_PASS2=${SKIP_PASS2}"
echo "COHERENCE_CANDIDATE_LETTERS=${COHERENCE_CANDIDATE_LETTERS} | COHERENCE_BATCH_SIZE=${COHERENCE_BATCH_SIZE} | COHERENCE_CHECKPOINT_EVERY=${COHERENCE_CHECKPOINT_EVERY} | COHERENCE_THRESHOLD=${COHERENCE_THRESHOLD} | COHERENCE_MARGIN=${COHERENCE_MARGIN} | SKIP_COHERENCE=${SKIP_COHERENCE}"

SKIP_PASS2_FLAG=()
if [[ "${SKIP_PASS2}" == "1" ]]; then
    SKIP_PASS2_FLAG=(--skip_pass2)
fi

SKIP_COHERENCE_FLAG=()
if [[ "${SKIP_COHERENCE}" == "1" ]]; then
    SKIP_COHERENCE_FLAG=(--skip_coherence)
fi

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
  --draft_max_retries       "$DRAFT_MAX_RETRIES" \
  --retry_temperature       "$RETRY_TEMPERATURE" \
  --coherence_candidate_letters   "$COHERENCE_CANDIDATE_LETTERS" \
  --coherence_batch_size          "$COHERENCE_BATCH_SIZE" \
  --coherence_checkpoint_every    "$COHERENCE_CHECKPOINT_EVERY" \
  --coherence_threshold           "$COHERENCE_THRESHOLD" \
  --coherence_margin              "$COHERENCE_MARGIN" \
  --num_tasks         "$NUM_TASKS" \
  "${SKIP_PASS2_FLAG[@]}" \
  "${SKIP_COHERENCE_FLAG[@]}"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
