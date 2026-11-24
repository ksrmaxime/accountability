#!/bin/bash -l
#SBATCH --job-name=step4a_remote_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step4a_remote_test_%j.out
#SBATCH --error=logs/step4a_remote_test_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# module/dcsrsoft scripts reference unset vars internally, so -u must stay
# off until after they load (same fix/comment as sbatch_step3_remote_test.sh).
set -eo pipefail

# =============================================================================
# step4a via the UNIL litellm proxy (UNIL-CI / CSCS-hosted models), CPU only.
#
# 2026-11-24 -- created at Maxime's request, once comparing Apertus-v1.5-70B
# against Apertus-8B-Instruct-2509 (local GPU) on step 3 showed the 70B
# remote model giving better results there with a broader prompt (see
# src/step3_prompts_remote.py's module docstring). This is the step4a
# counterpart of sbatch_step3_remote_test.sh: no GPU allocation needed at
# all, the model runs remotely, this job only sends HTTP requests.
# Everything else -- pass 1 (draft), the retry/force ladder, pass 2
# ("forced re-search"), pass 3 (pass-2 coherence check, logit-probing) -- is
# EXACTLY the same step4a script used on CURNAGL's GPU partition
# (sbatch_step4a_array.sh); only the --backend/--model_path/--api_* args
# differ, plus pass 1/pass 2's prompt module (src/step4a_prompts_remote.py,
# a byte-identical duplicate of src/step4a_prompts.py as of its creation --
# see that file's own module docstring). Pass 3's prompt module
# (src/step4a_pass2_coherence_logit_prompts.py) is SHARED, unchanged,
# regardless of backend.
#
# 2026-11-24 addition: pass "1.5" (pass1 coherence check, logit-probing,
# src/step4a_pass1_coherence_logit_prompts.py) -- see
# scripts/step4a_source_attribution.py's own module docstring for the full
# diagnosis/rationale. Runs BEFORE pass 2, directly on
# source_attribution_draft. This pass is gated STRICTLY to --backend
# litellm inside the script itself (it is a no-op for
# sbatch_step4a_array.sh's GPU/vllm/transformers path regardless of any
# flag here), so this sbatch file needs no conditional for that -- the
# PASS1_COHERENCE_*/SKIP_PASS1_COHERENCE knobs below only ever take effect
# on this remote run.
#
# API key: read from a .env file at the repo root (UNIL_LITELLM_API_KEY=...)
# -- see sbatch_step3_remote_test.sh's header comment for the full
# .env/gitignore convention, unchanged here.
#
# Output goes to its OWN directory (data/output/step4a_remote/), never
# data/output/step4a/, so this never collides with or gets confused for a
# real GPU/local run.
#
# Auto-chain: this script's own result is handed straight to
# sbatch_step4b_logit_remote_test.sh at the end (see the bottom of this
# file) -- same re-creation of the classic pipeline's merge-script auto-
# chain (sbatch_merge_step4a.sh's own `sbatch sbatch_step4b_logit_array.sh
# ...` line), just without the array/merge indirection since this is
# already a single job producing a single output file.
#
# Usage:
#   sbatch sbatch_step4a_remote_test.sh [<input>] [<model>]
#
#   <input>  : step 3 (remote) output (parquet/csv) -- when launched by
#              sbatch_step3_remote_test.sh's own auto-chain, this is that
#              run's own output parquet. Standalone default below points at
#              a plausible manual fallback.
#   <model>  : remote model identifier -- defaults to MODEL below, and is
#              passed through unchanged to the next chained step too, so
#              the whole pipeline run uses ONE consistent model end to end.
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
INPUT=${1:-"${WORKDIR}/data/output/step3_remote_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4a_remote
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un essai rapide

# Model / backend
MODEL_PATH=${2:-"swiss-ai/Apertus-v1.5-70B"}
BACKEND=litellm
API_BASE=${API_BASE:-"https://litellm.dcsr.unil.ch/v1"}
API_KEY_ENV=UNIL_LITELLM_API_KEY
CONCURRENCY=${CONCURRENCY:-6}

# Inference: pass 1 (draft) -- same meaning/defaults as
# sbatch_step4a_array.sh's pass 1 (--batch_size is unused by the litellm
# backend but still required by the script's argparse, so it's passed
# through harmlessly).
BATCH_SIZE=4
MAX_NEW_TOKENS=140
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# Pass 1 retries + last-resort force -- same as the GPU run.
DRAFT_MAX_RETRIES=2
RETRY_TEMPERATURE=0.4

# Pass "1.5" ("pass1 coherence check", logit-probing, 2026-11-24) --
# --backend litellm ONLY, see src/step4a_pass1_coherence_logit_prompts.py
# and scripts/step4a_source_attribution.py's own module docstring. Runs
# BEFORE pass 2 below, directly on source_attribution_draft.
PASS1_COHERENCE_CANDIDATE_LETTERS=A,B
PASS1_COHERENCE_BATCH_SIZE=32
PASS1_COHERENCE_CHECKPOINT_EVERY=200
PASS1_COHERENCE_THRESHOLD=${PASS1_COHERENCE_THRESHOLD:-0.5}
PASS1_COHERENCE_MARGIN=${PASS1_COHERENCE_MARGIN:-0.15}
SKIP_PASS1_COHERENCE=${SKIP_PASS1_COHERENCE:-0}

# Pass 2 ("forced re-search") -- same as the GPU run.
VERIFY_BATCH_SIZE=8
VERIFY_MAX_NEW_TOKENS=250
VERIFY_MAX_INPUT_TOKENS=16384
VERIFY_TEMPERATURE=0.0
SKIP_PASS2=${SKIP_PASS2:-0}

# Pass 3 ("pass2 coherence check", logit-probing) -- same as the GPU run.
# Uses client.logprob_many, same as step3's validation pass -- already
# confirmed working against this proxy once a valid key is in place.
COHERENCE_CANDIDATE_LETTERS=A,B
COHERENCE_BATCH_SIZE=32
COHERENCE_CHECKPOINT_EVERY=200
COHERENCE_THRESHOLD=${COHERENCE_THRESHOLD:-0.5}
COHERENCE_MARGIN=${COHERENCE_MARGIN:-0.15}
SKIP_COHERENCE=${SKIP_COHERENCE:-0}

# =============================================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$WORKDIR"

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

mkdir -p logs data/output

echo "=== RUN CONFIG (remote/litellm, CPU-only) ==="
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>} JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "DATE=$(date -Is)"
echo "INPUT=${INPUT} | OUTPUT_BASE=${OUTPUT_BASE} | TEXT_COL=${TEXT_COL} | N_ROWS=${N_ROWS}"
echo "BACKEND=${BACKEND} | MODEL_PATH=${MODEL_PATH} | API_BASE=${API_BASE} | CONCURRENCY=${CONCURRENCY}"
echo "BATCH=${BATCH_SIZE} | MAX_NEW_TOKENS=${MAX_NEW_TOKENS} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | TEMP=${TEMPERATURE}"
echo "DRAFT_MAX_RETRIES=${DRAFT_MAX_RETRIES} | RETRY_TEMPERATURE=${RETRY_TEMPERATURE}"
echo "PASS1_COHERENCE_CANDIDATE_LETTERS=${PASS1_COHERENCE_CANDIDATE_LETTERS} | PASS1_COHERENCE_BATCH_SIZE=${PASS1_COHERENCE_BATCH_SIZE} | PASS1_COHERENCE_CHECKPOINT_EVERY=${PASS1_COHERENCE_CHECKPOINT_EVERY} | PASS1_COHERENCE_THRESHOLD=${PASS1_COHERENCE_THRESHOLD} | PASS1_COHERENCE_MARGIN=${PASS1_COHERENCE_MARGIN} | SKIP_PASS1_COHERENCE=${SKIP_PASS1_COHERENCE}"
echo "VERIFY_BATCH=${VERIFY_BATCH_SIZE} | VERIFY_MAX_NEW_TOKENS=${VERIFY_MAX_NEW_TOKENS} | VERIFY_MAX_INPUT_TOKENS=${VERIFY_MAX_INPUT_TOKENS} | VERIFY_TEMP=${VERIFY_TEMPERATURE} | SKIP_PASS2=${SKIP_PASS2}"
echo "COHERENCE_CANDIDATE_LETTERS=${COHERENCE_CANDIDATE_LETTERS} | COHERENCE_BATCH_SIZE=${COHERENCE_BATCH_SIZE} | COHERENCE_CHECKPOINT_EVERY=${COHERENCE_CHECKPOINT_EVERY} | COHERENCE_THRESHOLD=${COHERENCE_THRESHOLD} | COHERENCE_MARGIN=${COHERENCE_MARGIN} | SKIP_COHERENCE=${SKIP_COHERENCE}"

SKIP_PASS1_COHERENCE_FLAG=()
if [[ "${SKIP_PASS1_COHERENCE}" == "1" ]]; then
    SKIP_PASS1_COHERENCE_FLAG=(--skip_pass1_coherence)
fi

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
  --backend           "$BACKEND" \
  --api_base          "$API_BASE" \
  --api_key_env       "$API_KEY_ENV" \
  --concurrency       "$CONCURRENCY" \
  --batch_size        "$BATCH_SIZE" \
  --max_new_tokens    "$MAX_NEW_TOKENS" \
  --max_input_tokens  "$MAX_INPUT_TOKENS" \
  --temperature       "$TEMPERATURE" \
  --draft_max_retries       "$DRAFT_MAX_RETRIES" \
  --retry_temperature       "$RETRY_TEMPERATURE" \
  --verify_batch_size       "$VERIFY_BATCH_SIZE" \
  --verify_max_new_tokens   "$VERIFY_MAX_NEW_TOKENS" \
  --verify_max_input_tokens "$VERIFY_MAX_INPUT_TOKENS" \
  --verify_temperature      "$VERIFY_TEMPERATURE" \
  --pass1_coherence_candidate_letters   "$PASS1_COHERENCE_CANDIDATE_LETTERS" \
  --pass1_coherence_batch_size          "$PASS1_COHERENCE_BATCH_SIZE" \
  --pass1_coherence_checkpoint_every    "$PASS1_COHERENCE_CHECKPOINT_EVERY" \
  --pass1_coherence_threshold           "$PASS1_COHERENCE_THRESHOLD" \
  --pass1_coherence_margin              "$PASS1_COHERENCE_MARGIN" \
  --coherence_candidate_letters   "$COHERENCE_CANDIDATE_LETTERS" \
  --coherence_batch_size          "$COHERENCE_BATCH_SIZE" \
  --coherence_checkpoint_every    "$COHERENCE_CHECKPOINT_EVERY" \
  --coherence_threshold           "$COHERENCE_THRESHOLD" \
  --coherence_margin              "$COHERENCE_MARGIN" \
  --task_id 0 \
  --num_tasks 1 \
  "${SKIP_PASS1_COHERENCE_FLAG[@]}" \
  "${SKIP_PASS2_FLAG[@]}" \
  "${SKIP_COHERENCE_FLAG[@]}"

echo "Job finished."

# -- Auto-chain -----------------------------------------------------------------
# 2026-11-24: single-job equivalent of the classic pipeline's
# sbatch_merge_step4a.sh -> `sbatch sbatch_step4b_logit_array.sh
# ${MERGED_PARQUET}` line. This script's own output IS already the one file
# the next step needs (--task_id 0 --num_tasks 1 means there is no array to
# merge), so it is handed straight to sbatch_step4b_logit_remote_test.sh,
# carrying the same MODEL_PATH forward so the whole chained run stays on one
# consistent model.
OUTPUT_PARQUET="${OUTPUT_BASE}_task0_job${SLURM_JOB_ID}.parquet"
if [[ -f "$OUTPUT_PARQUET" ]]; then
    sbatch "${WORKDIR}/sbatch_step4b_logit_remote_test.sh" "${OUTPUT_PARQUET}" "${MODEL_PATH}"
    echo "[chain] -> sbatch_step4b_logit_remote_test.sh submitted (input: ${OUTPUT_PARQUET})"
else
    echo "[chain] WARNING: expected output ${OUTPUT_PARQUET} not found -- NOT chaining to"
    echo "[chain] sbatch_step4b_logit_remote_test.sh. Check the log above for errors and, if"
    echo "[chain] the run actually succeeded under a different path, chain manually:"
    echo "[chain]   sbatch sbatch_step4b_logit_remote_test.sh <path> ${MODEL_PATH}"
fi
