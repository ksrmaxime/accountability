#!/bin/bash -l
#SBATCH --job-name=step4b_logit_remote_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step4b_logit_remote_test_%j.out
#SBATCH --error=logs/step4b_logit_remote_test_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# module/dcsrsoft scripts reference unset vars internally, so -u must stay
# off until after they load (same fix/comment as sbatch_step3_remote_test.sh).
set -eo pipefail

# =============================================================================
# step4b (logit-probing design) via the UNIL litellm proxy (UNIL-CI /
# CSCS-hosted models), CPU only.
#
# 2026-11-24 -- created at Maxime's request, once comparing Apertus-v1.5-70B
# against Apertus-8B-Instruct-2509 (local GPU) on step 3 showed the 70B
# remote model giving better results there with a broader prompt (see
# src/step3_prompts_remote.py's module docstring). This is the step4b
# counterpart of sbatch_step3_remote_test.sh: no GPU allocation needed at
# all, the model runs remotely, this job only sends HTTP requests.
# Everything else -- the "summarize-then-probe" pre-pass (generation), the
# 6-category logit-probing pass, the "validate-and-correct" pass 3 -- is
# EXACTLY the same step4b script used on CURNAGL's GPU partition
# (sbatch_step4b_logit_array.sh); only the --backend/--model_path/--api_*
# args differ, plus the prompt module (src/step4b_logit_prompts_remote.py,
# a byte-identical duplicate of src/step4b_logit_prompts.py as of its
# creation -- see that file's own module docstring). The summary pre-pass
# is plain generation (client.chat_many); the category/validation passes
# are logit-probing (client.logprob_many), same as step3/step4a's own
# validation passes -- already confirmed working against this proxy once a
# valid key is in place.
#
# API key: read from a .env file at the repo root (UNIL_LITELLM_API_KEY=...)
# -- see sbatch_step3_remote_test.sh's header comment for the full
# .env/gitignore convention, unchanged here.
#
# Output goes to its OWN directory (data/output/step4b_logit_remote/),
# never data/output/step4b_logit/, so this never collides with or gets
# confused for a real GPU/local run.
#
# Auto-chain: this script's own result is handed straight to
# sbatch_step5_remote_test.sh at the end (see the bottom of this file) --
# same re-creation of the classic pipeline's merge-script auto-chain
# (sbatch_merge_step4b_logit.sh's own `sbatch sbatch_step5_array.sh
# ${MERGED_PARQUET}` line), just without the array/merge indirection since
# this is already a single job producing a single output file.
#
# 2026-11-26 -- added the optional head-to-head "disambiguation" pass
# (src/step4b_logit_prompts_remote.py / scripts/step4b_source_category_logit.py,
# litellm-only, see those files' own changelogs). It is OFF by default
# (threshold 0.0): this script takes no positional flags, only the two
# positional args documented below, so the new pass is controlled the same
# way CONCURRENCY/SKIP_VALIDATION already are -- via an environment
# variable set BEFORE the `sbatch` call, e.g.:
#   DISAMBIGUATION_MARGIN_THRESHOLD=0.15 sbatch sbatch_step4b_logit_remote_test.sh
# Passing `--disambiguation_margin_threshold 0.15` directly to `sbatch` does
# NOT work and will break the run -- sbatch has no flag parsing here, so
# that string is consumed as $1 (INPUT), silently replacing the real input
# path and causing the python script to fail trying to read it as a file.
#
# Usage:
#   sbatch sbatch_step4b_logit_remote_test.sh [<input>] [<model>]
#
#   <input>  : step 4a (remote) output (parquet/csv) -- when launched by
#              sbatch_step4a_remote_test.sh's own auto-chain, this is that
#              run's own output parquet. Standalone default below points at
#              a plausible manual fallback.
#   <model>  : remote model identifier -- defaults to MODEL below, and is
#              passed through unchanged to the next chained step too.
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
INPUT=${1:-"${WORKDIR}/data/output/step4a_remote_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4b_logit_remote
TEXT_COL=text   # raw article column, read by the summary pre-pass only
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un essai rapide

# Model / backend
MODEL_PATH=${2:-"swiss-ai/Apertus-v1.5-70B"}
BACKEND=litellm
API_BASE=${API_BASE:-"https://litellm.dcsr.unil.ch/v1"}
API_KEY_ENV=UNIL_LITELLM_API_KEY
CONCURRENCY=${CONCURRENCY:-6}

# Pre-pass "summarize" -- plain generation, same meaning/defaults as
# sbatch_step4b_logit_array.sh (--summary_batch_size is unused by the
# litellm backend but still required by the script's argparse, so it's
# passed through harmlessly).
SUMMARY_BATCH_SIZE=8
SUMMARY_TEMPERATURE=0.0
SUMMARY_MAX_NEW_TOKENS=220
SUMMARY_CHECKPOINT_EVERY=10

# Pass 2 (logit-probing, no generation) -- same as the GPU run.
# --batch_size here counts individual (row,category,order) PROMPTS, not
# rows, but still unused as a GPU batch size under litellm -- CONCURRENCY
# above is what actually governs parallelism for this backend.
BATCH_SIZE=32
MAX_INPUT_TOKENS=16384
CANDIDATE_LETTERS=A,B
CHECKPOINT_EVERY=200

# Pass 3 ("validate-and-correct", logit-probing) -- same as the GPU run.
# 2026-10-29: disabled by default on the GPU pipeline (net-negative on
# job66672510) -- same default kept here.
VALIDATION_BATCH_SIZE=32
VALIDATION_CHECKPOINT_EVERY=200
SKIP_VALIDATION=${SKIP_VALIDATION:-1}

# Pass 3bis ("disambiguation", head-to-head logit-probing between pass 2's
# top-2 categories) -- litellm-only, added 2026-11-26. OFF by default
# (threshold 0.0, i.e. no row's margin is ever < 0.0). Override via env var,
# e.g. DISAMBIGUATION_MARGIN_THRESHOLD=0.15 sbatch sbatch_step4b_logit_remote_test.sh
DISAMBIGUATION_MARGIN_THRESHOLD=${DISAMBIGUATION_MARGIN_THRESHOLD:-0.0}
DISAMBIGUATION_BATCH_SIZE=${DISAMBIGUATION_BATCH_SIZE:-32}
DISAMBIGUATION_CHECKPOINT_EVERY=${DISAMBIGUATION_CHECKPOINT_EVERY:-200}

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
echo "SUMMARY_BATCH_SIZE=${SUMMARY_BATCH_SIZE} | SUMMARY_TEMPERATURE=${SUMMARY_TEMPERATURE} | SUMMARY_MAX_NEW_TOKENS=${SUMMARY_MAX_NEW_TOKENS} | SUMMARY_CHECKPOINT_EVERY=${SUMMARY_CHECKPOINT_EVERY}"
echo "BATCH_SIZE=${BATCH_SIZE} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | CANDIDATE_LETTERS=${CANDIDATE_LETTERS}"
echo "VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE} | VALIDATION_CHECKPOINT_EVERY=${VALIDATION_CHECKPOINT_EVERY} | SKIP_VALIDATION=${SKIP_VALIDATION}"
echo "DISAMBIGUATION_MARGIN_THRESHOLD=${DISAMBIGUATION_MARGIN_THRESHOLD} | DISAMBIGUATION_BATCH_SIZE=${DISAMBIGUATION_BATCH_SIZE} | DISAMBIGUATION_CHECKPOINT_EVERY=${DISAMBIGUATION_CHECKPOINT_EVERY}"

SKIP_VALIDATION_FLAG=()
if [[ "${SKIP_VALIDATION}" == "1" ]]; then
    SKIP_VALIDATION_FLAG=(--skip_validation)
fi

python scripts/step4b_source_category_logit.py \
  --input                    "$INPUT" \
  --output_base               "$OUTPUT_BASE" \
  --text_col                  "$TEXT_COL" \
  --n_rows                    "$N_ROWS" \
  --model_path                "$MODEL_PATH" \
  --backend                   "$BACKEND" \
  --api_base                  "$API_BASE" \
  --api_key_env               "$API_KEY_ENV" \
  --concurrency                "$CONCURRENCY" \
  --summary_batch_size        "$SUMMARY_BATCH_SIZE" \
  --summary_temperature       "$SUMMARY_TEMPERATURE" \
  --summary_max_new_tokens    "$SUMMARY_MAX_NEW_TOKENS" \
  --summary_checkpoint_every  "$SUMMARY_CHECKPOINT_EVERY" \
  --batch_size                "$BATCH_SIZE" \
  --max_input_tokens          "$MAX_INPUT_TOKENS" \
  --candidate_letters         "$CANDIDATE_LETTERS" \
  --checkpoint_every          "$CHECKPOINT_EVERY" \
  --validation_batch_size     "$VALIDATION_BATCH_SIZE" \
  --validation_checkpoint_every "$VALIDATION_CHECKPOINT_EVERY" \
  --disambiguation_margin_threshold "$DISAMBIGUATION_MARGIN_THRESHOLD" \
  --disambiguation_batch_size        "$DISAMBIGUATION_BATCH_SIZE" \
  --disambiguation_checkpoint_every  "$DISAMBIGUATION_CHECKPOINT_EVERY" \
  --task_id 0 \
  --num_tasks 1 \
  "${SKIP_VALIDATION_FLAG[@]}"

echo "Job finished."

# -- Auto-chain -----------------------------------------------------------------
# 2026-11-24: single-job equivalent of the classic pipeline's
# sbatch_merge_step4b_logit.sh -> `sbatch sbatch_step5_array.sh
# ${MERGED_PARQUET}` line.
OUTPUT_PARQUET="${OUTPUT_BASE}_task0_job${SLURM_JOB_ID}.parquet"
if [[ -f "$OUTPUT_PARQUET" ]]; then
    sbatch "${WORKDIR}/sbatch_step5_remote_test.sh" "${OUTPUT_PARQUET}" "${MODEL_PATH}"
    echo "[chain] -> sbatch_step5_remote_test.sh submitted (input: ${OUTPUT_PARQUET})"
else
    echo "[chain] WARNING: expected output ${OUTPUT_PARQUET} not found -- NOT chaining to"
    echo "[chain] sbatch_step5_remote_test.sh. Check the log above for errors and, if the run"
    echo "[chain] actually succeeded under a different path, chain manually:"
    echo "[chain]   sbatch sbatch_step5_remote_test.sh <path> ${MODEL_PATH}"
fi
