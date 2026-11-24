#!/bin/bash -l
#SBATCH --job-name=step6_remote_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step6_remote_test_%j.out
#SBATCH --error=logs/step6_remote_test_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# module/dcsrsoft scripts reference unset vars internally, so -u must stay
# off until after they load (same fix/comment as sbatch_step3_remote_test.sh).
set -eo pipefail

# =============================================================================
# step6 (admin response detection) via the UNIL litellm proxy (UNIL-CI /
# CSCS-hosted models), CPU only.
#
# 2026-11-24 -- created at Maxime's request, once comparing Apertus-v1.5-70B
# against Apertus-8B-Instruct-2509 (local GPU) on step 3 showed the 70B
# remote model giving better results there with a broader prompt (see
# src/step3_prompts_remote.py's module docstring). This is the step6
# counterpart of sbatch_step3_remote_test.sh: no GPU allocation needed at
# all, the model runs remotely, this job only sends HTTP requests.
# Everything else is EXACTLY the same step6 script used on CURNAGL's GPU
# partition (sbatch_step6_array.sh); only the --backend/--model_path/
# --api_* args differ, plus the prompt module
# (src/step6_prompts_remote.py, a byte-identical duplicate of
# src/step6_prompts.py as of its creation -- see that file's own module
# docstring). Pass 2 is removed in both the classic and remote paths (see
# scripts/step6_admin_response.py's own module docstring) -- this is a
# single, draft-only LLM call per eligible row.
#
# API key: read from a .env file at the repo root (UNIL_LITELLM_API_KEY=...)
# -- see sbatch_step3_remote_test.sh's header comment for the full
# .env/gitignore convention, unchanged here.
#
# Output goes to its OWN directory (data/output/step6_remote/), never
# data/output/step6/, so this never collides with or gets confused for a
# real GPU/local run.
#
# This is the LAST stage of the chained remote pipeline -- no further
# auto-chain after this one, same as the classic pipeline's
# sbatch_merge_step6.sh.
#
# Usage:
#   sbatch sbatch_step6_remote_test.sh [<input>] [<model>]
#
#   <input>  : step 5 (remote) output (parquet/csv) -- when launched by
#              sbatch_step5_remote_test.sh's own auto-chain, this is that
#              run's own output parquet. Standalone default below points at
#              a plausible manual fallback.
#   <model>  : remote model identifier -- defaults to MODEL below.
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
INPUT=${1:-"${WORKDIR}/data/output/step5_remote_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step6_remote
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un essai rapide

# Model / backend
MODEL_PATH=${2:-"swiss-ai/Apertus-v1.5-70B"}
BACKEND=litellm
API_BASE=${API_BASE:-"https://litellm.dcsr.unil.ch/v1"}
API_KEY_ENV=UNIL_LITELLM_API_KEY
CONCURRENCY=${CONCURRENCY:-6}

# Inference -- same meaning/defaults as sbatch_step6_array.sh's pass 1
# (--batch_size is unused by the litellm backend but still required by the
# script's argparse, so it's passed through harmlessly).
BATCH_SIZE=4
MAX_NEW_TOKENS=150
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# Pass 2 (verify, kept in the CLI for parity with the GPU script but never
# called -- see scripts/step6_admin_response.py's own module docstring).
VERIFY_BATCH_SIZE=16
VERIFY_MAX_NEW_TOKENS=40
VERIFY_MAX_INPUT_TOKENS=512
VERIFY_TEMPERATURE=0.0

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
echo "VERIFY_BATCH=${VERIFY_BATCH_SIZE} | VERIFY_MAX_NEW_TOKENS=${VERIFY_MAX_NEW_TOKENS} | VERIFY_MAX_INPUT_TOKENS=${VERIFY_MAX_INPUT_TOKENS} | VERIFY_TEMP=${VERIFY_TEMPERATURE}"

python scripts/step6_admin_response.py \
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
  --verify_batch_size       "$VERIFY_BATCH_SIZE" \
  --verify_max_new_tokens   "$VERIFY_MAX_NEW_TOKENS" \
  --verify_max_input_tokens "$VERIFY_MAX_INPUT_TOKENS" \
  --verify_temperature      "$VERIFY_TEMPERATURE" \
  --task_id 0 \
  --num_tasks 1

echo "Job finished."
echo ""
echo "[pipeline] step6 (admin response) is the last automated stage of the"
echo "[pipeline] remote/litellm chain -- same as the classic GPU pipeline."
echo "[pipeline] Final dataset for analysis: ${OUTPUT_BASE}_task0_job${SLURM_JOB_ID}.parquet"
echo "[pipeline] Further analysis is done externally in Stata -- nothing more to chain."
