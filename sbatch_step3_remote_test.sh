#!/bin/bash -l
#SBATCH --job-name=step3_remote_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step3_remote_test_%j.out
#SBATCH --error=logs/step3_remote_test_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# module/dcsrsoft scripts reference unset vars internally, so -u must stay
# off until after they load (same fix/comment as sbatch_step2.sh -- this
# script originally had `set -euo pipefail` here, which crashed with
# "STACK_20241118_NAME: unbound variable" inside /etc/profile.d/lmodstacks.sh
# the moment `dcsrsoft use 20241118` ran below, since -u was already active).
set -eo pipefail

# =============================================================================
# step3 via the UNIL litellm proxy (UNIL-CI / CSCS-hosted models), CPU only.
#
# 2026-11-18 -- first test of the --backend litellm path (src/client_remote.py)
# added to scripts/step3_criticism_detection.py. No GPU allocation needed at
# all: the model runs remotely, this job only sends HTTP requests. Everything
# else -- input, prompts, parsing, retry/force logic, the validation pass --
# is EXACTLY the same step3 script used on CURNAGL's GPU partition
# (sbatch_step3_array.sh); only the --backend/--model_path/--api_* args differ.
#
# API key: read from a .env file at the repo root (UNIL_LITELLM_API_KEY=...),
# same convention as src/download_src.py's existing SWISSDOX_API_KEY/
# SWISSDOX_API_SECRET -- see .env.example. .env is already gitignored, so add
# the real key there on EACH machine that runs this (git never carries .env
# between CURNAGL and your laptop -- that file simply isn't part of what
# push/pull moves). No "export" needed in your shell: src/client_remote.py
# calls load_dotenv() itself, exactly like the existing download step.
#
# BEFORE running this on real data, run the smoke test by hand once per model
# you intend to use (cheap, ~4 API calls, no sbatch needed):
#     python scripts/check_remote_api.py --model swiss-ai/Apertus-v1.5-70B
# It specifically checks whether logprob_many (needed by the validation pass
# below) actually works against this proxy -- this has NEVER been verified,
# unlike chat_many's plain generation, which almost any OpenAI-compatible
# gateway gets right. If that check's logprob_many step FAILS, either fix it
# first or set SKIP_VALIDATION=1 below (pass 1 + pass 2 generation-only is
# still a fully valid, just less complete, test).
#
# Output goes to its OWN directory (data/output/step3_remote/), never
# data/output/step3/, so this never collides with or gets confused for a
# real GPU/local run -- the two are meant to be compared, not merged, until
# you decide this backend is worth keeping.
#
# Usage:
#   sbatch sbatch_step3_remote_test.sh [<input>] [<model>]
#
#   <input>  : step 2 output (parquet/csv) -- defaults to the same
#              data/processed/.last_step2 pointer sbatch_step3_array.sh uses.
#   <model>  : remote model identifier -- defaults to MODEL below.
#              Models available through this key (per UNIL's 2026-11-18
#              message): UNIL-CI -- qwen3.8-flash-next, gemma-4-26b-a4b,
#              nemotron-3.5-super-vl ; CSCS -- zai-org/GLM-5.3,
#              swiss-ai/Apertus-v1.5-8B[-thinking], swiss-ai/Apertus-v1.5-70B[-thinking].
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO
# No bash-level API key check here -- same as sbatch_download.sh relies on
# src/download_src.py's own load_dotenv()+RuntimeError for SWISSDOX_API_KEY,
# src/client_remote.py raises a clear error itself if UNIL_LITELLM_API_KEY
# is missing from .env (checked right before the first request, inside the
# python process below -- see src/client_remote.py's module docstring).

# I/O
INPUT=${1:-"$(cat "${WORKDIR}/data/processed/.last_step2" 2>/dev/null || echo "${WORKDIR}/data/processed/step2_clean_GOLD800.parquet")"}
OUTPUT_BASE=${WORKDIR}/data/output/step3_remote
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un essai rapide avant le run complet

# Model / backend
MODEL_PATH=${2:-"swiss-ai/Apertus-v1.5-70B-thinking"}
BACKEND=litellm
API_BASE=${API_BASE:-"https://litellm.dcsr.unil.ch/v1"}
API_KEY_ENV=UNIL_LITELLM_API_KEY
CONCURRENCY=${CONCURRENCY:-6}   # concurrent HTTP requests -- replaces --batch_size's role
                                 # for this backend. No rate-limit guidance was given by
                                 # UNIL/CSCS for this key as of 2026-11-18, so this errs
                                 # conservative; raise it (env var, e.g. CONCURRENCY=12
                                 # sbatch ...) once you've confirmed the service can take more.

# Inference -- same meaning/defaults as sbatch_step3_array.sh's pass 1
# (--batch_size is unused by the litellm backend but still required by the
# script's argparse, so it's passed through harmlessly).
BATCH_SIZE=4
MAX_NEW_TOKENS=140
MAX_INPUT_TOKENS=24576
TEMPERATURE=0.0

# Pass 2 (verify) -- same as the GPU run.
VERIFY_BATCH_SIZE=16
VERIFY_MAX_NEW_TOKENS=8
VERIFY_MAX_INPUT_TOKENS=512
VERIFY_TEMPERATURE=0.0

# Pass 1 retries + last-resort force -- same as the GPU run.
DRAFT_MAX_RETRIES=2
RETRY_TEMPERATURE=0.4

# Validation pass (logit-probing) -- same as the GPU run's current default,
# BUT see the big warning in the header comment above: this needs
# logprob_many to actually work against litellm, which has not been
# verified yet. Set SKIP_VALIDATION=1 (env var) if check_remote_api.py's
# logprob_many check failed for your chosen model.
CANDIDATE_LETTERS=A,B
VALIDATION_BATCH_SIZE=32
VALIDATION_CHECKPOINT_EVERY=200
VALIDATION_INPUT=${VALIDATION_INPUT:-justification}
VALIDATION_THRESHOLD=${VALIDATION_THRESHOLD:-0.30}
SKIP_VALIDATION=${SKIP_VALIDATION:-0}

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
echo "DRAFT_MAX_RETRIES=${DRAFT_MAX_RETRIES} | RETRY_TEMPERATURE=${RETRY_TEMPERATURE}"
echo "CANDIDATE_LETTERS=${CANDIDATE_LETTERS} | VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE} | VALIDATION_CHECKPOINT_EVERY=${VALIDATION_CHECKPOINT_EVERY} | SKIP_VALIDATION=${SKIP_VALIDATION} | VALIDATION_INPUT=${VALIDATION_INPUT} | VALIDATION_THRESHOLD=${VALIDATION_THRESHOLD}"

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
  --draft_max_retries       "$DRAFT_MAX_RETRIES" \
  --retry_temperature       "$RETRY_TEMPERATURE" \
  --candidate_letters       "$CANDIDATE_LETTERS" \
  --validation_batch_size   "$VALIDATION_BATCH_SIZE" \
  --validation_checkpoint_every "$VALIDATION_CHECKPOINT_EVERY" \
  --validation_input        "$VALIDATION_INPUT" \
  --validation_threshold    "$VALIDATION_THRESHOLD" \
  --task_id 0 \
  --num_tasks 1 \
  "${SKIP_VALIDATION_FLAG[@]}"

echo "Job finished."
echo ""
echo "No auto-chain into step4a: this is a standalone test run, kept separate"
echo "from the GPU pipeline on purpose so the two can be compared. Once you've"
echo "looked at data/output/step3_remote/ and decided this is worth keeping,"
echo "chain it manually, e.g.:"
echo "  sbatch sbatch_step4a_array.sh <path to this run's merged parquet>"
