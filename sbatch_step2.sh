#!/bin/bash -l
#SBATCH --job-name=step2_clean
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO"
cd "${REPO_DIR}"

mkdir -p logs data/processed

module purge
module load python/3.12.1

source .venv/bin/activate

python --version
which python

# Usage:
#   sbatch sbatch_step2.sh [<input_parquet_or_csv>]
#
# If the input path is omitted, falls back to the download pointer left by
# sbatch_download.sh (data/input/.last_download).
INPUT=${1:-""}
if [[ -z "$INPUT" ]]; then
    DOWNLOAD_POINTER="${REPO_DIR}/data/input/.last_download"
    if [[ -f "$DOWNLOAD_POINTER" ]]; then
        INPUT="$(cat "$DOWNLOAD_POINTER")"
        echo "[INFO] Pas d'input fourni — utilisation du dernier download: ${INPUT}"
    else
        echo "[ERROR] Pas d'input fourni et aucun pointeur ${DOWNLOAD_POINTER} trouvé." >&2
        echo "[ERROR] Usage: sbatch sbatch_step2.sh <input_parquet_or_csv>" >&2
        exit 1
    fi
fi

if [[ ! -f "$INPUT" ]]; then
    echo "[ERROR] Fichier introuvable : ${INPUT}" >&2
    exit 1
fi

python scripts/step2_clean_targets.py \
  --input      "$INPUT" \
  --output_dir data/processed

echo "Job finished."

# ── Auto-chain ─────────────────────────────────────────────────────────────────
STEP2_POINTER="${REPO_DIR}/data/processed/.last_step2"
if [[ ! -f "$STEP2_POINTER" ]]; then
    echo "[ERROR] Pointeur step2 introuvable : ${STEP2_POINTER}" >&2
    exit 1
fi
STEP2_FILE="$(cat "$STEP2_POINTER")"
sbatch "${REPO_DIR}/sbatch_step3_array.sh" "${STEP2_FILE}"
echo "[chain] → sbatch_step3_array.sh submitted (input: ${STEP2_FILE})"
