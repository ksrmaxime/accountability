#!/bin/bash -l
#SBATCH --job-name=thesis_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/swissdox_%j.out
#SBATCH --error=logs/swissdox_%j.err
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

mkdir -p logs data/input

module purge
module load python/3.12.1

source .venv/bin/activate

python --version
which python

export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

python scripts/download.py \
  --start 2000-01-01 \
  --end   2025-12-31 \
  --max-results 1000000 \
  --outdir data/input

echo "Job finished."

# ── Auto-chain ─────────────────────────────────────────────────────────────────
DOWNLOAD_POINTER="${REPO_DIR}/data/input/.last_download"
if [[ ! -f "$DOWNLOAD_POINTER" ]]; then
    echo "[ERROR] Pointeur de download introuvable : ${DOWNLOAD_POINTER}" >&2
    exit 1
fi
DOWNLOADED_FILE="$(cat "$DOWNLOAD_POINTER")"
sbatch "${REPO_DIR}/sbatch_step2.sh" "${DOWNLOADED_FILE}"
echo "[chain] → sbatch_step2.sh submitted (input: ${DOWNLOADED_FILE})"
