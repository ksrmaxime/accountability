#!/bin/bash -l
#SBATCH --job-name=download_merge_years
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_merge_years_%j.out
#SBATCH --error=logs/download_merge_years_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# ═══════════════════════════════════════════════════════════════════════════
# Regroupe les années téléchargées par sbatch_download_yearly.sh en un seul
# fichier, au même format qu'un download classique :
#     data/input/download_job<ID de CE job>/swissdox_articles_lead.parquet
# puis met à jour data/input/.last_download et soumet sbatch_step2.sh.
#
# Soumis automatiquement à la fin de la chaîne. Lancement manuel possible :
#     sbatch sbatch_download_merge_years.sh <dossier de la chaîne>
# ═══════════════════════════════════════════════════════════════════════════

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO"
cd "${REPO_DIR}"
mkdir -p logs

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

CHAIN_DIR="${1:-}"
if [ -z "$CHAIN_DIR" ] || [ ! -d "$CHAIN_DIR" ]; then
    echo "[ERROR] Usage: sbatch sbatch_download_merge_years.sh <dossier de la chaîne>"
    exit 1
fi
OUTDIR="${REPO_DIR}/data/input/download_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== Regroupement des années ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "CHAIN_DIR=${CHAIN_DIR}"
echo "OUTDIR=${OUTDIR}"

python scripts/download_merge_years.py --chain_dir "$CHAIN_DIR" --outdir "$OUTDIR"

echo "Regroupement terminé."

# ── Auto-chain ─────────────────────────────────────────────────────────────────
DOWNLOADED_FILE="${OUTDIR}/swissdox_articles_lead.parquet"
echo "${DOWNLOADED_FILE}" > "${REPO_DIR}/data/input/.last_download"
sbatch "${REPO_DIR}/sbatch_step2.sh" "${DOWNLOADED_FILE}"
echo "[chain] → sbatch_step2.sh submitted (input: ${DOWNLOADED_FILE})"
