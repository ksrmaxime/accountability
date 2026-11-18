#!/bin/bash -l
#SBATCH --job-name=accountability_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# ═══════════════════════════════════════════════════════════════════════════
# Download Swissdox — UNE requête pour toute la période (START → END ci-dessous).
#
#   Lancement :                sbatch sbatch_download.sh
#   Reprise d'une requête :    sbatch sbatch_download.sh <QUERY_ID>
#
#   Pour une longue période (plusieurs années), préférer l'enchaînement annuel
#   (une requête Swissdox par année, jamais en parallèle) :
#       sbatch sbatch_download_yearly.sh 2000 2025
#
#   Sortie : data/input/download_job<ID>/swissdox_articles_lead.{parquet,csv}
#   puis data/input/.last_download pointe dessus et sbatch_step2.sh est soumis.
# ═══════════════════════════════════════════════════════════════════════════

START="2025-01-01"
END="2025-12-31"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -eo pipefail

# Logs écrits immédiatement dans le .out (sinon Python les garde en mémoire)
export PYTHONUNBUFFERED=1

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO"
cd "${REPO_DIR}"

# ── OUTPUT: dossier dédié à ce run (son propre job id) ────────────────────
OUTDIR="${REPO_DIR}/data/input/download_job${SLURM_JOB_ID}"
mkdir -p logs "${OUTDIR}"

# module/dcsrsoft scripts reference unset vars internally, so -u must stay off until after they load
module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

python --version
which python

export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# ── Reprise optionnelle : sbatch sbatch_download.sh <QUERY_ID> ────────────
# Si un job précédent a échoué APRÈS avoir soumis la requête, son .out contient
# "[Swissdox] queryId=XXXX". Passer cet ID reprend la même requête sans la renvoyer.
QUERY_ID="${1:-}"
QUERY_ID_ARGS=()
if [ -n "$QUERY_ID" ]; then
    echo "[INFO] Reprise de la requête Swissdox existante: $QUERY_ID"
    QUERY_ID_ARGS=(--query-id "$QUERY_ID")
fi

python scripts/download.py \
  --start "${START}" \
  --end "${END}" \
  --max-results 1000000 \
  --max-wait-hours 22 \
  --outdir "${OUTDIR}" \
  ${QUERY_ID_ARGS[@]+"${QUERY_ID_ARGS[@]}"}

echo "Job finished. Output: ${OUTDIR}"

# ── Auto-chain ─────────────────────────────────────────────────────────────────
DOWNLOADED_FILE="${OUTDIR}/swissdox_articles_lead.parquet"
if [[ ! -f "$DOWNLOADED_FILE" ]]; then
    echo "[ERROR] Fichier téléchargé introuvable : ${DOWNLOADED_FILE}" >&2
    exit 1
fi
echo "${DOWNLOADED_FILE}" > "${REPO_DIR}/data/input/.last_download"
sbatch "${REPO_DIR}/sbatch_step2.sh" "${DOWNLOADED_FILE}"
echo "[chain] → sbatch_step2.sh submitted (input: ${DOWNLOADED_FILE})"
