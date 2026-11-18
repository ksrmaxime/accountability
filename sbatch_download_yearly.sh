#!/bin/bash -l
#SBATCH --job-name=download_year
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_year_%j.out
#SBATCH --error=logs/download_year_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=FAIL

# ═══════════════════════════════════════════════════════════════════════════
# Download Swissdox UNE ANNÉE À LA FOIS, enchaîné automatiquement
#
#   Lancement (une seule commande pour toute la période) :
#       sbatch sbatch_download_yearly.sh 2000 2025
#
#   Chaque job télécharge UNE année, puis soumet lui-même le job de l'année
#   suivante. Les requêtes Swissdox sont donc strictement consécutives (jamais
#   deux en parallèle). Après la dernière année, le job de regroupement
#   (sbatch_download_merge_years.sh) est soumis automatiquement ; il soumet
#   ensuite sbatch_step2.sh.
#
#   Tout est rangé dans : data/input/download_years_chain<ID du 1er job>/
#       year_2000/  year_2001/  ...  (un sous-dossier par année)
#
#   Reprise après un échec (ex. l'année 2007 a échoué) : relancer à partir de
#   cette année en donnant le dossier de la chaîne ; les années déjà
#   téléchargées sont sautées :
#       sbatch sbatch_download_yearly.sh 2007 2025 <dossier de la chaîne>
# ═══════════════════════════════════════════════════════════════════════════

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -eo pipefail

# Logs écrits immédiatement dans le .out (sinon Python les garde en mémoire)
export PYTHONUNBUFFERED=1

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO"
cd "${REPO_DIR}"
mkdir -p logs

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

# ── Arguments ──────────────────────────────────────────────────────────────
YEAR="${1:-}"
END_YEAR="${2:-}"
CHAIN_DIR="${3:-}"
if ! [[ "$YEAR" =~ ^[0-9]{4}$ && "$END_YEAR" =~ ^[0-9]{4}$ ]] || [ "$YEAR" -gt "$END_YEAR" ]; then
    echo "[ERROR] Usage: sbatch sbatch_download_yearly.sh <ANNEE_DEBUT> <ANNEE_FIN> [DOSSIER_CHAINE]"
    exit 1
fi
if [ -z "$CHAIN_DIR" ]; then
    # 1er job de la chaîne : crée le dossier et note la période complète
    CHAIN_DIR="${REPO_DIR}/data/input/download_years_chain${SLURM_JOB_ID}"
    mkdir -p "$CHAIN_DIR"
    printf "start_year=%s\nend_year=%s\n" "$YEAR" "$END_YEAR" > "$CHAIN_DIR/chain_info.txt"
fi
mkdir -p "$CHAIN_DIR"
YEAR_DIR="${CHAIN_DIR}/year_${YEAR}"

echo "=== Download annuel : ${YEAR} (chaîne jusqu'à ${END_YEAR}) ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "CHAIN_DIR=${CHAIN_DIR}"

export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# ── Téléchargement de l'année (sauté si déjà fait) ──────────────────────────
if [ -f "${YEAR_DIR}/swissdox_articles_lead.parquet" ]; then
    echo "[INFO] ${YEAR} déjà téléchargée, on passe."
else
    mkdir -p "$YEAR_DIR"
    python scripts/download.py \
      --start "${YEAR}-01-01" \
      --end "${YEAR}-12-31" \
      --max-results 1000000 \
      --max-wait-hours 22 \
      --outdir "${YEAR_DIR}"
    echo "[INFO] ${YEAR} terminée → ${YEAR_DIR}"
fi

# ── Enchaînement ────────────────────────────────────────────────────────────
# (on n'arrive ici que si l'année a réussi : set -e arrête le script sinon)
if [ "$YEAR" -lt "$END_YEAR" ]; then
    NEXT=$((YEAR + 1))
    echo "[INFO] Soumission de l'année ${NEXT}…"
    sbatch "${REPO_DIR}/sbatch_download_yearly.sh" "$NEXT" "$END_YEAR" "$CHAIN_DIR"
else
    echo "[INFO] Dernière année faite. Soumission du regroupement…"
    sbatch "${REPO_DIR}/sbatch_download_merge_years.sh" "$CHAIN_DIR"
fi
