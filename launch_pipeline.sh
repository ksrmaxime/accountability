#!/bin/bash
# =============================================================================
# launch_pipeline.sh — Lance le pipeline complet en une seule commande.
#
# Usage :
#   bash launch_pipeline.sh              # pipeline complet depuis le download
#   bash launch_pipeline.sh step2        # depuis step2 (fichier téléchargé par défaut)
#   bash launch_pipeline.sh step2  /path/to/downloaded.parquet
#   bash launch_pipeline.sh step3  /path/to/step2_clean.parquet
#   bash launch_pipeline.sh step4a /path/to/step3_merged/results.parquet
#   bash launch_pipeline.sh step4b /path/to/step4a_merged/results.parquet
#   bash launch_pipeline.sh step5  /path/to/step4b_merged/results.parquet
#   bash launch_pipeline.sh step6  /path/to/step5_merged/results.parquet
#
# Chaque étape soumet automatiquement la suivante via --dependency=afterok
# (ou, pour step2 qui n'est pas un job array, directement après son succès).
# Un seul job actif à la fois — aucun superviseur, aucune limite de 3 jours.
#
# step6 (admin_response) est la dernière étape automatisée : son fichier
# fusionné (step6_merged_job<ID>/results.parquet) est le dataset final,
# analysé ensuite dans Stata en dehors de ce pipeline.
# =============================================================================

set -euo pipefail

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO
STEP=${1:-"download"}
INPUT=${2:-""}

cd "$WORKDIR"

case "$STEP" in
  download)
    JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_download.sh")
    echo "Pipeline lancé depuis : download  (job ${JOB_ID})"
    ;;
  step2)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step2.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step2.sh")
    fi
    echo "Pipeline lancé depuis : step2 (clean/classify targets)  (job ${JOB_ID})"
    ;;
  step3)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step3_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step3_array.sh")
    fi
    echo "Pipeline lancé depuis : step3 (criticism detection)  (array job ${JOB_ID})"
    ;;
  step4a)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step4a_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step4a_array.sh")
    fi
    echo "Pipeline lancé depuis : step4a (source attribution)  (array job ${JOB_ID})"
    ;;
  step4b)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step4b_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step4b_array.sh")
    fi
    echo "Pipeline lancé depuis : step4b (source category)  (array job ${JOB_ID})"
    ;;
  step5)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step5_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step5_array.sh")
    fi
    echo "Pipeline lancé depuis : step5 (content type)  (array job ${JOB_ID})"
    ;;
  step6)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step6_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_step6_array.sh")
    fi
    echo "Pipeline lancé depuis : step6 (admin response)  (array job ${JOB_ID})"
    ;;
  *)
    echo "Étape inconnue : '$STEP'"
    echo "Étapes valides : download | step2 | step3 | step4a | step4b | step5 | step6"
    exit 1
    ;;
esac

echo ""
echo "Suivi : squeue -u \$USER"
echo "Logs  : ls ${WORKDIR}/logs/"
