#!/bin/bash -l
#SBATCH --job-name=step4b_logit_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step4b_logit_array_%A_%a.out
#SBATCH --error=logs/step4b_logit_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8   # 9 taches (0-8) -- meme decoupage que le 3-pass

dcsrsoft use 20241118

set -euo pipefail

# =============================================================================
# step 4b -- REDESIGN "logit-probing" (2026-10-26+), en pause par rapport au
# design a 3 passes (sbatch_step4b_array.sh, toujours disponible tel quel).
#
# Au lieu de generer une reponse texte parmi 6 categories, ce script pose 6
# questions OUI/NON independantes (une par categorie) et lit directement les
# logits du prochain token (A/B) -- pas de generation, pas de parsing qui
# peut echouer, pas de retry/force. Voir scripts/step4b_source_category_logit.py
# et src/step4b_logit_prompts.py / src/step4b_logit_utils.py pour le detail.
#
# 2026-10-26 "summarize-then-probe" -- AJOUT d'un pre-pass de generation (le
# seul pas de ce script qui genere du texte) : on demande d'abord au modele un
# resume factuel de la critique (ce qui est reproche, et par qui), puis les 6
# questions OUI/NON de la passe logit lisent CE RESUME (source_summary) plutot
# que l'article brut -- objectif : reduire le bruit de l'article entier, qui
# semble avoir plafonne la precision du design direct-article (49.3%, job
# 66603355) au niveau du 3-pass. Nouveaux parametres SUMMARY_* ci-dessous.
# Resultat (job66603934) : 65.2% -- puis "boundary sharpening" des definitions
# de categories (job66608678, 62.3% -- regression en bonne partie du bruit,
# cf. src/step4b_logit_prompts.py pour le detail).
#
# 2026-10-27 "validate-and-correct" -- AJOUT d'une PASSE 3 : pour chaque ligne
# resolue par la passe 2, on relit LE MEME resume avec UNE question logit
# holistique ("la passe 2 a conclu categorie Y -- vrai ou faux ?"), meme
# machinerie que la passe 2 (un seul forward pass, order-swap, calibration
# propre a cette question -- voir compute_validation_calibration dans
# scripts/step4b_source_category_logit.py). Si la passe 3 repond FAUX, la
# ligne retombe sur le runner-up de la passe 2 (2e categorie la plus probable,
# deja capture par l'agregation de la passe 2) plutot que sur une 3e tentative.
# Une lecture de passe 3 inconcluante est fail-safe : le choix de la passe 2
# est conserve tel quel. Nouveaux parametres VALIDATION_* ci-dessous.
#
# IMPORTANT -- a faire AVANT de lancer ce sbatch (une seule fois, rapide) :
#   srun --pty --gres=gpu:1 --mem=24G --time=00:15:00 bash -l
#   module load python/3.12.1 && source .venv/bin/activate
#   python scripts/check_step4b_logit_tokens.py \
#       --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
#       --dtype bf16 --backend transformers --candidate_letters A,B
# Confirme que "A"/"B" sont bien des tokens uniques et des reponses
# plausibles pour ce tokenizer avant d'investir du temps GPU sur l'array.
# (Ce check porte sur le tokenizer, pas sur le contenu lu par la passe logit --
# pas besoin de le relancer juste pour le nouveau pre-pass summary.)
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO

# I/O
INPUT=${1:-"${WORKDIR}/data/output/step4a_merged/results.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/step4b_logit
TEXT_COL=text
N_ROWS=0        # 0 = toutes les lignes ; mettre ex. 20 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Pre-pass "summarize" -- generation classique (seule etape de ce script qui
# genere du texte), lit l'article brut (TEXT_COL) et produit un resume factuel
# de la critique et de sa source, stocke dans source_summary. Reutilise le
# meme runner generique que l'ancien design a 3 passes (src/runner.py).
SUMMARY_BATCH_SIZE=8
SUMMARY_TEMPERATURE=0.0
SUMMARY_MAX_NEW_TOKENS=220
SUMMARY_CHECKPOINT_EVERY=10

# Inference (passe logit) -- pas de temperature/max_new_tokens : une seule
# passe forward (lecture de logits), jamais de generation. batch_size compte
# des PROMPTS individuels (row x categorie x ordre), pas des lignes -- chaque
# ligne eligible (avec un resume exploitable) produit 6 categories x 2 ordres
# = 12 prompts, qui lisent source_summary et non l'article brut.
BATCH_SIZE=32
MAX_INPUT_TOKENS=16384
CANDIDATE_LETTERS=A,B   # changer ici si check_step4b_logit_tokens.py recommande une autre paire
CHECKPOINT_EVERY=200

# Passe 3 "validate-and-correct" -- meme machinerie logit que la passe 2
# (pas de temperature/max_new_tokens), reutilise CANDIDATE_LETTERS et
# MAX_INPUT_TOKENS ci-dessus. batch_size compte des prompts (row x ordre),
# 2 par ligne validee -- bien moins que la passe 2 (qui en compte 12/ligne).
#
# 2026-10-29 -- DESACTIVEE pour le moment (SKIP_VALIDATION=1). Sur
# job66672510 evalue contre le gold corrige (4 lignes mal etiquetees
# trouvees et corrigees le meme jour), la passe 3 reste net-negative
# (-1 : 0 corrigee, 1 cassee, 1 neutre) sans qu'aucune categorie n'en
# beneficie de facon fiable -- voir le changelog de
# src/step4b_logit_prompts.py et scripts/step4b_source_category_logit.py
# pour le detail de l'investigation (biais "Interest Group" corrige
# le 28/10, puis constat que les erreurs restantes sont des trous
# d'info au resume, pas des erreurs de raisonnement pass2/3). Remettre
# SKIP_VALIDATION=0 ci-dessous pour la reactiver sans rien supprimer :
# le flag --skip_validation est ajoute conditionnellement juste avant
# l'appel python, pas en dur.
VALIDATION_BATCH_SIZE=32
VALIDATION_CHECKPOINT_EVERY=200
SKIP_VALIDATION=${SKIP_VALIDATION:-1}

NUM_TASKS=${NUM_TASKS:-9}   # doit correspondre au nombre de taches dans --array

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/output

# -- Auto-chain : task 0 soumet le merge des le debut --
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "0" ]]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        "${WORKDIR}/sbatch_merge_step4b_logit.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_step4b_logit.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
fi

echo "=== SLURM ARRAY ==="
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>} TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

echo "=== RUN CONFIG ==="
echo "INPUT=${INPUT} | OUTPUT_BASE=${OUTPUT_BASE}"
echo "TEXT_COL=${TEXT_COL} | NUM_TASKS=${NUM_TASKS} | TASK=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL_PATH} | DTYPE=${DTYPE} | BACKEND=${BACKEND}"
echo "SUMMARY_BATCH_SIZE=${SUMMARY_BATCH_SIZE} | SUMMARY_TEMPERATURE=${SUMMARY_TEMPERATURE} | SUMMARY_MAX_NEW_TOKENS=${SUMMARY_MAX_NEW_TOKENS} | SUMMARY_CHECKPOINT_EVERY=${SUMMARY_CHECKPOINT_EVERY}"
echo "BATCH_SIZE=${BATCH_SIZE} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | CANDIDATE_LETTERS=${CANDIDATE_LETTERS}"
echo "VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE} | VALIDATION_CHECKPOINT_EVERY=${VALIDATION_CHECKPOINT_EVERY} | SKIP_VALIDATION=${SKIP_VALIDATION}"

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
  --dtype                     "$DTYPE" \
  --backend                   "$BACKEND" \
  --trust_remote_code \
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
  --num_tasks                 "$NUM_TASKS" \
  "${SKIP_VALIDATION_FLAG[@]}"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminee."
