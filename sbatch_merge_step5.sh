#!/bin/bash -l
#SBATCH --job-name=merge_step5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_step5_%j.out
#SBATCH --error=logs/merge_step5_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

# Usage:
#   ARRAY_ID=$(sbatch --parsable sbatch_step5_array.sh)
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_step5.sh ${ARRAY_ID}
#
# Ou en une commande :
#   ARRAY_ID=$(sbatch --parsable sbatch_step5_array.sh) && \
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_step5.sh ${ARRAY_ID}

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_merge_step5.sh <ARRAY_JOB_ID>"
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/step5"
MERGED_PARQUET="${OUTDIR}/step5_merged_job${ARRAY_JOB_ID}.parquet"
MERGED_CSV="${OUTDIR}/step5_merged_job${ARRAY_JOB_ID}.csv"

echo "=== MERGE step5 array job ${ARRAY_JOB_ID} ==="
echo "DATE=$(date -Is)"

export OUTDIR OUTBASE ARRAY_JOB_ID MERGED_PARQUET MERGED_CSV

python3 - <<'PYEOF'
import sys
import os
import glob
import pandas as pd

outbase        = os.environ["OUTBASE"]
job_id         = os.environ["ARRAY_JOB_ID"]
merged_parquet = os.environ["MERGED_PARQUET"]
merged_csv     = os.environ["MERGED_CSV"]

# ---------------------------------------------------------------------------
# 1. Gather task outputs (one row per article+keyword, all rows preserved)
# ---------------------------------------------------------------------------
pattern = f"{outbase}_task*_job{job_id}.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*_job{job_id}.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] Aucun fichier trouve avec le pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
else:
    dfs = [pd.read_parquet(f) for f in files]

print(f"[merge] {len(files)} fichiers trouves:")
for f, df_f in zip(files, dfs):
    print(f"  {f}  ->  {len(df_f):,} lignes")

merged = pd.concat(dfs, ignore_index=True)

sort_cols = [c for c in ["article_id", "keyword"] if c in merged.columns]
if not sort_cols:
    sort_cols = [merged.columns[0]]
merged = merged.sort_values(sort_cols).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Save
# ---------------------------------------------------------------------------
merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes (une par article+keyword)")
print(f"[merge] Trie par: {sort_cols}")
print(f"[merge] -> {merged_parquet}")
print(f"[merge] -> {merged_csv}")

# Summary of answer distribution
answer_col = "content_type"
if answer_col in merged.columns:
    counts = merged[answer_col].value_counts(dropna=False)
    total = len(merged)
    print(f"\n[merge] Distribution {answer_col} ({total:,} lignes):")
    for ans, count in counts.items():
        pct = 100 * count / total
        print(f"  {ans}: {count:,} ({pct:.1f}%)")

PYEOF

# =============================================================================
# ARCHIVE -- merged results + prompts + sbatch under one run folder
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/step5_merged_job${ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"                    "${RUN_DIR}/results.csv"            || true
cp "$MERGED_PARQUET"                "${RUN_DIR}/results.parquet"        || true
cp "src/step5_prompts.py"               "${RUN_DIR}/prompts_used.py"        || true
cp "sbatch_step5_array.sh"       "${RUN_DIR}/sbatch_array_used.sh"   || true
cp "$0"                             "${RUN_DIR}/sbatch_merge_used.sh"   || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"
echo "  results.csv / results.parquet  (une ligne par article+keyword)"
echo "  prompts_used.py"
echo "  sbatch_array_used.sh / sbatch_merge_used.sh"

echo "Merge termine."
echo "${RUN_DIR}" > "${WORKDIR}/data/output/.last_step5_archive"

# -- Auto-chain -----------------------------------------------------------------
sbatch "${WORKDIR}/sbatch_step6_array.sh" "${MERGED_PARQUET}"
echo "[chain] -> sbatch_step6_array.sh submitted (input: ${MERGED_PARQUET})"
