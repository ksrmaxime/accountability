"""
scripts/make_gold400_subset.py

Builds a small step-2-shaped input file containing ONLY the 400 (article_id,
keyword) rows used in the gold-standard sample, so the pipeline can be run
end-to-end on just those 400 rows instead of the full ~25,790-row corpus
while iterating on prompt changes.

Why this exists: step3/4a/4b/5/6 each expect an input parquet/csv in the
exact shape produced by step2_clean_targets.py (one row per article+target,
with `article_id` + `keyword` uniquely identifying each row -- see that
script's own dedup step). This script does nothing more than filter that
file down to the 400 rows the gold-standard CSV already scored, keeping
every original column and dtype untouched, so step3's prompt-building code
sees exactly the same row shape it always does.

It does NOT touch the gold CSV's own columns (keyword_answer, justification,
etc.) -- only article_id/keyword are used to select which rows to keep. The
output has zero gold-standard columns in it; it's a pure subset of step 2's
own output.

Usage (on the cluster, from the repo root):
  python scripts/make_gold400_subset.py \
      --step2_input data/processed/step2_clean.parquet \
      --gold_csv    data/output/Accountability_GOLD_sample400.csv \
      --output      data/processed/step2_clean_GOLD400.parquet

Then feed that file as the first positional argument to any *_array.sh
sbatch script instead of the default full-corpus step2_clean.parquet (see
docs/gold_validation_runs.md for the full recipe, including how to keep the
array jobs to a single task instead of spreading 400 rows over 9 GPUs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step2_input", default="data/processed/step2_clean.parquet",
                     help="Full-corpus step 2 output (parquet or csv).")
    ap.add_argument("--gold_csv", default="data/output/Accountability_GOLD_sample400.csv",
                     help="The gold-standard sample CSV (needs article_id + keyword columns).")
    ap.add_argument("--output", default="data/processed/step2_clean_GOLD400.parquet",
                     help="Where to write the filtered subset.")
    args = ap.parse_args()

    step2_path = Path(args.step2_input)
    gold_path = Path(args.gold_csv)

    if not step2_path.exists():
        print(f"[ERROR] step2 input not found: {step2_path}", file=sys.stderr)
        return 1
    if not gold_path.exists():
        print(f"[ERROR] gold CSV not found: {gold_path}", file=sys.stderr)
        return 1

    step2 = (pd.read_parquet(step2_path) if step2_path.suffix == ".parquet"
              else pd.read_csv(step2_path, low_memory=False))
    gold = pd.read_csv(gold_path, low_memory=False)

    for col in ("article_id", "keyword"):
        if col not in step2.columns:
            print(f"[ERROR] '{col}' missing from step2 input columns: {list(step2.columns)}", file=sys.stderr)
            return 1
        if col not in gold.columns:
            print(f"[ERROR] '{col}' missing from gold CSV columns: {list(gold.columns)}", file=sys.stderr)
            return 1

    step2 = step2.copy()
    gold = gold.copy()
    step2["_join_id"] = step2["article_id"].astype(str).str.strip() + "\x1f" + step2["keyword"].astype(str).str.strip()
    gold["_join_id"] = gold["article_id"].astype(str).str.strip() + "\x1f" + gold["keyword"].astype(str).str.strip()

    gold_ids = set(gold["_join_id"])
    subset = step2[step2["_join_id"].isin(gold_ids)].drop(columns=["_join_id"]).copy()

    found_ids = set(step2.loc[step2["_join_id"].isin(gold_ids), "_join_id"])
    missing = gold_ids - found_ids

    print(f"[make_gold400_subset] step2 input: {len(step2):,} rows")
    print(f"[make_gold400_subset] gold CSV:    {len(gold):,} rows ({len(gold_ids):,} distinct article_id+keyword pairs)")
    print(f"[make_gold400_subset] matched:     {len(subset):,} rows")

    if missing:
        print(f"[WARNING] {len(missing):,} gold (article_id, keyword) pairs were NOT found in {step2_path}.")
        print("          This means step2_clean.parquet has changed since the gold sample was drawn,")
        print("          or the gold CSV's article_id/keyword values don't match step2's exactly.")
        print("          The output below is missing these rows -- fix before trusting a comparison against it.")
        for jid in list(missing)[:10]:
            aid, kw = jid.split("\x1f")
            print(f"    missing: article_id={aid}  keyword={kw}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if len(subset) != len(gold_ids):
        dup_count = len(subset) - len(found_ids)
        if dup_count > 0:
            print(f"[WARNING] {dup_count:,} extra row(s) matched -- step2_clean.parquet likely has "
                  f"duplicate (article_id, keyword) rows that step2_clean_targets.py's own dedup should "
                  f"normally prevent. Investigate before using this file.")

    args_output = Path(args.output)
    args_output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args_output, index=False)
    print(f"[make_gold400_subset] -> {args_output}  ({len(subset):,} rows, {len(subset.columns)} columns)")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
