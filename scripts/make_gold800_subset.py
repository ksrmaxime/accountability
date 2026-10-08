"""
scripts/make_gold800_subset.py

Builds a small step-2-shaped input file containing ONLY the 800 (article_id,
keyword) rows used in the expanded gold-standard sample (the original 400
plus the new 400 added across gold_batch 9-16), so the pipeline can be run
end-to-end on just those 800 rows instead of the full corpus.

This generalizes make_gold400_subset.py (see that script's own docstring
for the full rationale) to accept several gold CSVs at once and union their
(article_id, keyword) pairs before filtering step2's output. It does NOT
touch any gold-standard columns -- only article_id/keyword are used to
select which rows to keep, and every original step2 column/dtype is kept
untouched.

Usage (on the cluster, from the repo root):
  python scripts/make_gold800_subset.py \
      --step2_input data/processed/step2_clean.parquet \
      --gold_csv    data/output/Accountability_GOLD_sample400.csv \
      --gold_csv    data/output/Accountability_GOLD_new400_batch9-16.csv \
      --output      data/processed/step2_clean_GOLD800.parquet
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
    ap.add_argument("--gold_csv", action="append", required=True,
                     help="A gold-standard CSV (needs article_id + keyword columns). "
                          "Repeat this flag to combine several gold CSVs (e.g. the "
                          "original 400 and the new 400).")
    ap.add_argument("--output", default="data/processed/step2_clean_GOLD800.parquet",
                     help="Where to write the filtered subset.")
    args = ap.parse_args()

    step2_path = Path(args.step2_input)
    if not step2_path.exists():
        print(f"[ERROR] step2 input not found: {step2_path}", file=sys.stderr)
        return 1

    step2 = (pd.read_parquet(step2_path) if step2_path.suffix == ".parquet"
              else pd.read_csv(step2_path, low_memory=False))
    for col in ("article_id", "keyword"):
        if col not in step2.columns:
            print(f"[ERROR] '{col}' missing from step2 input columns: {list(step2.columns)}", file=sys.stderr)
            return 1
    step2 = step2.copy()
    step2["_join_id"] = step2["article_id"].astype(str).str.strip() + "\x1f" + step2["keyword"].astype(str).str.strip()

    gold_ids: set[str] = set()
    total_gold_rows = 0
    for gold_csv in args.gold_csv:
        gold_path = Path(gold_csv)
        if not gold_path.exists():
            print(f"[ERROR] gold CSV not found: {gold_path}", file=sys.stderr)
            return 1
        gold = pd.read_csv(gold_path, low_memory=False)
        for col in ("article_id", "keyword"):
            if col not in gold.columns:
                print(f"[ERROR] '{col}' missing from gold CSV columns ({gold_path}): {list(gold.columns)}", file=sys.stderr)
                return 1
        gold = gold.copy()
        gold["_join_id"] = gold["article_id"].astype(str).str.strip() + "\x1f" + gold["keyword"].astype(str).str.strip()
        this_ids = set(gold["_join_id"])
        overlap = this_ids & gold_ids
        if overlap:
            print(f"[WARNING] {len(overlap):,} (article_id, keyword) pairs in {gold_path} "
                  f"already appeared in a previously combined gold CSV -- they will be "
                  f"counted once in the combined set.")
        print(f"[make_gold800_subset] {gold_path}: {len(gold):,} rows ({len(this_ids):,} distinct pairs)")
        total_gold_rows += len(gold)
        gold_ids |= this_ids

    subset = step2[step2["_join_id"].isin(gold_ids)].drop(columns=["_join_id"]).copy()
    found_ids = set(step2.loc[step2["_join_id"].isin(gold_ids), "_join_id"])
    missing = gold_ids - found_ids

    print(f"[make_gold800_subset] step2 input:      {len(step2):,} rows")
    print(f"[make_gold800_subset] combined gold:     {total_gold_rows:,} rows across {len(args.gold_csv)} file(s), "
          f"{len(gold_ids):,} distinct article_id+keyword pairs")
    print(f"[make_gold800_subset] matched in step2:  {len(subset):,} rows")

    if missing:
        print(f"[WARNING] {len(missing):,} gold (article_id, keyword) pairs were NOT found in {step2_path}.")
        for jid in list(missing)[:10]:
            aid, kw = jid.split("\x1f")
            print(f"    missing: article_id={aid}  keyword={kw}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if len(subset) != len(gold_ids):
        dup_count = len(subset) - len(found_ids)
        if dup_count > 0:
            print(f"[WARNING] {dup_count:,} extra row(s) matched -- step2_clean.parquet likely has "
                  f"duplicate (article_id, keyword) rows. Investigate before using this file.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)
    print(f"[make_gold800_subset] -> {output_path}  ({len(subset):,} rows, {len(subset.columns)} columns)")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
