"""
scripts/step2_clean_targets.py

STEP 2 of the accountability pipeline — turns the raw Swissdox download
(one row = one article) into the clean, LLM-ready dataset (one row = one
article x target combination).

What this step does, in order:

  1. Drop duplicate articles: when two rows have the exact same (non-empty)
     title, keep only the one published first (earliest pubtime).
  2. Scan each remaining article's text for every department, administrative
     unit, independent agency and federal councillor the pipeline tracks.
     Matching is alias-aware and language-filtered (a German alias is never
     matched against a French article and vice-versa) and every alias of the
     same real-world entity (abbreviated/spelled-out, French/German) is
     collapsed to ONE canonical name — see src/target_taxonomy.py.
  3. Explode: one output row per (article, canonical target) pair. The same
     article appears multiple times if it mentions multiple targets; the
     same target's full canonical name appears only once per article no
     matter how many of its aliases were found in the text. Each row keeps
     both the canonical name (`keyword`, for grouping/counting) and the
     literal alias found in that article's own text (`matched_alias`, for
     anything shown to a human or fed into an LLM prompt).
  4. Classify each target: target_type (Federal Department / Administrative
     Unit / Independent Agency / Federal Councillor) and, where applicable,
     parent_dept (the department a unit/councillor belongs to).
  5. For Federal Councillor targets only: drop the row if the article's
     publication date falls outside that person's actual term of office
     (computed from the federal council composition history).

Usage
-----
  python scripts/step2_clean_targets.py \\
      --input  data/raw/swissdox_all_<timestamp>.parquet \\
      --output_dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.target_taxonomy import (
    FEDERAL_COUNCILLOR,
    classify_target,
    get_council_for_date,
    is_councillor_in_office,
    match_targets_detailed,
)


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def drop_duplicate_titles(df: pd.DataFrame, *, title_col: str = "title", pubtime_col: str = "pubtime") -> pd.DataFrame:
    """Keep one row per exact title (earliest pubtime wins). Rows with a
    missing/empty title are never considered duplicates of one another."""
    title = df[title_col].astype("string").str.strip()
    has_title = title.notna() & (title != "")

    with_title = df[has_title].copy()
    without_title = df[~has_title]

    with_title["_title_norm"] = title[has_title]
    with_title = with_title.sort_values(pubtime_col, na_position="last")
    before = len(with_title)
    with_title = with_title.drop_duplicates(subset="_title_norm", keep="first")
    with_title = with_title.drop(columns="_title_norm")
    dropped = before - len(with_title)

    out = pd.concat([with_title, without_title], ignore_index=False).sort_index()
    print(f"[dedup] {dropped:,} duplicate-title articles dropped "
          f"({before:,} -> {len(with_title):,} titled articles; "
          f"{len(without_title):,} untitled articles kept as-is)")
    return out


def explode_targets(df: pd.DataFrame, *, text_col: str, lang_col: str) -> pd.DataFrame:
    """One input row -> zero or more output rows, one per canonical target found.

    Each output row carries both:
      - keyword        the canonical target name (unified across FR/DE and
                        abbreviated/spelled-out aliases) — use this to group
                        or count occurrences of "the same target".
      - matched_alias   the literal alias actually found in this article's
                         own text (e.g. "EDA" in a German article) — use
                         this for anything shown to a human or an LLM, so it
                         reads naturally against the article it came from.
    """
    records = []
    for row in df.itertuples(index=False):
        row_d = row._asdict()
        text = row_d.get(text_col, "")
        lang = row_d.get(lang_col, "both")
        text = "" if pd.isna(text) else str(text)
        lang = "both" if pd.isna(lang) else str(lang)

        for canonical, matched_alias in match_targets_detailed(text, lang):
            new_row = dict(row_d)
            new_row["keyword"] = canonical
            new_row["matched_alias"] = matched_alias
            records.append(new_row)

    exploded = pd.DataFrame.from_records(records)
    print(f"[explode] {len(df):,} articles -> {len(exploded):,} article-target rows "
          f"({exploded['keyword'].nunique() if len(exploded) else 0} distinct targets)")
    return exploded


def classify_and_filter_councillors(df: pd.DataFrame, *, pubtime_col: str = "pubtime") -> pd.DataFrame:
    target_types = []
    parent_depts = []
    for keyword in df["keyword"]:
        info = classify_target(keyword)
        target_types.append(info["target_type"])
        parent_depts.append(info["parent_dept"])
    df = df.copy()
    df["target_type"] = target_types
    df["parent_dept"] = parent_depts

    is_councillor = df["target_type"] == FEDERAL_COUNCILLOR
    in_office = pd.Series(True, index=df.index)
    in_office[is_councillor] = [
        is_councillor_in_office(name, pubtime)
        for name, pubtime in zip(df.loc[is_councillor, "keyword"], df.loc[is_councillor, pubtime_col])
    ]
    # Fill parent_dept for councillor rows that are in office (department they headed at pubtime).
    for idx in df.index[is_councillor]:
        if in_office[idx]:
            comp = get_council_for_date(df.at[idx, pubtime_col])
            dept = next((d for d, name in comp.items() if name == df.at[idx, "keyword"]), None)
            df.at[idx, "parent_dept"] = dept

    dropped = int((is_councillor & ~in_office).sum())
    print(f"[councillors] {dropped:,} rows dropped (article published outside the "
          f"named councillor's term of office)")
    return df[in_office].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Step 1 (download) output: parquet or csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--lang_col", default="language")
    ap.add_argument("--pubtime_col", default="pubtime")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step2] Reading {in_path} ...")
    df = load(in_path)
    print(f"[step2] {len(df):,} articles loaded")

    df = drop_duplicate_titles(df, title_col=args.title_col, pubtime_col=args.pubtime_col)
    exploded = explode_targets(df, text_col=args.text_col, lang_col=args.lang_col)
    final = classify_and_filter_councillors(exploded, pubtime_col=args.pubtime_col)

    # Safety net: article_id is expected to uniquely identify one article, but
    # a handful of Swissdox entries reuse the same article_id for a live
    # article that was re-titled after publication (same id, different
    # title) — title-based dedup above does not catch these since the
    # titles differ. Guard the "one row per article-target pair" guarantee
    # regardless, keeping the earliest-published version.
    before = len(final)
    final = final.sort_values(args.pubtime_col, na_position="last")
    final = final.drop_duplicates(subset=["article_id", "keyword"], keep="first")
    reused_id_dropped = before - len(final)
    if reused_id_dropped:
        print(f"[dedup] {reused_id_dropped:,} rows dropped for reused article_id "
              f"(same article_id, same target, different title — likely a re-titled "
              f"live article; kept the earliest-published version)")

    # Column order: identifiers + classification first, article content/metadata after.
    front = ["article_id", "keyword", "matched_alias", "target_type", "parent_dept"]
    other = [c for c in final.columns if c not in front]
    final = final[[c for c in front if c in final.columns] + other]

    out_parquet = out_dir / "step2_clean.parquet"
    out_csv = out_dir / "step2_clean.csv"
    final.to_parquet(out_parquet, index=False)
    final.to_csv(out_csv, index=False)

    print(f"\n[step2] Final dataset: {len(final):,} article-target rows "
          f"({final['article_id'].nunique():,} distinct articles, "
          f"{final['keyword'].nunique():,} distinct targets)")
    print(final["target_type"].value_counts().to_string())
    print(f"[step2] Saved -> {out_parquet}")
    print(f"[step2] Saved -> {out_csv}")

    pointer = out_dir / ".last_step2"
    pointer.write_text(str(out_parquet), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
