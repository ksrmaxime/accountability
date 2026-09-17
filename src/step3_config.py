# src/step3_config.py
"""
STEP 3 — criticism detection: row selection.

The legacy src/run3_config.py masked rows on a NOT-YET-EXPLODED file (one
row per article, a pipe-separated `matched_keywords` column) — run3 itself
did the exploding into one row per (article, keyword) before running the
LLM.

That exploding, plus the alias canonicalisation and target typing, now
happens once in step 2. Step 3's input is already one row per (article,
target) with a single, valid `keyword` value on every row, so the only
thing left to guard against here is a missing/empty article text.
"""
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows with non-empty article text and a non-empty target keyword."""
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    has_keyword = df["keyword"].notna() & (df["keyword"].astype(str).str.strip() != "")
    return has_text & has_keyword
