# src/step4a_config.py
"""
STEP 4a — source attribution: row selection.

Only rows step 3 flagged as an actual criticism (`keyword_answer == "YES"`)
are sent to the LLM here. Every other row (NO, or not evaluated) stays in
the dataset untouched, per the pipeline's rule from step 4 onward: once an
(article, target) pair isn't a confirmed criticism, it is no longer
analysed by step 4 and later, but it is never dropped from the file.
"""
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows confirmed as criticism by step 3, with non-empty text."""
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    is_criticism = df["keyword_answer"].astype(str).str.strip().str.upper() == "YES"
    return has_text & is_criticism
