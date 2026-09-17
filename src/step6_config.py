# src/step6_config.py
"""
STEP 6 — admin response detection: row selection.

Runs on every row step 3 confirmed as a criticism (`keyword_answer == "YES"`),
independently of what step 4a/4b/5 found about the source or content type of
that criticism — exactly like step 5, every confirmed-criticism row is
checked for whether the target responds.
"""
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows confirmed as criticism by step 3, with non-empty text."""
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    is_criticism = df["keyword_answer"].astype(str).str.strip().str.upper() == "YES"
    return has_text & is_criticism
