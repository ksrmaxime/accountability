# src/step5_config.py
"""
STEP 5 — content type (Policy vs Entity): row selection.

Runs on every row step 3 confirmed as a criticism (`keyword_answer == "YES"`),
regardless of what step 4a/4b found about the source of that criticism —
content type and source are independent dimensions of the same criticism.
"""
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows confirmed as criticism by step 3, with non-empty text."""
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    is_criticism = df["keyword_answer"].astype(str).str.strip().str.upper() == "YES"
    return has_text & is_criticism
