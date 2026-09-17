# src/step4b_config.py
"""
STEP 4b — source category: row selection.

Only runs on rows step 4a marked SOURCED (a specific external actor is
quoted/referenced). Rows step 4a marked JOURNALIST, or that step 3 never
confirmed as criticism in the first place, are left untouched here.
"""
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows step 4a identified as backed by a specific external actor."""
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    is_sourced = df["source_attribution"].astype(str).str.strip().str.upper() == "SOURCED"
    return has_text & is_sourced
