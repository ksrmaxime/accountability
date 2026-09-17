# src/step3_prompts.py
"""
STEP 3 — criticism detection. Renamed 1:1 from the legacy src/run3_prompts.py;
prompt wording and logic are UNCHANGED on purpose (this stage already worked
well in the base pipeline).

One thing step 2 changed structurally: `keyword` is now a canonical name
that can unify several aliases of the same target (e.g. "EDA" and "DFAE"
both become "DFAE/EDA"). Showing that composite label to the LLM would be a
real behaviour change even though this file's prompt text never moved, so
the prompt uses `matched_alias` instead — the literal term step 2 actually
found in *this* article's own text — which is exactly what the legacy
pipeline showed the model. `keyword` remains available on every row for
grouping/counting once the LLM's answers come back.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = ""

USER_TEMPLATE = """\
You will receive an article to analyze. Your task is to tell if "{keyword}" is being criticized in this article. Criticism can be express even if the overall evaluation is positive in the article. The criticism can be about who it is or what it did. Answer ONLY by YES or NO nothing else

ARTICLE:
{article_text}\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
