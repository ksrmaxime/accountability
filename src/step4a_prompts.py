# src/step4a_prompts.py
"""
STEP 4a — source attribution (first half of what "step4" used to be).

Small-model-friendly by design: a single binary question, answered in one
word, exactly like step 3's YES/NO. This replaces the old, much heavier
run4 -> run4eval -> run4arbitre chain (summarise, then detect, then
classify, cross-checked by extra verification prompts) built to pin down
the *exact person* criticising. That precision isn't needed anymore since
only the broad category of source matters now — so the question is split
into two small, separate steps instead: this one asks only whether a
specific external actor is quoted/referenced at all; step 4b (run only on
the rows answered SOURCED here) asks which broad category that actor
belongs to.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = ""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is this criticism backed by the testimony, statement, or quote of one or more specific external actors, or is it purely the journalist's own narrative framing, with no specific actor identified as making the criticism?

Answer with exactly one word:
SOURCED — if one or more specific actors are quoted, named, or clearly referenced as making the criticism.
JOURNALIST — if the criticism is presented as the article's own narrative, with no specific actor identified.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
