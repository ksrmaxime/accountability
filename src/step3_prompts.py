# src/step3_prompts.py
"""
STEP 3 — criticism detection.

Originally ported 1:1 from the legacy src/run3_prompts.py with the wording
left untouched. After a real full run, two problems showed up in the actual
results: (1) no justification was captured at all, so a wrong answer could
not be diagnosed, and (2) the model sometimes seems to flag "{keyword}" as
YES when it is actually the one doing the criticizing (or merely mentioned
near criticism of something else), not the one being criticized. This
version keeps the same core binary question but:
  - explicitly tells the model not to confuse "{keyword} is criticized"
    with "{keyword} is the one criticizing",
  - asks for a one-sentence justification BEFORE the final YES/NO answer,
    to force the small model to reason about the text instead of pattern-
    matching straight to a label (same "reasoning before conclusion"
    pattern already used in step 6, now applied consistently everywhere).

`matched_alias` is still used instead of the canonical `keyword` for the
same reason as before: it's the literal term step 2 found in *this*
article's own text.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You read a newspaper article and judge whether one specific, named entity
is criticized in it. Be careful to distinguish an entity being criticized
from an entity that is itself doing the criticizing, or one that is simply
mentioned in a neutral or unrelated context.\
"""

USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Your task is to determine whether "{keyword}" ITSELF is being criticized in this article -- for who it is or what it did or decided. Criticism can be present even if the article's overall tone is neutral or positive elsewhere.

Do not confuse this with a case where "{keyword}" is the one expressing criticism of someone or something else, or is only mentioned in passing -- neither of those counts as "{keyword}" being criticized.

First, in one sentence, briefly justify your answer based on the article.
Then, on a new line, answer with exactly one word: YES or NO.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
