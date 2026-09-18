# src/step5_prompts.py
"""
STEP 5 — content type: does the criticism target a specific public POLICY
choice, or the target as an ENTITY (its behaviour, competence, integrity,
efficiency, or a personal/institutional scandal)?

Simplified from the legacy run6 stage, which used a 4-way typology (Person /
Policy / Both / Unclear) with a long example bank. Only the Policy/Entity
distinction is needed now, so the model is asked a single forced-choice
question, in the same spirit as step 3's YES/NO and step 4a's SOURCED/
JOURNALIST.

The category definitions (SYSTEM_PROMPT) are UNCHANGED — only the user
template now asks for a one-sentence justification before the final
POLICY/ENTITY label (same pattern as steps 3, 4a, 4b and 6).
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to classify the NATURE of this criticism into ONE of two categories.

=== CATEGORIES ===

  Entity  — the criticism targets the target ITSELF: its behaviour, competence,
            efficiency, management, integrity, morality, or a personal or
            institutional scandal involving it or the people who run it.
            Key signal: no specific named policy, decision, or proposal is
            being criticized — the attack is on WHO the target is or HOW it
            behaves/performs, even if that behaviour has political consequences.

  Policy  — the criticism targets a specific, identifiable PUBLIC POLICY
            CHOICE: a named bill, reform, regulation, budget, decision, or
            proposal the target is responsible for.
            Key signal: you could name the specific policy or decision being
            criticized.

If the article criticizes both the target's policy choices AND its behaviour
or competence, choose whichever is the more central focus of the criticism.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is this criticism about a specific PUBLIC POLICY choice made by "{keyword}", or about "{keyword}" as an ENTITY (its behaviour, competence, efficiency, integrity, or a personal/institutional scandal)?

First, in one sentence, justify your answer based on the article.
Then, on a new line, answer with exactly one word: POLICY or ENTITY.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
