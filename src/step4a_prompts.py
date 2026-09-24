# src/step4a_prompts.py
"""
STEP 4a — source attribution (first half of what "step4" used to be).

A real full run showed this stage badly miscalibrated: out of ~13,000
confirmed-criticism rows, only 7 were labelled SOURCED, the rest almost all
JOURNALIST -- yet a spot check of JOURNALIST-labelled articles found named
actors, letters, and institutional sources quoted directly. The original
one-shot binary question ("is this backed by testimony... or is it purely
the journalist's own narrative framing...") packs two dense clauses into a
single sentence and asks for an immediate one-word answer with no reasoning
step; for an 8B model that combination collapses to "JOURNALIST" as a
default almost every time.

Fix: make the model search for and name a candidate actor FIRST (forcing it
to actually engage with the text), and only then commit to the SOURCED /
JOURNALIST label on a separate line -- the same justification-before-answer
pattern used everywhere else in this pipeline.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to determine whether that criticism is attributed to a
specific, identifiable actor (a named person, organization, or
institution), or whether it is simply the journalist's own narrative
framing, with no specific actor identified as making the criticism.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Look for a specific person, organization, or institution that is quoted, named, or clearly identified as making this criticism (for example: a named politician, an interest group, another authority, a civil servant, or a member of the public).

First, in one sentence, say who -- if anyone -- is identified as making the criticism, or state that no specific actor is identified.
Then, on a new line, answer with exactly one word:
SOURCED — if a specific actor is quoted, named, or clearly identified as making the criticism.
JOURNALIST — if no specific actor is identified and the criticism is presented as the article's own narrative.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify): classify the pass-1 justification alone, without the
# article. See src/verify_utils.py for why this second pass exists. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a short explanation of who, if anyone, is
identified as making a criticism. You are not shown the article itself --
your only task is to read that explanation and decide which final answer
it supports.\
"""

VERIFY_USER_TEMPLATE = """\
The question was: is the criticism of "{keyword}" attributed to a specific, identifiable actor, or is it just the journalist's own narrative framing?

Here is the explanation given by the other analyst:
"{justification}"

Based solely on this explanation, answer with exactly one word:
SOURCED — if the explanation names or identifies a specific actor making the criticism.
JOURNALIST — if the explanation says no specific actor is identified.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)


# --- Last resort (force): a bare one-word decision, no justification asked
# for at all, used only for rows that still have no parseable pass-1 answer
# after every retry of the full prompt above. Deliberately minimal -- the
# goal here is just to stop a row from being silently dropped, not to
# produce a reasoned judgment. Any row answered this way is flagged via
# source_attribution_forced=True (see scripts/step4a_source_attribution.py)
# and never goes through pass 2, since there is no justification to verify.

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one word, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is that criticism attributed to a specific, identifiable actor (a named person, organization, or institution), or is it just the journalist's own narrative framing with no such actor identified?

Answer with exactly one word: SOURCED or JOURNALIST.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
