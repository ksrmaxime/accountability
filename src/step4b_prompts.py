# src/step4b_prompts.py
"""
STEP 4b — source category (second half of what "step4" used to be).

Only runs on rows step 4a marked SOURCED. Classifies the actor(s) behind
the criticism into one of 6 broad categories — replacing the old pipeline's
attempt to pin down the exact person, which is no longer needed now that
only the broad group matters.

The category definitions (SYSTEM_PROMPT) are UNCHANGED — only the user
template now asks for a one-sentence justification before the final
category label, so the model names who it thinks is speaking before
committing to a category (same pattern as steps 3, 4a, 5 and 6).
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a criticism is backed by the
testimony, statement, or quote of one or more specific external actors.
Your task is to classify those actors into ONE of the following categories.

=== CATEGORIES ===

  Interest Group                     — any company, association, union, lobby, or other
                                        external organisation defending its own specific
                                        interest and speaking in its own name.
  Civil Servant                      — a state employee speaking in a personal/field
                                        capacity rather than on behalf of their institution
                                        (e.g. a police officer's or nurse's firsthand account).
  General Public                     — an ordinary citizen quoted with no other role,
                                        title, or affiliation.
  Politician                         — an elected or appointed political figure from any
                                        party (e.g. a state councillor, federal councillor,
                                        national councillor, member of parliament).
  Administrative Unit of the State   — a department, administrative unit, or regulatory
                                        agency of the state that is itself the source of
                                        the criticism (e.g. another department, an
                                        oversight body).
  Other                              — none of the above categories apply.

If several actors are quoted and they belong to different categories, choose the
category of whichever actor's criticism is most central to the article.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized, and the criticism is backed by the \
testimony or statement of one or more specific actors.

ARTICLE:
{article_text}

First, in one sentence, identify who is making this criticism.
Then, on a new line, answer with exactly one of these category names, and nothing else:
Interest Group, Civil Servant, General Public, Politician, Administrative Unit of the State, Other.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify): classify the pass-1 justification alone, without the
# article. See src/verify_utils.py for why this second pass exists.
#
# Unlike step 3/4a/5/6 (binary choice), this is a 6-way classification --
# the verify system prompt repeats the full category legend (unchanged
# from SYSTEM_PROMPT above) so the second pass has the same category
# boundaries to work with, even though it only sees a one-sentence
# description instead of the article. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a one-sentence description of who is making
a criticism against a specific target. You are not shown the article
itself -- your only task is to read that description and classify the
actor it describes into ONE of the following categories.

=== CATEGORIES ===

  Interest Group                     — any company, association, union, lobby, or other
                                        external organisation defending its own specific
                                        interest and speaking in its own name.
  Civil Servant                      — a state employee speaking in a personal/field
                                        capacity rather than on behalf of their institution
                                        (e.g. a police officer's or nurse's firsthand account).
  General Public                     — an ordinary citizen quoted with no other role,
                                        title, or affiliation.
  Politician                         — an elected or appointed political figure from any
                                        party (e.g. a state councillor, federal councillor,
                                        national councillor, member of parliament).
  Administrative Unit of the State   — a department, administrative unit, or regulatory
                                        agency of the state that is itself the source of
                                        the criticism (e.g. another department, an
                                        oversight body).
  Other                              — none of the above categories clearly apply, or the
                                        description does not clearly identify an actor.\
"""

VERIFY_USER_TEMPLATE = """\
Here is a description, written by another analyst, of who is criticizing "{keyword}" in a newspaper article:
"{justification}"

Based solely on this description, answer with exactly one of these category names, and nothing else:
Interest Group, Civil Servant, General Public, Politician, Administrative Unit of the State, Other.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)
