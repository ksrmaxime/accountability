# src/step4b_prompts.py
"""
STEP 4b — source category (second half of what "step4" used to be).

Only runs on rows step 4a marked SOURCED. Classifies the actor(s) behind
the criticism into one of 6 broad categories — replacing the old pipeline's
attempt to pin down the exact person, which is no longer needed now that
only the broad group matters.

2026-09-29 update: a 400-row gold-standard comparison, plus the user's own
review of the full-corpus category distribution, found two related
problems with the original category definitions (now revised below):
  - "Civil Servant" was defined too narrowly (state employee speaking in a
    "personal/field capacity", illustrated only with a frontline example
    like a police officer or nurse). In practice, named officials speaking
    about their own institution's affairs -- an agency director, an
    auditor, a diplomat -- were being pushed elsewhere (often into
    "Administrative Unit of the State" or, when neither felt like a clean
    fit, into "Other") instead of into "Civil Servant", even though they
    are exactly the kind of named state employee this category is meant
    to cover.
  - "Other" was being over-used as a catch-all whenever the model was not
    confident an actor cleanly matched one of the five specific
    categories, rather than as a last resort.
The fix is a clearer, mutually-exclusive rule for telling "Administrative
Unit of the State" (the institution itself, or an anonymous/unattributed
statement made in its name) apart from "Civil Servant" (a specific NAMED
individual employed by the state, personally quoted -- about a field
matter or about institutional affairs, doesn't matter, as long as they are
not an elected/appointed political figure), an explicit instruction not to
default to "Other" out of uncertainty, and a short set of contrastive
examples grounded in the kind of cases that were actually observed being
misclassified. The user template's "identify who is speaking, then
classify" pattern is otherwise unchanged.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a criticism is backed by the
testimony, statement, or quote of one or more specific external actors.
Your task is to classify those actors into ONE of the following categories.

=== CATEGORIES ===

  Interest Group                     — a company, association, union, lobby group, or other
                                        organisation acting in its own economic or advocacy
                                        interest and speaking in its own name (not a state
                                        body).
  Civil Servant                      — a specific NAMED individual employed by or acting on
                                        behalf of the state (an official, agency director,
                                        auditor, diplomat, police officer, expert, etc.) who
                                        is personally quoted or interviewed giving their own
                                        account or opinion -- whether about a hands-on/field
                                        matter or about their institution's affairs, it makes
                                        no difference -- as long as they are not an elected or
                                        appointed political figure, and the criticism is
                                        presented as their own personal statement rather than
                                        a formal position issued in the institution's name.
  General Public                     — an ordinary citizen quoted with no other role,
                                        title, or affiliation.
  Politician                         — an elected or appointed political figure from any
                                        party (e.g. a state councillor, federal councillor,
                                        national councillor, member of parliament).
  Administrative Unit of the State   — the criticism is presented as coming from the
                                        institution itself, not from a named individual: an
                                        official report, audit, investigation, communiqué, or
                                        unattributed statement issued in the name of a
                                        department, administrative unit, or regulatory agency
                                        (e.g. "the Finance Delegation's report states...",
                                        "the Federal Audit Office found...").
  Other                              — use ONLY when the criticizing actor is anonymous with
                                        no institution named either, is a media outlet or
                                        journalist acting as the source in their own right
                                        (not merely reporting someone else's criticism), is a
                                        foreign government or foreign organisation, or
                                        otherwise clearly does not match any category above.
                                        Do NOT use Other merely because the actor's exact role
                                        feels ambiguous or hard to pin down -- in that case,
                                        pick the closest matching category using the rules
                                        above instead.

A few contrastive examples to fix the Administrative Unit / Civil Servant boundary,
the main source of confusion:
  - A financial audit office's official report criticizes a project -> Administrative Unit
    of the State (the institution itself, via an unattributed report, is the source).
  - The retired former director of that same audit office gives a personal interview
    about the same case -> Civil Servant (a specific named individual's own account).
  - The head of a federal procurement agency explains or defends a decision in an
    interview, under their own name -> Civil Servant, not Administrative Unit of the
    State, because it is a named person speaking personally, not an anonymous statement
    issued in the institution's name.
  - An unnamed "source close to the department", with no institution or individual
    identified -> Other.

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
# the verify system prompt repeats the full category legend (kept in sync
# with SYSTEM_PROMPT above, including the 2026-09-29 revision) so the
# second pass has the same category boundaries to work with, even though
# it only sees a one-sentence description instead of the article. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a one-sentence description of who is making
a criticism against a specific target. You are not shown the article
itself -- your only task is to read that description and classify the
actor it describes into ONE of the following categories.

=== CATEGORIES ===

  Interest Group                     — a company, association, union, lobby group, or other
                                        organisation acting in its own economic or advocacy
                                        interest and speaking in its own name (not a state
                                        body).
  Civil Servant                      — a specific NAMED individual employed by or acting on
                                        behalf of the state (an official, agency director,
                                        auditor, diplomat, police officer, expert, etc.) who
                                        is personally quoted or interviewed giving their own
                                        account or opinion -- whether about a hands-on/field
                                        matter or about their institution's affairs, it makes
                                        no difference -- as long as they are not an elected or
                                        appointed political figure, and the criticism is
                                        presented as their own personal statement rather than
                                        a formal position issued in the institution's name.
  General Public                     — an ordinary citizen quoted with no other role,
                                        title, or affiliation.
  Politician                         — an elected or appointed political figure from any
                                        party (e.g. a state councillor, federal councillor,
                                        national councillor, member of parliament).
  Administrative Unit of the State   — the criticism is presented as coming from the
                                        institution itself, not from a named individual: an
                                        official report, audit, investigation, communiqué, or
                                        unattributed statement issued in the name of a
                                        department, administrative unit, or regulatory agency
                                        (e.g. "the Finance Delegation's report states...",
                                        "the Federal Audit Office found...").
  Other                              — use ONLY when the description does not clearly
                                        identify an actor, points to an anonymous source with
                                        no institution named, a media outlet or journalist
                                        acting as the source in their own right, or a foreign
                                        government or organisation. Do NOT use Other merely
                                        because the actor's exact role feels ambiguous -- pick
                                        the closest matching category above instead.\
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


# --- Last resort (force): a bare category decision, no justification asked
# for at all, used only for rows that still have no parseable pass-1 answer
# after every retry of the full prompt above. Mirrors step4a's own force
# prompt (src/step4a_prompts.py) -- deliberately minimal, the goal is just
# to stop a row from being silently dropped, not to produce a reasoned
# judgment. Any row answered this way is flagged via source_category_forced
# = "TRUE" (see scripts/step4b_source_category.py) and never goes through
# pass 2, since there is no justification to verify. Left unchanged in the
# 2026-09-29 update since it affects a small fraction of rows and is
# deliberately kept minimal.

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one category name, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized, and the criticism is backed by the \
testimony or statement of one or more specific actors.

ARTICLE:
{article_text}

Classify the actor(s) making this criticism into ONE of these categories:
Interest Group, Civil Servant, General Public, Politician, Administrative Unit of the State, Other.

Answer with exactly one of these category names, and nothing else.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
