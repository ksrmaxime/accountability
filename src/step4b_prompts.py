# src/step4b_prompts.py
"""
STEP 4b — source category (second half of what "step4" used to be).

Only runs on rows step 4a marked SOURCED. Classifies the actor(s) behind
the criticism into one of 6 broad categories — replacing the old pipeline's
attempt to pin down the exact person, which is no longer needed now that
only the broad group matters.

2026-09-29 update (superseded by the 2026-09-30 rewrite below): a gold
comparison found "Civil Servant" too narrowly defined and "Other" heavily
over-used as a catch-all. The fix at the time redefined "Civil Servant"
more broadly and appended a block of contrastive examples on top of the
existing flat category list.

2026-09-30 rewrite: running that version against the gold standard showed
the broadened "Civil Servant" definition had overcorrected in three
concrete ways, all visible in the model's own justifications: (a) it now
absorbed private-sector actors -- a company/association director or an
academic researcher -- who should stay "Interest Group"; (b) it pulled in
named officials who were actually presenting their institution's official
position (a report, an audit finding), which should stay "Administrative
Unit of the State" regardless of whether a named person delivered it; (c)
it pulled in a retired former official with no current state role, who
should be "Other". A fourth, pre-existing confusion (a senior but
non-elected civil servant mistaken for "Politician") was also visible.

Rather than layering a fourth caveat onto the existing flat list plus its
appendix of contrastive examples, the category legend below is rewritten
as a single ordered decision procedure: check categories 1-5 in order and
stop at the first one that fits, falling through to "Other" only when none
apply. This resolves (a) structurally, since "Interest Group" is checked
before "Civil Servant" ever comes into play, and folds (b), (c) into a
one-sentence qualifier inside the "Civil Servant" step itself instead of a
separate examples block -- there is no longer a standalone "contrastive
examples" section to keep in sync with the category definitions. The user
template's "identify who is speaking, then classify" pattern is unchanged.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a criticism is backed by the
testimony, statement, or quote of one or more specific actors. Classify
the actor whose criticism is most central to the article into ONE of the
following six categories. Go through them IN ORDER and stop at the first
one that fits -- do not skip ahead.

1. Politician — the actor holds an elected or appointed POLITICAL office
   (e.g. a federal/state/national councillor, a member of parliament, a
   party figure speaking in that capacity). A senior civil servant or
   agency head is NOT a politician, however senior their title sounds --
   check category 4 for them instead.

2. Interest Group — the actor speaks for or as a private company,
   business/professional association, union, lobby group, think tank, or
   an independent expert/researcher not employed by the state -- defending
   or reflecting its own economic, professional, or advocacy interest.

3. Administrative Unit of the State — the criticism is presented as the
   institution's own position rather than one person's opinion: an official
   report, audit, investigation, communique, or statement issued in the
   name of a department, administrative unit, or regulatory agency (e.g.
   "the Finance Delegation's report states...", "the Federal Audit Office
   found..."). This still applies when a named spokesperson or director
   delivers it, as long as they present the institution's formal position
   rather than their own personal opinion.

4. Civil Servant — a specific, named individual, CURRENTLY employed by the
   state (an official, agency director, auditor, diplomat, police officer,
   etc.), gives their own personal opinion or account -- distinct from a
   formal institutional position, which belongs to category 3 instead. A
   retired or former official with no current state role does not qualify
   here; use Other instead.

5. General Public — an ordinary citizen quoted with no other role, title,
   or affiliation.

6. Other — use only when none of categories 1-5 fit: an anonymous source
   with no institution named, a media outlet or journalist acting as the
   source in their own right, a foreign government or organisation, a
   retired/former official with no current state role, or any actor that
   genuinely matches nothing above. Do not use Other merely because the
   actor's exact role feels ambiguous -- work through categories 1-5 first.

If several actors are quoted and they belong to different categories,
classify the one whose criticism is most central to the article.\
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
# the verify system prompt repeats the same ordered decision procedure as
# SYSTEM_PROMPT above (kept in sync through the 2026-09-30 rewrite) so the
# second pass applies the same category boundaries, even though it only
# sees a one-sentence description instead of the article. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a one-sentence description of who is making
a criticism against a specific target. You are not shown the article
itself -- your only task is to read that description and classify the
actor it describes into ONE of the following six categories. Go through
them IN ORDER and stop at the first one that fits -- do not skip ahead.

1. Politician — the actor holds an elected or appointed POLITICAL office
   (e.g. a federal/state/national councillor, a member of parliament, a
   party figure speaking in that capacity). A senior civil servant or
   agency head is NOT a politician, however senior their title sounds --
   check category 4 for them instead.

2. Interest Group — the actor speaks for or as a private company,
   business/professional association, union, lobby group, think tank, or
   an independent expert/researcher not employed by the state -- defending
   or reflecting its own economic, professional, or advocacy interest.

3. Administrative Unit of the State — the criticism is presented as the
   institution's own position rather than one person's opinion: an official
   report, audit, investigation, communique, or statement issued in the
   name of a department, administrative unit, or regulatory agency. This
   still applies when a named spokesperson or director delivers it, as
   long as they present the institution's formal position rather than
   their own personal opinion.

4. Civil Servant — a specific, named individual, CURRENTLY employed by the
   state, gives their own personal opinion or account -- distinct from a
   formal institutional position, which belongs to category 3 instead. A
   retired or former official with no current state role does not qualify
   here; use Other instead.

5. General Public — an ordinary citizen quoted with no other role, title,
   or affiliation.

6. Other — use only when none of categories 1-5 fit, or the description
   does not clearly identify an actor. Do not use Other merely because the
   actor's exact role feels ambiguous -- work through categories 1-5 first.\
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
# pass 2, since there is no justification to verify. Left unchanged through
# the 2026-09-29 and 2026-09-30 updates since it affects a small fraction
# of rows and is deliberately kept minimal.

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
