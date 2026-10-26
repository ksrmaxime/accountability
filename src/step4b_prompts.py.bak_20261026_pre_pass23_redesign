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

2026-10-12 update: the corrected 800-row gold (job65610203) found this
step's errors clustering around inconsistent handling of articles where
several actors from different categories are quoted -- the model tends to
settle on whichever actor it notices first rather than checking the full
ordered hierarchy against every actor present, and SYSTEM_PROMPT's own
"most central" tie-break (a subjective criterion with no fixed order) was
likely reinforcing that. The tie-break is replaced with an explicit
instruction to enumerate every criticizing actor across the whole article
first, then apply categories 1-6 in order to that full set. No change to
the six category definitions themselves or to VERIFY_SYSTEM_PROMPT/
VERIFY_USER_TEMPLATE, which never had this tie-break sentence.

2026-10-22 rewrite: a manual, category-by-category review of 68 job66237092
rows where both gold and Apertus assigned a category (see session notes)
found two separate problems, one definitional and one structural.

Definitional problem -- the old boundaries were driven by job-title pattern
matching ("agency director" -> Civil Servant, "party name mentioned" ->
Interest Group) rather than by a single, consistent underlying question.
The user's own working definitions make that underlying question explicit:
for every actor, ask IN WHOSE NAME they are speaking. That single test
cleanly resolves several of the recurring confusions below:
  - A cantonal/communal executive member (Regierungsrat / conseiller
    d'Etat / Stadtrat) was twice misclassified as Civil Servant purely
    because their job title sounded like an agency head. Under the "in
    whose name" test, the same person is Politician when giving their own
    opinion/stance, and Administrative Unit of the State when their
    office issues a formal institutional position -- never Civil Servant,
    which the rewrite below now reserves for state employees NOT speaking
    for their institution (a police officer testifying, a civil servant
    giving a personal account).
  - Independent researchers/economists with no state employer were twice
    pulled into Civil Servant (a recurrence of the exact 2026-09-30
    regression). The rewrite follows the user's redefinition and routes
    an unaffiliated expert/researcher to General Public instead of
    Interest Group -- Interest Group is reserved for someone speaking for
    an organized entity (company, association, union, lobby, NGO); an
    individual expert with no such affiliation is closer to an informed
    citizen than to an organization defending its own interest.
  - "Other" had drifted into a catch-all for any actor whose role felt
    atypical (a former official, a political commentator, a private
    investor), which buried the one thing it should mean: a criticism
    that is clearly present in the article but whose source is vague or
    unnamed (e.g. "X has faced mounting criticism in recent weeks",
    nobody named, the journalist not taking ownership either). The
    rewrite narrows "Other" back to that, and sends an identifiable actor
    with no current institutional role (a retired official, an
    independent consultant) through the same "in whose name" test as
    everyone else -- typically landing on General Public, since they are
    then speaking only for themselves. A FORMER politician is the one
    exception kept: the gold standard consistently keeps e.g. a former
    Federal Councillor as Politician (their public political identity
    persists after leaving office in a way an ex-civil-servant's
    institutional authority does not), so Politician's definition below
    explicitly covers this.
  - A straightforward subject/object confusion recurred several times
    (the model names the CRITICIZED target itself, "{keyword}", as if it
    were the source of the criticism). The user template now says
    explicitly that the actor to classify is never "{keyword}".

Structural problem -- of the five "Administrative Unit of the State" gold
rows Apertus got wrong, four shared one exact pattern: an official
institutional statement (a court ruling, an audit, a report) co-occurred
in the same article with a lower-numbered-category actor (an individual
politician's quote, a private expert's opinion), and the 2026-10-12
"lowest-numbered category that fits ANY actor wins" tie-break mechanically
handed the answer to that other actor -- even though gold consistently
treated the formal institutional statement as the more central critique
regardless of its position in the list. The category order was never
designed as a priority ranking; using it as one via the tie-break
systematically fought the gold standard's own judgment. The fix is to move
"Administrative Unit of the State" to position 1 so an official
institutional position wins the tie-break whenever one is present,
matching what the gold data actually does. Politician, Interest Group,
Civil Servant, General Public keep their relative order after it; Other
remains the fallback.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a criticism is backed by the
testimony, statement, or quote of one or more specific actors. Classify
the actor whose criticism is most central to the article into ONE of the
following six categories.

For every actor you consider, ask first: IN WHOSE NAME are they speaking --
their own, their institution's, an organization's, or no one's in
particular? That question is what separates the categories below. Go
through them IN ORDER and stop at the first one that fits -- do not skip
ahead.

1. Administrative Unit of the State — the criticism (or the response to
   it) is presented as the FORMAL POSITION of a state institution, at any
   level -- a federal/cantonal/communal department or office, a city or
   commune, a cantonal or federal executive body, a court, a regulatory
   or oversight agency. This applies however the statement is delivered --
   an official report, audit, ruling, communique -- and ALSO when a named
   individual delivers it (a spokesperson, a director, or even an elected
   executive member), as long as that individual is speaking IN THE NAME
   of the institution rather than giving their own personal opinion. If
   an official institutional statement like this appears anywhere in the
   article, it wins this category even when a different actor (a
   politician's personal quote, an outside expert) is quoted more
   prominently elsewhere.

2. Politician — an elected or appointed political office-holder (a
   federal/cantonal/communal councillor, a member of parliament, a party
   leader, a cantonal or communal executive member) is speaking IN THEIR
   OWN NAME or political capacity -- their own opinion or stance, not
   their institution's official position (that was category 1). A
   political party issuing its own collective statement (a press release,
   a parliamentary group's position) also belongs here. The same person
   can be category 1 in one article and category 2 in another -- it
   depends only on whether they spoke for themselves/their party or for
   the institution they head. A FORMER elected politician (e.g. a retired
   Federal Councillor) still counts here when speaking about public
   affairs -- their political identity does not disappear on retirement
   the way an ex-civil-servant's institutional authority does.

3. Interest Group — the actor speaks FOR an organized entity representing
   a particular interest: a private company, business/professional
   association, union, lobby group, think tank, NGO, or an organized
   advocacy/protest group -- even an informal one, as long as it is acting
   as a collective, not a lone individual. Excludes political
   parties/politicians (category 2) and state administrations
   (category 1).

4. Civil Servant — a specific, named individual CURRENTLY employed by the
   state (an official, auditor, diplomat, police officer, etc.) gives
   their OWN personal opinion or account, explicitly NOT as their
   institution's formal position (that is category 1). E.g. a police
   officer testifying about what they personally witnessed, an employee
   voicing a personal grievance. A retired or former official with no
   current state role does NOT qualify here -- check category 5 instead.

5. General Public — an individual speaking only for themselves, with no
   institution or organized interest behind them: an ordinary citizen, a
   survey respondent, an independent expert or researcher with no
   organizational affiliation being represented, or a formerly
   state-employed person (other than a former politician, see category 2)
   who no longer holds any current institutional role.

6. Other — use ONLY when a criticism is clearly present in the article but
   its source is vague or unnamed -- e.g. the article says "X has faced
   mounting criticism in recent weeks" without identifying who is
   criticizing, and the journalist is not taking personal ownership of the
   claim either. Do NOT use Other just because an identifiable actor's
   role feels unusual or hard to pin down -- apply the "in whose name are
   they speaking" test and work through categories 1-5 first; almost every
   named actor fits one of them.

If several actors are quoted, first identify every actor across the WHOLE
article who criticizes the target, then apply categories 1-6 IN ORDER to
that full set of actors -- the lowest-numbered category that fits ANY of
them wins, even if a different actor was quoted first or more
prominently.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized, and the criticism is backed by the \
testimony or statement of one or more specific actors.

ARTICLE:
{article_text}

First, in one sentence, identify who is making this criticism. This is always someone \
OTHER than "{keyword}" itself -- "{keyword}" is the target being criticized, never the \
source of the criticism.
Then, on a new line, answer with exactly one of these category names, and nothing else:
Administrative Unit of the State, Politician, Interest Group, Civil Servant, General Public, Other.\
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
# SYSTEM_PROMPT above (kept in sync through the 2026-09-30 and 2026-10-22
# rewrites) so the second pass applies the same category boundaries, even
# though it only sees a one-sentence description instead of the article. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a one-sentence description of who is making
a criticism against a specific target. You are not shown the article
itself -- your only task is to read that description and classify the
actor it describes into ONE of the following six categories.

For the actor described, ask first: IN WHOSE NAME are they speaking --
their own, their institution's, an organization's, or no one's in
particular? That question is what separates the categories below. Go
through them IN ORDER and stop at the first one that fits -- do not skip
ahead.

1. Administrative Unit of the State — the criticism is presented as the
   FORMAL POSITION of a state institution, at any level -- a department,
   office, city/commune, cantonal or federal executive body, a court, a
   regulatory or oversight agency. This still applies when a named
   spokesperson, director, or elected executive member delivers it, as
   long as they speak IN THE NAME of the institution rather than giving
   their own personal opinion.

2. Politician — an elected or appointed political office-holder (a
   federal/cantonal/communal councillor, a member of parliament, a party
   leader, a cantonal or communal executive member) speaking IN THEIR OWN
   NAME or political capacity, not their institution's official position
   (category 1). A political party's own collective statement also
   belongs here. A FORMER elected politician still counts here.

3. Interest Group — the actor speaks FOR an organized entity representing
   a particular interest: a private company, business/professional
   association, union, lobby group, think tank, NGO, or an organized
   advocacy/protest group. Excludes political parties/politicians and
   state administrations.

4. Civil Servant — a specific, named individual CURRENTLY employed by the
   state gives their OWN personal opinion or account, explicitly not as
   their institution's formal position (category 1). A retired or former
   official with no current state role does NOT qualify here -- use
   category 5 instead.

5. General Public — an individual speaking only for themselves: an
   ordinary citizen, a survey respondent, an independent expert or
   researcher with no organizational affiliation being represented, or a
   formerly state-employed person (other than a former politician) with
   no current institutional role.

6. Other — use only when the description does not clearly identify an
   actor, or the criticism's source is described as vague/unnamed. Do not
   use Other merely because the actor's exact role feels ambiguous -- work
   through categories 1-5 first.\
"""

VERIFY_USER_TEMPLATE = """\
Here is a description, written by another analyst, of who is criticizing "{keyword}" in a newspaper article:
"{justification}"

Based solely on this description, answer with exactly one of these category names, and nothing else:
Administrative Unit of the State, Politician, Interest Group, Civil Servant, General Public, Other.\
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
# the 2026-09-29, 2026-09-30 and 2026-10-22 updates since it affects a small
# fraction of rows and is deliberately kept minimal.

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one category name, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized, and the criticism is backed by the \
testimony or statement of one or more specific actors.

ARTICLE:
{article_text}

Classify the actor(s) making this criticism into ONE of these categories:
Administrative Unit of the State, Politician, Interest Group, Civil Servant, General Public, Other.

Answer with exactly one of these category names, and nothing else.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
