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

Two-pass design (added after the justification-first run showed the label
sometimes contradicting its own justification -- see src/verify_utils.py):

  Pass 1 (draft)  — reads the article, writes the justification, then a
    first-guess label right after it. Both are kept: the justification
    under `keyword_justification`, the first-guess label under
    `keyword_answer_draft` (QA only).
  Pass 2 (verify) — reads ONLY `keyword_justification` (no article) and
    classifies it into YES/NO. This is the FINAL label, saved as
    `keyword_answer` — the name every downstream step's mask uses.

This is the version that measured best on the 400-row gold comparison
(job65205108: accuracy=75.7%, precision=56.4%, recall=78.8%, F1=65.7%,
kappa=0.478).

2026-09-29/30 and 2026-10-01 updates (all superseded below): three rounds
of rewrites tried to fix two error patterns found in job65205108 -- (a)
false negatives when the thing actually criticized is a policy/project/
decision rather than "{keyword}" named directly, and (b) false positives
from reading a neutral or merely-factual statement as criticism. Every one
of these rewrites (two rounds of bolted-on exclusion rules in pass 1, one
unified-definition rewrite, and finally a full architectural change moving
the whole definition into a much longer pass 2 applied to a plain factual
summary instead of the article) measured *worse* than job65205108 on every
metric, the last one by a wide margin (job65277505: recall collapsed to
11.9% because pass 2 was asked to make the entire judgment, cold, under an
8-token budget with no room to reason -- see that run's analysis). A
manual review of all its false negatives also showed the pass-1 summary
itself was the bigger single cause (around 55% of cases), not just pass 2:
an unguided 1-2 sentence summary written by a small model tends to pick the
article's most generic facet rather than searching for a reproach that may
sit elsewhere in a long article.

2026-10-02, reverted + reinforced: back to the exact job65205108
architecture and wording below (full article in pass 1, one-sentence
justification + YES/NO label, pass 2 verifies that justification alone),
on the reasoning that job65205108's remaining errors are worth fixing
*from* its own wording rather than by replacing it again. Rather than
re-adding exclusion rules to pass 1 (the thing that made every later
rewrite worse -- a precise list of NO-cases reads, in practice, as "if a
case isn't on this list, it's YES"), this version adds a small, genuinely
balanced set of worked examples to each pass, calibrating both directions
at once instead of only warning against false positives or only against
false negatives:
  - Pass 1 gets three YES examples and three NO examples illustrating the
    SYSTEM_PROMPT's own existing distinction (criticized vs. criticizing,
    vs. merely mentioned), including one case of institutional
    responsibility (a project/policy run by the entity) so that pattern
    isn't systematically missed -- but as one example among six balanced
    ones, not as a dedicated bolted-on rule.
  - Pass 2's job is unchanged (classify the pass-1 justification alone,
    still without the article) and its prompt is still the short, simple
    original -- but it now gets a few worked examples specifically aimed
    at job65205108's main false-positive pattern: a justification that
    only speculates or infers a reproach ("this could be seen as...",
    "this suggests that...", "one might read this as...") without
    describing an actual one, which should be corrected to NO even when
    the justification's own stated conclusion was YES.

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

For example:
- "A parliamentary committee accuses the ministry of mismanaging a defense procurement project it runs, citing cost overruns." -> the ministry IS criticized: the reproach targets a project that is its own direct responsibility, even though the ministry itself is not named.
- "Critics say the mayor's housing policy has worsened the rent crisis in the city." -> the mayor IS criticized: a clear reproach targets her decision.
- "\"The agency has been negligent and failed to act in time,\" the auditor's report states." -> the agency IS criticized: a direct, explicit reproach.
- "The agency reports a 3% increase in applications this year." -> NOT criticized: a plain neutral fact, no reproach is expressed.
- "The minister dismisses the opposition's proposal as unrealistic." -> the minister is NOT criticized: here the minister is the one doing the criticizing, not the target.
- "The report notes that the department was informed of the problem and tasked with following up." -> NOT criticized: being informed or involved is not itself a reproach.

First, in one sentence, briefly justify your answer based on the article.
Then, on a new line, answer with exactly one word: YES or NO.\
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
newspaper article and written a short explanation of whether a specific
entity is criticized in it. You are not shown the article itself -- your
only task is to read that explanation and decide which final answer it
supports.\
"""

VERIFY_USER_TEMPLATE = """\
The question was: is "{keyword}" criticized in the article?

Here is the explanation given by the other analyst:
"{justification}"

Judge what the explanation actually describes, not just which label it ends with: an explanation that only speculates or infers a possible reproach, without describing one that is actually made in the article, does not support YES even if it concludes with YES.

For example:
- "The report could be seen as implying criticism of the agency." -> NO: this only speculates about a possible reading, it does not describe an actual reproach.
- "One might infer that the minister's policy is being questioned here." -> NO: an inference, not a reported criticism.
- "The committee explicitly accuses the ministry of mismanaging the project." -> YES: a concrete, actual reproach is described.
- "The article neutrally reports the agency's budget figures, with no reproach mentioned." -> NO: nothing to verify, no criticism is described.

Based solely on this explanation, answer with exactly one word: YES or NO.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)


# --- Last resort (force): a bare one-word decision, no justification asked
# for at all, used only for rows that still have no usable pass-1 answer
# after every retry. Mirrors step4a's own force prompt (src/step4a_prompts.py)
# -- deliberately minimal, the goal is just to stop a row from being
# silently dropped, not to produce a reasoned judgment. Any row answered
# this way is flagged via keyword_answer_forced = "TRUE" (see
# scripts/step3_criticism_detection.py) and never goes through pass 2.
# Unchanged across every rewrite this file has had.

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one word, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Is "{keyword}" ITSELF being criticized in this article -- for who it is or what it did or decided?

Answer with exactly one word: YES or NO.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
