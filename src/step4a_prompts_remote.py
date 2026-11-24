# src/step4a_prompts_remote.py
"""
STEP 4a -- source attribution, REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared
Apertus-v1.5-70B (via --backend litellm, the UNIL/CSCS proxy) against
Apertus-8B-Instruct-2509 (local GPU) on step 3 and found the 70B remote
model gives better results there with a broader prompt (see
src/step3_prompts_remote.py's own module docstring), Maxime wants to move
the rest of the pipeline (step4a onward) onto the same remote backend.
This file exists purely so that any future prompt experimentation for
step4a's remote path never touches src/step4a_prompts.py, which must stay
exactly as it is: the extensively-tuned production wording for the local
8B/vllm/transformers path (see that file's own module docstring for its
full tuning history -- the removed/reverted pass-2 validation-on-
justification design, the 2026-11-17 "forced re-search" redesign, the
2026-11-18 pass-3 coherence check, etc.).

As of this creation, NO WORDING HAS BEEN CHANGED ANYWHERE BELOW: every
prompt string (SYSTEM_PROMPT, the pass-2 "forced re-search"
VERIFY_SYSTEM_PROMPT, FORCE_SYSTEM_PROMPT) and every worked example is
byte-identical to src/step4a_prompts.py at the time of this duplication --
only this module-level docstring (documentation, not prompt content) was
written fresh, replacing that file's own tuning-history docstring, which
documents a history specific to that file and not (yet) to this one. This
is intentionally just a safe, isolated starting point for later remote-
specific tuning, not a tuning pass itself.

scripts/step4a_source_attribution.py picks this module instead of
src/step4a_prompts.py only when --backend litellm is used; every other
backend (vllm/transformers) is completely unaffected, byte-for-byte, by
anything in this file. The pass-3 coherence check
(src/step4a_pass2_coherence_logit_prompts.py) is a separate, SHARED file,
deliberately NOT duplicated here -- same reasoning as step3's
src/step3_logit_prompts.py: a fixed, narrow validation question is meant
to act as the same independent, skeptical second read regardless of which
draft prompt (this file, or the original) produced pass 2's justification.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to determine whether this specific criticism of the target is
attributed to an outside voice -- a source external to the journalist,
however broadly or narrowly identified (a named person, a group, an
institution, a category of people, etc.) -- or whether it is simply the
journalist's own narrative framing, with no outside voice behind it.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Focus specifically on the criticism of "{keyword}": is it attributed to an
outside voice, or is it the article's own narrative framing?

An outside voice can be identified as loosely as a named individual, a
political party, a canton, an administration, a company, an association, a
category of people ("residents," "opponents," "experts"), or survey
respondents -- none of this needs a specific person's name to qualify.
"{keyword}" itself never counts as that voice just because it is named as
the one who made the decision, took the action, or is otherwise the
subject of the criticism.

Two things do NOT settle the question by themselves: a signed byline does
not mean the article has no outside sources behind the criticism of
"{keyword}" -- a signed piece can still quote or cite someone else's
criticism of the target. And sources quoted elsewhere in the article, for
other claims or about other people, do not count unless they are
specifically the ones credited with the criticism of "{keyword}" itself.

The answer is JOURNALIST only when, specifically for the criticism of
"{keyword}", no outside voice of any kind -- named or not -- is credited
with holding or voicing it: the judgment is presented as the article's own.

For example:
- "Residents interviewed for the piece say the new bus route is useless and badly planned." -> SOURCED: an outside group ("residents") is credited with the criticism, even though no individual is named.
- "This signed column, after quoting opposition MP Jane Doe calling the reform 'reckless', goes on to describe the minister's week." -> SOURCED: even though the piece is signed, the criticism itself is attributed to the quoted MP, not to the columnist.
- "The cantonal government and a coalition of municipalities jointly oppose the project, citing its excessive cost." -> SOURCED: a named collective is credited with the criticism, even with no individual inside it named.
- "A year-end roundup lists, among several federal councillors, that Martin Pfister had a difficult year." -> JOURNALIST: Pfister's name appears only as the subject being discussed; no one is credited with criticizing him.
- "The piece quotes an economist about the state of the labour market, then separately argues in its own voice that the minister's reform is misguided." -> JOURNALIST: an outside voice is quoted, but not for the criticism of the minister -- that judgment is the article's own.
- "The piece argues, in its own narrative voice, that the reform is short-sighted and will hurt small businesses, without attributing that judgment to any critic, resident, expert, or other outside voice." -> JOURNALIST: no outside voice is credited with the criticism at all.

First, in one sentence, say whether the criticism of "{keyword}" specifically is attributed to an outside voice (and who, even loosely), or whether it is the article's own narrative framing.
Then, on a new line, answer with exactly one word:
SOURCED — if the criticism of "{keyword}" is attributed to any outside voice, named or not.
JOURNALIST — if no outside voice is credited with that specific criticism and it is the article's own narrative framing.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify) -- 2026-11-17 redesign ("forced re-search"). See this
# file's own module docstring for the full history and rationale. Re-reads
# the FULL ARTICLE (never just pass 1's justification -- that was the old,
# reverted design's fatal blind spot), keeps pass 1's own already-tuned
# exclusions verbatim, and explicitly forces an enumerate-before-answer
# step to counter the diagnosed retrieval/attention failure. Scoped by the
# caller (scripts/step4a_source_attribution.py) to source_attribution ==
# "JOURNALIST" rows only -- this prompt's own wording assumes that. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs. A first, quick pass over \
a newspaper article already concluded that a specific criticism in it is not attributed to any \
outside voice. Your task is to re-examine the SAME article once more, more carefully, specifically \
to check whether that first pass missed a real outside voice.\
"""

VERIFY_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized. A first pass already concluded that this \
criticism is JOURNALIST -- the article's own narrative framing, with no outside voice credited.

ARTICLE:
{article_text}

An outside voice can be identified as loosely as a named individual, a political party, a canton, \
an administration, a company, an association, a category of people ("residents," "opponents," \
"experts"), or survey respondents -- none of this needs a specific person's name to qualify. \
"{keyword}" itself never counts as that voice just because it is named as the one who made the \
decision, took the action, or is otherwise the subject of the criticism. A signed byline does not \
mean the article has no outside sources behind the criticism of "{keyword}" -- a signed piece can \
still quote or cite someone else's criticism of the target. And sources quoted elsewhere in the \
article, for other claims or about other people, do not count unless they are specifically the \
ones credited with the criticism of "{keyword}" itself.

Before answering, go through the article once more and list, one by one, every named individual, \
political party, canton, administration, company, association, category of people, or other actor \
mentioned anywhere in connection with the criticism of "{keyword}" -- even if mentioned only \
briefly, in a quote, in an interview answer or question, or as part of a named group with no \
individual singled out. For each one you find, check specifically: are THEY the one credited with \
the criticism of "{keyword}", or are they mentioned for some other reason?

Then say in one or two sentences what you found (name the actor if you found one, or say plainly \
that the search found none), and finish on a new line with exactly one word:
SOURCED — if you found an outside voice credited with the criticism of "{keyword}" that the first pass missed.
JOURNALIST — if, even after this closer search, no outside voice of any kind is credited with that specific criticism.\
"""


def build_verify_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


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
