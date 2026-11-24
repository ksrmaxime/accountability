# src/step5_prompts_remote.py
"""
STEP 5 -- content type classification, REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared Apertus-v1.5-70B
(via --backend litellm, the UNIL/CSCS proxy) against Apertus-8B-Instruct-2509
(local GPU) on step 3 and found the 70B remote model gives better results
there with a broader prompt (see src/step3_prompts_remote.py's own module
docstring), Maxime wants to move the rest of the pipeline (step5 onward)
onto the same remote backend. This file exists purely so that any future
prompt experimentation for step5's remote path never touches
src/step5_prompts.py, which must stay exactly as it is: the extensively-
tuned production wording for the local 8B/vllm/transformers path (see that
file's own module docstring for its full tuning history -- the pass-2
removal/reactivation/removal cycle documented there, the ENTITY->POLICY
boilerplate-echo bias, etc.).

As of this creation, NO WORDING HAS BEEN CHANGED ANYWHERE BELOW: SYSTEM_PROMPT,
build_user_prompt, VERIFY_SYSTEM_PROMPT and build_verify_prompt are all
byte-identical to src/step5_prompts.py at the time of this duplication --
only this module-level docstring (documentation, not prompt content) was
written fresh, replacing that file's own tuning-history docstring, which
documents a history specific to that file and not (yet) to this one. This
is intentionally just a safe, isolated starting point for later remote-
specific tuning, not a tuning pass itself.

scripts/step5_content_type.py picks this module instead of
src/step5_prompts.py only when --backend litellm is used; every other
backend (vllm/transformers) is completely unaffected, byte-for-byte, by
anything in this file.
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

  Policy  — the criticism targets a specific, identifiable PUBLIC POLICY
            CHOICE: a named bill, reform, regulation, budget, decision, or
            proposal the target is responsible for.

A specific policy, decision, or project being named in the article does NOT
by itself make this POLICY, and neither does your own justification simply
using the words "policy", "choice", or "decision" -- the decisive question
is what the criticism is actually about. Ask yourself: if the target had
executed this flawlessly, would the criticism disappear? If yes, the
criticism is about delivery, competence, or communication -> answer ENTITY,
even though a specific policy, decision, or project is the backdrop. If the
criticism would remain even with flawless execution -- because critics
object to the content or wisdom of the choice itself, and a different
decision should have been made -- answer POLICY.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is this criticism about a specific PUBLIC POLICY choice made by "{keyword}", or about "{keyword}" as an ENTITY (its behaviour, competence, efficiency, integrity, or a personal/institutional scandal)?

Naming a specific policy, project, or decision does not by itself decide the answer -- ask: would the criticism disappear if "{keyword}" had executed it flawlessly? If yes, this is ENTITY. If the criticism would remain even with flawless execution, this is POLICY.

For example: "The ministry's housing subsidy program is criticized for chronic processing delays and errors that left applicants waiting months." -> ENTITY: the criticism targets how poorly the program was run, not whether the subsidy itself was the right choice -- flawless execution would have resolved it.

First, in one sentence, justify your answer based on the article.
Then, on a new line, answer with exactly one word: POLICY or ENTITY.\
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
newspaper article and written a short explanation of the nature of a
criticism directed at a specific target. You are not shown the article
itself -- your only task is to read that explanation and classify the
criticism it describes into ONE of two categories.

=== CATEGORIES ===

  Policy — the explanation describes a disagreement about WHAT was
           decided, or what should be decided: the measure's relevance,
           usefulness, objectives, feasibility, the political motives for
           proposing or changing it, or its concrete effects/results.

  Entity — the explanation describes a disagreement about HOW the target
           carried out its public-policy tasks or its normal functioning:
           its competence, organization, representativeness, style,
           personality, or productivity.

Judge what the explanation's own reasoning is actually about, not just
whether it mentions the word "policy" or names a specific measure as
background -- an explanation can name a policy and still be about HOW it
was handled (Entity), or describe someone's conduct and still be about
WHAT was decided (Policy).\
"""

VERIFY_USER_TEMPLATE = """\
Here is an explanation, written by another analyst, of a criticism directed at "{keyword}":
"{justification}"

Based solely on this explanation, answer with exactly one word: POLICY or ENTITY.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)
