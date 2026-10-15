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

2026-10-07 update: a 400-row gold comparison (job65468999 chain) found the
model's two error directions were roughly balanced (ENTITY mistaken for
POLICY and vice versa), both traceable to the same cause -- the old "Key
signal" wording let the model decide ENTITY vs POLICY by whether a specific
policy/decision/project was merely nameable in the text, rather than by
what the criticism itself was about. That let "poor execution of a named
project" get called POLICY, and "a specific, nameable decision described in
behaviour/competence language" get called ENTITY. SYSTEM_PROMPT and
VERIFY_SYSTEM_PROMPT both replace the old per-category "Key signal" lines
with one shared test: content/wisdom of the choice -> POLICY, quality of
execution/delivery/communication -> ENTITY, regardless of whether a project
or policy is named as the backdrop.

2026-10-12 update: the corrected 800-row gold (job65610203) found the
2026-10-07 fix above has partly backfired -- 38 ENTITY->POLICY errors show
justifications that literally echo "policy choice" to justify POLICY, the
exact lexical shortcut that edit tried to eliminate, now apparently
reinforcing it instead (confirmed via draft-vs-final: 26/38 already wrong
at pass 1). SYSTEM_PROMPT and VERIFY_SYSTEM_PROMPT now add an explicit
warning against deciding from the word "policy"/"choice"/"decision"
itself, and reformulate the test as a counterfactual: would the criticism
disappear if the target had executed this flawlessly? (yes -> ENTITY,
no -> POLICY). USER_TEMPLATE also gets its first worked example (this
step previously had none, unlike steps 3/4a/4b/6) illustrating the ENTITY
side of that test, since the backdrop-execution confusion is the larger
and clearer of the two error directions.
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

  Entity  — the criticism targets the target ITSELF: its behaviour, competence,
            efficiency, management, integrity, morality, or a personal or
            institutional scandal involving it or the people who run it.

  Policy  — the criticism targets a specific, identifiable PUBLIC POLICY
            CHOICE: a named bill, reform, regulation, budget, decision, or
            proposal the target is responsible for.

A specific policy, decision, or project being named in the explanation does
NOT by itself make this POLICY, and neither does the explanation simply
using the words "policy", "choice", or "decision" -- judge what the
criticism is actually about. Ask: would the criticism disappear if the
target had executed this flawlessly? If yes, answer ENTITY, even if a
specific policy, decision, or project is mentioned as the backdrop. If the
criticism would remain even with flawless execution -- because it targets
the content or wisdom of the choice itself -- answer POLICY.\
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
