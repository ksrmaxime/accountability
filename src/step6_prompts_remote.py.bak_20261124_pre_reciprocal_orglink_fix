# src/step6_prompts_remote.py
"""
STEP 6 -- admin response detection, REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared Apertus-v1.5-70B
(via --backend litellm, the UNIL/CSCS proxy) against Apertus-8B-Instruct-2509
(local GPU) on step 3 and found the 70B remote model gives better results
there with a broader prompt (see src/step3_prompts_remote.py's own module
docstring), Maxime wants to move the rest of the pipeline (step6 onward)
onto the same remote backend. This file exists purely so that any future
prompt experimentation for step6's remote path never touches
src/step6_prompts.py, which must stay exactly as it is: the extensively-
tuned production wording for the local 8B/vllm/transformers path (see that
file's own module docstring for its full tuning history -- the pass-2
removal/reactivation/removal cycle documented there, the "organisational
link" paragraph addressing office/spokesperson confusion, etc.).

At creation, NO WORDING HAD BEEN CHANGED ANYWHERE BELOW: SYSTEM_PROMPT,
build_user_prompt, VERIFY_SYSTEM_PROMPT and build_verify_prompt (and the
unused strip_negative_boilerplate helper) were all byte-identical to
src/step6_prompts.py at the time of this duplication -- only the module-
level docstring (documentation, not prompt content) was written fresh.
See the entry just below for the first real wording change made to this
file.

2026-11-24 reinforcement pass (USER_TEMPLATE only -- pass 1, the only pass
this step's script actually calls): diagnosed from job20905947 (the first
full remote-chain run on GOLD800). step6's oracle accuracy on that run
(79.8%, n=89) was already the best of the 4 analyzed remote steps and
above the classic 8B baseline (62-72%), but Maxime asked for the 18
errors (15 FN, 3 FP) to be read anyway. Not a parsing bug (checked the
same way as step4a/step5 -- no opening/closing bare-word mismatch). One
of the 15 FN (DDPS/VBS, the Ritter interview row) is a gold annotation
error, not a pipeline error -- labelled YES but its own justification
says "Aucune reponse officielle du VBS... n'est rapportee"; flagged to
Maxime, left as-is in gold. Of the other 17:
  - ~6 FN are genuine retrieval misses: the model's own justification
    flatly states no response exists anywhere in the article (e.g.
    armasuisse, SEM, the SRC director), when gold quotes one that does.
    Maxime explicitly decided NOT to add a pass 2 re-read for this
    (unlike step4a's "forced re-search") -- these will likely persist.
  - ~3 FN are self-contradictions: the model's OWN justification already
    states the organisation's justification near-verbatim to gold's own
    (DETEC's privacy-harm reversal, the EFD/UBS manager-responsibility
    defense) and still answers NO; one (OFCL/BBL) reads as the model
    silently requiring the justification to come AFTER the criticism in
    the text, which USER_TEMPLATE already says not to require ("may
    appear anywhere in the article") but apparently not forcefully
    enough to survive to the final word.
  - 3 FN (Baume-Schneider declining a round table, the EFD filing a legal
    appeal, Cassis changing his public position) are gold counting a
    reported ACTION as itself a sufficient response, with no quoted
    statement -- directly against this prompt's own explicit exclusion
    ("the article merely reports that {keyword} took some action...
    without actually saying anything about it"). Maxime explicitly chose
    to KEEP this exclusion as-is (not change it to match gold's
    practice) -- these 3 are accepted as disagreements with gold, not
    something this prompt should chase.
  - All 3 FP share one shape: the model credits a real statement as a
    response, but it addresses a DIFFERENT topic than the actual reproach
    being evaluated (Rosti speaking on unrelated matters in the same
    piece, the NDB regretting only the leak and not the underlying
    unlawfulness finding), and once (OSAV/BLV) the credited statement is
    from a DIFFERENT, unlinked agency (Swissmedic) entirely -- a direct
    violation of this prompt's own existing organisational-link rule.
Fix below targets only the two patterns Maxime asked to address by
prompt wording alone (not the retrieval misses, not the action-as-
response gold disagreement, both explicitly left alone): (1) a new
paragraph requiring the credited response to address the SAME reproach
being evaluated, not a different topic or a different, unlinked entity,
with a worked negative example mirroring the Swissmedic/BLV mix-up; (2)
a closing check-before-answering reminder that a justification/defense
already identified in the model's own reasoning must be counted as YES
regardless of where in the article it appears or how it is framed --
aimed at the DETEC/EFD-style self-contradictions. Not yet re-run/
confirmed on real data -- see sbatch_step6_remote_test.sh for the next
test run.

scripts/step6_admin_response.py picks this module instead of
src/step6_prompts.py only when --backend litellm is used; every other
backend (vllm/transformers) is completely unaffected, byte-for-byte, by
anything in this file.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
Your task is to determine whether, in a newspaper article, the entity being
criticised responds to the criticism/reproach directed at them.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticised by {source}.

ARTICLE:
{article_text}

Your task is to determine whether, in the article, "{keyword}" gives a response \
to the reproach addressed to them.

Count as a response: a direct statement from "{keyword}" itself; a statement given on its behalf by an official spokesperson, a subordinate service, or a unit acting under its authority (for example, a specialised office responding in the name of the department it belongs to); or any explanation, justification, defense, acknowledgment of the problem, promise to fix or review it, or attempt to minimize its importance, regarding the decision or situation being criticized -- even if that statement is brief, general, or not explicitly framed as a reply to the criticism. The response may appear anywhere in the article, not only right after the criticism -- read the whole article before answering.

When such a response comes from an office, spokesperson, or minister rather than "{keyword}" by name, decide whether that speaker counts for "{keyword}" using only what the article itself states -- look for an explicit link written in the text (for example, "Bundesamt X von Bundesrat {keyword}", "{keyword}'s department/office", or a shared ministry named earlier in the article), not your own assumptions about how an administration is organised. If the article instead marks the speaker as belonging to a different level or place (a canton, a different country, an unrelated body) even under a similar-sounding title, that speaker's statement does not count for "{keyword}" -- this includes a different federal office with related subject-matter expertise (for example, a statement from a different, separately-named agency about a related technical question does not count for "{keyword}" unless the article itself explicitly ties that agency's statement to "{keyword}").

The statement you count must address the SAME reproach you are evaluating here, not a different criticism, a different topic, or only one narrow part of a broader accusation while leaving its actual substance untouched. "{keyword}" (or the article) covering other matters elsewhere in the piece, or reacting to only a side aspect of what is being criticized while ignoring the real substance of the reproach, does not count as a response to THIS reproach. For example, if "{keyword}" is accused of favouring one interest group but only speaks in the article about an unrelated procedural question, or if an office only expresses regret that an internal report was leaked without addressing what that report actually found, that does not count.

Do not count a response as present just because "{keyword}" is quoted or mentioned elsewhere on an unrelated matter, because another party unconnected to it comments on the criticism, or because the article merely reports that "{keyword}" took some action, changed course, or came under pressure, without "{keyword}" (or its spokesperson) actually saying anything about it.

Before you answer, check your own reasoning above: if you have identified ANY sentence, however brief or routine-sounding, where "{keyword}" (or a spokesperson/unit explicitly linked to it, as defined above) explains, justifies, defends, or minimizes the criticized decision, and that sentence addresses the same reproach being evaluated here, your answer must be YES -- even if that sentence also serves another purpose in the article, even if it is not explicitly framed as a reply, and regardless of whether it appears before or after the criticism in the text. Do not talk yourself out of a justification you have already found.

First, in one sentence, briefly justify your answer based solely on the article.
Then, on a new line, answer with exactly one word: YES or NO.\
"""

# step 4b source_category -> natural-language phrase, in the register of the
# original free-text critic_answer_final values it replaces.
_CATEGORY_PHRASES = {
    "Interest Group": "an interest group",
    "Civil Servant": "a civil servant",
    "General Public": "a member of the general public",
    "Politician": "a politician",
    "Administrative Unit of the State": "another administrative unit of the state",
    "Other": "an external actor",
}


def _source_phrase(row: pd.Series) -> str:
    attribution = row.get("source_attribution", pd.NA)
    attribution = "" if pd.isna(attribution) else str(attribution).strip().upper()

    if attribution == "JOURNALIST":
        return "the journalist"

    if attribution == "SOURCED":
        category = row.get("source_category", pd.NA)
        category = "" if pd.isna(category) else str(category).strip()
        return _CATEGORY_PHRASES.get(category, "an unspecified source")

    # source_attribution missing/unparsed (should not normally happen since
    # step 4a covers every keyword_answer == "YES" row) -> fall back exactly
    # like the legacy code did when critic_answer_final was NA.
    return "an unspecified source"


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    source = _source_phrase(row)
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text, source=source)


# --- Pass 2 (verify): classify the pass-1 justification alone, without the
# article. See src/verify_utils.py for why this second pass exists. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a short explanation of whether a specific
target responds to a criticism directed at it. You are not shown the
article itself -- your only task is to read that explanation and decide
which final answer it supports.\
"""

VERIFY_USER_TEMPLATE = """\
The question was: does "{keyword}" give a response to the criticism/reproach directed at them? Count as a response: a direct statement from "{keyword}" itself; a statement made on its behalf by a spokesperson or a subordinate unit acting under its authority; or any explanation, justification, defense, acknowledgment of the problem, promise to fix or review it, or attempt to minimize it, regarding the decision or situation being criticized -- even if brief, general, indirect, or narrated in the third person.

Here is the explanation given by the other analyst:
"{justification}"

First, quote the specific words or acts in this explanation that constitute \
such a response, if any -- answer "none" if there is none.
Then, on a new line, answer with exactly one word: YES or NO.\
"""


import re

_NEGATIVE_BOILERPLATE_RE = re.compile(
    r"does not (contain|provide)\b.{0,80}?\b(direct (statement|response)|"
    r"response or acknowledgment|acknowledgment)\b"
    r"|there is no\b.{0,60}?\b(direct )?(response|statement|acknowledgment)\b",
    re.IGNORECASE,
)


def strip_negative_boilerplate(text) -> str:
    """Remove the reflexive 'no direct response/statement' sentence(s) that
    pass 1 tends to write regardless of its own final answer (see the
    2026-10-20 docstring entry above), wherever in the text they occur --
    not just at the start. Returns "" if nothing substantive remains."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    kept = [s for s in sentences if not _NEGATIVE_BOILERPLATE_RE.search(s)]
    return " ".join(kept).strip()


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)
