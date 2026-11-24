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

At creation, NO WORDING HAD BEEN CHANGED ANYWHERE BELOW: SYSTEM_PROMPT,
build_user_prompt, VERIFY_SYSTEM_PROMPT and build_verify_prompt were all
byte-identical to src/step5_prompts.py at the time of this duplication --
only the module-level docstring (documentation, not prompt content) was
written fresh. See the entry just below for the first real wording change
made to this file.

2026-11-24 "specific-choice" redefinition (SYSTEM_PROMPT + USER_TEMPLATE
only -- pass 1, the only pass this step's script actually calls; see
below): diagnosed from job20905947 (the first full remote-chain run on
GOLD800). step5's oracle accuracy on that run (71.9%, n=89) was below the
classic 8B baseline (76-78%), with a heavily FN-directional error pattern
(22 FN "pred ENTITY/gold POLICY" vs 3 FP) and weak POLICY recall (29%).
Reading all 25 error rows' own justifications (pipeline's and gold's)
found two distinct things going on, NOT a parsing bug (unlike step4a's
2026-11-24 fix just above in this file's sibling -- none of these 25
justifications open or close on the wrong bare word):
  (a) 4/25 rows (VBS, Ignazio Cassis, OFC/BAK, DFAE/EDA) are gold
      annotation errors, not pipeline errors -- gold's own stored
      content_type_justification argues the OPPOSITE of gold's own
      stored content_type label (e.g. VBS: labelled POLICY, but its own
      justification says "et non sur un choix de politique publique
      precis"). Same failure mode as the already-revised OFROU/ASTRA row
      (see src/step5_prompts.py's own changelog), just never caught for
      these four. Flagged to Maxime, left as-is in gold for him to review
      separately -- not something a prompt change can fix.
  (b) The other ~21 rows are a genuine, systematic criterion mismatch.
      This prompt's old test asked "if executed flawlessly, would the
      criticism disappear?", which pushed the model to answer ENTITY
      whenever the criticism was phrased in terms of competence,
      communication, judgment, or integrity -- even when one specific,
      named policy/decision/measure was squarely the thing being
      criticized (e.g. "the agency's choice to END the Masterplan
      Biomedizin is a bad signal" -> model said ENTITY ("the agency's
      choice", "bad signal" read as a competence/judgment complaint);
      gold said POLICY (it IS a specific measure -- its suppression).
      Gold's actual practice, read across these cases, counts ANY
      criticism aimed at one specific, identifiable policy/measure as
      POLICY regardless of whether that criticism is framed as about the
      measure's wisdom, its communication, or the competence/integrity
      behind it -- ENTITY is for criticism with no single specific choice
      at its center, a value judgment about the target itself.
Fix (Maxime's explicit rule, given verbatim): POLICY now covers any stage
of a public policy's life (proposal, development, implementation, effect,
modification, suppression) whenever the reproach is PRECISELY the choice
to do one of those things -- however harshly phrased (a mistake, a
betrayal, cowardice, etc. are all still POLICY if what's being judged is
the choice itself). ENTITY is reserved for a value judgment about the
target's clairvoyance, competence, behaviour, track record, management,
or morality, detached from any single specific choice. Maxime's own two
worked examples ("M. X regrette la proposition de Z de soutenir la
reforme X" -> POLICY; "M. X trouve Z vraiment lache de soutenir une
reforme pareille" -> ENTITY) are reproduced in both SYSTEM_PROMPT and
USER_TEMPLATE below. Deliberately does NOT add a third "BOTH" answer
(discussed and explicitly deferred by Maxime): gold's own 800 rows only
have POLICY/ENTITY today, so a BOTH answer could never be scored against
it without first re-annotating gold itself -- a separate, larger task.
VERIFY_SYSTEM_PROMPT/VERIFY_USER_TEMPLATE (pass 2) are dead code for this
step -- scripts/step5_content_type.py never calls them (pass 2 was
removed, see src/step5_prompts.py's own changelog) -- and were
deliberately NOT updated to match this new rule; if pass 2 is ever
reactivated here, revisit them too. Not yet re-run/confirmed on real
data -- see sbatch_step5_remote_test.sh for the next test run.

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

  Policy  — the criticism is aimed at a specific, identifiable PUBLIC
            POLICY or measure, at any stage of its life: it being
            proposed, developed, implemented, having an effect, being
            modified, or being dropped/suppressed. This applies whenever
            what is reproached to the target is PRECISELY its choice to
            propose, develop, implement, change, or suppress that
            specific policy or measure -- even when the reproach is
            phrased harshly, as a mistake, a betrayal, cowardice, or bad
            judgment, as long as the thing actually being judged is that
            choice itself.

  Entity  — the criticism does NOT bear on a specific policy choice, but
            is instead a value judgment about the target itself: its
            clairvoyance, competence, behaviour, track record,
            management, or morality -- judging the target as an actor,
            detached from any single specific choice.

The test is not whether a specific policy, decision, or project is named
in the article, and not whether your own justification uses the words
"policy", "choice", or "decision" -- a named measure can still be nothing
more than the backdrop to a purely ENTITY-style character judgment. The
test is: is the reproach aimed AT a specific choice concerning a public
policy (its proposal, development, implementation, effect, modification,
or suppression) -- POLICY -- or is it a value judgment about the target's
qualities or conduct, detached from any single specific choice -- ENTITY?

For example:
- "Mr X regrets that entity Z proposed to support reform X." -> POLICY: the reproach is aimed at the choice to propose/support that specific reform.
- "Mr X finds entity Z really cowardly for supporting such a reform." -> ENTITY: the reproach is a value judgment about Z's character (cowardice), not about the choice itself.
- "The ministry's housing subsidy program is criticized for chronic processing delays and errors that left applicants waiting months, with no specific decision about the program itself being questioned." -> ENTITY: a general judgment about how the program is run day to day, not about one specific choice.
- "The agency is criticized for deciding to end the housing subsidy program despite its proven results." -> POLICY: the reproach is precisely the choice to end (suppress) that specific policy.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is this criticism aimed at a specific PUBLIC POLICY choice made by "{keyword}" (its proposal, development, implementation, effect, modification, or suppression), or is it a value judgment about "{keyword}" as an ENTITY (its clairvoyance, competence, behaviour, track record, management, or morality), detached from any single specific choice?

Naming a specific policy, project, or decision does not by itself decide the answer. Ask: is the reproach precisely the choice to propose, develop, implement, change, or suppress that specific policy -- even if phrased harshly, as a mistake or a betrayal? If yes, this is POLICY. Or is the reproach a value judgment about "{keyword}" itself (its clairvoyance, competence, behaviour, track record, management, or morality), with no single specific choice actually being the target? If yes, this is ENTITY.

For example:
- "Mr X regrets that "{keyword}" proposed to support reform X." -> POLICY: the reproach is aimed at the choice to propose/support that reform.
- "Mr X finds "{keyword}" really cowardly for supporting such a reform." -> ENTITY: a value judgment about "{keyword}"'s character, not about the choice itself.

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
