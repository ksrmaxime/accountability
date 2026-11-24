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
reactivated here, revisit them too.

2026-11-24 "two-test" redefinition (SYSTEM_PROMPT + USER_TEMPLATE only --
replaces the "specific-choice" redefinition just above, which was NEVER
run on a real sbatch job -- see that entry's own last line): the
specific-choice wording above was confirmed as a SEVERE regression on
job20934804 (71.9%->46.7% oracle accuracy, precision 75%->36%, recall
29%->96.4%): the model started treating virtually ANY describable action
by the target -- appointing an unsuitable risk manager (armasuisse),
dimensioning server capacity ahead of a public coin sale (AFF/EFV),
maintaining a diplomatic posture (Cassis) -- as "a specific policy
choice", because almost any human conduct CAN be redescribed as "a
choice". Root cause, found by re-reading the specific-choice prompt
itself rather than more error rows: it is internally self-contradictory.
Its own POLICY definition says harsh phrasing "as a mistake, a betrayal,
cowardice, or bad judgment" can still be POLICY "as long as... the choice
itself" is what's judged -- then its own worked example uses "finds Z
really cowardly" to illustrate ENTITY. Telling the model that
harshly-phrased choice-criticism (explicitly including the word
"cowardice") stays POLICY, right next to an example where "cowardly"
means ENTITY, plausibly pushed it to resolve the tension by over-applying
POLICY to any choice-shaped criticism at all, including personnel/
operational/managerial ones that were never meant to qualify.

Discussed with Maxime at length (no code touched until this entry) before
redesigning: a single test ("is there a specific choice?") cannot work,
because it is never false -- naming who to appoint, how to size a
server, or what tone to keep in a negotiation are all "choices" in the
trivial sense. Two SEPARATE, successive tests are needed instead:

  TEST 1 (public-policy gate): is the criticized decision a genuine
  PUBLIC measure/program/rule/course of action -- as opposed to an
  INTERNAL operational/managerial/personnel matter (who to hire, how to
  staff or run a project internally, technical/logistical execution, a
  spokesperson's communication style)? Anything internal/operational
  fails this gate and is ENTITY outright, regardless of how deliberate or
  specific the internal choice was -- this is what should have excluded
  armasuisse/AFF-EFV above, and didn't, because the old prompt had no such
  gate at all.

  TEST 2 (content vs. trait, only reached if test 1 passes): does the
  reproach depend on WHAT this specific public measure concretely does
  (its content, its target, its effects) -- POLICY -- or is it a general
  CHARACTER/COMPETENCE judgment about the target (clairvoyance,
  competence, courage, honesty, management...) that would transfer
  unchanged to a totally different decision made in a similarly-described
  manner -- ENTITY, even with one specific measure as the article's
  pretext. Operationalised as a concrete "swap test": would the exact
  same reproach still land if the specific measure were replaced by a
  completely different one, handled the same way by the same target? If
  yes, it's a trait (ENTITY); if the reproach stops making sense without
  THIS measure's own content, it's POLICY. This resolves the cowardice
  contradiction cleanly: "regrets the proposal to support reform X"
  depends on reform X's own content (fails the swap test) -> POLICY;
  "finds Z cowardly for supporting such a reform" would describe Z
  identically for any other reform backed under similar pressure (passes
  the swap test) -> ENTITY -- matching both of Maxime's original worked
  examples without contradicting itself.

Theoretical grounding (not just an ad hoc patch): test 1 mirrors the
classic policy/administration distinction in public-administration theory
(the content of a public policy vs. the operational management that
executes it); test 2 mirrors the dispositional/situational distinction in
social-attribution theory (judging a stable trait of the actor vs.
judging one act in its specific context). Both tests are reproduced
practically identically in SYSTEM_PROMPT and USER_TEMPLATE below, with
six worked examples each (the original two from Maxime, plus the
armasuisse/AFF-EFV/housing-subsidy/Masterplan-Biomedizin cases this
redesign specifically targets). The model is also asked to name which
test decided its answer (test 1 or test 2) in its one-sentence
justification -- a new, free diagnostic signal for the next error
analysis, at no extra cost. Not yet re-run/confirmed on real data -- see
sbatch_step5_remote_test.sh for the next test run.

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
Your task is to classify the NATURE of this criticism into ONE of two categories, using TWO
successive tests -- do not stop at test 1 passing, and do not skip straight to test 2.

=== TEST 1: is this a PUBLIC POLICY matter at all? ===

Ask first: does the criticized decision concern an official PUBLIC measure, program, rule, or
course of action -- something by which the target exercises its public mandate toward a defined
public, sector, or cause -- as opposed to an INTERNAL, OPERATIONAL, MANAGERIAL, or PERSONNEL
matter (who to hire or appoint, how to organise or staff a project, technical or logistical
execution, a spokesperson's communication style, day-to-day internal running)?

If the matter is internal/operational/managerial/personnel, the answer is ENTITY, full stop --
no matter how specific or deliberate that internal choice was. A decision only ever qualifies for
POLICY if it is genuinely about a public measure or program, never about how the target runs
itself internally.

=== TEST 2 (only reached if test 1 says "yes, genuine public-policy matter"): content vs. trait ===

  Policy  — the reproach depends on the CONTENT of that specific public measure itself: whether
            it was the right or wrong thing to do, GIVEN WHAT THAT MEASURE CONCRETELY DOES (its
            proposal, development, implementation, effect, modification, or suppression). The
            reproach would NOT transfer, unchanged, to a completely different measure -- it only
            makes sense in light of what THIS specific measure actually does or fails to do.

  Entity  — the reproach is a general CHARACTER or COMPETENCE judgment about the target as an
            actor (its clairvoyance, competence, behaviour, track record, management, morality,
            courage, honesty...) that would transfer just as well to a completely different
            decision made by the same target in a similarly-described manner (e.g. under
            pressure, hastily, self-servingly), regardless of what that other decision actually
            did. A specific measure can still be the article's occasion or pretext for this kind
            of judgment -- that alone does not make it POLICY.

Swap test for TEST 2: imagine replacing the specific measure in the article with a completely
different one, handled by the same target in a similarly-described manner. Does the exact same
reproach still land, unchanged? If yes, you are looking at a TRAIT judgment -- ENTITY. If the
reproach stops making sense without THIS measure's own specific content -- POLICY.

Harsh phrasing (a mistake, a betrayal, bad judgment, cowardice) never decides the answer by
itself -- apply test 1 and then test 2 to work out which category that harsh phrasing actually
belongs to; the same harsh word can land on either side depending on what it is actually
predicated of.

For example:
- "Mr X regrets that entity Z proposed to support reform X." -> POLICY: test 1 passes (reform X is a public measure); test 2: the regret depends on reform X's own content, and would not transfer unchanged to a different reform.
- "Mr X finds entity Z really cowardly for supporting such a reform." -> ENTITY: test 1 passes, but test 2 fails -- "cowardly" would describe Z just as well had Z backed any other measure under similar pressure, regardless of that measure's content.
- "The agency is criticized for appointing an unqualified risk manager." -> ENTITY: fails test 1 outright -- a personnel/internal-management decision, not a public policy, however specific and deliberate that appointment was.
- "The office is criticized for under-provisioning server capacity ahead of a public coin sale." -> ENTITY: fails test 1 -- a technical/operational execution matter, not a public policy.
- "The ministry's housing subsidy program is criticized for chronic processing delays and errors that left applicants waiting months, with no specific decision about the program itself being questioned." -> ENTITY: fails test 1 -- day-to-day administrative execution, not a policy choice.
- "The agency is criticized for deciding to end the housing subsidy program despite its proven results." -> POLICY: test 1 passes (the program itself is the public measure); test 2 passes -- the reproach depends on THIS program's own track record, and would not transfer unchanged to ending a different, unproven program.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Classify the NATURE of this criticism using TWO successive tests -- do not stop at test 1 passing, and do not skip straight to test 2.

TEST 1 -- is this a PUBLIC POLICY matter at all? Does the criticized decision concern an official PUBLIC measure, program, rule, or course of action by "{keyword}" -- as opposed to an INTERNAL, OPERATIONAL, MANAGERIAL, or PERSONNEL matter (who to hire or appoint, how to organise or staff a project, technical or logistical execution, a spokesperson's communication style, day-to-day internal running)? If it is internal/operational/managerial/personnel, the answer is ENTITY -- no matter how specific or deliberate that internal choice was.

TEST 2 (only if test 1 says yes) -- content vs. trait. Does the reproach depend on the CONTENT of that specific public measure itself -- whether it was the right or wrong thing to do, GIVEN WHAT THAT MEASURE CONCRETELY DOES -- such that the reproach would NOT transfer, unchanged, to a completely different measure? That is POLICY. Or is the reproach a general CHARACTER or COMPETENCE judgment about "{keyword}" as an actor (its clairvoyance, competence, behaviour, track record, management, morality, courage, honesty...) that would transfer just as well to a totally different decision handled in a similarly-described manner, regardless of what that other decision actually did? That is ENTITY, even if one specific measure is the article's occasion for the remark.

Swap test for TEST 2: imagine replacing the specific measure with a completely different one, handled by "{keyword}" in a similarly-described manner. Does the exact same reproach still land, unchanged? If yes -- a trait judgment, ENTITY. If the reproach stops making sense without THIS measure's own specific content -- POLICY.

Harsh phrasing (a mistake, a betrayal, bad judgment, cowardice) never decides the answer by itself -- apply test 1 and then test 2; the same harsh word can land on either side depending on what it is actually predicated of.

For example:
- "Mr X regrets that "{keyword}" proposed to support reform X." -> POLICY: test 1 passes (a public measure); test 2: the regret depends on reform X's own content.
- "Mr X finds "{keyword}" really cowardly for supporting such a reform." -> ENTITY: test 1 passes, but test 2 fails -- "cowardly" would describe "{keyword}" just as well for any other measure backed under similar pressure.
- "The agency is criticized for appointing an unqualified risk manager." -> ENTITY: fails test 1 -- a personnel decision, not a public policy.
- "The ministry's housing subsidy program is criticized for chronic processing delays, with no specific decision about the program itself being questioned." -> ENTITY: fails test 1 -- day-to-day administrative execution, not a policy choice.
- "The agency is criticized for deciding to end the housing subsidy program despite its proven results." -> POLICY: test 1 passes (the program is the public measure); test 2 passes -- the reproach depends on THIS program's own track record.

First, in one or two sentences, justify your answer, naming which test decided it (test 1 or test 2) and why.
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
