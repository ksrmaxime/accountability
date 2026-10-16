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

2026-10-14 attempt (reverted): job65774182 found 16/19 POLICY->ENTITY
errors had a pass-1 justification arguing POLICY exclusively while the
recorded label said ENTITY -- the label disagreeing with its own
justification. Tried fixing this by adding a second worked example (the
POLICY side, mirroring the existing ENTITY one) plus an instruction that
the final answer must match the justification's own conclusion. Result on
job65965608: accuracy collapsed from 78.1% to 45.0% (FN 9->61) -- the
justification text was already saturated with "policy choice" boilerplate
regardless of the true answer (the exact bias named in the 2026-10-12
entry above), and forcing the label to always agree with it just made
that pre-existing bias deterministic instead of occasionally
self-correcting. Reverted in full.

2026-10-15 update: rather than add more examples or rules on top of the
counterfactual-disappear-if-executed-flawlessly test -- itself one of the
phrases the model was mechanically echoing as filler -- SYSTEM_PROMPT and
USER_TEMPLATE are rewritten around a different articulation of the same
distinction: POLICY is a disagreement about WHAT was decided (the
measure's relevance, usefulness, objectives, feasibility, the political
motives for proposing or changing it, or its concrete effects/results);
ENTITY is a disagreement about HOW the target carried out its
public-policy tasks or normal functioning (competence, organization,
representativeness, style, personality, productivity). No worked example
and no secondary rule is included this round, on purpose, to test whether
simplifying and rewording the core distinction -- without anything extra
for the model to latch onto or recite -- does better than either of the
two example-based attempts above.

2026-10-15 update #2: job65998611 confirmed the simplification direction
helps (45.0% -> 63.4%) but still falls short of the original 78.1%
baseline -- errors now echo this prompt's own vocabulary ("relevance and
usefulness", "political motives") as filler almost verbatim, the same
surface-echo failure mode as before, just with new words. Per request, a
small EXAMPLES block is added back to USER_TEMPLATE: 2 POLICY + 2 ENTITY,
each a different concrete scenario (tax threshold design / subsidy
assumptions for POLICY; office response time / director leadership style
for ENTITY) written in deliberately distinct vocabulary and sentence
shape, specifically so no single phrase is common to all four and
copy-paste-able as generic filler the way earlier single-example or
repeated-template attempts were.

2026-10-16 update: job66020284 (post _find_label fix, see
src/verify_utils.py) showed the remaining errors split into three kinds:
(1) truncated pass-1 responses that never reach the final label line --
see sbatch_step5_array.sh's MAX_NEW_TOKENS increase; (2) the target of the
criticism being confused with the policy topic merely mentioned in the
article (left alone for now, per request); and (3) a genuine pass-1
label/justification contradiction on otherwise clean, complete responses
(e.g. a justification concluding "...the disagreement is about what was
decided, not how it was carried out" with ENTITY still recorded as the
final answer) -- this is exactly the failure pass 2 was built to catch
(see src/verify_utils.py's module docstring) and _find_label cannot fix
it, since there is no textual ambiguity to resolve, just a generation-time
inconsistency. Pass 2 is reactivated for this step (scripts/step5_content_type.py)
to address kind (3) specifically. VERIFY_SYSTEM_PROMPT is updated to match
SYSTEM_PROMPT's current WHAT/HOW dichotomy (it still carried the old
2026-10-12 counterfactual-test wording, which no longer matches pass 1's
actual category definitions) -- VERIFY_USER_TEMPLATE is unchanged, it was
already generic. Unlike the step5 pass 2 removed back on 2026-10-13 (which
read justifications written under the OLD counterfactual-test prompt and
inherited its "policy choice" boilerplate bias), pass 2 now reads
justifications written under the current, more concretely-grounded
prompt, and is judging a definition that actually matches what pass 1 was
asked. Whether this holds up has to be measured on the next run; if pass 2
turns out to reintroduce the same one-directional bias as before, revert
to this file's pre-reactivation state (see the .bak file from this date).
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to classify the NATURE of this criticism into ONE of two categories.

=== CATEGORIES ===

  Policy — what is being criticized is the public policy itself: its
           relevance, its usefulness, its objectives, its feasibility, the
           political motives behind proposing it or wanting to change it,
           or its concrete effects and results. The disagreement is about
           WHAT was decided, or what should be decided.

  Entity — what is being criticized is how the target carried out its
           public-policy tasks, or its normal functioning: its competence,
           organization, representativeness, style, personality, or its
           efficiency (productivity) as a unit. The disagreement is about
           HOW the target acted, not about the substance of the policy
           itself.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is this criticism about the public POLICY itself made by "{keyword}" -- its relevance, usefulness, objectives, feasibility, the political motives behind it, or its concrete effects and results -- or about "{keyword}" as an ENTITY -- how it carried out its tasks or its normal functioning: its competence, organization, representativeness, style, personality, or efficiency?

=== EXAMPLES ===

  "The new threshold was set far too low and will end up penalising small
  businesses the measure was never meant to target." -> POLICY (the
  complaint is about the design of the measure, not about who carries it
  out)

  "Even if rolled out perfectly, the subsidy would still fail to cut
  emissions the way it promises to -- the whole plan rests on the wrong
  assumptions." -> POLICY (the complaint is about what the plan is meant
  to achieve, not about execution)

  "It took the office eighteen months to answer a routine request, far
  longer than comparable agencies manage." -> ENTITY (the complaint is
  about how slowly and poorly the office operates, not about any decision
  it made)

  "Former staff describe a director who surrounded herself with loyalists,
  sidelined internal experts, and let the workplace atmosphere turn toxic."
  -> ENTITY (the complaint is about leadership and conduct, not about a
  policy choice)

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
