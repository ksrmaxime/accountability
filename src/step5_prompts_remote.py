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
analysis, at no extra cost.

2026-11-24 "WHAT vs HOW" rewrite (COMPLETE replacement of SYSTEM_PROMPT +
USER_TEMPLATE, not a patch on top of the "two-test" entry above -- see
that entry's own text, which is kept here only as a historical record of
what was tried and why it failed): the two-test design above was
confirmed on job20943266 as an even WORSE regression than the
specific-choice wording it replaced -- oracle accuracy 71.9%->46.7%->
39.3%, with the model now predicting POLICY on 80/89 rows (53 FP). Root
cause, found by reading the FP justifications rather than more aggregate
numbers: the "swap test" in test 2 is practically vacuous. Any detailed,
concrete criticism has unique supporting facts (a specific drone's
autonomy failure, a specific webshop's server capacity, a specific
percentage in a trade deal) that trivially "wouldn't transfer unchanged"
to a different episode -- so the model answered POLICY almost every time,
for exactly the reason test 2 was supposed to prevent. Reading gold's own
justifications on these 53 FP rows (armasuisse's procurement failures,
the VBS drone program, fedpol's caseload, the AFF/EFV coin-sale webshop,
Cassis's Gaza posture twice, Amherd/VBS data security, Beat Jans twice,
COMCO/Weko) shows gold is NOT asking "would this exact reproach transfer
to a different episode" -- it is asking something simpler and more
robust: is the criticism about WHAT was decided (the content, target, or
substance of one specific measure) or about HOW the target functions as
an organisation/individual (its competence, planning, risk management,
organisation, leadership, communication, or character) -- even when one
specific, named project or dossier is the article's illustration of a
HOW-type judgment. Gold's own vocabulary makes this near-literal across
these rows: "comportement institutionnel general", "gestion des risques",
"pratique de l'administration", "Kompetenz und Verwaltungspraxis",
"allgemeine Fuehrungsqualitaeten". Worth noting: this WHAT/HOW framing
already existed, unused, in this very file's own VERIFY_SYSTEM_PROMPT
below (the dead pass-2 prompt, inherited unchanged from
src/step5_prompts.py and never touched by either redefinition above) --
its POLICY/ENTITY definitions already said almost exactly this
("disagreement about WHAT was decided" vs "disagreement about HOW the
target carried out its ... normal functioning"). Neither of the two
redesigns above drew on it; this one does, consciously, while rebuilding
SYSTEM_PROMPT/USER_TEMPLATE for pass 1 from scratch -- per Maxime's
explicit instruction, this is a full rewrite, not another layer: the
"public-policy gate" and the "swap test" language from the two-test
design above are REMOVED entirely, not kept alongside the new framing.
Personnel/operational/managerial matters (armasuisse's risk-manager
appointment, AFF/EFV's server sizing) are no longer handled by a separate
gate -- they fall out naturally as HOW-type competence/management
judgments under the new single test. One additional, explicitly flagged
wrinkle: negotiation-outcome criticisms ("gave away too much", "got a bad
deal") are ambiguous between a competence judgment on negotiating skill
(ENTITY) and a content judgment on the deal's substance (POLICY); a
worked warning about this is included below because two FP rows (Guy
Parmelin, Karin Keller-Sutter) have this exact shape. For those two rows
specifically, gold's own content_type_justification text argues FOR
POLICY in so many words ("un choix de politique commerciale identifiable
et nommable", "eine spezifische Angelegenheit") while the stored label is
ENTITY -- the same gold label/justification self-contradiction pattern
already known from the OFROU/ASTRA and VBS/Cassis/OFC-BAK/DFAE rows,
flagged to Maxime, not something this prompt can or should chase. A third
row (Martin Pfister/F-35) is murkier -- its own justification mixes "une
politique specifique et nommee" with "et sa gestion" (its management) in
the same sentence -- and is noted as a genuinely hard boundary case
rather than a clear-cut error. Not yet re-run/confirmed on real data --
see sbatch_step5_remote_test.sh for the next test run.

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
Your task is to classify the NATURE of this criticism into ONE of two categories: POLICY or ENTITY.

The distinction is WHAT was decided versus HOW the target functions as an organisation or
individual.

  Policy  — the criticism targets WHAT was decided: the actual content, substance, objectives,
            scope, target, or direction of a specific public measure, at any stage of its life
            (it being proposed, developed, implemented, having an effect, being modified, or
            being dropped/suppressed). The question being asked is: was THIS decision, on its own
            merits -- given what it actually does, to whom, and why -- the right or wrong thing to
            do? The reproach would disappear if the measure's content had been different, even if
            the target had otherwise acted in exactly the same manner.

  Entity  — the criticism targets HOW the target functions as an actor: its competence, planning,
            risk management, organisation, leadership, communication, staffing, internal
            practices, track record, or character (clairvoyance, courage, honesty, morality). The
            question being asked is: is this target GOOD AT what it does and how it conducts
            itself -- regardless of which specific decision or episode happens to illustrate that
            judgment? The reproach would survive even if the measure's content had been completely
            different, as long as it had been handled in the same manner.

A simple way to tell them apart: try rephrasing the criticism as "the target is bad at managing,
planning, organising, leading, or communicating" (or an equivalent character judgment: cowardly,
dishonest, incompetent, out of touch). If that rephrasing keeps the criticism's real point intact,
it is ENTITY, even when one specific, named project or decision is the article's illustration of
it. If that rephrasing loses the point -- because the actual complaint is about what the decision
itself does or fails to do, not about the target's competence or character -- it is POLICY.

Naming a specific policy, project, or decision does NOT by itself decide the answer, and neither
does the tone of the criticism: a harshly-worded reproach (a mistake, a betrayal, cowardice,
incompetence) can land on either side, depending on whether it is judging the CONTENT of a
decision or the MANNER/CHARACTER of the actor. A single named project described in technical
detail (cost overruns, delays, a disqualified appointee, a botched rollout) is usually evidence
being offered FOR a competence/management judgment (ENTITY), not itself the subject of a POLICY
judgment -- a POLICY criticism is about whether proposing, adopting, changing, or ending that
measure was the right call given its actual purpose and effects, not about how well it was
planned, staffed, or executed.

For example:
- "Mr X regrets that entity Z proposed to support reform X." -> POLICY: the reproach concerns the content of this specific choice (supporting reform X), not Z's competence or character.
- "Mr X finds entity Z really cowardly for supporting such a reform." -> ENTITY: "cowardly" is a character judgment about Z that would apply just as well had Z made any other decision under similar pressure.
- "The agency's procurement project suffered cost overruns, delays, and a disqualified risk manager due to poor planning and risk management." -> ENTITY: a judgment about the agency's competence in managing projects, illustrated by this one project -- the same complaint would be made of a different project handled with the same poor planning.
- "The office's online shop crashed due to insufficient server capacity, and employees may have been favoured in the allocation that followed." -> ENTITY: a judgment about the office's operational competence and conduct, not about the public measure's content.
- "The ministry's housing subsidy program is criticized for chronic processing delays and errors that left applicants waiting months." -> ENTITY: a judgment about how the program is run day to day, not about the program's content.
- "The agency is criticized for deciding to end the housing subsidy program despite its proven results." -> POLICY: the reproach is precisely about the content of this choice (ending a program that works), not about competence.
- "The minister is criticized for a prolonged, cautious diplomatic posture throughout an international crisis." -> ENTITY: a judgment of the minister's general conduct and character (timidity, lack of resolve) sustained over time, not about the content of one specific, named decision.
- "The minister is criticized because a specific proposed decision -- to recognise X, or to sign agreement Y -- was the wrong call." -> POLICY: the reproach is about the content of that one specific, nameable decision.

Negotiation outcomes are a common trap: a critic complaining that a negotiator "gave away too
much" or "got a bad deal" is usually judging that person's negotiating SKILL (ENTITY) -- the same
complaint would be made of any outcome perceived as weak, regardless of what was actually
negotiated -- unless the article's own reproach is specifically about the agreement's terms and
their concrete consequences on their own merits (POLICY).\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Classify the NATURE of this criticism: is it about WHAT "{keyword}" decided, or about HOW "{keyword}" functions as an organisation or individual?

POLICY means the criticism targets the actual content, substance, objectives, scope, or direction of a specific public measure (its proposal, development, implementation, effect, modification, or suppression). Was THIS decision, given what it actually does, to whom, and why, the right or wrong thing to do? The reproach would disappear if the measure's content had been different, even if "{keyword}" had otherwise acted in exactly the same manner.

ENTITY means the criticism targets HOW "{keyword}" functions as an actor: its competence, planning, risk management, organisation, leadership, communication, staffing, internal practices, track record, or character (clairvoyance, courage, honesty, morality). Is "{keyword}" good at what it does and how it conducts itself -- regardless of which specific decision or episode illustrates that judgment? The reproach would survive even if the measure's content had been completely different, as long as it had been handled in the same manner.

A simple test: try rephrasing the criticism as «"{keyword}" is bad at managing, planning, organising, leading, or communicating» (or a character judgment: cowardly, dishonest, incompetent, out of touch). If that rephrasing keeps the criticism's real point intact, this is ENTITY, even when one specific, named project or decision illustrates it. If the rephrasing loses the point -- because the real complaint is about what the decision itself does or fails to do -- this is POLICY.

Naming a specific policy, project, or decision does not by itself decide the answer, and neither does harsh phrasing (a mistake, a betrayal, cowardice, incompetence) -- these can land on either side depending on whether the content of a decision, or the manner/character of "{keyword}", is what is actually being judged. A single named project described in technical detail (cost overruns, delays, a disqualified appointee, a botched rollout) is usually evidence offered FOR a competence judgment (ENTITY), not itself a POLICY question.

For example:
- "Mr X regrets that "{keyword}" proposed to support reform X." -> POLICY: the reproach concerns the content of that specific choice.
- "Mr X finds "{keyword}" really cowardly for supporting such a reform." -> ENTITY: "cowardly" is a character judgment that would apply to "{keyword}" regardless of which decision prompted it.
- "A procurement project suffered cost overruns, delays, and a disqualified risk manager due to poor planning." -> ENTITY: a judgment of "{keyword}"'s competence in managing projects, illustrated by this one project.
- "The housing subsidy program is criticized for chronic processing delays and errors." -> ENTITY: how the program is run day to day, not its content.
- ""{keyword}" is criticized for deciding to end the housing subsidy program despite its proven results." -> POLICY: the reproach is precisely about the content of that choice.
- ""{keyword}" is criticized for a prolonged, cautious posture throughout an international crisis." -> ENTITY: a judgment of general conduct and character sustained over time, not of one specific decision's content.

Watch out for negotiation outcomes: a critic saying "{keyword}" "gave away too much" or "got a bad deal" is usually judging negotiating skill (ENTITY), unless the reproach is specifically about the agreement's terms and their concrete consequences on their own merits (POLICY).

First, in one or two sentences, justify your answer by saying whether the criticism is fundamentally about WHAT was decided or HOW "{keyword}" acted.
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
