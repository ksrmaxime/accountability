# src/step4b_logit_prompts.py
"""
STEP 4b — logit-probing redesign (2026-10-26+).

2026-10-26 "summarize-then-probe" addition: job66603355 (direct-article
logit-probing on all 800 rows) came back at 49.3% accuracy on the gold
population -- flat/slightly below the old 3-pass design's 52.9%, with a
very low mean winning margin (0.088, median 0.070) suggesting the 6
binary questions were picking up a lot of article noise rather than a
clean signal. The user's hypothesis: reading the full, often long and
multi-topic article directly for each of the 12 (category x order)
binary probes is noisy -- a short, targeted summary of just "what is
{keyword} criticized for, and by whom" would give pass 2's probes a much
cleaner, shorter signal to work from.

This adds a PRE-PASS (SUMMARY_TEMPLATE / build_summary_prompt): a plain
generation step (unlike pass 2, this one DOES generate text -- there is
no fixed answer set to read logits for) that reads the full article once
and produces a 2-4 sentence factual summary of the criticism and its
source, explicitly asked NOT to classify the source into a category
itself (to avoid anchoring pass 2 on a premature label, same anti-
anchoring rationale as the old pass 2's bare-category-only design).
Pass 2's 6 binary questions then read THIS SUMMARY instead of the raw
article -- build_question_prompt/build_calibration_prompt are unchanged
in their signature (still just take whatever text is in `text_col`), the
caller (scripts/step4b_source_category_logit.py) simply points them at
the new `source_summary` column instead of the raw article column. The
only prompt-level change needed here is `context_label`: the question
template said "ARTICLE" throughout, which would be misleading once what
follows is a summary, not the article itself -- made a parameter instead
of a hardcoded string, defaulting to "ARTICLE" so the original
direct-article flow (job66603355) is unaffected if anyone re-runs it for
comparison.

Replaces the generation-based 3-pass design (src/step4b_prompts.py,
scripts/step4b_source_category.py) with a different architecture
entirely, proposed by the user based on a method that worked well on a
separate task: instead of asking the model to GENERATE a category label
(risking it defaulting to one or two over-used categories when faced with
a 6-way choice, and anchoring on whatever actor it names first), ask 6
independent YES/NO questions -- one per category -- and read the answer
directly off the model's next-token logits rather than generating text.

For article a and category c, this gives a calibrated probability
p_corrected(a, c) that "the criticism of {keyword} in this article comes
from {c}". The final answer is simply argmax_c p_corrected(a, c).

Three ideas make this work (see src/step4b_logit_utils.py for the math):

1. One forward pass, not generation. The model's response is never
   sampled; we read the raw logits at the position where the answer
   letter would go, restrict to just the candidate letters (A/B), and
   softmax over those two. This can't produce a malformed or missing
   answer the way free-text generation can, and costs one forward pass
   instead of a multi-token decode loop.

2. Order-swap averaging. A model can have an intrinsic preference for
   the letter "A" or "B" independent of content. Every question is
   therefore asked twice -- once with A=Yes/B=No, once with A=No/B=Yes --
   and the two P(yes) readings are averaged. See build_question_prompt's
   `order` parameter.

3. Context-free calibration (Zhao et al. 2021, "Calibrate Before Use").
   Each of the 6 category questions has its own intrinsic yes-bias
   independent of any article (a broader category like "Politician" may
   just sound more plausible by default than "Civil Servant"). This is
   measured ONCE per category -- not per row -- by asking the exact same
   question with the article replaced by "N/A" and the target replaced by
   a neutral placeholder (build_calibration_prompt), then removed from
   every row's reading in logit space. See step4b_logit_utils.calibrate.

Known limitation (flagged by the user's own reference doc): the "empty
article" approximation can under/over-correct a category whose true
definition is itself about ABSENCE of a named actor -- here, "Other" --
since an empty article resembles a vague/unnamed-source article by
construction. Worth watching in the per-category QA, same as the
reference doc's own "no mechanism" caveat.

2026-10-27 "validate-and-correct" addition: a diagnostic read of
job66608678's 69-row population found that swapping every WRONG row to
pass 2's own runner-up category (2nd-highest calibrated probability)
would recover 14 of 26 errors (62.3% -> 82.6% ceiling) -- but a blind,
margin-based swap ("low confidence -> take runner-up") only pays off in
a narrow band (<=0.05 margin: +3 net) and turns net NEGATIVE beyond
~0.10, because plenty of CORRECT rows also sit at thin margins. Several
of the recoverable errors are also NOT low-margin at all (Albert Rosti:
p1=0.97 and still wrong) -- pass 2's 6 independent yes/no questions can
each be confidently swayed by locally-strong but not-actually-decisive
textual signal (lots of generic citizen language drowning out one
specific named politician, say), a failure mode a margin threshold
cannot see by construction. Pass 3 below asks a genuinely different
question to catch exactly these: not "which of 6 categories fits best"
but "pass 2 concluded category Y -- true or false", in ONE holistic
read of the same summary, instead of 6 independent isolated ones. Where
pass 3 says false, the row falls back to pass 2's own runner-up rather
than a 3rd fresh guess. Deliberately NOT the same shape as the old
3-pass design's draft/verify/force/reclassify machinery (abandoned in
favour of logit-probing on 2026-10-26): this is a single calibrated
logit read, same machinery as pass 2 itself (one forward pass,
order-swap averaging, context-free calibration), not a generation step.

Category definitions below are adapted from src/step4b_prompts.py's
SYSTEM_PROMPT (same six categories, same underlying "in whose name are
they speaking" test), but rewritten as STANDALONE paragraphs: the old
"ordered decision procedure, stop at first match" framing doesn't apply
here, since every category is now asked about independently and
symmetrically rather than checked in sequence.

2026-10-26 "boundary sharpening" addition: job66603934 (summarize-then-
probe) reached 65.2% accuracy (vs 49.3% for direct-article logit-probing
and 52.9% for the 3-pass design) -- a real jump, with mean margin up from
0.088 to 0.298. A row-by-row read of its 24 errors against their
source_summary text showed that roughly half of them were genuine
PASS-2 misclassifications (the summary already contained the correct,
unambiguous signal) rather than the summary itself having picked the
wrong source -- the latter is a pass-1/gold-adaptation question, left
alone here. Three concrete, recurring boundary bugs accounted for most
of the pass-2 misses, each now addressed by a targeted edit below
(definitions/examples only, no change to the overall architecture or to
SUMMARY_TEMPLATE):

 (a) 'Other' was being predicted even when the summary named a specific
     person, group, or outlet as the critic -- directly contradicting
     Other's own "vague or unnamed source" definition (6/24 errors, e.g.
     a named economist, a named activist, a named media outlet). Fix:
     Other's definition now explicitly forbids this, and media/press is
     added to Interest Group's list of organized-entity types.
 (b) An elected official or appointed civil servant explaining/
     defending their own actions "in an official capacity" was
     repeatedly read as 'Administrative Unit of the State', when the
     correct category (Politician / Civil Servant) requires only that
     the article does NOT present it as a formal, collective
     institutional statement (~5/24 errors). Fix: added an explicit
     tie-break to all three of these categories' definitions.
 (c) A company executive discussing policy (UBS's CEO on bank capital
     rules) was read as 'Politician' -- a clean, confident
     misclassification despite an unambiguous summary. Fix: Politician's
     definition now explicitly excludes business executives (routed to
     Interest Group instead), reinforced with a matching canonical
     example.

Also scoped 'Politician' to Swiss office-holders only (a foreign head of
state/minister does not qualify -- see the new narrow clause in
'Other'), and added "political commentator/analyst" to General Public's
definition and examples, after a named independent analyst was
misclassified as Other. Deliberately NOT touched: SUMMARY_TEMPLATE
itself, and the handful of errors traced to the summary locking onto a
different (but real) critic than the one gold used -- those are being
evaluated for a gold-label adaptation instead, since the logit-probing
read of the summary was correct in those cases.
"""
from __future__ import annotations

import pandas as pd

from src.step4b_prompts import CATEGORY_PHRASE  # reuse the exact same phrasing

CATEGORIES = [
    "Administrative Unit of the State",
    "Politician",
    "Interest Group",
    "Civil Servant",
    "General Public",
    "Other",
]

CATEGORY_DEFINITION = {
    "Administrative Unit of the State": (
        "The criticism (or the response to it) is presented as the FORMAL POSITION of a "
        "state institution, at any level -- a federal, cantonal or communal department or "
        "office, a city or commune, a cantonal or federal executive body, a court, or a "
        "regulatory/oversight agency. This includes an official report, audit, ruling or "
        "communique, and ALSO a named individual (a spokesperson, a director, or even an "
        "elected executive member) speaking explicitly IN THE NAME of that institution, "
        "rather than giving a personal opinion. Reserve this category for a genuinely "
        "FORMAL, COLLECTIVE statement -- a press release, an official report, a ruling, "
        "or a spokesperson explicitly presenting the institution's own position. Merely "
        "holding an official job title, or explaining/defending one's own actions 'in an "
        "official capacity', is NOT enough by itself: if a specific named person is "
        "really just giving their own account or opinion rather than the institution's "
        "formal position, prefer 'Politician' (if they hold elected/appointed political "
        "office) or 'Civil Servant' (if they are an appointed state employee) instead."
    ),
    "Politician": (
        "An elected or appointed political office-holder (a federal, cantonal or communal "
        "councillor, a member of parliament, a party leader, a cantonal or communal "
        "executive member) is speaking IN THEIR OWN NAME or political capacity -- their "
        "own opinion or stance, not their institution's official position. A political "
        "party issuing its own collective statement also counts, as does a FORMER elected "
        "politician commenting on public affairs -- their political identity does not "
        "disappear on retirement. Restricted to SWISS political office-holders (federal, "
        "cantonal, or communal) -- a foreign head of state, minister, or politician does "
        "NOT qualify here, even if clearly named and clearly political in their own "
        "country (see 'Other'). This also excludes business executives, company "
        "spokespeople, or other representatives of a private company or economic "
        "interest commenting on economic, financial, or regulatory policy -- even when "
        "the topic is a government decision -- route those to 'Interest Group' instead. "
        "This category takes priority over 'Administrative Unit of the State' whenever "
        "the elected official is giving their own assessment or opinion, even one "
        "phrased around their official duties, UNLESS the article explicitly presents "
        "it as a formal, collective institutional statement."
    ),
    "Interest Group": (
        "The actor speaks FOR an organized entity representing a particular interest: a "
        "private company, a business or professional association, a union, a lobby group, "
        "a think tank, an NGO, a media outlet or journalistic investigation, or an "
        "organized advocacy/protest group -- even an informal one, as long as it acts as "
        "a collective, not a lone individual. This excludes political parties/politicians "
        "and state administrations. Also includes a business executive or company "
        "representative commenting on economic, financial, or regulatory matters, even "
        "when framed as a personal view -- their professional/economic interest counts "
        "as the organized entity here."
    ),
    "Civil Servant": (
        "A specific, named individual CURRENTLY employed by the state (an official, "
        "auditor, diplomat, police officer, etc.) gives their OWN personal opinion or "
        "account, explicitly NOT as their institution's formal position. A retired or "
        "former official with no current state role does NOT qualify here. This "
        "category takes priority over 'Administrative Unit of the State' whenever a "
        "named, currently-employed official is giving their own account or opinion -- "
        "even one framed using their job title or official duties -- UNLESS the "
        "article explicitly presents it as the institution's own formal, collective "
        "position."
    ),
    "General Public": (
        "An individual speaking only for themselves, with no institution or organized "
        "interest behind them: an ordinary citizen, a survey respondent, an independent "
        "expert, researcher, or political commentator/analyst with no organizational "
        "affiliation, or a formerly state-employed person (other than a former "
        "politician) with no current institutional role."
    ),
    "Other": (
        "Used ONLY when a criticism is clearly present in the article but its source is "
        "vague or unnamed -- e.g. the article says someone 'has faced mounting criticism "
        "in recent weeks' without identifying who is criticizing, and the journalist is "
        "not taking personal ownership of the claim either. Do NOT use this category "
        "just because a named source's precise organizational nature is unclear, or "
        "because several different critics are mentioned together -- if the article "
        "names ANY specific person, group, or outlet as the source, pick whichever of "
        "the other five categories fits best instead. The one other valid use of this "
        "category: the source IS clearly named but is a foreign government or "
        "political actor outside Switzerland's own public institutions (e.g. a foreign "
        "head of state or minister) -- a case the other five categories, being "
        "Swiss-specific, cannot otherwise capture."
    ),
}

# Two short, self-contained illustrative scenarios per category -- these
# serve as this category's own "Yes" examples, and as every OTHER
# category's "No" example (reused verbatim: "here is a case that is
# clearly NOT this category, it's actually {other_category}"). Kept short
# on purpose (a sentence or two, not a full fake article) since they are
# few-shot illustrations of the underlying test, not training data.
CANONICAL_EXAMPLES = {
    "Administrative Unit of the State": [
        "A federal audit office publishes a report concluding that a government agency "
        "mismanaged public funds.",
        "A spokesperson for a cantonal health department states, explicitly in the "
        "department's name, that current hospital wait times are unacceptable and the "
        "department's own responsibility.",
    ],
    "Politician": [
        "A member of the national parliament says in a TV interview that, in her own "
        "opinion, the government's new policy is a mistake.",
        "A city councillor publicly criticizes a mayor's decision, speaking only in his "
        "own political capacity, not as part of the city executive.",
    ],
    "Interest Group": [
        "A spokesperson for a national business association criticizes a proposed "
        "regulation on behalf of its member companies.",
        "A bank's CEO personally argues against proposed capital-requirement "
        "regulations that would affect his company, even though he is speaking in his "
        "own words rather than reading an official press release.",
    ],
    "Civil Servant": [
        "A currently serving police officer, in a personal interview, says she is "
        "frustrated with outdated equipment, speaking only for herself and not for the "
        "police department.",
        "A government auditor, speaking informally to a journalist, personally criticizes "
        "how a colleague's team handled a case, explicitly not as an official departmental "
        "statement.",
    ],
    "General Public": [
        "A local resident tells a journalist that construction noise near her home has "
        "made daily life unbearable.",
        "An independent economist or political commentator, with no organizational "
        "affiliation, publishes an opinion piece or analysis arguing that a proposed "
        "policy -- or the performance of a group of officials -- is poorly designed or "
        "underwhelming.",
    ],
    "Other": [
        "The article states that a company 'has faced mounting criticism in recent weeks' "
        "without naming who is criticizing it.",
        "The article mentions that 'critics say the project is too expensive' without "
        "identifying any specific person or organization.",
    ],
}

# The two candidate letters read off the model's next-token logits. Must
# each be a SINGLE token for the target tokenizer in context -- verified
# at runtime by src/client.py's TransformersClient.logprob_many (and can
# be checked standalone with scripts/check_step4b_logit_tokens.py before
# committing any GPU time).
LETTERS = ("A", "B")

QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. You are given \
information about a newspaper article in which "{keyword}" is criticized, and asked \
whether the criticism comes from a specific type of source.

{context_label}:
{article_text}

CATEGORY TO EVALUATE: {category_phrase}
Definition: {category_definition}

Example (answer: Yes): {yes_example_1}
Example (answer: Yes): {yes_example_2}
Example (answer: No, this is {no_cat_1} instead): {no_example_1}
Example (answer: No, this is {no_cat_2} instead): {no_example_2}
Example (answer: No, this is {no_cat_3} instead): {no_example_3}
Example (answer: No, this is {no_cat_4} instead): {no_example_4}
Example (answer: No, this is {no_cat_5} instead): {no_example_5}

Question: based on the {context_label} above, is the criticism of "{keyword}" genuinely \
coming from {category_phrase}?
Answer with exactly one letter and nothing else:
{letter_yes} = Yes
{letter_no} = No\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_ARTICLE = "N/A"
DEFAULT_CONTEXT_LABEL = "ARTICLE"


def _build_prompt(
    category: str, keyword: str, article_text: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """order=1 -> A=Yes/B=No ; order=2 -> A=No/B=Yes (see LETTERS).
    context_label names whatever `article_text` actually is -- "ARTICLE"
    for the original direct-article flow, "SUMMARY OF THE CRITICISM" for
    the 2026-10-26 summarize-then-probe flow (see build_summary_prompt)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_yes, letter_no = LETTERS if order == 1 else (LETTERS[1], LETTERS[0])

    other_cats = [c for c in CATEGORIES if c != category]
    no_examples = [CANONICAL_EXAMPLES[c][0] for c in other_cats]

    return QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase=CATEGORY_PHRASE[category],
        category_definition=CATEGORY_DEFINITION[category],
        yes_example_1=CANONICAL_EXAMPLES[category][0],
        yes_example_2=CANONICAL_EXAMPLES[category][1],
        no_cat_1=other_cats[0], no_example_1=no_examples[0],
        no_cat_2=other_cats[1], no_example_2=no_examples[1],
        no_cat_3=other_cats[2], no_example_3=no_examples[2],
        no_cat_4=other_cats[3], no_example_4=no_examples[3],
        no_cat_5=other_cats[4], no_example_5=no_examples[4],
        letter_yes=letter_yes, letter_no=letter_no,
    )


def build_question_prompt(
    row: pd.Series, text_col: str, category: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The real, article- (or summary-) grounded question for one row and
    one category, in one of the two letter orderings. `text_col` is
    whatever column holds the text to reason from -- the raw article for
    the original flow, or `source_summary` for the summarize-then-probe
    flow; pass the matching `context_label` so the prompt's own wording
    doesn't call a summary an "ARTICLE"."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_prompt(category, keyword, article_text, order, context_label=context_label)


def build_calibration_prompt(
    category: str, order: int, context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The SAME question with the article/summary and target neutralised
    -- used ONCE per category (not per row) to measure that category's
    own intrinsic yes-bias, independent of any article content. See
    src/step4b_logit_utils.calibrate. Pass the same `context_label` the
    real questions use, so the calibration prompt matches them exactly
    except for the row-specific content."""
    return _build_prompt(category, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order, context_label=context_label)


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_yes, letter_no) for the given order, mirroring
    _build_prompt's own logic -- used by the pipeline script to know
    which letter's probability is P(yes) for a given order."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])


# ---------------------------------------------------------------------------
# Pre-pass (2026-10-26 "summarize-then-probe" addition): a plain GENERATION
# step, unlike everything above. Reads the full article once and produces a
# short, factual summary of the criticism and its source for pass 2's 6
# binary probes to read instead of the raw article -- see the module
# docstring for why. Explicitly told NOT to classify the source itself, so
# it doesn't anchor pass 2 on a premature label the way the old pass 1's
# draft category did.
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the article \
below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Summarize, in 2 to 4 sentences:
1. What exactly "{keyword}" is being criticized for.
2. Who or what is making this criticism -- name the specific actor, institution, or \
organization if the article identifies one (never "{keyword}" itself, which is only the \
target, never the source, of the criticism).
3. If a named individual is speaking, make clear whether they are speaking personally / in \
their own political capacity, or explicitly in the name of their institution or \
organization -- this distinction matters more than their job title.

Do NOT classify the source into a category yourself -- just describe who they are and what \
they said, as factually and specifically as possible. If the article does not clearly name \
who is criticizing "{keyword}", say so explicitly rather than guessing.\
"""


def build_summary_prompt(row: pd.Series, text_col: str) -> str:
    """The pre-pass prompt: reads the FULL ARTICLE (text_col should be the
    raw article column here, never source_summary itself) and asks for a
    short factual summary -- see SUMMARY_TEMPLATE. The caller runs this
    through plain generation (src/client.py's chat_many, e.g. via
    src/runner.py's run_llm_dataframe), not logprob_many -- there is no
    fixed answer set to read logits for here."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return SUMMARY_TEMPLATE.format(keyword=keyword, article_text=article_text)


# ---------------------------------------------------------------------------
# Pass 3 (2026-10-27 "validate-and-correct" addition): a second
# LOGIT-PROBING step -- same machinery as pass 2 (one forward pass,
# order-swap averaging, context-free calibration), reading the SAME
# summary, but asking ONE holistic question instead of 6 independent
# ones: "pass 2 concluded the source is category Y -- true or false?".
# See the module docstring's 2026-10-27 paragraph for why this is not
# just pass 2 repeated, and src/step4b_source_category_logit.py's
# compute_validation_calibration / pass-3 loop for how it's used.
# ---------------------------------------------------------------------------

VALIDATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the {context_label} \
below, "{keyword}" is criticized.

{context_label}:
{article_text}

A prior automated analysis concluded that the source of this criticism belongs to the \
following category:

CLAIMED CATEGORY: {category_phrase}
Definition: {category_definition}

Question: based ONLY on the {context_label} above, is this conclusion correct -- does the \
criticism of "{keyword}" genuinely come from {category_phrase}?
Answer with exactly one letter and nothing else:
{letter_true} = True
{letter_false} = False\
"""


def _build_validation_prompt(
    claimed_category: str, keyword: str, article_text: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """order=1 -> A=True/B=False ; order=2 -> A=False/B=True. Reuses
    LETTERS/letters_for_order verbatim -- same letter-bias cancellation
    as pass 2, just relabelled True/False instead of Yes/No."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_true, letter_false = letters_for_order(order)
    return VALIDATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase=CATEGORY_PHRASE[claimed_category],
        category_definition=CATEGORY_DEFINITION[claimed_category],
        letter_true=letter_true, letter_false=letter_false,
    )


def build_validation_prompt(
    row: pd.Series, text_col: str, claimed_category: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The real, row-grounded validation question: does pass 2's own
    chosen category for this row hold up under one holistic re-read of
    the same summary? `claimed_category` is whatever pass 2 wrote to
    `source_category` for this row -- the caller reads it off the row
    and passes it in, this function doesn't look it up itself."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_validation_prompt(claimed_category, keyword, article_text, order, context_label=context_label)


def build_validation_calibration_prompt(
    claimed_category: str, order: int, context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """Same empty-summary neutralisation as build_calibration_prompt, for
    pass 3's own question shape -- measures, once per claimed category
    (not per row), how often THIS validation question's wording defaults
    to True independent of any real content."""
    return _build_validation_prompt(claimed_category, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order, context_label=context_label)
