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

Category definitions below are adapted from src/step4b_prompts.py's
SYSTEM_PROMPT (same six categories, same underlying "in whose name are
they speaking" test), but rewritten as STANDALONE paragraphs: the old
"ordered decision procedure, stop at first match" framing doesn't apply
here, since every category is now asked about independently and
symmetrically rather than checked in sequence.
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
        "rather than giving a personal opinion."
    ),
    "Politician": (
        "An elected or appointed political office-holder (a federal, cantonal or communal "
        "councillor, a member of parliament, a party leader, a cantonal or communal "
        "executive member) is speaking IN THEIR OWN NAME or political capacity -- their "
        "own opinion or stance, not their institution's official position. A political "
        "party issuing its own collective statement also counts, as does a FORMER elected "
        "politician commenting on public affairs -- their political identity does not "
        "disappear on retirement."
    ),
    "Interest Group": (
        "The actor speaks FOR an organized entity representing a particular interest: a "
        "private company, a business or professional association, a union, a lobby group, "
        "a think tank, an NGO, or an organized advocacy/protest group -- even an informal "
        "one, as long as it acts as a collective, not a lone individual. This excludes "
        "political parties/politicians and state administrations."
    ),
    "Civil Servant": (
        "A specific, named individual CURRENTLY employed by the state (an official, "
        "auditor, diplomat, police officer, etc.) gives their OWN personal opinion or "
        "account, explicitly NOT as their institution's formal position. A retired or "
        "former official with no current state role does NOT qualify here."
    ),
    "General Public": (
        "An individual speaking only for themselves, with no institution or organized "
        "interest behind them: an ordinary citizen, a survey respondent, an independent "
        "expert or researcher with no organizational affiliation, or a formerly "
        "state-employed person (other than a former politician) with no current "
        "institutional role."
    ),
    "Other": (
        "Used ONLY when a criticism is clearly present in the article but its source is "
        "vague or unnamed -- e.g. the article says someone 'has faced mounting criticism "
        "in recent weeks' without identifying who is criticizing, and the journalist is "
        "not taking personal ownership of the claim either."
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
        "An environmental NGO issues a statement condemning a construction project as "
        "damaging to local wildlife.",
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
        "An independent economist with no organizational affiliation argues in an opinion "
        "piece that a proposed tax reform is poorly designed.",
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
