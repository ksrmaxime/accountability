# src/step4b_logit_prompts.py
"""
STEP 4b — logit-probing redesign (2026-10-26+).

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
You are a media analysis assistant specialised in Swiss public affairs. You are given a \
newspaper article in which "{keyword}" is criticized, and asked whether the criticism \
comes from a specific type of source.

ARTICLE:
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

Question: in the ARTICLE above, is the criticism of "{keyword}" genuinely coming from \
{category_phrase}?
Answer with exactly one letter and nothing else:
{letter_yes} = Yes
{letter_no} = No\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_ARTICLE = "N/A"


def _build_prompt(category: str, keyword: str, article_text: str, order: int) -> str:
    """order=1 -> A=Yes/B=No ; order=2 -> A=No/B=Yes (see LETTERS)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_yes, letter_no = LETTERS if order == 1 else (LETTERS[1], LETTERS[0])

    other_cats = [c for c in CATEGORIES if c != category]
    no_examples = [CANONICAL_EXAMPLES[c][0] for c in other_cats]

    return QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
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


def build_question_prompt(row: pd.Series, text_col: str, category: str, order: int) -> str:
    """The real, article-grounded question for one row and one category,
    in one of the two letter orderings."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_prompt(category, keyword, article_text, order)


def build_calibration_prompt(category: str, order: int) -> str:
    """The SAME question with the article and target neutralised -- used
    ONCE per category (not per row) to measure that category's own
    intrinsic yes-bias, independent of any article content. See
    src/step4b_logit_utils.calibrate."""
    return _build_prompt(category, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order)


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_yes, letter_no) for the given order, mirroring
    _build_prompt's own logic -- used by the pipeline script to know
    which letter's probability is P(yes) for a given order."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])
