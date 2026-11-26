# src/step4b_logit_prompts_remote.py
"""
STEP 4b (logit-probing design) -- REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared Apertus-v1.5-70B
(via --backend litellm, the UNIL/CSCS proxy) against Apertus-8B-Instruct-2509
(local GPU) on step 3 and found the 70B remote model gives better results
there with a broader prompt (see src/step3_prompts_remote.py's own module
docstring), Maxime wants to move the rest of the pipeline (step4b onward)
onto the same remote backend. This file exists purely so that any future
prompt experimentation for step4b's remote path never touches
src/step4b_logit_prompts.py, which must stay exactly as it is: the
extensively-tuned production wording for the local 8B/vllm/transformers
path (see that file's own module docstring for its full tuning history --
the "summarize-then-probe" pre-pass, the category-boundary sharpening
rounds, the "validate-and-correct" pass 3, etc.).

As of this creation, NO WORDING HAS BEEN CHANGED ANYWHERE BELOW: the
summary pre-pass prompt, all 6 category question templates, the
calibration templates, and the pass-3 validation templates are all
byte-identical to src/step4b_logit_prompts.py at the time of this
duplication -- only this module-level docstring (documentation, not
prompt content) was written fresh, replacing that file's own tuning-
history docstring, which documents a history specific to that file and
not (yet) to this one. This is intentionally just a safe, isolated
starting point for later remote-specific tuning, not a tuning pass
itself.

2026-11-25 "summary reliability" pass -- SUMMARY_TEMPLATE only, nothing
else in this file touched. Diagnosed by reading all 72 (article, generated
source_summary) pairs from job21026402's oracle population against the
final classification outcome: every row with an imperfect summary ended up
misclassified (15/15), while a faithful summary still left a ~21% residual
error rate (12/56) attributable to pass 2's probes rather than the summary
-- so summary quality is a real, separate lever worth pulling on its own.
Four GENERAL weaknesses recurred across multiple, unrelated articles (not
single one-off mistakes, so fixed as general principles, not per-case
patches):
  1. Inferring/asserting a personal-vs-institutional framing the article
     itself never actually states (e.g. adding a committee/session context
     not supported by the text) -- now only stated when the article makes
     it explicit, "not specified by the article" otherwise.
  2. Paraphrasing a named, specific actor (a person, a party, an
     organization) into a vaguer generic description -- now the article's
     own exact name/party/organization must be kept, never generalised.
  3. Describing an anonymous or pseudonymous commenter as if they were an
     identified individual -- now flagged explicitly when the source is
     not a real identified name.
  4. Truncation risk (SUMMARY_MAX_NEW_TOKENS is a fixed sbatch budget, see
     sbatch_step4b_logit_remote_test.sh) cutting off a summary mid-sentence
     before the decisive fact was written -- mitigated by asking for a
     tighter, complete answer; raising that sbatch budget is a separate,
     complementary fix on the launch-script side, not a prompt change.
A fifth, structural issue (an article with more than one distinct critic
of the same keyword, for different reasons) is only made VISIBLE by this
pass (each critic now described separately instead of blended into one),
not resolved -- which one gold then treats as "the" source for that row is
a separate, unresolved question this prompt change does not attempt to
settle.

scripts/step4b_source_category_logit.py picks this module instead of
src/step4b_logit_prompts.py only when --backend litellm is used; every
other backend (vllm/transformers) is completely unaffected, byte-for-byte,
by anything in this file. Unlike step3/step4a, this step's entire
machinery (summary generation + 6-category logit probe + validate-and-
correct) lives in ONE prompt module in the original design, so the whole
file is duplicated here rather than splitting a "shared validation" file
out separately.
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

Summarize, in 2 to 4 short, complete sentences (a few more only if point 4 below applies and \
more than one distinct critic must be described separately):
1. What exactly "{keyword}" is being criticized for.
2. Who or what is making this criticism -- name the specific actor, institution, or \
organization EXACTLY as the article identifies them (the precise person's name, party, \
office, or organization -- never a vaguer paraphrase or generalisation of it, and never \
"{keyword}" itself, which is only the target, never the source, of the criticism). If the \
source is not identified by a real name -- an anonymous commenter, a pseudonym, an unnamed \
"reader" or "expert" -- say so explicitly instead of describing them as if they were an \
identified individual.
3. Whether this actor is speaking personally / in their own capacity, or explicitly in the \
name of an institution or organization -- this distinction matters more than their job \
title. State this ONLY when the article itself makes it clear either way; if the article \
does not say, write that it is not specified rather than inferring or assuming one or the \
other.
4. If the article presents more than one distinct actor independently criticizing \
"{keyword}" for separate reasons, describe each one on its own (following points 1-3 for \
each) instead of merging them into a single blended description.

Do NOT classify the source into a category yourself -- just describe who they are and what \
they said, as factually and specifically as possible. If the article does not clearly name \
who is criticizing "{keyword}", say so explicitly rather than guessing. Keep the answer \
tight and self-contained -- a short, complete answer is better than a longer one that risks \
being cut off before it is finished.\
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

# Pass-3-ONLY reminder, keyed by claimed_category, empty ("") for any
# category not listed. See the module docstring's 2026-10-28b changelog
# entry: these are the specific clauses pass 3 was shown to under-apply
# even though pass 2 (reading the identical CATEGORY_DEFINITION text)
# applies them correctly -- evidence-driven, not a general restatement of
# the definition. Deliberately NOT added to QUESTION_TEMPLATE/_build_prompt
# (pass 2), which is not broken here.
VALIDATION_REMINDER = {
    "Interest Group": (
        "\nReminder for this category specifically: do NOT reject this just because the "
        "source is described as speaking \"personally\", informally, or without naming a "
        "formal organization -- that framing does not by itself rule out Interest Group. "
        "(1) A business executive, investor, or company representative giving their own "
        "view on economic, financial, or regulatory policy is still Interest Group -- "
        "their professional/economic interest is what counts, exactly as in the second "
        "example above, even when the article frames it as a personal opinion rather than "
        "an official statement. (2) A group of several named or unnamed activists/"
        "protesters acting together is still Interest Group (an informal advocacy group), "
        "even if no formal institution is named -- acting as a collective is what matters, "
        "not formal registration. (3) A media outlet or journalistic investigation raising "
        "the criticism also counts, even when the underlying topic is a government body's "
        "own decision -- what matters is WHO is voicing the criticism, not what it is "
        "about.\n"
    ),
}

VALIDATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the {context_label} \
below, "{keyword}" is criticized.

{context_label}:
{article_text}

A prior automated analysis concluded that the source of this criticism belongs to the \
following category:

CLAIMED CATEGORY: {category_phrase}
Definition: {category_definition}{validation_reminder}

Example (answer: True): {yes_example_1}
Example (answer: True): {yes_example_2}
Example (answer: False -- the source is actually {no_cat_1} instead): {no_example_1}
Example (answer: False -- the source is actually {no_cat_2} instead): {no_example_2}
Example (answer: False -- the source is actually {no_cat_3} instead): {no_example_3}
Example (answer: False -- the source is actually {no_cat_4} instead): {no_example_4}
Example (answer: False -- the source is actually {no_cat_5} instead): {no_example_5}

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
    as pass 2, just relabelled True/False instead of Yes/No.

    2026-10-28: now also includes the SAME CANONICAL_EXAMPLES (2 "yes"
    illustrations of claimed_category + 5 contrastive "no, this is
    actually {other category}" illustrations) that pass 2's own
    _build_prompt has always used -- see the module docstring's
    2026-10-28 changelog entry for why: a row-by-row read of job66621913's
    pass-3 errors found that in most cases the summary text ALREADY
    contained everything needed to apply a boundary rule we'd already
    written (e.g. "Trump is a foreign head of state" -> Other; a named
    CEO speaking personally -> Interest Group per its business-executive
    clause), yet pass 3 -- asking the SAME information a second time,
    holistically -- applied these rules less reliably than pass 2 did on
    the very first read. The one structural difference between the two
    prompts was these examples: pass 2 had them, pass 3 didn't, despite
    both reading identical CATEGORY_DEFINITION text. This is a prompt
    fix, not an information-source fix -- still reads source_summary,
    not the raw article.

    2026-10-28b: also inserts VALIDATION_REMINDER[claimed_category] (empty
    string for any category not in that dict) right after the definition
    -- see the module docstring's 2026-10-28b entry for the specific,
    evidence-driven clauses this reinforces for 'Interest Group'."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_true, letter_false = letters_for_order(order)

    other_cats = [c for c in CATEGORIES if c != claimed_category]
    no_examples = [CANONICAL_EXAMPLES[c][0] for c in other_cats]

    return VALIDATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase=CATEGORY_PHRASE[claimed_category],
        category_definition=CATEGORY_DEFINITION[claimed_category],
        validation_reminder=VALIDATION_REMINDER.get(claimed_category, ""),
        yes_example_1=CANONICAL_EXAMPLES[claimed_category][0],
        yes_example_2=CANONICAL_EXAMPLES[claimed_category][1],
        no_cat_1=other_cats[0], no_example_1=no_examples[0],
        no_cat_2=other_cats[1], no_example_2=no_examples[1],
        no_cat_3=other_cats[2], no_example_3=no_examples[2],
        no_cat_4=other_cats[3], no_example_4=no_examples[3],
        no_cat_5=other_cats[4], no_example_5=no_examples[4],
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
