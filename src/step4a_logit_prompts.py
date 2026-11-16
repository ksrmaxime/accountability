# src/step4a_logit_prompts.py
"""
STEP 4a -- validation pass (2026-11-16+), logit-probing.

Context: step4a's pass 1 (src/step4a_prompts.py, free-text generation,
unchanged by this file) is well-tuned after 7 rounds of prompt revision
since 2026-10-02, but job20479169 (800-row gold comparison, the current
best step3 lineage) still shows a clear, directional error pattern on its
95 YES-gated rows: 20 FN (gold=SOURCED, Apertus=JOURNALIST) vs only 4 FP
(gold=JOURNALIST, Apertus=SOURCED) -- accuracy 74.7%, actually BELOW the
82.1% majority-class ("always SOURCED") baseline. Reading these 20 FN
justifications shows pass 1 reasoning explicitly about an "outside voice"
and still concluding JOURNALIST, sometimes with the word "SOURCED" itself
appearing earlier in the same justification before the final JOURNALIST
conclusion -- i.e. pass 1 is aware of the question's substance but
resolves close calls toward its documented default (JOURNALIST, the exact
collapse-to-default failure mode this file's SYSTEM_PROMPT/USER_TEMPLATE
history has fought since 2026-10-02).

Design choice -- why a SEPARATE validation pass, not a first-pass logit
replacement (step4b's design): step4a's pass 1 free-text justification
("search for and name a candidate actor FIRST") is explicitly what fixed
the ORIGINAL one-shot collapse-to-JOURNALIST failure (see
src/step4a_prompts.py's 2026-10-02 entry) -- a single calibrated-logit
forward pass with no generated reasoning step would reproduce exactly
that one-shot structure. Keeping pass 1 untouched and adding a second,
calibrated pass that re-reads pass 1's own justification (mirroring
step3's "validate-on-justification" design, src/step3_logit_prompts.py)
preserves the reasoning step while adding a continuous, calibrated,
threshold-tunable signal instead of a second blind generation call.

Why this is NOT a repeat of step4a's own removed pass 2 (reverted
2026-10-13, see scripts/step4a_source_attribution.py): that pass 2 used
free-text generation + a binary regex rule ("names ANY actor, even
briefly, counts as SOURCED") with no calibration -- it flipped
JOURNALIST->SOURCED indiscriminately whenever pass 1's justification
happened to name the article's own byline journalist, with no way to
weigh how confidently it should fire. This pass reuses step4b's proven
logit-probing machinery instead (one forward pass per order, order-swap
averaging, context-free calibration against a neutral placeholder) to
get a continuous P(should be SOURCED) and a tunable decision threshold,
rather than an unconditional rule.

Direction -- the MIRROR of step3's validation pass, not a copy: step3's
pass 1 over-predicted YES (FP-heavy), so its validation pass could only
ever flip YES->NO. step4a's pass 1 under-predicts SOURCED (FN-heavy, 20
vs 4) -- the opposite imbalance -- so this validation pass is scoped to
rows where pass 1 said JOURNALIST, and can only ever flip
JOURNALIST->SOURCED, never the reverse. This cannot create a new error on
a row pass 1 already got right as SOURCED (58/58 correct on job20479169),
and cannot touch the much rarer JOURNALIST->SOURCED-should-stay-JOURNALIST
false-positive direction (only 4 rows) by construction.

Validation question reuses step4a's OWN already-tuned definition and all
6 worked examples verbatim from src/step4a_prompts.py's SYSTEM_PROMPT/
USER_TEMPLATE (unchanged since the 2026-10-20 revert) -- same convention
as step3_logit_prompts.py: this file does not re-derive new wording, it
reframes the already-tuned question for a single calibrated-logit read
with order-swap letters instead of free-text generation. Nothing here
edits src/step4a_prompts.py.

--validation_input article|justification (default: justification, by
analogy with step3's own finding that re-reading the full article a
second time measured net-negative there -- not yet independently
confirmed for step4a, this default is a prior, not a result. If a real
run shows justification-mode underperforms here, article mode is kept
available precisely to test that).

Toggle via --validation_input, --validation_threshold,
--skip_validation -- see scripts/step4a_source_attribution.py.
"""
from __future__ import annotations
import pandas as pd

LETTERS = ("A", "B")

# Verbatim from src/step4a_prompts.py's SYSTEM_PROMPT (unchanged since the
# 2026-10-20 revert -- see that file's own module docstring for the full
# tuning history behind this exact wording).
VALIDATION_DEFINITION = """\
A criticism of a target is SOURCED when it is attributed to an outside voice -- a source \
external to the journalist, however broadly or narrowly identified (a named person, a group, an \
institution, a category of people, etc.). It is JOURNALIST when it is simply the journalist's own \
narrative framing, with no outside voice behind it. An outside voice can be identified as loosely \
as a named individual, a political party, a canton, an administration, a company, an association, \
a category of people ("residents," "opponents," "experts"), or survey respondents -- none of this \
needs a specific person's name to qualify. The target itself never counts as that voice just \
because it is named as the one who made the decision, took the action, or is otherwise the \
subject of the criticism. Two things do NOT settle the question by themselves: a signed byline \
does not mean there is no outside source behind the criticism -- a signed piece can still quote \
or cite someone else's criticism of the target. And sources quoted elsewhere in the article, for \
other claims or about other people, do not count unless they are specifically the ones credited \
with the criticism of THIS target.\
"""

# Verbatim from src/step4a_prompts.py's USER_TEMPLATE worked examples (same
# 6, same order, same wording) -- reformatted as (scenario, label,
# explanation) triples for this file's own template below.
VALIDATION_EXAMPLES = [
    ("Residents interviewed for the piece say the new bus route is useless and badly planned.",
     "SOURCED", "an outside group (\"residents\") is credited with the criticism, even though no "
     "individual is named."),
    ("This signed column, after quoting opposition MP Jane Doe calling the reform 'reckless', "
     "goes on to describe the minister's week.",
     "SOURCED", "even though the piece is signed, the criticism itself is attributed to the "
     "quoted MP, not to the columnist."),
    ("The cantonal government and a coalition of municipalities jointly oppose the project, "
     "citing its excessive cost.",
     "SOURCED", "a named collective is credited with the criticism, even with no individual "
     "inside it named."),
    ("A year-end roundup lists, among several federal councillors, that Martin Pfister had a "
     "difficult year.",
     "JOURNALIST", "Pfister's name appears only as the subject being discussed; no one is "
     "credited with criticizing him."),
    ("The piece quotes an economist about the state of the labour market, then separately "
     "argues in its own voice that the minister's reform is misguided.",
     "JOURNALIST", "an outside voice is quoted, but not for the criticism of the minister -- "
     "that judgment is the article's own."),
    ("The piece argues, in its own narrative voice, that the reform is short-sighted and will "
     "hurt small businesses, without attributing that judgment to any critic, resident, expert, "
     "or other outside voice.",
     "JOURNALIST", "no outside voice is credited with the criticism at all."),
]


def _examples_block() -> str:
    lines = []
    for text, label, explanation in VALIDATION_EXAMPLES:
        lines.append(f'- "{text}" -> {label}: {explanation}')
    return "\n".join(lines)


_EXAMPLES_BLOCK = _examples_block()


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_sourced, letter_journalist) for the given order --
    order=1 -> A=SOURCED/B=JOURNALIST ; order=2 -> A=JOURNALIST/B=SOURCED.
    Same bias-cancellation idea as step3/step4b's own letters_for_order,
    duplicated locally so this file stays self-contained."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])


# --- "validate-on-article" variant: re-reads the full original article
# (never pass 1's justification). Kept available via --validation_input
# article, same reason step3 keeps its own article-mode toggle, but NOT
# the default -- see module docstring. ---

VALIDATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. A prior automated \
analysis concluded that, in the article below, the criticism of "{keyword}" is NOT attributed to \
any outside voice (i.e. it is JOURNALIST -- the article's own narrative framing). Your task is to \
independently re-examine the article and confirm or reject that conclusion.

{definition}

ARTICLE:
{article_text}

{examples_block}

Question: based ONLY on the article above, is the criticism of "{keyword}" specifically \
attributed to ANY outside voice (named or not), making the prior JOURNALIST conclusion wrong -- \
or does no outside voice back it, confirming JOURNALIST?
Answer with exactly one letter and nothing else:
{letter_sourced} = SOURCED, the prior JOURNALIST conclusion is wrong, an outside voice IS credited
{letter_journalist} = JOURNALIST, the prior conclusion is correct, no outside voice is credited\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_ARTICLE = "N/A"


def _build_prompt(keyword: str, article_text: str, order: int) -> str:
    letter_sourced, letter_journalist = letters_for_order(order)
    return VALIDATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        definition=VALIDATION_DEFINITION,
        article_text=article_text,
        examples_block=_EXAMPLES_BLOCK,
        letter_sourced=letter_sourced,
        letter_journalist=letter_journalist,
    )


def build_validation_prompt(row: pd.Series, text_col: str, order: int) -> str:
    """The real, row-grounded validation question: does pass 1's own
    JOURNALIST verdict for this row hold up under an independent,
    full-article re-read? `text_col` should be the RAW ARTICLE column."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_prompt(keyword, article_text, order)


def build_validation_calibration_prompt(order: int) -> str:
    """Same empty-article neutralisation as step3/step4b's calibration
    prompts -- measures, once per task, how often this validation
    question's wording defaults to SOURCED independent of any real
    content."""
    return _build_prompt(CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order)


# --- "validate-on-justification" variant (DEFAULT, see module docstring):
# re-reads only pass 1's own source_attribution_justification. ---

VALIDATION_QUESTION_TEMPLATE_JUSTIFICATION = """\
You are a media analysis assistant specialised in Swiss public affairs. Below is the \
justification an automated analysis wrote to support its conclusion that, for "{keyword}", NO \
outside voice is credited with the criticism (i.e. it is JOURNALIST -- the article's own \
narrative framing). Your task is to validate whether this justification actually holds up.

{definition}

JUSTIFICATION TO VALIDATE:
{justification_text}

{examples_block}

Question: does the justification above, read against the definition, actually show that an \
outside voice (named or not) IS credited with the criticism of "{keyword}" -- making the prior \
JOURNALIST conclusion wrong -- or does it confirm that no outside voice backs it?
Answer with exactly one letter and nothing else:
{letter_sourced} = SOURCED, this justification shows an outside voice IS credited after all
{letter_journalist} = JOURNALIST, this justification is validated, no outside voice is credited\
"""

CALIBRATION_JUSTIFICATION = "N/A"


def _build_prompt_from_justification(keyword: str, justification_text: str, order: int) -> str:
    letter_sourced, letter_journalist = letters_for_order(order)
    return VALIDATION_QUESTION_TEMPLATE_JUSTIFICATION.format(
        keyword=keyword,
        definition=VALIDATION_DEFINITION,
        justification_text=justification_text,
        examples_block=_EXAMPLES_BLOCK,
        letter_sourced=letter_sourced,
        letter_journalist=letter_journalist,
    )


def build_validation_prompt_from_justification(row: pd.Series, order: int) -> str:
    """Grounded in pass 1's own source_attribution_justification instead
    of the raw article -- see this file's module docstring for why this
    is the default. `row` must already have a non-null
    source_attribution_justification (callers should filter for that,
    same as they already filter for source_attribution == "JOURNALIST"
    and not-forced)."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = (
        "" if pd.isna(row.get("source_attribution_justification"))
        else str(row["source_attribution_justification"]).strip()
    )
    return _build_prompt_from_justification(keyword, justification, order)


def build_validation_calibration_prompt_from_justification(order: int) -> str:
    """Context-free calibration for the justification-input variant --
    same idea as build_validation_calibration_prompt, but using the
    neutral justification-shaped placeholder instead of an empty
    article."""
    return _build_prompt_from_justification(CALIBRATION_KEYWORD, CALIBRATION_JUSTIFICATION, order)
