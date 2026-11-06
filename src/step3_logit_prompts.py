# src/step3_logit_prompts.py
"""
STEP 3 — pass 2 "validate-on-full-article" (2026-11-02+), logit-probing.

Context: a deep manual read of job66790113's 148 step3 errors (789-row
comparable population, gold corrected) found a heavily FP-skewed pattern --
109 false positives ("{keyword}" flagged as criticized when gold says NO)
against only 39 false negatives, concentrated almost entirely on named
Federal Councillors (Keller-Sutter 11, Rösti 10, Baume-Schneider 9, Cassis
7, Jans 5, Parmelin 5, Pfister 5 -- all 7 Federal Councillors appear). A
close reading of ~20 of these FPs' own pass-1 justifications (all quoted
verbatim in the user-facing analysis) showed the SAME root cause every
time: pass 1 treats ANY negative-valence or controversy-adjacent mention of
the named entity as "criticism of it personally", even when (a) the fault
actually belongs to the collective government or a different, specific
actor, (b) it is ordinary political/policy disagreement rather than a
reproach, (c) the entity is a victim, not at fault, (d) the mention is
purely neutral/factual (a poll ranking, a succession scenario, a report
presenter), or (e) the entity's own statement/warning is later vindicated
and still gets coded as criticism of it.

`src/step3_prompts.py`'s own module docstring documents roughly a dozen
rounds of pass-1 prompt tuning since 2026-09-29, every attempt to add
exclusion rules or worked examples directly to pass 1's SYSTEM_PROMPT/
USER_TEMPLATE either reverted or measured net-negative -- adding a clause
for one FP shape reliably un-tuned something else (the "non-local
recalibration" effect documented there). A generation-based pass 2 was
also tried and removed on 2026-10-13: it read ONLY pass 1's own
justification text, never the article, so it could only react to however
pass 1's justification happened to frame things -- it could not
independently catch a target-attribution confusion pass 1 had already
made, and its blind spot was symmetric (NO->YES overrides were just as
unreliable as YES->NO ones: on job65556910, 30 NO->YES overrides, 24
wrong).

This pass 2 is deliberately NOT a repeat of either abandoned approach:

  1. It re-reads the FULL ORIGINAL ARTICLE, not pass 1's justification --
     genuinely new information for a second, independent read, the same
     reason step4b's own validation pass was found most promising when
     given the raw article rather than a lossy summary.
  2. It is scoped ONLY to rows pass 1 already said YES to (never NO), by
     design -- since the measured problem is FP-skewed roughly 3:1, a
     pass that can only ever turn a YES into a NO cannot create a new
     false negative on a row pass 1 got right, and cannot touch the
     (much rarer, structurally different -- see the FN root-cause write-
     up: self-criticism and institutional-ownership misses, not over-
     eager flagging) false-negative problem at all. That is an accepted,
     deliberate scope limit, not an oversight.
  3. It does not touch pass 1's SYSTEM_PROMPT/USER_TEMPLATE at all (zero
     risk of the non-local recalibration effect that sank every previous
     prompt-tuning round on THIS file, since that effect only bites when
     the SAME instruction block governs every row).
  4. Mechanically, it reuses the logit-probing design already proven out
     for step4b (src/step4b_logit_utils.py, scripts/step4b_source_category_logit.py):
     one forward pass per (row, order), no generation/parsing failure
     mode, order-swap averaging to cancel letter-position bias, and
     context-free calibration (Zhao et al. 2021) against an empty-
     article variant of the same question. Binary here (YES/NO), not a
     6-way category choice, so there is only one calibration value (p0)
     to measure per task, not six.

The underlying question and its worked examples are carried over VERBATIM
from src/step3_prompts.py's SYSTEM_PROMPT/USER_TEMPLATE (the version in
force since the 2026-10-12 revert, i.e. the one this file's pass 1 run
actually used) -- this file does not re-derive new wording, it reuses the
already-tuned definition and all 9 existing worked examples, just
reformatted for a single calibrated-logit question with order-swap letters
instead of free-text generation with a justification. Nothing here edits
src/step3_prompts.py.

2026-11-06 "validate-on-justification" ablation: job20002898 (the
"article" mode above, --validation_input=article) measured clearly net-
negative against the corrected gold -- 218/236 scored candidates rejected
(0 kept), of which only 96 were genuine FPs correctly caught but 122 were
genuine TPs wrongly turned into new FNs (step3 accuracy 77.9%, BELOW the
78.7% naive-NO baseline; micro-accuracy over the full cascade fell from
68.8% to 44.2%). The job's own printed p0 (0.3932) ruled out a degenerate
calibration baseline as the cause -- the raw (pre-calibration) P(yes) read
off the REAL article was itself crushed to ~0.17 on average even for true
positives, while the content-free "N/A" calibration article alone scored
HIGHER (p0=0.39) than almost every real read. A weak-but-real directional
signal survived (mean p_true 0.0755 for true-YES vs 0.0570 for true-NO),
so the model is not reading randomly, just far too conservatively on real
articles. Leading hypothesis: a real, long, multi-paragraph Swiss news
article typically mentions the keyword entity in several neutral/
distractor contexts (quotes, policy statements, routine mentions) in
addition to the one passage that actually criticizes it, whereas the 10
worked examples are all short, single-passage, clean scenarios -- forcing
the model to locate the one relevant passage itself, from scratch, inside
a noisy real article is a much harder retrieval problem than judging a
short clean scenario, and the model appears to default toward the
"neutral mention" (NO) pattern when it can't confidently relocate the
criticism.

This ablation tests the fix suggested by that diagnosis directly: give the
validation pass pass 1's own JUSTIFICATION instead of the raw article --
pass 1 has already done the retrieval work of locating and stating the
specific claimed criticism in one or two clean sentences, so the
validation question becomes "does this stated reasoning actually hold up
under the definition", not "can you re-find the needle in the haystack".
This is NOT simply reverting to the pre-2026-10-13 removed pass 2:
  - That pass 2 ran on ALL rows (both YES and NO) and used free-text
    generation with no calibration; this ablation keeps the YES-only scope
    (can only ever flip YES->NO, by construction) and the full logit-
    probing/order-swap/context-free-calibration machinery.
  - The question itself is reframed as an explicit VALIDATION REQUEST
    ("does this justification validate the claim") rather than a
    confirm/reject-a-prior-conclusion framing, to avoid any residual
    anchoring-on-an-external-claim effect.
  - The acknowledged risk carried over from that old pass 2's own history
    still applies and is not eliminated by this change: reading only pass
    1's own justification means this pass can only ever judge pass 1's
    OWN framing of the evidence, not independently re-derive it from the
    source -- if pass 1's justification itself already launders a target-
    attribution mistake into confident-sounding prose, this pass cannot
    catch that specific failure mode. It can still catch cases where the
    justification, read coldly against the strict definition, doesn't
    actually support a YES (e.g. restates a neutral mention or someone
    else's fault) even though pass 1 labeled it YES.
Toggle via --validation_input article|justification (default: article,
unchanged) -- see scripts/step3_criticism_detection.py.
"""
from __future__ import annotations
import pandas as pd

LETTERS = ("A", "B")

# Verbatim from src/step3_prompts.py's SYSTEM_PROMPT (unchanged since the
# 2026-10-12 revert -- see that file's own module docstring for the full
# tuning history behind this exact wording).
VALIDATION_DEFINITION = """\
Be careful to distinguish an entity being criticized from an entity that is itself doing the \
criticism, or one that is simply mentioned in a neutral or unrelated context. Only answer YES \
if the article itself attributes fault, blame, failure, or wrongdoing directly to this specific \
entity: an entity that merely faces a difficult or negative situation, carries out its normal \
role, is cited as a neutral source, states its own position or decision, or is mentioned near \
criticism that explicitly targets a different, specific person, institution, or policy, is not \
itself being criticized.\
"""

# Verbatim from src/step3_prompts.py's USER_TEMPLATE worked examples (same
# 9, same order, same wording) -- reformatted as (scenario, label,
# explanation) triples for this file's own template below.
VALIDATION_EXAMPLES = [
    ("A parliamentary committee accuses the ministry of mismanaging a defense procurement "
     "project it runs, citing cost overruns.",
     "YES", "the ministry IS criticized: the reproach targets a project that is its own direct "
     "responsibility, even though the ministry itself is not named."),
    ("Critics say the mayor's housing policy has worsened the rent crisis in the city.",
     "YES", "the mayor IS criticized: a clear reproach targets her decision."),
    ("\"The agency has been negligent and failed to act in time,\" the auditor's report states.",
     "YES", "the agency IS criticized: a direct, explicit reproach."),
    ("The agency reports a 3% increase in applications this year.",
     "NO", "a plain neutral fact, no reproach is expressed."),
    ("The minister dismisses the opposition's proposal as unrealistic.",
     "NO", "the minister is the one doing the criticizing here, not the target."),
    ("The report notes that the department was informed of the problem and tasked with "
     "following up.",
     "NO", "being informed or involved is not itself a reproach."),
    ("\"The agency's own negligence and repeated failures to act allowed the crisis to "
     "worsen,\" the audit concludes.",
     "YES", "the fault is placed directly on the agency itself, not merely on the situation."),
    ("A parliamentary committee criticizes a new government savings package; the office that "
     "helped carry out part of it is only mentioned as having implemented measures decided "
     "elsewhere.",
     "NO", "the criticism targets the package/decision, not this office, which is not itself "
     "named as being at fault."),
    ("The agency's own internal report warns that, without additional funding, its insurance "
     "fund could become insolvent within a few years.",
     "NO", "it is disclosing a risk about its own situation, and nobody in the text blames it "
     "for that risk."),
    ("The financial oversight office questioned the fixed-price terms of a defense contract; "
     "the defense agency firmly rejected its findings.",
     "NO", "the oversight office is the one raising the criticism, and having its findings "
     "disputed by the party it criticized does not make it the target."),
]

VALIDATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. A prior automated \
analysis concluded that "{keyword}" IS criticized in the article below. Your task is to \
independently re-examine the article and confirm or reject that conclusion.

{definition}

ARTICLE:
{article_text}

{examples_block}

Question: based ONLY on the article above, is "{keyword}" genuinely, itself, being criticized \
in it -- for who it is or what it did or decided? Do not confuse this with "{keyword}" being \
the one expressing criticism of someone or something else, or being only mentioned in passing.
Answer with exactly one letter and nothing else:
{letter_yes} = YES, the prior conclusion is correct
{letter_no} = NO, the prior conclusion is wrong\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_ARTICLE = "N/A"

# 2026-11-06 "validate-on-justification" ablation (see module docstring).
# Reads only pass 1's own keyword_justification, reframed as an explicit
# validation request rather than a confirm/reject-a-prior-conclusion
# question, to test whether a shorter, already-focused input avoids the
# near-universal over-rejection measured in --validation_input=article
# mode (job20002898).
VALIDATION_QUESTION_TEMPLATE_JUSTIFICATION = """\
You are a media analysis assistant specialised in Swiss public affairs. Below is the \
justification an automated analysis wrote to support its claim that "{keyword}" is criticized \
in a news article. Your task is to validate whether this justification actually holds up.

{definition}

JUSTIFICATION TO VALIDATE:
{justification_text}

{examples_block}

Question: does the justification above genuinely establish, under the definition given, that \
"{keyword}" is itself being criticized -- for who it is or what it did or decided? Validate it \
ONLY if the reasoning given actually shows direct fault, blame, failure, or wrongdoing \
attributed to "{keyword}" itself -- not a difficult situation it merely faces, a neutral \
mention, its own statement or decision, or criticism that in fact targets someone or something \
else.
Answer with exactly one letter and nothing else:
{letter_yes} = YES, this justification is validated
{letter_no} = NO, this justification is not validated\
"""

CALIBRATION_JUSTIFICATION = "N/A"


def _examples_block() -> str:
    lines = []
    for text, label, explanation in VALIDATION_EXAMPLES:
        lines.append(f'- "{text}" -> {label}: {explanation}')
    return "\n".join(lines)


_EXAMPLES_BLOCK = _examples_block()


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_yes, letter_no) for the given order -- order=1 ->
    A=Yes/B=No ; order=2 -> A=No/B=Yes. Same bias-cancellation idea as
    step4b's own letters_for_order, duplicated locally so this file stays
    self-contained (no cross-step import)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])


def _build_prompt(keyword: str, article_text: str, order: int) -> str:
    letter_yes, letter_no = letters_for_order(order)
    return VALIDATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        definition=VALIDATION_DEFINITION,
        article_text=article_text,
        examples_block=_EXAMPLES_BLOCK,
        letter_yes=letter_yes,
        letter_no=letter_no,
    )


def build_validation_prompt(row: pd.Series, text_col: str, order: int) -> str:
    """The real, row-grounded validation question: does pass 1's own YES
    verdict for this row hold up under an independent, full-article
    re-read? `text_col` should be the RAW ARTICLE column (never a
    summary) -- the whole point of this pass is genuinely new
    information, not a re-read of pass 1's own justification."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_prompt(keyword, article_text, order)


def build_validation_calibration_prompt(order: int) -> str:
    """Same empty-article neutralisation as step4b's calibration prompts --
    measures, once per task (not per row), how often this validation
    question's wording defaults to YES independent of any real content."""
    return _build_prompt(CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order)


def _build_prompt_from_justification(keyword: str, justification_text: str, order: int) -> str:
    letter_yes, letter_no = letters_for_order(order)
    return VALIDATION_QUESTION_TEMPLATE_JUSTIFICATION.format(
        keyword=keyword,
        definition=VALIDATION_DEFINITION,
        justification_text=justification_text,
        examples_block=_EXAMPLES_BLOCK,
        letter_yes=letter_yes,
        letter_no=letter_no,
    )


def build_validation_prompt_from_justification(row: pd.Series, order: int) -> str:
    """2026-11-06 ablation: same validation question, but grounded in pass
    1's own keyword_justification instead of the raw article -- see this
    file's module docstring for why. `row` must already have a non-null
    keyword_justification (callers should filter for that, same as they
    already filter for keyword_answer == "YES" and not-forced)."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row.get("keyword_justification")) else str(row["keyword_justification"]).strip()
    return _build_prompt_from_justification(keyword, justification, order)


def build_validation_calibration_prompt_from_justification(order: int) -> str:
    """Context-free calibration for the justification-input variant --
    same idea as build_validation_calibration_prompt, but using the
    neutral justification-shaped placeholder instead of an empty article."""
    return _build_prompt_from_justification(CALIBRATION_KEYWORD, CALIBRATION_JUSTIFICATION, order)
