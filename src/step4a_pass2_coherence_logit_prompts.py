# src/step4a_pass2_coherence_logit_prompts.py
"""
STEP 4a -- pass 3, "pass2 coherence check" (2026-11-18), logit-probing.

Context: the 2026-11-17 "forced re-search" pass 2 (src/step4a_prompts.py's
VERIFY_SYSTEM_PROMPT/build_verify_prompt) measured a real net gain on
job20620441 (95-row YES-gated population: accuracy 75.8% -> 83.2%, now
for the first time ABOVE the 82.1% majority baseline). But reading all 32
rows pass 2 touched line by line found roughly half of its 27 flips to
SOURCED are self-contradictory: the model's OWN written reasoning
concludes things like "no outside voice is credited", "framed as the
author's own opinion", "framed by the journalist", or even explicitly
names the article's byline journalist as "the actor" -- textbook
JOURNALIST reasoning -- and then the bare final word it commits to is
SOURCED anyway. Measured precisely: of 27 flips, 11 are "clean" (the
reasoning genuinely names a real outside voice) and those are reliable,
82% precision (9/11 correct) -- exactly what the forced-enumeration idea
was for. The other 16 are this self-contradiction pattern, and behave
like a coin flip: 8 land right, 8 land wrong, purely because this
touched population happens to be ~59% SOURCED, not because the mechanism
found anything. Reverting every contradictory row to JOURNALIST (i.e.
trusting the reasoning over the lazy final word) gives the EXACT SAME
83.2% headline accuracy on this one run, via a much healthier FP=6/FN=9
(vs the actual FP=14/FN=2) -- i.e. today's gain is partly a real signal
and partly a coin flip that happened to net to zero. On a different
draw of rows that coin flip would not reliably land 50/50 -- it is noise,
not a result, and should not be trusted going forward as-is.

Two fixes were considered for this self-contradiction (user's question,
2026-11-18): deterministic Python rules, or a third model pass. Both
were actually tried against this run's real 32 justification texts
before picking one:

  Deterministic rules, attempt 1 (bag-of-phrases: JOURNALIST-leaning
  phrases vs SOURCED-leaning phrases, whichever the text contains):
  18/32 rows came out AMBIGUOUS -- the model very often mentions a
  candidate actor AND THEN negates their relevance in the same
  paragraph ("the article cites X and Y, but they are not credited with
  THIS specific criticism"), so a same-text keyword count cannot tell
  "found, and it counts" from "found, but it doesn't count".

  Deterministic rules, attempt 2 ("last occurring phrase wins", the same
  principle src/verify_utils.py's own _find_label already uses for
  pass 1): resolves the ambiguity, but changes 17/32 decisions, of which
  only 9 are real corrections and 8 are regressions -- it breaks almost
  as many already-correct answers as it fixes, because sentence order in
  a free-text paragraph is not a reliable signal of which clause is the
  model's real conclusion. Net effect on this run: +1 row out of 32 --
  indistinguishable from noise, and not something to trust on a
  different draw.

  Conclusion: detecting self-contradiction from the justification TEXT
  is itself a genuine reading-comprehension task, not a keyword-matching
  one -- which is exactly why this pipeline already uses a model (never
  hand-written rules) everywhere else it needs to classify a short piece
  of text (pass 2, throughout step3/step4b/step5/step6, per
  src/verify_utils.py's own module docstring). This file gives step4a's
  new pass 2 the same treatment, as a narrow THIRD pass.

Design -- why this is NOT a repeat of the 2026-11-16/17 validation pass
that was already tried and removed (src/step4a_logit_prompts.py, kept
there for the record): that removed pass re-read either pass 1's
justification or the full article and tried to answer the REAL-WORLD
question ("is this row actually SOURCED or JOURNALIST") from limited
information -- exactly the kind of long-range search a single logit read
cannot force. This pass asks something much narrower and already fully
answerable from the text in front of it: "what does THIS SHORT PARAGRAPH
(pass 2's own justification, ~150-350 characters) itself conclude?" --
pure reading comprehension of a paragraph that already states its own
conclusion in plain words, nothing left to search for. This is the same
"classify a short, clean, already-written text" pattern src/verify_utils.py
documents as the reason pass-2-reads-only-the-justification design works
well everywhere else in this pipeline.

Decision rule (see scripts/step4a_source_attribution.py): only OVERRIDE
pass 2's stated answer when this check's calibrated reading clearly
disagrees (a margin around the 0.5 decision boundary, --coherence_margin,
not yet empirically tuned -- a prior, like every other threshold in this
project's history). An unclear/inconclusive read, or one that agrees,
leaves pass 2's own answer untouched -- fail-safe, same convention as
every other validation/verification pass in this codebase.

Scope: only the rows pass 2 actually produced a parseable answer for
(both flips and confirms -- self-contradiction is possible in EITHER
direction in principle, even though only the SOURCED direction was
observed on job20620441, so this check is not one-directional the way
pass 2 itself is). Cheap: ~4% of eligible rows on job20620441 (32/800),
and the input here is a short justification, not the article, so no
large-context forward pass is needed.
"""
from __future__ import annotations
import pandas as pd

LETTERS = ("A", "B")

QUESTION_TEMPLATE = """\
You are a careful reading-comprehension assistant. Below is a short paragraph of reasoning that \
a prior automated step wrote while re-examining whether a criticism of "{keyword}" is attributed \
to an outside voice (SOURCED) or is just the article's own narrative framing, with no outside \
voice behind it (JOURNALIST).

REASONING TO READ:
{justification_text}

Read ONLY what this paragraph itself says. Do not re-decide the question yourself and do not use \
any outside knowledge about "{keyword}" -- your only job is to report what this paragraph's own \
stated conclusion is. Note: if the paragraph says the criticism is attributed to the article's \
author, its byline journalist, or "the journalist", that is NOT an outside voice -- the \
paragraph's own conclusion in that case is still that no outside voice is credited.

For example:
- "The article mentions Jane Doe, but she is not credited with this specific criticism. The criticism is framed as the author's own opinion." -> this paragraph's own conclusion is JOURNALIST (no outside voice).
- "After checking again, I found that the criticism is attributed to the opposition party, which is quoted criticizing the minister's decision." -> this paragraph's own conclusion is SOURCED (an outside voice is credited).
- "The criticism is attributed to the author of the piece, who is the journalist himself." -> this paragraph's own conclusion is JOURNALIST (the journalist is not an outside voice, even though the word "attributed" appears).
- "I found no named individual, group, or institution credited with this criticism; it remains the article's own narrative framing." -> this paragraph's own conclusion is JOURNALIST.
- "The cantonal governments are credited with raising this exact criticism in the article." -> this paragraph's own conclusion is SOURCED.
- "Several actors are mentioned in the article, but none of them are credited with this particular criticism." -> this paragraph's own conclusion is JOURNALIST.

Question: what is this paragraph's OWN stated conclusion -- that an outside voice IS credited with the criticism of "{keyword}" (SOURCED), or that NO outside voice is credited (JOURNALIST)?
Answer with exactly one letter and nothing else:
{letter_sourced} = this paragraph's own conclusion is SOURCED (an outside voice IS credited)
{letter_journalist} = this paragraph's own conclusion is JOURNALIST (no outside voice is credited)\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_JUSTIFICATION = "N/A"


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_sourced, letter_journalist) for the given order --
    order=1 -> A=SOURCED/B=JOURNALIST ; order=2 -> A=JOURNALIST/B=SOURCED.
    Same bias-cancellation idea as step3/step4b/the removed step4a
    validation pass's own letters_for_order, duplicated locally so this
    file stays self-contained."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])


def _build_prompt(keyword: str, justification_text: str, order: int) -> str:
    letter_sourced, letter_journalist = letters_for_order(order)
    return QUESTION_TEMPLATE.format(
        keyword=keyword,
        justification_text=justification_text,
        letter_sourced=letter_sourced,
        letter_journalist=letter_journalist,
    )


def build_coherence_prompt(row: pd.Series, order: int) -> str:
    """Grounded in pass 2's OWN source_attribution_pass2_justification --
    never the article, never pass 1's justification. Callers should
    filter for a non-null source_attribution_pass2_justification (same
    convention as every other logit-probing pass in this codebase)."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification_text = str(row["source_attribution_pass2_justification"]).strip()
    return _build_prompt(keyword, justification_text, order)


def build_coherence_calibration_prompt(order: int) -> str:
    """Same empty-input neutralisation as every other calibrated pass in
    this codebase -- measures, once per task, how often this reading-
    comprehension question's own wording defaults to SOURCED independent
    of any real paragraph content."""
    return _build_prompt(CALIBRATION_KEYWORD, CALIBRATION_JUSTIFICATION, order)
