# src/step4a_pass1_coherence_logit_prompts.py
"""
STEP 4a -- "pass1 coherence check" (2026-11-24), logit-probing.

Context: job20934804 (the second full remote-chain run on GOLD800,
after the 2026-11-24 "verdict-last" fix to src/step4a_prompts_remote.py
eliminated the earlier parsing bug -- see that file's own changelog)
still sits below the classic 8B baseline, and reading the remaining
errors' own justifications found a NEW self-contradiction pattern,
specific to pass 1 (the draft), not pass 2 like the one
src/step4a_pass2_coherence_logit_prompts.py already addresses.

Measured on job20934804: of 113 rows with both a draft label and a
justification, 18 (16%) have a justification whose own plain-language
reasoning clearly argues one thing ("no outside voice is credited",
"framed as the author's own analysis", "without crediting any specific
outside voice") while the stored source_attribution_draft is the
OPPOSITE (SOURCED) -- e.g. Viola Amherd: "...without crediting any
specific outside voice such as a named individual, group, or
institution with those particular criticisms. SOURCED". Only 2 rows show
the reverse direction. This is not the 2026-11-24 parsing bug (which was
the model putting its verdict FIRST, confusing a last-line parser) --
checked specifically and ruled out, there are zero rows where the
justification OPENS on the opposite bare word. The mechanism here looks
different: asked to end with "one final new line containing nothing but
a single word", the model does not always insert a real line break
before that word -- it is simply appended as the last token of the same
paragraph. The extraction itself works exactly as designed (it does grab
the true last word), but that last word, chosen at that specific
generation step, does not reliably reflect the reasoning that precedes
it. Separating the verdict from the reasoning onto a distant final line
seems to have given the model more room for this kind of slip, not less.

Maxime's suggestion (2026-11-24), which this file implements: rather
than trying to out-word-smith the pass 1 prompt further to prevent this
slip at generation time, add the exact same remedy already proven for
pass 2's own version of this problem (src/step4a_pass2_coherence_logit_prompts.py,
2026-11-18, kept 11/27 "clean" SOURCED flips and reverted the other 16
self-contradictory ones) -- a narrow THIRD-PARTY reading-comprehension
check, via calibrated logit-probing, that reads ONLY the draft's own
justification text (never the article, never any outside knowledge) and
reports what that paragraph itself concludes. This file is that same
design, applied one stage earlier (pass 1's justification instead of
pass 2's). The question template, calibration mechanism, and override
philosophy are deliberately identical to that file's -- see its own
module docstring for the full rationale on why this is a reading-
comprehension task and not a keyword-matching one (the same "found a
candidate actor, then negates their relevance in the same paragraph"
ambiguity that defeated deterministic rules for pass 2 applies here too).

Decision rule (see scripts/step4a_source_attribution.py): only OVERRIDE
source_attribution_draft when this check's calibrated reading clearly
disagrees (a margin around the 0.5 decision boundary,
--pass1_coherence_margin, not yet empirically tuned -- a prior, like
every other threshold in this project's history). An unclear/
inconclusive read, or one that agrees, leaves the draft label untouched
-- fail-safe, same convention as every other validation/verification
pass in this codebase.

Scope: every row with a parsed source_attribution_draft AND a non-bare
source_attribution_justification (i.e. every row that went through a
genuine pass 1, not a forced one-word fallback, which has no
justification to read). Runs BEFORE pass 2 ("forced re-search") in
scripts/step4a_source_attribution.py, on source_attribution_draft itself
-- not after the draft is copied into source_attribution -- so that a
row this check corrects from SOURCED back to JOURNALIST becomes a
legitimate pass 2 candidate (a fair second chance like any other
genuine JOURNALIST draft), and a row corrected from JOURNALIST to
SOURCED is correctly excluded from pass 2 (the ambiguity is already
resolved, no re-search needed). Cheap, same as pass 2's own coherence
check: the input is a short justification, not the article.

2026-11-24: added to scripts/step4a_source_attribution.py gated STRICTLY
behind --backend litellm (see that script's own comment at the call
site) -- this problem was diagnosed only on the remote/70B path, and the
production local-GPU path (src/step4a_prompts.py, --backend
vllm/transformers) must stay completely unaffected, byte-for-byte,
unless Maxime explicitly asks to extend this there too. This prompts
module itself is backend-agnostic (a pure "what does this paragraph
conclude" reading-comprehension question, like pass 2's own coherence
check, which is already shared across backends) -- it is the ORCHESTRATION
script's call site that is backend-gated, not this file.
"""
from __future__ import annotations
import pandas as pd

LETTERS = ("A", "B")

QUESTION_TEMPLATE = """\
You are a careful reading-comprehension assistant. Below is a short paragraph of reasoning that \
a prior automated step wrote while examining whether a criticism of "{keyword}" is attributed to \
an outside voice (SOURCED) or is just the article's own narrative framing, with no outside voice \
behind it (JOURNALIST).

REASONING TO READ:
{justification_text}

Read ONLY what this paragraph itself says. Do not re-decide the question yourself and do not use \
any outside knowledge about "{keyword}" -- your only job is to report what this paragraph's own \
stated conclusion is. Note: if the paragraph says the criticism is attributed to the article's \
author, its byline journalist, or "the journalist", that is NOT an outside voice -- the \
paragraph's own conclusion in that case is still that no outside voice is credited.

For example:
- "The article mentions Jane Doe, but she is not credited with this specific criticism. The criticism is framed as the author's own opinion." -> this paragraph's own conclusion is JOURNALIST (no outside voice).
- "The criticism of the agency is attributed to the opposition party, which is quoted criticizing the minister's decision." -> this paragraph's own conclusion is SOURCED (an outside voice is credited).
- "The criticism is attributed to the author of the piece, who is the journalist himself." -> this paragraph's own conclusion is JOURNALIST (the journalist is not an outside voice, even though the word "attributed" appears).
- "No named individual, group, or institution is credited with this criticism; it is framed as the article's own analysis." -> this paragraph's own conclusion is JOURNALIST, even if a single unrelated word like "SOURCED" appears elsewhere in or after the paragraph -- judge the paragraph's actual stated reasoning, not any stray trailing word.
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
    Duplicated locally (same as src/step4a_pass2_coherence_logit_prompts.py's
    own copy) so this file stays self-contained."""
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
    """Grounded in pass 1's OWN source_attribution_justification -- never
    the article, never pass 2's justification. Callers should filter for
    a non-null source_attribution_justification (same convention as
    every other logit-probing pass in this codebase)."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification_text = str(row["source_attribution_justification"]).strip()
    return _build_prompt(keyword, justification_text, order)


def build_coherence_calibration_prompt(order: int) -> str:
    """Same empty-input neutralisation as every other calibrated pass in
    this codebase -- measures, once per task, how often this reading-
    comprehension question's own wording defaults to SOURCED independent
    of any real paragraph content."""
    return _build_prompt(CALIBRATION_KEYWORD, CALIBRATION_JUSTIFICATION, order)
