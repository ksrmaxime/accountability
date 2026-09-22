# src/verify_utils.py
"""
Shared helper for the pipeline's two-pass "draft + verify" pattern.

Every LLM stage (step 3, 4a, 4b, 5, 6) now runs in two passes:

  pass 1 (draft)  -- reads the full article, writes a one-sentence
    justification, then a first-guess label right after it, in the same
    generation. Both are kept: the justification under the field's usual
    "*_justification" name, and the first-guess label under a new
    "*_draft" name (kept only for QA -- to measure how often the two
    passes disagree).

  pass 2 (verify) -- reads ONLY the pass-1 justification text (never the
    article), and is asked to classify that short text into one of the
    task's valid labels. This is the FINAL, validated label, saved under
    the field's normal name (e.g. "keyword_answer") -- the one every
    downstream step's eligibility mask and the final dataset actually use.

Why split them: on the first justification-first run, a manual review
found that because pass 1 asks for the label immediately after the
justification (same generation, same continuation), the forced final
token sometimes drifted away from what the justification it had just
written actually supported -- e.g. a justification plainly arguing "no
criticism found" followed by a YES label anyway. Measured on the real
run: 9.3% of step 3 YES rows and 40% of step 6 YES rows had a
justification that contradicted their own label. Pass 2 is a much
narrower, easier task (classify one short sentence) and cannot see the
long article, so it has nothing left to drift on.
"""
from __future__ import annotations
import re
import pandas as pd


def parse_label_only(raw: str, labels: dict) -> object:
    """
    Parse a pass-2 (verify) response expected to be a single label word
    (or short category name).

    `labels` maps the UPPERCASE label the model should answer with to the
    canonical value to store, e.g. {"YES": "YES", "NO": "NO"} or
    {"INTEREST GROUP": "Interest Group", ...}.

    Tries, in order:
      1. the first non-empty line, checked with .startswith() -- handles
         a clean one-word answer, or one word plus trailing punctuation;
      2. a whole-word search over the entire response -- handles a model
         that ignores the "one word only" instruction but still names a
         valid label somewhere in its answer. Uses word boundaries (not a
         plain substring check) so a short label like "NO" doesn't false-
         match inside an unrelated word like "NOTHING", and "OTHER"
         doesn't false-match inside "ANOTHER".

    Returns pd.NA if neither step finds a valid label.
    """
    if not raw:
        return pd.NA
    text = raw.strip().upper()
    if not text:
        return pd.NA

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    first = lines[0] if lines else text
    for key, value in labels.items():
        # anchored at the start of the first line, but still requires a
        # whole-word match: plain .startswith("NO") would wrongly fire on
        # "Nothing in the text..." or "Not applicable" -- both start with
        # the two characters "NO" without meaning it as an answer.
        if re.match(r"\b" + re.escape(key) + r"\b", first):
            return value

    for key, value in labels.items():
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            return value

    return pd.NA
