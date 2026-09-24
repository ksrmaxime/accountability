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


_PREAMBLE_WORDS = {
    "ANSWER", "FINAL", "MY", "IS", "THE", "LABEL", "DECISION", "THIS",
    "THEREFORE", "SO", "THUS", "CONCLUSION", "RESPONSE", "CATEGORY",
}


def _is_bare(text: str, labels: dict) -> bool:
    """True if `text` reduces to nothing once every valid label word/phrase
    and a short list of filler words ("answer", "final", "the", ...) are
    stripped out -- i.e. it is not a real explanatory sentence, just a label
    dressed up as one ("Answer: SOURCED", "JOURNALIST", or even the OTHER
    label sitting alone, as in a garbled "JOURNALIST\nSOURCED" response).
    Checked against every key in `labels`, not just the one already
    matched, so a bare answer that names the wrong label is still caught.
    """
    su = text.upper()
    for key in labels:
        su = re.sub(r"\b" + re.escape(key) + r"\b", " ", su)
    su = re.sub(r"[^A-Z]", " ", su)
    words = [w for w in su.split() if w not in _PREAMBLE_WORDS]
    return len(words) == 0


def _find_label(text: str, labels: dict):
    """Word-boundary search for any of `labels`' keys anywhere in `text`
    (not just as a prefix) -- catches "Answer: SOURCED", "The answer is
    JOURNALIST.", etc., not just a clean isolated label word."""
    su = text.upper()
    for key, value in labels.items():
        if re.search(r"\b" + re.escape(key) + r"\b", su):
            return value
    return None


def parse_draft_labeled(raw: str, labels: dict) -> tuple:
    """
    Shared parser for a pass-1 (draft) response: a short justification
    sentence plus a label word, in either order, possibly on the same line,
    possibly with a preamble ("Answer: SOURCED"). `labels` is the same
    {"UPPER_KEY": "canonical value"} mapping already used for pass 2
    (each script's own `VERIFY_LABELS`).

    Returns (label, justification) on success, or (None, None) on failure.

    Found on the real job65119601 run: about 1 in 5 step 4a "final" answers
    turned out to rest on an empty or near-empty justification, because the
    model sometimes answers with the label word alone (no explanation), or
    with two label-like lines and nothing else (e.g. "JOURNALIST\nSOURCED").
    The old per-script parser accepted these as a successful draft with a
    blank justification; pass 2 then had nothing to verify and fell back to
    a fixed default every single time -- silently, with no signal that the
    answer wasn't a real judgment. This parser treats "found a label but the
    rest of the response isn't a real explanation" as a FULL failure (both
    label and justification come back empty), matching the two-pass
    design's own premise: pass 2 can only verify something pass 1 actually
    wrote down.

    Also broader than the original per-script parsers on purpose: the old
    version only recognised the label as an exact prefix of the first or
    last line, so a response like "Answer: SOURCED" or a single merged
    sentence ("This is SOURCED because X quotes Y directly.") was silently
    dropped to NA even though it plainly states an answer. This version
    searches for the label word anywhere (word boundaries, not prefix) on
    the first/last line, on a lone single-line response, and finally
    anywhere across the whole response -- intended to close a real chunk of
    the pass-1 NA rate, not just the "answer with no justification" bug.
    """
    if not raw or not raw.strip():
        return None, None
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return None, None

    # 1) label found on the last line (the requested format)
    label = _find_label(lines[-1], labels)
    if label is not None:
        justification = " ".join(lines[:-1]).strip()
        if justification and not _is_bare(justification, labels):
            return label, justification

    # 2) label found on the first line (a model that answers out of order)
    label = _find_label(lines[0], labels)
    if label is not None:
        justification = " ".join(lines[1:]).strip()
        if justification and not _is_bare(justification, labels):
            return label, justification

    # 3) a single line/paragraph carrying both the reasoning and the label
    # together (e.g. "This is SOURCED because a named source is quoted.")
    if len(lines) == 1:
        label = _find_label(lines[0], labels)
        if label is not None and not _is_bare(lines[0], labels):
            return label, lines[0]

    # 4) label buried in the middle of a multi-line response, with real
    # content in the other lines
    for i, line in enumerate(lines):
        label = _find_label(line, labels)
        if label is None:
            continue
        rest = " ".join(lines[:i] + lines[i + 1:]).strip()
        if rest and not _is_bare(rest, labels):
            return label, rest

    return None, None
