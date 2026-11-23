# src/step3_prompts_remote.py
"""
STEP 3 — criticism detection, REMOTE-BACKEND-ONLY prompt fork.

2026-11-19: created at Maxime's explicit request, after comparing
Apertus-v1.5-70B and 70B-thinking (via --backend litellm, the UNIL/CSCS
proxy) against the GOLD800 sample (job20650690, job20654710) and finding
the exact same error signature on BOTH remote models: strong precision
(0.84-0.92), weak recall (0.36-0.47), and a dominant false-negative shape
(~90 of ~800 rows) where the model's own justification explicitly reasons
itself out of a YES despite real criticism being present -- see the
conversation's qualitative breakdown of job20650690's 89 false negatives
for the full pattern tally.

WHY THIS IS A SEPARATE FILE, not an edit to src/step3_prompts.py: that
file's own module docstring documents a dozen-plus rounds of pass-1
prompt tuning on the production vllm/transformers (8B, local GPU) path
since 2026-09-29, several explicitly reverted after measuring net-negative
on the real gold set (see its 2026-09-29/30, 2026-10-01, 2026-10-12
entries) because of what that docstring calls a "non-local recalibration
effect" -- loosening the definition to catch one missed pattern reliably
creates new false positives elsewhere, since the SAME instruction block
governs every row. Maxime's access to the UNIL/CSCS proxy is personal and
may not last, so src/step3_prompts.py (and its sibling
src/step3_logit_prompts.py) must stay exactly as they are: whatever is
tried here must never put the proven, extensively-tuned production
8B/local-GPU wording at risk. scripts/step3_criticism_detection.py now
picks this module instead of src/step3_prompts.py specifically when
--backend litellm is used (see that script's own 2026-11-19 entry) --
every other backend (vllm/transformers) is completely unaffected, byte-
for-byte, by anything in this file.

BASELINE: SYSTEM_PROMPT, USER_TEMPLATE (its first 10 worked examples,
in the same order and wording), build_user_prompt, FORCE_SYSTEM_PROMPT,
FORCE_USER_TEMPLATE, build_force_prompt, VERIFY_SYSTEM_PROMPT,
VERIFY_USER_TEMPLATE and build_verify_prompt are copied VERBATIM from
src/step3_prompts.py as it stood after that file's 2026-10-13 entry (the
version actually used, unmodified, for every remote-backend run analysed
so far, since build_client()/step3_criticism_detection.py previously had
no remote-specific prompt at all). This file does not re-derive or
re-litigate any of that wording's own history -- see src/step3_prompts.py
for the full changelog behind it. VERIFY_* and the plain FORCE prompt are
kept here only for interface parity with that file (the detection script
never actually calls pass 2, same as the production path); they are not
touched or reasoned about below.

THE ONE CHANGE, added here and here only: a new, balanced pair of worked
examples (11-12) in USER_TEMPLATE, targeting a specific false-negative
shape that is NOT covered by any of the existing 10 examples and was NOT
among the patterns any of the already-reverted production-path tuning
rounds targeted (see src/step3_prompts.py's history and
src/step3_logit_prompts.py's ADMIN_UNIT_CAUTION entry -- that one tried
to make the model MORE cautious on administrative/departmental targets to
cut false positives; this is the opposite direction, aimed at recall, on
a pattern that attempt never addressed):

  A department/ministry's own named political leadership (a minister, a
  director) is blamed for a management or leadership failure, and the
  article never uses the department's own name in the same sentence as
  the fault -- e.g. "VBS's decisions, overseen by Amherd, were misguided"
  or "the department's reorganization, led by its minister, caused
  chaos". On job20650690/job20654710, roughly a quarter of false
  negatives (22 of 89, and a comparable share of the thinking model's 93)
  read almost exactly like this, with justifications explicitly drawing a
  "leadership vs. the institution itself" line that none of the existing
  10 examples addresses one way or the other.

Example 11 (new, YES) establishes that a department's own leadership
making a management decision about the department's own internal affairs
IS the department's own responsibility, under the same "direct
responsibility" logic example 1 already uses for a project the ministry
runs. Example 12 (new, NO) is the deliberately paired contrast, carried
over from the same "always calibrate both directions at once" discipline
src/step3_prompts.py's own 2026-10-02 entry describes: a minister's
UNRELATED personal conduct (nothing to do with running the department)
must NOT transfer to the department, guarding against exactly the kind of
over-broad swing that sank several of the production-path's own past
attempts.

STATUS: UNTESTED as of creation. The plan is to re-run the 800-row
GOLD800 remote test (swiss-ai/Apertus-v1.5-70B, --backend litellm) with
this prompt and compare against the already-measured baseline for that
exact model/sample (job20650690: accuracy=87.0%, precision=0.842,
recall=0.473, F1=0.606 on keyword_answer_draft) before deciding whether
to keep, iterate further, or revert this one addition -- the same
measure-before-trusting discipline documented throughout
src/step3_prompts.py and src/step3_logit_prompts.py. Do not assume this
helps until that comparison has actually been run.
"""
from __future__ import annotations
import pandas as pd

# --- Verbatim from src/step3_prompts.py -- see that file for the tuning
# history behind this exact wording. Not touched here. ---
SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You read a newspaper article and judge whether one specific, named entity
is criticized in it. Be careful to distinguish an entity being criticized
from an entity that is itself doing the criticizing, or one that is simply
mentioned in a neutral or unrelated context. Only answer YES if the article
itself attributes fault, blame, failure, or wrongdoing directly to this
specific entity: an entity that merely faces a difficult or negative
situation, carries out its normal role, is cited as a neutral source,
states its own position or decision, or is mentioned near criticism that
explicitly targets a different, specific person, institution, or policy,
is not itself being criticized.\
"""

# --- USER_TEMPLATE: examples 1-10 verbatim from src/step3_prompts.py.
# Examples 11-12 are NEW -- see module docstring. Everything else
# (framing, closing instruction) is unchanged. ---
USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Your task is to determine whether "{keyword}" ITSELF is being criticized in this article -- for who it is or what it did or decided. Criticism can be present even if the article's overall tone is neutral or positive elsewhere.

Do not confuse this with a case where "{keyword}" is the one expressing criticism of someone or something else, or is only mentioned in passing -- neither of those counts as "{keyword}" being criticized.

For example:
- "A parliamentary committee accuses the ministry of mismanaging a defense procurement project it runs, citing cost overruns." -> the ministry IS criticized: the reproach targets a project that is its own direct responsibility, even though the ministry itself is not named.
- "Critics say the mayor's housing policy has worsened the rent crisis in the city." -> the mayor IS criticized: a clear reproach targets her decision.
- "\"The agency has been negligent and failed to act in time,\" the auditor's report states." -> the agency IS criticized: a direct, explicit reproach.
- "The agency reports a 3% increase in applications this year." -> NOT criticized: a plain neutral fact, no reproach is expressed.
- "The minister dismisses the opposition's proposal as unrealistic." -> the minister is NOT criticized: here the minister is the one doing the criticizing, not the target.
- "The report notes that the department was informed of the problem and tasked with following up." -> NOT criticized: being informed or involved is not itself a reproach.
- "\"The agency's own negligence and repeated failures to act allowed the crisis to worsen,\" the audit concludes." -> the agency IS criticized: the fault is placed directly on the agency itself, not merely on the situation.
- "A parliamentary committee criticizes a new government savings package; the office that helped carry out part of it is only mentioned as having implemented measures decided elsewhere." -> the office is NOT criticized: the criticism targets the package/decision, not this office, which is not itself named as being at fault.
- "The agency's own internal report warns that, without additional funding, its insurance fund could become insolvent within a few years." -> the agency is NOT criticized: it is disclosing a risk about its own situation, and nobody in the text blames it for that risk.
- "The financial oversight office questioned the fixed-price terms of a defense contract; the defense agency firmly rejected its findings." -> the oversight office is NOT criticized: it is the one raising the criticism, and having its findings disputed by the party it criticized does not make it the target of criticism.
- "The ministry's own reorganization, overseen directly by the minister heading it, left staff confused and triggered an internal crisis within the department." -> the ministry IS criticized: the fault concerns how the department itself was led and managed -- its own leadership's decisions about its own internal affairs -- which is squarely the ministry's own responsibility, not a separate, unrelated actor's doing.
- "The minister heading the department is mocked online for a joke he made at an unrelated public event, with no mention of how he runs the department." -> the department is NOT criticized: this is about the minister's unrelated personal conduct, not about how the department itself is managed or led -- nothing here reflects on the department.

First, in one sentence, briefly justify your answer based on the article.
Then, on a new line, answer with exactly one word: YES or NO.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify): kept only for interface parity with
# src/step3_prompts.py -- never called by scripts/step3_criticism_detection.py
# for either backend. Verbatim, unchanged. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant. Another analyst has already read a
newspaper article and written a short explanation of whether a specific
entity is criticized in it. You are not shown the article itself -- your
only task is to read that explanation and decide which final answer it
supports.\
"""

VERIFY_USER_TEMPLATE = """\
The question was: is "{keyword}" criticized in the article?

Here is the explanation given by the other analyst:
"{justification}"

Based solely on this explanation, answer with exactly one word: YES or NO.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)


# --- Last resort (force): verbatim from src/step3_prompts.py, unchanged. ---

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one word, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Is "{keyword}" ITSELF being criticized in this article -- for who it is or what it did or decided?

Answer with exactly one word: YES or NO.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
