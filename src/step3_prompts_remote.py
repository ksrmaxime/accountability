# src/step3_prompts_remote.py
"""
STEP 3 — criticism detection, REMOTE-BACKEND-ONLY prompt fork.

2026-11-19: created at Maxime's explicit request, after comparing
Apertus-v1.5-70B and 70B-thinking (via --backend litellm, the UNIL/CSCS
proxy) against the GOLD800 sample (job20650690, job20654710) and finding
the exact same error signature on BOTH remote models: strong precision
(0.84-0.92), weak recall (0.36-0.47), and a dominant false-negative shape
(~90 of ~800 rows) where the model's own justification explicitly reasons
itself out of a YES despite real criticism being present.

WHY THIS IS A SEPARATE FILE, not an edit to src/step3_prompts.py: that
file's own module docstring documents a dozen-plus rounds of pass-1
prompt tuning on the production vllm/transformers (8B, local GPU) path
since 2026-09-29, several explicitly reverted after measuring net-negative
on the real gold set, because of what that docstring calls a "non-local
recalibration effect" -- loosening the definition to catch one missed
pattern reliably creates new false positives elsewhere, since the SAME
instruction block governs every row. Maxime's access to the UNIL/CSCS
proxy is personal and may not last, so src/step3_prompts.py (and its
sibling src/step3_logit_prompts.py) must stay exactly as they are:
whatever is tried here must never put the proven, extensively-tuned
production 8B/local-GPU wording at risk. scripts/step3_criticism_detection.py
picks this module instead of src/step3_prompts.py specifically when
--backend litellm is used -- every other backend (vllm/transformers) is
completely unaffected, byte-for-byte, by anything in this file.

2026-11-19, first revision: added a balanced pair of worked examples
(11-12) targeting one specific false-negative shape (a department's own
political leadership blamed for a management/leadership failure, with the
department's name never in the same sentence as the fault -- e.g. "VBS's
decisions, overseen by Amherd, were misguided"). Measured on job20762627
against the job20650690 baseline (same model, same GOLD800 sample):
draft precision 0.842->0.786, recall 0.473->0.544, F1 0.606->0.643;
after validation, precision 0.924->0.868, recall 0.361->0.468, F1
0.519->0.608 -- a real, targeted win (17/89 of the original false
negatives flipped correctly, concentrated almost exactly on the intended
DDPS/VBS and SRC/NDB leadership-attribution cases), for a modest and
favourable (not runaway) precision cost.

2026-11-23, full rewrite of the governing definition (SYSTEM_PROMPT and
USER_TEMPLATE's framing paragraphs only -- see below): a follow-up
justification analysis of job20762627's remaining 77 false negatives
found the SAME dominant failure shape persisting even after the above fix
-- 73% of them still read almost verbatim as "does not attribute fault,
blame, failure, or wrongdoing directly to X itself", including cases
(project/policy managed by the entity, reported third-party criticism)
that worked examples 1 and 2 already illustrate. That is: the problem is
not primarily missing examples, it is the governing definition itself
(SYSTEM_PROMPT's "Only answer YES if the article itself attributes fault,
blame, failure, or wrongdoing DIRECTLY to this specific entity") training
the model to demand an explicit, direct, blame-shaped accusation and
discount anything reported, indirect, or phrased as a setback/
controversy/disputed outcome rather than a verdict of fault.

At Maxime's explicit request, this revision tests a different hypothesis:
a 70B-class model may not need a narrow, heavily operationalised
definition the way the original wording (tuned against the local 8B,
which needed a tight, literal rule to avoid freewheeling into false
positives) does -- a larger model may be able to apply a broader, more
naturally-worded definition of "criticism" (explicitly covering direct,
reported, and indirect criticism; explicitly covering the entity's own
conduct, its management/leadership, and positions or policies it
proposed, defended, or carried out politically) without the same
precision collapse the production-path's own past broadening attempts
suffered on the 8B. SYSTEM_PROMPT and USER_TEMPLATE's framing paragraphs
are rewritten below to state that broader definition directly, instead of
accumulating more worked examples around the old narrow one.

Per Maxime's explicit instruction, the worked examples themselves
(1-12, i.e. the original 10 plus the leadership pair added 2026-11-19)
are LEFT UNCHANGED, verbatim, same order, same wording -- only the
SYSTEM_PROMPT and the two framing paragraphs immediately before "For
example:" in USER_TEMPLATE were rewritten. All 12 examples remain valid
illustrations under the new, broader wording (none of the YES examples
needs a broader rule to justify them, and none of the NO examples relies
on excluding indirect/reported criticism -- they exclude on other
grounds: wrong direction, neutral fact, unrelated personal conduct, the
entity's own risk disclosure, or someone else being the actual target).

STATUS: UNTESTED as of this rewrite. The plan is the same discipline as
every prior change here: re-run the 800-row GOLD800 remote test
(swiss-ai/Apertus-v1.5-70B, --backend litellm) and compare against the
job20762627 baseline just above (draft: acc=87.25%, precision=0.786,
recall=0.544, F1=0.643) before deciding whether this broader definition
is a net improvement, a net regression (i.e. the precision collapse the
8B always suffered from broadening, now happening on the 70B too), or
a wash. Do not assume either outcome until that comparison has actually
been run.
"""
from __future__ import annotations
import pandas as pd

# --- 2026-11-23: rewritten to state a broader definition of criticism
# directly (direct, reported, indirect; behaviour, management, political
# proposals) -- see module docstring for the full rationale and the
# baseline this is measured against. The 12 worked examples below this
# point are UNCHANGED from the previous (2026-11-19) revision. ---
SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You read a newspaper article and judge whether one specific, named entity
is itself the target of criticism in it. Criticism can take several forms:
a direct reproach stated in the article's own narration, a criticism
reported from or attributed to someone else (a journalist, an official, an
expert, a political opponent, an auditor, members of the public, and so
on), or an indirect criticism conveyed by describing a failure, a
shortcoming, a controversy, a setback, a reversal, or a disputed outcome
that reflects on the entity -- it does not need to use words like "fault"
or "blame" to count. Criticism can concern the entity's own conduct or
behaviour, how it manages or leads itself or what it runs, a decision it
made, or a position, measure, or policy it proposed, defended, or carried
out politically. Be careful, however, to distinguish an entity being
criticized from an entity that is itself doing the criticizing of someone
or something else, or one that is simply mentioned in a neutral or
unrelated context: the criticism must concern this specific entity itself
-- for what it is, does, decided, manages, or proposed -- and not a
different, specific person, institution, or policy that merely happens to
be mentioned nearby.\
"""

# --- USER_TEMPLATE: framing paragraphs rewritten 2026-11-23 (see module
# docstring); the 12 worked examples below and the closing instruction are
# UNCHANGED, verbatim, from the previous (2026-11-19) revision. ---
USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Your task is to determine whether "{keyword}" ITSELF is the target of criticism in this article. That criticism can be direct, reported from someone else, or conveyed indirectly through a failure, a controversy, a setback, or a disputed outcome -- it does not need to be worded as an explicit accusation to count, and it can concern "{keyword}"'s own conduct or behaviour, its management or leadership, a decision it made, or a position, measure, or policy it proposed, defended, or carried out politically. Criticism can be present even if the article's overall tone is neutral or positive elsewhere.

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
