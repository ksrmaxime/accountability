# src/step3_prompts.py
"""
STEP 3 — criticism detection.

Originally ported 1:1 from the legacy src/run3_prompts.py with the wording
left untouched. After a real full run, two problems showed up in the actual
results: (1) no justification was captured at all, so a wrong answer could
not be diagnosed, and (2) the model sometimes seems to flag "{keyword}" as
YES when it is actually the one doing the criticizing (or merely mentioned
near criticism of something else), not the one being criticized. This
version keeps the same core binary question but:
  - explicitly tells the model not to confuse "{keyword} is criticized"
    with "{keyword} is the one criticizing",
  - asks for a one-sentence justification BEFORE the final YES/NO answer,
    to force the small model to reason about the text instead of pattern-
    matching straight to a label (same "reasoning before conclusion"
    pattern already used in step 6, now applied consistently everywhere).

2026-09-29 update (superseded below): a 400-row gold comparison had found
two roughly opposite error patterns -- false positives from treating
neutral statements as criticism, and false negatives from refusing to
attribute criticism of a project/subordinate unit to the entity
responsible for it. Two separate rules with four examples were bolted onto
USER_TEMPLATE, on top of the original two paragraphs, to address each.

2026-09-30 update (superseded below): running that version against the
gold standard showed the two bolted-on rules did not hold up well
together -- new false positives from hedged-then-YES reasoning and from
"responsibility" being used to launder a criticism of a genuinely
different actor onto "{keyword}". USER_TEMPLATE was restructured into an
explicit two-step checklist ("find the reproach, then check who it
targets"), trimmed to two inline examples, and a "no hedging" format
requirement was added.

2026-10-01 rewrite (current): a second 400-row gold comparison showed
*every* metric for keyword_answer had gotten worse across both prior
updates, not better -- accuracy, precision and F1 declined monotonically
from the original version through both revisions, and a specific new
failure mode appeared (the model attributing a predecessor's or an
unrelated actor's decision to "{keyword}" through a chain of reasoning,
e.g. "{keyword} is the new person in the role, so this counts"). Looking
at the two prior updates side by side, the real design flaw was not that
the original definition was wrong, but that it was *incomplete*: it never
told the model what to do with the two edge cases (institutional
responsibility, and a neutral-sounding statement) in the first place. Both
updates tried to patch that gap by bolting a separate "rule" onto the
untouched original paragraphs, and gave each bolted-on rule its own
concrete examples -- which gave the edge-case rules a stronger, more
vivid operationalization than the base definition itself ever had, so the
model had much more to go on for the rare edge cases than for the
ordinary one.

This version does not add a third rule. It rewrites USER_TEMPLATE's
definition of "{keyword} is criticized" as a single, complete paragraph
that already states the institutional-responsibility case and the
neutral-statement exclusion as part of what "criticized" means -- not as
exceptions bolted on afterwards -- and then gives one worked example for
*every* case the definition distinguishes (ordinary criticism, a neutral
non-criticism, "{keyword}" as the critic rather than the target,
attributable institutional responsibility, a genuinely separate actor,
mere mention/involvement, a predecessor's decision, and a policy
disagreement that is not a reproach), so that no single case is better
illustrated than the others. The explicit "do not hedge" instruction from
the 2026-09-30 version is dropped -- the gold-comparison justifications
suggested the abstract instruction did not stop hedge-then-YES reasoning
and, combined with the stacked rules, may have pushed the opposite error
(refusing to name a reproach at all) in the following run. Grounding is
instead expected to come from the examples themselves, plus the existing
requirement to name the specific reproach found.

`matched_alias` is still used instead of the canonical `keyword` for the
same reason as before: it's the literal term step 2 found in *this*
article's own text.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You read a newspaper article and judge whether one specific, named entity
is criticized in it. Be careful to distinguish an entity that is actually
criticized from one that is itself doing the criticizing, one that is
simply mentioned in a neutral context, and one that is only mentioned
nearby a reproach that in fact targets someone else entirely.\
"""

USER_TEMPLATE = """\
In the article below, focus specifically on "{keyword}".

ARTICLE:
{article_text}

Your task is to determine whether "{keyword}" ITSELF is being criticized in this article.

"{keyword}" is criticized when the article contains an actual reproach, negative judgment, accusation, or denunciation -- made by a journalist, a quoted source, a political actor, or anyone else -- and that reproach targets either "{keyword}" by name, or a project, program, decision, or subordinate unit that is clearly "{keyword}"'s own, current, direct responsibility (for instance a procurement project run by a department, even when the reproach names the project or a supplier rather than the department itself).

"{keyword}" is NOT criticized when any of the following is true: the statement is a neutral fact, forecast, statistic, or plain description of a decision, with no reproach actually expressed, even if a reader could imagine reading it negatively; "{keyword}" is the one expressing criticism of someone or something else rather than the target of it; the reproach targets a genuinely separate actor -- another department, a foreign government, a private company, a parliamentary committee's own conduct, or a predecessor who held the role before "{keyword}" took over -- even if "{keyword}" is simply mentioned nearby as being informed, associated, or in charge of following up, without being blamed directly; or the article merely reports a policy disagreement or a contested position ("{keyword}" defends a stance some find wrong), without anyone actually reproaching "{keyword}" for wrongdoing or failure.

Examples covering the full range of cases above:
- "The ministry forecasts a budget deficit of 3 billion next year." -> NOT criticism: a neutral statement, no reproach expressed.
- "Critics accuse the ministry of hiding the true scale of the deficit." -> Criticism of the ministry: an explicit reproach naming it directly.
- "{keyword} denounces the opposition's handling of the file." -> NOT criticism of "{keyword}": here "{keyword}" is the one criticizing, not the target.
- "A parliamentary committee denounces cost overruns and delays in a defense-equipment project run by the ministry." -> Criticism of the ministry: the project is its own direct, current responsibility, even though the ministry itself is not named.
- "A private contractor is blamed for a manufacturing defect; no reproach is made against the ministry's own decisions." -> NOT criticism of the ministry: the reproach targets a genuinely separate actor.
- "The ministry is mentioned as having been informed of the problem and tasked with following up." -> NOT criticism: mere involvement or being informed is not itself a reproach.
- "The cost overrun was decided two years before {keyword} took office, under their predecessor." -> NOT criticism of "{keyword}": the decision belongs to a separate actor in time, not to their own current responsibility.
- "Some lawmakers disagree with {keyword}'s support for the free-trade deal." -> NOT criticism: a policy disagreement about a position is not a reproach for wrongdoing.

First, in one sentence, state the specific reproach you found in the article and who it targets -- or say plainly that you found no reproach at all.
Then, on a new line, answer with exactly one word: YES or NO.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify): classify the pass-1 justification alone, without the
# article. See src/verify_utils.py for why this second pass exists. ---

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


# --- Last resort (force): a bare one-word decision, no justification asked
# for at all, used only for rows that still have no parseable pass-1 answer
# after every retry of the full prompt above. Mirrors step4a's own force
# prompt (src/step4a_prompts.py) -- deliberately minimal, the goal is just
# to stop a row from being silently dropped, not to produce a reasoned
# judgment. Any row answered this way is flagged via keyword_answer_forced
# = "TRUE" (see scripts/step3_criticism_detection.py) and never goes
# through pass 2, since there is no justification to verify. Left
# unchanged through the 2026-09-29, 2026-09-30 and 2026-10-01 updates
# since it affects a negligible fraction of rows and is deliberately kept
# minimal.

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
