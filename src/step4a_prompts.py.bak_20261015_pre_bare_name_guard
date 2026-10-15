# src/step4a_prompts.py
"""
STEP 4a — source attribution (first half of what "step4" used to be).

A real full run showed this stage badly miscalibrated: out of ~13,000
confirmed-criticism rows, only 7 were labelled SOURCED, the rest almost all
JOURNALIST -- yet a spot check of JOURNALIST-labelled articles found named
actors, letters, and institutional sources quoted directly. The original
one-shot binary question ("is this backed by testimony... or is it purely
the journalist's own narrative framing...") packs two dense clauses into a
single sentence and asks for an immediate one-word answer with no reasoning
step; for an 8B model that combination collapses to "JOURNALIST" as a
default almost every time.

Fix: make the model search for and name a candidate actor FIRST (forcing it
to actually engage with the text), and only then commit to the SOURCED /
JOURNALIST label on a separate line -- the same justification-before-answer
pattern used everywhere else in this pipeline.

2026-10-07 update: a 400-row gold comparison (job65468999 chain) found two
error patterns. (1) The model often names "{keyword}" itself as the actor
making the criticism whenever it is named as the one who took the
decision/action being criticized, even though it is the TARGET of the
criticism, not its source -- USER_TEMPLATE now explicitly excludes this.
(2) A handful of pass-2 flips turned a clearly-SOURCED justification (e.g.
one that just names an organization) into a final JOURNALIST answer --
VERIFY_USER_TEMPLATE's SOURCED line now explicitly covers a bare named
actor, not only a full quotation.

2026-10-12 update: a fresh breakdown on the corrected 800-row gold
(job65610203) found the dominant error (gold=SOURCED, Apertus=JOURNALIST)
is 73.5% self-contradictory -- pass 1's own draft already said SOURCED,
and pass 2 flips it to JOURNALIST on the very same justification text,
despite the 2026-10-07 SOURCED clarification above already telling it a
bare named actor counts. Since the rule is already stated and still not
reliably applied, VERIFY_USER_TEMPLATE now asks pass 2 to first quote the
specific actor it finds (or "none") before committing to the label --
the same justification-before-answer structure already used for pass 1,
rather than more wording added to a rule that demonstrably isn't the
bottleneck. This needs a larger verify token budget than the previous
bare one-word answer; VERIFY_MAX_NEW_TOKENS was raised from 8 to 40 in
sbatch_step4a_array.sh accordingly.

2026-10-14 update: a draft-vs-gold breakdown on job65774182 (800-row run,
pass 2 now removed for this step -- see scripts/step4a_source_attribution.py)
found 13 FP (gold=JOURNALIST, Apertus=SOURCED) on the 130-row step3-TP
population, of which 12 fall into two sub-patterns absent from any prior
fix: a named political party/canton/coalition treated as a qualifying actor
without a specific individual inside it ever being named (5 rows), and a
signed column's own byline author mistaken for an external source (3 rows),
plus the already-known target's-own-name confusion recurring in a milder
form (4 rows, already covered by the exclusion sentence below -- no new
rule needed there). USER_TEMPLATE now explicitly excludes collectives
without a named individual and byline authors, with four worked examples
(two SOURCED, two JOURNALIST) -- balanced on purpose, since the prompt
previously had zero worked examples at all and two one-sided additions
risked re-introducing the original default-to-JOURNALIST collapse this
file's first fix (above) was built to escape.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to determine whether that criticism is attributed to a
specific, identifiable actor (a named person, organization, or
institution), or whether it is simply the journalist's own narrative
framing, with no specific actor identified as making the criticism.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Look for a specific person, organization, or institution that is quoted, named, or clearly identified as making this criticism (for example: a named politician, an interest group, another authority, a civil servant, or a member of the public). "{keyword}" itself does not count as that actor just because it is named as the one who made the decision, took the action, or is otherwise the subject of the criticism -- the actor you are looking for is whoever is voicing or levelling the criticism, not the one the criticism targets.

A political party, parliamentary group, canton, or other named collective does not count as that actor unless a specific individual within it is also named or quoted as voicing the criticism. The article's own byline author -- a signed column, editorial, or portrait piece -- is the journalist, not an external actor, even when their name appears prominently in the text.

For example:
- "Opposition MP Jane Doe called the decision 'reckless and irresponsible'." -> SOURCED: a specific named individual, distinct from the target, is quoted directly criticizing it.
- "Peter Müller, head of the civil servants' union, said the reform was 'a betrayal of public sector workers'." -> SOURCED: a named individual with a specific role is quoted, even though they speak on behalf of an organization.
- "The SP and a coalition of cantons jointly criticized the decision." -> JOURNALIST: a party and a group of cantons are named, but no specific individual within them is identified as voicing the criticism.
- "This signed column by journalist Christophe Passer pokes fun at Guy Parmelin's recent gaffe." -> JOURNALIST: Christophe Passer is the article's own author, not an external source, even though he is named.

First, in one sentence, say who -- if anyone -- is identified as making the criticism, or state that no specific actor is identified.
Then, on a new line, answer with exactly one word:
SOURCED — if a specific actor is quoted, named, or clearly identified as making the criticism.
JOURNALIST — if no specific actor is identified and the criticism is presented as the article's own narrative.\
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
newspaper article and written a short explanation of who, if anyone, is
identified as making a criticism. You are not shown the article itself --
your only task is to read that explanation and decide which final answer
it supports.\
"""

VERIFY_USER_TEMPLATE = """\
The question was: is the criticism of "{keyword}" attributed to a specific, identifiable actor, or is it just the journalist's own narrative framing?

Here is the explanation given by the other analyst:
"{justification}"

First, quote the specific name, title, or role of any actor this explanation \
identifies as making or supporting the criticism, if any -- answer "none" if \
there is none.
Then, on a new line, answer with exactly one word:
SOURCED — if the explanation names, quotes, or otherwise identifies any specific actor making the criticism, even briefly or without a full quotation.
JOURNALIST — if the explanation says no specific actor is identified.\
"""


def build_verify_prompt(row: pd.Series, justification_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    justification = "" if pd.isna(row[justification_col]) else str(row[justification_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, justification=justification)


# --- Last resort (force): a bare one-word decision, no justification asked
# for at all, used only for rows that still have no parseable pass-1 answer
# after every retry of the full prompt above. Deliberately minimal -- the
# goal here is just to stop a row from being silently dropped, not to
# produce a reasoned judgment. Any row answered this way is flagged via
# source_attribution_forced=True (see scripts/step4a_source_attribution.py)
# and never goes through pass 2, since there is no justification to verify.

FORCE_SYSTEM_PROMPT = """\
You are a media analysis assistant. Answer with exactly one word, nothing else.\
"""

FORCE_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Is that criticism attributed to a specific, identifiable actor (a named person, organization, or institution), or is it just the journalist's own narrative framing with no such actor identified?

Answer with exactly one word: SOURCED or JOURNALIST.\
"""


def build_force_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return FORCE_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
