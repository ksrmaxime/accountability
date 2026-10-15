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

2026-10-15 update: job65965608 (first run with the 2026-10-14 examples
above) moved accuracy from 74.6% to 78.5% on the 130-row step3-TP
population (still short of the 80% target) but flipped the error profile:
FN fell from 20 to 2 (recall 79%->98%, the collective/byline fix worked),
while FP rose from 13 to 26. Reading the new FP justifications shows the
model copied the surface FORM of the new SOURCED examples ("Name did X" ->
SOURCED) rather than the discriminating principle -- new errors read
literally "SOURCED: Martin Pfister" or "SOURCED: Albert Rösti" (the bare
target name, the exact case the first paragraph below already excludes in
prose) and even one self-contradicting "SOURCED: The journalist is making
the criticism." USER_TEMPLATE now adds an explicit rule that a bare name
with no description of what was said or done is not evidence, an explicit
guard against this self-contradiction, and two more JOURNALIST-side
examples in the same terse "bare name" surface form that was being
copied -- deliberately unbalanced this round (2 SOURCED / 4 JOURNALIST)
because the imbalance is now calibrated against a measured, data-backed
error direction (FP running 2x FN), not an untested guess.

2026-10-15 update #2 (full rewrite): job65998611's 28 FN (gold=SOURCED,
Apertus=JOURNALIST) showed the 2026-10-14 collective-exclusion rule above
is miscalibrated against the actual gold standard -- named but
non-individuated collectives ("the parliamentary group and the canton",
"CTSO and Alliance des villes", "the SP, the Swiss Union of Civil
Servants, and the EU", "the community and some users", poll respondents)
are scored SOURCED in gold, not JOURNALIST, even with zero named
individual inside them. Confirmed directly with the user: the real test
is binary and broader than what was coded -- SOURCED applies whenever the
criticism of the target is attributed to ANY outside voice, however vague
or unnamed (a collective, "opponents," "residents," etc.); JOURNALIST
applies only when no outside voice of any kind is credited with that
specific criticism and it is the article's own narrative framing. Two
further nuances: a signed byline does not by itself mean there is no real
outside source behind the criticism (a signed piece can still quote an
external critic), and sources quoted elsewhere in the article for
unrelated claims do not count unless they are specifically the ones
credited with criticizing THIS target. SYSTEM_PROMPT and USER_TEMPLATE are
rewritten around this broader, context-sensitive test rather than the
narrower named-individual checklist, keeping only the target's-own-name
exclusion and the genuine-signed-opinion case from the prior version. Six
worked examples (3 SOURCED / 3 JOURNALIST), each a different concrete
scenario and phrasing chosen to avoid a single copy-pasteable template,
replace the previous four. The self-contradiction guard and the
"bare name is not evidence" sentence from the 2026-10-15 update above are
dropped as separate rules since the rewritten test already subsumes them.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a newspaper article in which a specific target is criticized.
Your task is to determine whether this specific criticism of the target is
attributed to an outside voice -- a source external to the journalist,
however broadly or narrowly identified (a named person, a group, an
institution, a category of people, etc.) -- or whether it is simply the
journalist's own narrative framing, with no outside voice behind it.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

Focus specifically on the criticism of "{keyword}": is it attributed to an
outside voice, or is it the article's own narrative framing?

An outside voice can be identified as loosely as a named individual, a
political party, a canton, an administration, a company, an association, a
category of people ("residents," "opponents," "experts"), or survey
respondents -- none of this needs a specific person's name to qualify.
"{keyword}" itself never counts as that voice just because it is named as
the one who made the decision, took the action, or is otherwise the
subject of the criticism.

Two things do NOT settle the question by themselves: a signed byline does
not mean the article has no outside sources behind the criticism of
"{keyword}" -- a signed piece can still quote or cite someone else's
criticism of the target. And sources quoted elsewhere in the article, for
other claims or about other people, do not count unless they are
specifically the ones credited with the criticism of "{keyword}" itself.

The answer is JOURNALIST only when, specifically for the criticism of
"{keyword}", no outside voice of any kind -- named or not -- is credited
with holding or voicing it: the judgment is presented as the article's own.

For example:
- "Residents interviewed for the piece say the new bus route is useless and badly planned." -> SOURCED: an outside group ("residents") is credited with the criticism, even though no individual is named.
- "This signed column, after quoting opposition MP Jane Doe calling the reform 'reckless', goes on to describe the minister's week." -> SOURCED: even though the piece is signed, the criticism itself is attributed to the quoted MP, not to the columnist.
- "The cantonal government and a coalition of municipalities jointly oppose the project, citing its excessive cost." -> SOURCED: a named collective is credited with the criticism, even with no individual inside it named.
- "A year-end roundup lists, among several federal councillors, that Martin Pfister had a difficult year." -> JOURNALIST: Pfister's name appears only as the subject being discussed; no one is credited with criticizing him.
- "The piece quotes an economist about the state of the labour market, then separately argues in its own voice that the minister's reform is misguided." -> JOURNALIST: an outside voice is quoted, but not for the criticism of the minister -- that judgment is the article's own.
- "The piece argues, in its own narrative voice, that the reform is short-sighted and will hurt small businesses, without attributing that judgment to any critic, resident, expert, or other outside voice." -> JOURNALIST: no outside voice is credited with the criticism at all.

First, in one sentence, say whether the criticism of "{keyword}" specifically is attributed to an outside voice (and who, even loosely), or whether it is the article's own narrative framing.
Then, on a new line, answer with exactly one word:
SOURCED — if the criticism of "{keyword}" is attributed to any outside voice, named or not.
JOURNALIST — if no outside voice is credited with that specific criticism and it is the article's own narrative framing.\
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
