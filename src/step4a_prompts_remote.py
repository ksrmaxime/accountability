# src/step4a_prompts_remote.py
"""
STEP 4a -- source attribution, REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared
Apertus-v1.5-70B (via --backend litellm, the UNIL/CSCS proxy) against
Apertus-8B-Instruct-2509 (local GPU) on step 3 and found the 70B remote
model gives better results there with a broader prompt (see
src/step3_prompts_remote.py's own module docstring), Maxime wants to move
the rest of the pipeline (step4a onward) onto the same remote backend.
This file exists purely so that any future prompt experimentation for
step4a's remote path never touches src/step4a_prompts.py, which must stay
exactly as it is: the extensively-tuned production wording for the local
8B/vllm/transformers path (see that file's own module docstring for its
full tuning history -- the removed/reverted pass-2 validation-on-
justification design, the 2026-11-17 "forced re-search" redesign, the
2026-11-18 pass-3 coherence check, etc.).

At creation, NO WORDING HAD BEEN CHANGED ANYWHERE BELOW: every prompt
string (SYSTEM_PROMPT, the pass-2 "forced re-search" VERIFY_SYSTEM_PROMPT,
FORCE_SYSTEM_PROMPT) and every worked example was byte-identical to
src/step4a_prompts.py at the time of this duplication -- only the
module-level docstring (documentation, not prompt content) was written
fresh. See the entries below for the real wording changes made to this
file since.

2026-11-24 "verdict-last" fix (USER_TEMPLATE pass 1, VERIFY_USER_TEMPLATE
pass 2): diagnosed from job20905947 (the first full remote-chain run on
GOLD800). step4a's oracle accuracy on that run (62.9%, n=89) was well
below the classic 8B baseline (82.1% majority / 83.2% best run), with a
heavily FN-directional error pattern (30 FN vs 3 FP). Reading the FN
rows' own stored justifications found that 24/30 (80%) were NOT genuine
reasoning failures: the draft's own justification text literally opens
with "SOURCED -- <correct reasoning, closely matching gold's own
justification>", yet the row's stored source_attribution_draft is
JOURNALIST. Across the full 800-row run, 47/62 (76%) of all
draft-JOURNALIST rows show this same contradiction.

Root cause: this prompt's own closing instruction ("Then, on a new line,
answer with exactly one word: SOURCED -- if ... / JOURNALIST -- if ...")
models the answer as a "LABEL -- condition" pair, which the 70B tends to
take literally and front-load as "SOURCED -- <its reasoning>" at the very
START of its response, sometimes with no separate concluding label line
at all. src/verify_utils.py's shared parse_draft_labeled()/_find_label()
(used unconditionally by every step/backend, deliberately NOT duplicated
here) resolve a label by scanning the LAST line of the response first --
tuned against the local 8B's own tendency to reason-then-conclude. When
the 70B's actual last line instead continues the reasoning and happens to
use the word "journalist" as ordinary language (e.g. contrasting with
"the journalist's own narrative"), the parser locks onto that incidental
mention instead of the real, already-stated opening verdict -- while the
justification text it saves (everything except that misread last line)
still keeps the correct opening "SOURCED --" sentence, which is exactly
how the contradiction was caught after the fact.

Fix (Maxime's explicit request, scoped to this remote-only file, per his
instruction: ask the model to justify first and answer last, so the
single-word verdict is unambiguously the final line): both USER_TEMPLATE
and VERIFY_USER_TEMPLATE now explicitly forbid opening with (or using
inside the reasoning) the bare words SOURCED/JOURNALIST, and require the
single verdict word, alone and with no trailing dash/explanation, as the
very last line of the response -- so _find_label(lines[-1]) sees nothing
but that one clean word regardless of how the preceding reasoning is
phrased. FORCE_USER_TEMPLATE (the no-justification last-resort pass) was
already unambiguous ("Answer with exactly one word: SOURCED or
JOURNALIST.") and is left untouched. Confirmed as fixing the verdict-
FIRST failure mode (see the 2026-11-25 entry below for the residual,
different self-contradiction mode found after this fix).

2026-11-24 "attribution != outsideness" clarification (USER_TEMPLATE and
VERIFY_USER_TEMPLATE, pass 1 and pass 2, SUPERSEDED by the 2026-11-25 full
rewrite below -- kept here for the historical record): reading
job20943266's 14 remaining FN oracle errors found that 7/14 (50%) shared
one conceptual root cause -- gold counting a criticism as SOURCED when
attributed to a court ruling, a federal audit finding, a parliamentary
oversight letter, an agency chief's own self-critical admission, named
letter-signatories, or a signed guest author, none of which are
"external to the organisation" in the classic adversarial sense. Patched
by broadening the existing "outside voice" definition's worked examples
and exclusion clause, without touching the core question wording itself.
Confirmed on job20947346 (see below): the broadening measurably improved
RETRIEVAL (the model started correctly naming the self-critical chief and
the guest author by name and role) but did NOT reliably change the
VERDICT -- in 2/2 observed cases the model still concluded JOURNALIST
after correctly identifying the right voice, reasoning that "the
journalist synthesises/frames this into the narrative". This showed the
problem was not (only) a missing clause but the core question's own
framing -- see the full rewrite below.

2026-11-25 "two modes of criticism" full rewrite (Maxime's explicit
instruction, modeled on the step5 "WHAT vs HOW" precedent: diagnose
together conceptually first, then rewrite the ENTIRE core definition from
scratch rather than patch another clause onto the existing one):
job20947346's full-chain analysis found two residual problems in step4a,
neither fixable by another added clause:

(1) 7 of 11 oracle false positives (Rösti, Amherd, Armee suisse, SRC/NDB,
DDPS/VBS, Keller-Sutter x2) have a draft justification that is COMPLETE,
clean, and unambiguously argues JOURNALIST ("the judgment is framed as
the journalist's own assessment", "no external entity is quoted or
credited") -- with NO glomming, no missing line break, nothing the
2026-11-24 "glomming" detection (scripts/step4a_source_attribution.py)
can catch, since that check only looks at whether the bare word is
cleanly isolated on its own line, which it is here. The bare final word is
simply the opposite of what the reasoning just said. This is a genuine
CONTENT self-contradiction, not a formatting one.

(2) 2 of 11 oracle false negatives (Baume-Schneider/Samir Jamal Aldin,
armasuisse/Urs Loher) show the 2026-11-24 clarification's retrieval half
working but its verdict half not: pass 1 now correctly names Samir Jamal
Aldin as "a film-maker and co-president of the Swiss Film Academy" and
Urs Loher's own quoted self-criticism, by name -- but still concludes
JOURNALIST both times, reasoning that the journalist "synthesises" or
"frames" these statements "into a narrative". Diagnosis, confirmed with
Maxime: the question this prompt has always asked -- "is it attributed to
an outside voice, or is it the article's own narrative framing?" -- never
actually distinguishes "the journalist is reporting/relaying someone
else's judgment" from "the journalist formed this judgment himself",
because EVERY sentence in an article is, trivially, part of "the
article's own narrative framing" in the sense of being written by the
journalist -- so the model always has an easy escape hatch back to
JOURNALIST even after correctly finding the right voice. Patching one more
exclusion clause onto the existing wording (as the 2026-11-24 entry
above tried) does not fix this, because the flaw is in what the base
question itself asks the model to look for, not in which actors count --
the same diagnosis, and the same prescribed fix (discuss the core
distinction together, then rewrite the WHOLE prompt from scratch rather
than stack a clause), as step5's "two-test" failure and its "WHAT vs HOW"
replacement.

Fix: SYSTEM_PROMPT now opens with a general explanation (Maxime's own
framing, used near-verbatim) of the two different places a critical
judgment in an article can come from -- the journalist's own analysis,
formed by himself with no one else credited for it, versus a judgment
someone ELSE already formed (a testimony, a document, a formal finding, a
quoted statement) that the journalist merely reports, summarises, or
quotes -- explicitly stating that the journalist being the one who
physically writes the article does NOT by itself make a relayed judgment
his own. USER_TEMPLATE's question is rephrased to ask directly which of
these two the specific criticism is, and adds a concrete operational test
(complementing Maxime's conceptual framing): "try to finish the sentence
'According to ___, {keyword} [the criticism].' If you can fill that blank
with a real, specific person, document, ruling, or group -- even if the
journalist is only reporting it -- SOURCED. If the only honest way to
finish it is 'according to the article itself/the journalist', with no
other real candidate -- JOURNALIST." This directly targets both residual
problems: it gives the model a mechanical check to run even when its
reasoning might otherwise waver (addressing failure (1)), and it makes
explicit, for the first time, that relaying/synthesising someone else's
already-formed judgment does not make it the journalist's own (addressing
failure (2)). Two new worked examples added specifically for failure (2)'s
exact shape (a named guest author's signed column; an agency chief's own
quoted admission that the journalist then "weaves into a broader
narrative"). The existing broadened "outside voice can be internal to the
organisation" examples and exclusions from the 2026-11-24 clarification
above are KEPT, recast into the new framing rather than discarded --
Maxime's instruction was specifically to not stack a new layer on top of
the OLD base question, not to discard the earlier correct substance. Same
rewrite applied to VERIFY_USER_TEMPLATE (pass 2), including its forced-
enumeration instruction, so the re-search pass reasons with the same
"who actually formed this judgment" framing. FORCE_SYSTEM_PROMPT/
FORCE_USER_TEMPLATE (the bare-word last resort) are, as always, left
untouched -- already unambiguous and orthogonal to this question. Not yet
confirmed on a real run.

scripts/step4a_source_attribution.py picks this module instead of
src/step4a_prompts.py only when --backend litellm is used; every other
backend (vllm/transformers) is completely unaffected, byte-for-byte, by
anything in this file. The pass-3 coherence check
(src/step4a_pass2_coherence_logit_prompts.py) is a separate, SHARED file,
deliberately NOT duplicated here -- same reasoning as step3's
src/step3_logit_prompts.py: a fixed, narrow validation question is meant
to act as the same independent, skeptical second read regardless of which
draft prompt (this file, or the original) produced pass 2's justification.
"""
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs. When a journalist writes
an article that criticizes someone or something, that critical judgment can come from one of two
different places. First, the journalist can form the judgment entirely himself: based on his own
analysis of the facts, he personally concludes that the target deserves criticism, without anyone
else being credited with holding that specific view. Second, the journalist can instead be relaying
a judgment that someone else has already formed -- a testimony, a document, a formal finding, or a
quoted statement -- which he reports, summarises, or quotes, but which originates from that other
party, not from his own analysis. The fact that the journalist is the one physically writing the
article does NOT by itself make the criticism his own: what matters is whose judgment it actually
is, not who is reporting it. Your task is to determine, for one specific criticism in a given
article, which of these two it is.\
"""

USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized.

ARTICLE:
{article_text}

We want to determine: is this specific criticism of "{keyword}" built on a point of view that
comes from testimonies, documents, or statements provided by someone else -- even if the journalist
is the one reporting, summarising, or quoting them -- or is it purely the journalist who, based on
his own analysis, without crediting anyone else with holding this view, gives this critical
judgment himself?

A simple way to check: try to finish the sentence "According to ___, '{keyword}' <the criticism>."
If you can fill that blank with a real, specific person, document, ruling, or group who can be said
to hold that judgment -- even if the journalist is only reporting it -- the answer is SOURCED. If
the only honest way to finish that sentence is "according to the article itself" or "according to
the journalist," with no other real candidate, the answer is JOURNALIST.

That "someone else" can be identified as loosely as a named individual, a political party, a
canton, an administration, a company, an association, a category of people ("residents,"
"opponents," "experts"), or survey respondents -- none of this needs a specific person's name to
qualify, as long as it is a real, identifiable party rather than a vague or anonymous reference. It
also does not need to be external to the organisation being criticized: a court, tribunal, audit
office, or other oversight body's own formal ruling or report counts, even though it is part of the
state apparatus; a specific named individual's own quoted words acknowledging or admitting the
criticism counts, even when that individual IS "{keyword}" or speaks for it -- what matters is that
it is that person's own judgment, not the journalist's; a specific, identifiable, or countable
collective counts (e.g. "250 employees who signed an internal letter"), even if internal to the
criticized organisation; and a signed opinion piece or guest column written by an identified
outside figure counts as that author's own judgment, even though it is signed and even though the
journalist's own outlet published it.

"{keyword}" itself never counts as that "someone else" just because it is named as the one who made
the decision, took the action, or is otherwise the subject of the criticism -- that is different
from "{keyword}" itself (or someone speaking for it) being actively quoted acknowledging or
admitting the criticism in their own words, which DOES count, per above. And a vague,
undifferentiated reference with no specific nameable or countable referent (e.g. "critics say,"
"some observers," "the community") does not count either -- that is still the journalist's own
narration, just using vague language, not a real judgment traceable to someone else.

Two things do NOT settle the question by themselves: a signed byline does not mean the article has
no outside judgment behind the criticism of "{keyword}" -- a signed piece can still relay or quote
someone else's criticism of the target, or itself be the voice of an identified outside author (see
above). And sources quoted elsewhere in the article, for other claims or about other people, do not
count unless they are specifically the ones whose judgment the criticism of "{keyword}" itself
comes from.

For example:
- "Residents interviewed for the piece say the new bus route is useless and badly planned." -> SOURCED: "according to residents interviewed for the piece" -- a real, identifiable party holds this judgment, even though no individual is named.
- "This signed column, after quoting opposition MP Jane Doe calling the reform 'reckless', goes on to describe the minister's week." -> SOURCED: the criticism traces back to the quoted MP's own judgment, not to the columnist's, even though the piece is signed.
- "The cantonal government and a coalition of municipalities jointly oppose the project, citing its excessive cost." -> SOURCED: a named collective holds this judgment, even with no individual inside it named.
- "A year-end roundup lists, among several federal councillors, that Martin Pfister had a difficult year." -> JOURNALIST: there is no one else's judgment to trace this to -- Pfister's name appears only as the subject being discussed.
- "The piece quotes an economist about the state of the labour market, then separately argues in its own voice that the minister's reform is misguided." -> JOURNALIST: the economist's judgment is about something else; the criticism of the minister has no other candidate holder besides the journalist himself.
- "The piece argues, in its own narrative voice, that the reform is short-sighted and will hurt small businesses, without attributing that judgment to any critic, resident, expert, or other outside voice." -> JOURNALIST: the only honest answer to "according to whom?" is "according to the article itself."
- "A federal audit office's report concludes that the agency mismanaged the funds, and the article cites this finding." -> SOURCED: "according to the federal audit office's report" -- the judgment is the audit office's, not the journalist's, even though it is a state body rather than an adversarial outsider.
- "The agency's own director, quoted directly, says: 'We made serious mistakes and should have acted sooner.'" -> SOURCED: "according to the agency's own director" -- the judgment is his own quoted admission, not the journalist's, even though he heads the criticized agency itself.
- "250 employees of the ministry sign an internal letter criticizing its leadership, and the article reports on this letter." -> SOURCED: "according to the 250 signatories of the internal letter" -- a specific, countable, documented group holds this judgment, even though they work inside the criticized ministry.
- "A guest column, signed by an outside economist, argues that the minister's reform is misguided." -> SOURCED: "according to [the named economist]" -- the signed author is an identified outside figure, not the paper's own journalist.
- "A guest column, written by a named outside filmmaker and signed with his name, argues through his own perspective that the ministry's funding policy harms the industry; the journalist's own editorial framing elsewhere in the piece is neutral." -> SOURCED: the judgment belongs to the named outside author, not to the journalist's own analysis, even though the journalist's outlet published and framed the piece around it.
- "An agency's own chief, quoted directly, admits in his own words that his team handled the matter poorly; the journalist then weaves this admission into a broader narrative about the agency's challenges." -> SOURCED: the judgment originates from the chief's own quoted admission -- the journalist synthesising it into a larger narrative does not change who actually formed that judgment.
- "A roundup piece vaguely mentions that 'collectors have grumbled' about the agency's new policy, without naming or identifying any specific person, group, or document." -> JOURNALIST: there is no real, identifiable "someone else" to point to -- just vague language standing in for the journalist's own narration.

Write your reasoning first: in one or two full sentences, explain who you believe actually formed
the judgment behind the criticism of "{keyword}" specifically -- the journalist himself, or someone
else (and who, even loosely) -- whichever the "according to ___" test points to. Do not begin your
response with the word SOURCED or JOURNALIST, and do not use either of those two words anywhere
inside this reasoning -- describe what you found in plain language instead.

Only after that reasoning, add one final new line containing nothing but a single word -- no dash,
no explanation, no punctuation, nothing else on that line:
SOURCED, if someone other than the journalist can be said to hold the judgment behind the criticism of "{keyword}", named or not.
JOURNALIST, if the only honest answer to "according to whom?" is the article itself, with no one else's judgment behind it.
That single word must be the very last thing in your response.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify) -- 2026-11-17 redesign ("forced re-search"), rewritten
# 2026-11-25 alongside pass 1 to use the same "two modes of criticism" /
# "according to whom?" framing -- see this file's own module docstring for
# the full history and rationale. Re-reads the FULL ARTICLE (never just
# pass 1's justification -- that was the old, reverted design's fatal
# blind spot), and explicitly forces an enumerate-before-answer step to
# counter the diagnosed retrieval/attention failure. Scoped by the caller
# (scripts/step4a_source_attribution.py) to source_attribution ==
# "JOURNALIST" rows only -- this prompt's own wording assumes that. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs. When a journalist writes an
article that criticizes someone or something, that critical judgment can come from one of two
different places: the journalist's own analysis, formed by himself with no one else credited for
it, or a judgment someone else already formed -- a testimony, a document, a formal finding, a
quoted statement -- that the journalist merely reports, summarises, or quotes. The journalist being
the one who physically writes the article does NOT by itself make a relayed judgment his own. A
first, quick pass over a newspaper article already concluded that a specific criticism in it has no
one else's judgment behind it. Your task is to re-examine the SAME article once more, more
carefully, specifically to check whether that first pass missed a real other party whose judgment
it actually is.\
"""

VERIFY_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized. A first pass already concluded that this \
criticism is JOURNALIST -- that the journalist's own analysis is the only judgment behind it, with \
no one else credited.

ARTICLE:
{article_text}

We want to determine: is this specific criticism of "{keyword}" built on a point of view that \
comes from testimonies, documents, or statements provided by someone else -- even if the \
journalist is the one reporting, summarising, or quoting them -- or is it purely the journalist \
who, based on his own analysis, without crediting anyone else with holding this view, gives this \
critical judgment himself? A simple way to check: try to finish the sentence "According to ___, \
'{keyword}' <the criticism>." If you can fill that blank with a real, specific person, document, \
ruling, or group -- even if the journalist is only reporting it -- SOURCED. If the only honest way \
to finish it is "according to the article itself" or "according to the journalist," with no other \
real candidate, JOURNALIST.

That "someone else" can be identified as loosely as a named individual, a political party, a \
canton, an administration, a company, an association, a category of people ("residents," \
"opponents," "experts"), or survey respondents -- none of this needs a specific person's name to \
qualify, as long as it is a real, identifiable party rather than a vague or anonymous reference. It \
also does not need to be external to the organisation being criticized: a court, tribunal, audit \
office, or other oversight body's own formal ruling or report counts; a specific named \
individual's own quoted words acknowledging or admitting the criticism counts, even when that \
individual IS "{keyword}" or speaks for it; a specific, identifiable, or countable collective \
counts (e.g. "250 employees who signed an internal letter"), even if internal to the criticized \
organisation; and a signed opinion piece or guest column written by an identified outside figure \
counts as that author's own judgment, even though the journalist's own outlet published it and even \
though the journalist may weave it into a broader narrative elsewhere in the piece. "{keyword}" \
itself never counts just because it is named as the one who made the decision or is otherwise the \
subject of the criticism -- but that is different from "{keyword}" itself (or someone speaking for \
it) being actively quoted acknowledging the criticism in their own words, which DOES count. A \
vague, undifferentiated reference with no specific nameable or countable referent ("critics say," \
"some observers," "the community") does NOT count -- that is still the journalist's own narration. \
And a signed byline does not settle the question either way by itself: a signed piece can still \
relay or quote someone else's criticism, or itself be the voice of an identified outside author.

Before answering, go through the article once more and list, one by one, every named individual, \
political party, canton, administration, company, association, category of people, court or \
tribunal, audit or oversight body, signed author, or other actor mentioned anywhere in connection \
with the criticism of "{keyword}" -- even if mentioned only briefly, in a quote, in an interview \
answer or question, as part of a named group with no individual singled out, as a formal ruling or \
report, or as "{keyword}" itself (or someone speaking for it) openly acknowledging the criticism in \
their own words. For each one you find, check specifically: does the "according to ___" test point \
to THEM as the one whose judgment this is, or are they mentioned for some other reason?

Then say in one or two sentences what you found (name the actor if you found one, or say plainly \
that the search found none) -- without using the words SOURCED or JOURNALIST anywhere in that \
explanation.

Only after that explanation, add one final new line containing nothing but a single word -- no \
dash, no explanation, no punctuation, nothing else on that line:
SOURCED, if you found someone else whose judgment this criticism of "{keyword}" actually is, that the first pass missed.
JOURNALIST, if, even after this closer search, the only honest answer to "according to whom?" remains the article itself.
That single word must be the very last thing in your response.\
"""


def build_verify_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return VERIFY_USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Last resort (force): a bare one-word decision, no justification asked
# for at all, used only for rows that still have no parseable pass-1 answer
# after every retry of the full prompt above. Deliberately minimal -- the
# goal here is just to stop a row from being silently dropped, not to
# produce a reasoned judgment. Any row answered this way is flagged via
# source_attribution_forced=True (see scripts/step4a_source_attribution.py)
# and never goes through pass 2, since there is no justification to verify.
# UNCHANGED by the 2026-11-25 rewrite above -- already unambiguous and
# orthogonal to the "two modes of criticism" framing (it asks for a bare
# word, with no reasoning for the model to contradict itself against).

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
