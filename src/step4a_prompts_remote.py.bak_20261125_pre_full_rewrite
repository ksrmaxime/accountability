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
fresh. See the entry just below for the first real wording change made
to this file.

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
phrased. FORCE_USER_TEMPLATE
(the no-justification last-resort pass) was already unambiguous ("Answer
with exactly one word: SOURCED or JOURNALIST.") and is left untouched.
Not yet re-run/confirmed on real data -- see sbatch_step4a_remote_test.sh
for the next test run.

2026-11-24 "attribution != outsideness" clarification (USER_TEMPLATE and
VERIFY_USER_TEMPLATE, pass 1 and pass 2): reading job20943266's 14
remaining FN oracle errors found that 7/14 (50%) share one conceptual root
cause, across six different-looking surface patterns -- a court ruling
(Tribunal administratif federal, x2), a federal audit office's finding
(EFK), a parliamentary oversight committee's formal letter
(Finanzdelegation), an agency chief's own quoted self-critical admission
(armasuisse/Urs Loher), 250 named employees who signed an internal letter
(DFAE/Cassis), and a signed guest op-ed's own named author (a filmmaker,
not a staff journalist, writing for Baume-Schneider's portfolio). In every
case, pass 1's own justification correctly identifies the actor by name
but still concludes JOURNALIST, reasoning that nothing "external" or
"adversarial" was credited with the criticism -- while gold counts all six
as SOURCED. Root cause: although this file's own SYSTEM_PROMPT already
defines "outside voice" as "external to the journalist" (never "external
to the organisation"), the USER_TEMPLATE/VERIFY_USER_TEMPLATE's worked
examples and exclusion clause are all built from classic adversarial-
external-individual cases, which appears to have implicitly anchored the
model on a narrower reading than the definition actually states. Maxime's
own framing, used verbatim as the fix: "la voix exterieure, c'est une voix
autre que le journaliste, et pas une voix exterieure a l'entite" -- a
specific, identifiable, named/countable source counts regardless of
whether it is organisationally external to "{keyword}" (a court or audit
office's own formal finding; a named individual's own quoted admission,
even if that individual IS "{keyword}" or speaks for it; a specific,
countable collective such as named letter-signatories, even if internal to
the criticized organisation; a signed guest author distinct from a staff
journalist) -- as opposed to a vague, anonymous, undifferentiated
reference (no specific nameable/countable referent) or the journalist's
own unattributed narration, which still correctly stays JOURNALIST (kept
an explicit worked example for this, mirroring the already-correctly-
classified "diffuse numismatic community" JOURNALIST row from the same
run, to guard against over-broadening). Added four new worked examples per
template (audit finding, self-critical chief, named-collective letter,
guest op-ed author) plus the diffuse-reference JOURNALIST counter-example,
and widened VERIFY_USER_TEMPLATE's forced-enumeration instruction to name
these categories explicitly, so pass 2's re-search actually looks for
them. The existing "{keyword}' itself never counts... just because it is
named as the subject" exclusion is UNCHANGED and still correct -- it is
about passive naming-as-subject, not about "{keyword}" (or its
representative) being actively quoted admitting the criticism, which is a
different case, now explicitly carved out. Not yet confirmed on a real
run.

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
subject of the criticism -- that is different from "{keyword}" itself (or
someone speaking for it) being actively QUOTED acknowledging or admitting
the criticism in their own words; see below.

"Outside voice" means a voice other than the journalist's own narrating
voice -- it does NOT need to be external to the organisation or person
being criticized. In particular, all of the following count as an outside
voice, as long as they are specifically credited with the criticism of
"{keyword}":
- a court, tribunal, audit office, or other oversight body's own formal
  ruling, finding, or report (e.g. a court declaring a decision unlawful,
  an audit office's report, a parliamentary committee's letter) -- the
  institution itself is making that determination, not the journalist,
  even though it is part of the state apparatus rather than an adversarial
  outsider;
- a specific, named individual's own quoted words acknowledging or
  admitting the criticism, even when that individual IS "{keyword}" or
  speaks for/within the entity being criticized (e.g. an agency's own head
  candidly admitting, in his own words, that something went wrong) -- what
  matters is that it is that person's own voice, not the journalist's;
- a specific, identifiable, or countable collective credited by name or
  clear description, even if internal to the criticized organisation (e.g.
  "250 employees who signed an internal letter") -- unlike a vague,
  undifferentiated reference with no specific nameable or countable
  referent (e.g. "critics say," "some observers," "the community"), which
  does NOT count and stays JOURNALIST;
- a signed opinion piece or guest column written by an identified outside
  figure (an expert, a stakeholder, an affected party), as distinct from a
  staff journalist's own byline -- that counts as the named author's own
  voice, even though it is signed.

Two things do NOT settle the question by themselves: a signed byline does
not mean the article has no outside sources behind the criticism of
"{keyword}" -- a signed piece can still quote or cite someone else's
criticism of the target (or, per above, be itself the voice of an
identified outside author). And sources quoted elsewhere in the article,
for other claims or about other people, do not count unless they are
specifically the ones credited with the criticism of "{keyword}" itself.

The answer is JOURNALIST only when, specifically for the criticism of
"{keyword}", no outside voice of any kind -- named or not, internal or
external to the criticized party -- is credited with holding or voicing
it: the judgment is presented as the article's own.

For example:
- "Residents interviewed for the piece say the new bus route is useless and badly planned." -> SOURCED: an outside group ("residents") is credited with the criticism, even though no individual is named.
- "This signed column, after quoting opposition MP Jane Doe calling the reform 'reckless', goes on to describe the minister's week." -> SOURCED: even though the piece is signed, the criticism itself is attributed to the quoted MP, not to the columnist.
- "The cantonal government and a coalition of municipalities jointly oppose the project, citing its excessive cost." -> SOURCED: a named collective is credited with the criticism, even with no individual inside it named.
- "A year-end roundup lists, among several federal councillors, that Martin Pfister had a difficult year." -> JOURNALIST: Pfister's name appears only as the subject being discussed; no one is credited with criticizing him.
- "The piece quotes an economist about the state of the labour market, then separately argues in its own voice that the minister's reform is misguided." -> JOURNALIST: an outside voice is quoted, but not for the criticism of the minister -- that judgment is the article's own.
- "The piece argues, in its own narrative voice, that the reform is short-sighted and will hurt small businesses, without attributing that judgment to any critic, resident, expert, or other outside voice." -> JOURNALIST: no outside voice is credited with the criticism at all.
- "A federal audit office's report concludes that the agency mismanaged the funds, and the article cites this finding." -> SOURCED: the audit office's own formal finding is credited with the criticism, even though it is a state body, not an adversarial outsider.
- "The agency's own director, quoted directly, says: 'We made serious mistakes and should have acted sooner.'" -> SOURCED: the director's own quoted admission is his own voice, not the journalist's narration -- even though he heads the criticized agency itself.
- "250 employees of the ministry sign an internal letter criticizing its leadership, and the article reports on this letter." -> SOURCED: a specific, countable, documented group is credited with the criticism, even though they work inside the criticized ministry.
- "A guest column, signed by an outside economist, argues that the minister's reform is misguided." -> SOURCED: the signed author is an identified outside figure, not the paper's own journalist, so the criticism is attributed to her, not to the article's own narrative.
- "A roundup piece vaguely mentions that 'collectors have grumbled' about the agency's new policy, without naming or identifying any specific person, group, or document." -> JOURNALIST: the reference is too diffuse and unidentified to count as a specific outside voice.

Write your reasoning first: in one or two full sentences, explain whether
the criticism of "{keyword}" specifically is attributed to an outside
voice (and who, even loosely), or whether it is the article's own
narrative framing. Do not begin your response with the word SOURCED or
JOURNALIST, and do not use either of those two words anywhere inside this
reasoning -- describe what you found in plain language instead.

Only after that reasoning, add one final new line containing nothing but
a single word -- no dash, no explanation, no punctuation, nothing else on
that line:
SOURCED, if the criticism of "{keyword}" is attributed to any outside voice, named or not.
JOURNALIST, if no outside voice is credited with that specific criticism and it is the article's own narrative framing.
That single word must be the very last thing in your response.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)


# --- Pass 2 (verify) -- 2026-11-17 redesign ("forced re-search"). See this
# file's own module docstring for the full history and rationale. Re-reads
# the FULL ARTICLE (never just pass 1's justification -- that was the old,
# reverted design's fatal blind spot), keeps pass 1's own already-tuned
# exclusions verbatim, and explicitly forces an enumerate-before-answer
# step to counter the diagnosed retrieval/attention failure. Scoped by the
# caller (scripts/step4a_source_attribution.py) to source_attribution ==
# "JOURNALIST" rows only -- this prompt's own wording assumes that. ---

VERIFY_SYSTEM_PROMPT = """\
You are a media analysis assistant specialised in Swiss public affairs. A first, quick pass over \
a newspaper article already concluded that a specific criticism in it is not attributed to any \
outside voice. Your task is to re-examine the SAME article once more, more carefully, specifically \
to check whether that first pass missed a real outside voice.\
"""

VERIFY_USER_TEMPLATE = """\
In the article below, "{keyword}" is criticized. A first pass already concluded that this \
criticism is JOURNALIST -- the article's own narrative framing, with no outside voice credited.

ARTICLE:
{article_text}

An outside voice can be identified as loosely as a named individual, a political party, a canton, \
an administration, a company, an association, a category of people ("residents," "opponents," \
"experts"), or survey respondents -- none of this needs a specific person's name to qualify. \
"{keyword}" itself never counts as that voice just because it is named as the one who made the \
decision, took the action, or is otherwise the subject of the criticism -- but that is different \
from "{keyword}" itself (or someone speaking for it) being actively QUOTED acknowledging or \
admitting the criticism in their own words, which DOES count (see below). A signed byline does not \
mean the article has no outside sources behind the criticism of "{keyword}" -- a signed piece can \
still quote or cite someone else's criticism of the target, or itself be the voice of an identified \
outside author (see below). And sources quoted elsewhere in the article, for other claims or about \
other people, do not count unless they are specifically the ones credited with the criticism of \
"{keyword}" itself.

"Outside voice" means a voice other than the journalist's own narrating voice -- it does NOT need \
to be external to the organisation or person being criticized. All of the following count, as long \
as they are specifically credited with the criticism of "{keyword}": a court, tribunal, audit \
office, or other oversight body's own formal ruling, finding, or report (even though it is part of \
the state apparatus, not an adversarial outsider); a specific, named individual's own quoted words \
acknowledging or admitting the criticism, even when that individual IS "{keyword}" or speaks for/ \
within the entity being criticized (e.g. an agency's own head candidly admitting, in his own words, \
that something went wrong); a specific, identifiable, or countable collective credited by name or \
clear description, even if internal to the criticized organisation (e.g. "250 employees who signed \
an internal letter") -- unlike a vague, undifferentiated reference with no specific nameable or \
countable referent ("critics say," "some observers," "the community"), which does NOT count; and a \
signed opinion piece or guest column written by an identified outside figure (an expert, a \
stakeholder, an affected party), as distinct from a staff journalist's own byline.

Before answering, go through the article once more and list, one by one, every named individual, \
political party, canton, administration, company, association, category of people, court or \
tribunal, audit or oversight body, signed author, or other actor mentioned anywhere in connection \
with the criticism of "{keyword}" -- even if mentioned only briefly, in a quote, in an interview \
answer or question, as part of a named group with no individual singled out, as a formal ruling or \
report, or as "{keyword}" itself (or someone speaking for it) openly acknowledging the criticism in \
their own words. For each one you find, check specifically: are THEY the one credited with the \
criticism of "{keyword}", or are they mentioned for some other reason?

Then say in one or two sentences what you found (name the actor if you found one, or say plainly \
that the search found none) -- without using the words SOURCED or JOURNALIST anywhere in that \
explanation.

Only after that explanation, add one final new line containing nothing but a single word -- no \
dash, no explanation, no punctuation, nothing else on that line:
SOURCED, if you found an outside voice credited with the criticism of "{keyword}" that the first pass missed.
JOURNALIST, if, even after this closer search, no outside voice of any kind is credited with that specific criticism.
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
