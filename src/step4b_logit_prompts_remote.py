# src/step4b_logit_prompts_remote.py
"""
STEP 4b (logit-probing design) -- REMOTE-BACKEND-ONLY prompt fork.

2026-11-24: created at Maxime's request. Having compared Apertus-v1.5-70B
(via --backend litellm, the UNIL/CSCS proxy) against Apertus-8B-Instruct-2509
(local GPU) on step 3 and found the 70B remote model gives better results
there with a broader prompt (see src/step3_prompts_remote.py's own module
docstring), Maxime wants to move the rest of the pipeline (step4b onward)
onto the same remote backend. This file exists purely so that any future
prompt experimentation for step4b's remote path never touches
src/step4b_logit_prompts.py, which must stay exactly as it is: the
extensively-tuned production wording for the local 8B/vllm/transformers
path (see that file's own module docstring for its full tuning history --
the "summarize-then-probe" pre-pass, the category-boundary sharpening
rounds, the "validate-and-correct" pass 3, etc.).

As of this creation, NO WORDING HAS BEEN CHANGED ANYWHERE BELOW: the
summary pre-pass prompt, all 6 category question templates, the
calibration templates, and the pass-3 validation templates are all
byte-identical to src/step4b_logit_prompts.py at the time of this
duplication -- only this module-level docstring (documentation, not
prompt content) was written fresh, replacing that file's own tuning-
history docstring, which documents a history specific to that file and
not (yet) to this one. This is intentionally just a safe, isolated
starting point for later remote-specific tuning, not a tuning pass
itself.

2026-11-25 "summary reliability" pass -- SUMMARY_TEMPLATE only, nothing
else in this file touched. Diagnosed by reading all 72 (article, generated
source_summary) pairs from job21026402's oracle population against the
final classification outcome: every row with an imperfect summary ended up
misclassified (15/15), while a faithful summary still left a ~21% residual
error rate (12/56) attributable to pass 2's probes rather than the summary
-- so summary quality is a real, separate lever worth pulling on its own.
Four GENERAL weaknesses recurred across multiple, unrelated articles (not
single one-off mistakes, so fixed as general principles, not per-case
patches):
  1. Inferring/asserting a personal-vs-institutional framing the article
     itself never actually states (e.g. adding a committee/session context
     not supported by the text) -- now only stated when the article makes
     it explicit, "not specified by the article" otherwise.
  2. Paraphrasing a named, specific actor (a person, a party, an
     organization) into a vaguer generic description -- now the article's
     own exact name/party/organization must be kept, never generalised.
  3. Describing an anonymous or pseudonymous commenter as if they were an
     identified individual -- now flagged explicitly when the source is
     not a real identified name.
  4. Truncation risk (SUMMARY_MAX_NEW_TOKENS is a fixed sbatch budget, see
     sbatch_step4b_logit_remote_test.sh) cutting off a summary mid-sentence
     before the decisive fact was written -- mitigated by asking for a
     tighter, complete answer; raising that sbatch budget is a separate,
     complementary fix on the launch-script side, not a prompt change.
A fifth, structural issue (an article with more than one distinct critic
of the same keyword, for different reasons) is only made VISIBLE by this
pass (each critic now described separately instead of blended into one),
not resolved -- which one gold then treats as "the" source for that row is
a separate, unresolved question this prompt change does not attempt to
settle.

2026-11-26 "clean rewrite from scratch" -- SUMMARY_TEMPLATE and
CATEGORY_DEFINITION/CANONICAL_EXAMPLES rewritten together from scratch, at
Maxime's explicit request, rather than layering yet another caveat onto
wording that had already drifted through several stacked, increasingly
narrow patches -- most recently the 2026-11-25 pass above, which measurably
REGRESSED job21066855's accuracy (62.0% -> 50.7% vs job21026402, same
71-row population). Maxime's point: this file's starting point
(src/step4b_prompts.py, duplicated via src/step4b_logit_prompts.py) was
tuned sentence-by-sentence for the 8B local model's much narrower
interpretive range; the 70B remote model doesn't need nearly as much
hand-holding, and stacking targeted caveats on an 8B-era base text was
fighting the model rather than trusting it.

SUMMARY_TEMPLATE: cut back from a 4-point structured checklist (what /
who / personal-vs-institutional capacity / anonymous-source flagging /
multi-actor handling) to a single, open paragraph request -- explain what
is being criticized and who says so, on what statement in the article,
without classifying. The capacity/anonymity/multi-actor instructions
added in the 2026-11-25 pass are DELETED outright, not reworded: in
particular, that pass's point 3 ("speaking personally... or explicitly in
the name of an institution or organization") is what caused job21066855's
regression, by teaching the model to phrase ANY representational
statement -- a politician's party, a company, an association -- as "in
the name of X", which lexically collides with Administrative Unit's own
definition below and dragged predictions toward it regardless of the
actual entity type (quantified: pred=Administrative Unit in 35% of rows
whose summary contained that phrase vs 16% of rows that didn't,
job21066855 vs job21026402 comparison). Removing the instruction that
produces the phrase resolves this structurally, instead of special-casing
the phrase itself -- which is also why the definitions below no longer
share that exact wording across categories (see next paragraph).

CATEGORY_DEFINITION / CANONICAL_EXAMPLES: rewritten around the single
underlying test Maxime actually applies when reading these articles
himself, per category:
  - Politician: is the source a Swiss political office-holder or party?
    Often signalled by a party name/abbreviation in parentheses after the
    person's name. A personal opinion or an official party statement both
    count -- this category does not distinguish the two.
  - Administrative Unit of the State vs. Civil Servant: both are state
    employees; the ONLY question is whether the article attaches the
    administrative unit's own name to the statement as its authority
    (-> Administrative Unit -- e.g. "the director of the cantonal
    building department said...") or the person is speaking with no
    institutional name invoked, just for themselves (-> Civil Servant).
    This personal-vs-institutional test is the one place in the whole
    taxonomy where it actually matters.
  - Interest Group: companies, business/professional associations,
    unions, NGOs, lobby groups, media outlets, informal collectives. The
    2026-11-25 regression's core confusion was treating this category as
    if it had the same personal-vs-institutional split as Administrative
    Unit/Civil Servant -- it does not: an entity's own statement and an
    individual explicitly speaking for that entity (a CEO defending his
    own company) are BOTH Interest Group, with no distinction to make.
    This is now stated explicitly in the definition, specifically to
    close off the lexical overlap the deleted summary instruction used
    to create.
  - General Public: ordinary individuals speaking only for themselves,
    regardless of job or title, plus abstract/aggregate references
    ("public opinion", "many residents") with no organization behind
    them.
  - Other: unchanged in substance (unnamed/vague source, or a named
    foreign/non-Swiss political actor) -- a genuine "doesn't fit", not a
    default for uncertainty.
VALIDATION_REMINDER's "Interest Group" entry is removed: the clauses it
used to restate (business executive speaking personally, informal
collectives, media outlets) are now stated directly in the main
definition above, so pass 3 (currently disabled via --skip_validation)
would read the same text pass 2 does, with nothing left to restate.

QUESTION_TEMPLATE, VALIDATION_QUESTION_TEMPLATE, the calibration/letter-
order machinery, and every build_*/letters_for_order function are
UNCHANGED -- this is a wording-only rewrite of the pre-pass prompt and the
six category definitions/examples, nothing structural.

2026-11-26b: job21084261 (the first run under the clean rewrite above)
reached 66.7% accuracy (48/72), the best of the whole series so far (vs
62.0% pre-rewrite with the p0 floor alone, 50.7% on the regressed
2026-11-25 pass) -- see the orchestration script's analysis notes. Its
confusion matrix showed a real, if smaller, residual confusion between
'Politician' and 'Interest Group' -- Maxime's diagnosis: a political party
is, in the everyday sense of the term, itself a group representing a
particular interest, so the two definitions above didn't clearly enough
rule a party OUT of 'Interest Group'. Two concrete misses matched this
exactly: a party ("the Greens") and a named "SVP faction president" both
pulled a confident (margin 0.52 and 0.80) 'Interest Group' prediction
instead of 'Politician' -- "faction" likely reading, to the model, like
"an organized association" rather than "the party's own members in
parliament". Both CATEGORY_DEFINITION entries now say explicitly, and
symmetrically (each category's definition is shown to the model in
isolation -- see QUESTION_TEMPLATE -- so a rule needed by both sides can't
just live in one of them), that a party/faction is Politician, never
Interest Group, regardless of how interest-driven its position is.
CANONICAL_EXAMPLES unchanged.

2026-11-26c: job21084261 ALSO showed source_category_p_civil_servant as
null for ALL 97 SOURCED rows (not just the 4 gold-labelled ones Maxime
asked about) -- categorically different from the other five categories,
which each had at most a handful of scattered nulls (0-5/97, ordinary
occasional per-row logit-read misses). An all-97 failure, with nothing
category-specific about which rows failed, points at the ONE-TIME
calibration call for this category (compute_calibration runs once per
category, not per row -- see scripts/step4b_source_category_logit.py) --
if THAT single call fails to read a clean A/B logit, p0['Civil Servant']
is NaN and the per-row aggregation loop then nulls out every row's
Civil Servant probability regardless of how the real per-row reads went
(pd.isna(p0[cat]) short-circuits to NaN). Checking the CATEGORY_DEFINITION
text itself for anything that singles Civil Servant out: it was the ONLY
one of the six containing an embedded question mark ("...-> Civil
Servant)? A retired..."), sitting unusually close to the template's own
real "Question:" section below it -- on the calibration prompt's
content-free "N/A" variant, where there is nothing else for the model to
anchor on, this is a plausible trigger for an off-format first token that
breaks the A/B logit read. Rewritten as a declarative sentence instead
(still states the exact same test), purely to remove this one structural
oddity -- this is a plausible, not confirmed, mechanism (no cluster shell
access to inspect the actual failed API calls), but it is a safe,
strictly-improving change either way: no other category's definition uses
an embedded question, so this just brings Civil Servant in line with the
other five. Maxime separately clears the stale, non-job-specific
checkpoint file (OUTPUT_BASE has no job id, so
.../step4b_logit_remote_task0_checkpoint.parquet is reused/extended
across every test-script run) before the next run, since the resume logic
has no way to tell a cached NaN apart from a cached real value and would
otherwise keep reusing this failure forever regardless of any prompt fix.

2026-11-27 "three qualitative fixes" -- CATEGORY_DEFINITION only
(CANONICAL_EXAMPLES unchanged, same precedent as 2026-11-26b/c), from
reading job21121556's confusion matrix and every one of its 17 remaining
errors' source_summary (see that analysis), and Maxime's explicit answers
to the three questions it raised:

  1. Civil Servant: an unnamed/anonymous source identified only by where
     they work (e.g. "sources inside Fedpol say...") was being predicted
     as 'Other' instead (job21121556's fedpol/ceaf826f row) -- Maxime: if
     they're known to work inside a department and are sharing their own
     work experience without speaking for the institution, that is
     clearly Civil Servant, named or not. Added a sentence making this
     explicit: not being personally named does not change the category,
     only whether they speak for themselves vs. for the institution does.

  2. Interest Group: a neighbourhood/civic association (job21121556's
     OFROU-ASTRA/Pia Meier row) was being predicted as 'General Public'
     instead -- Maxime: yes, that is clearly an Interest Group (an
     organized, non-political entity), add it to the list. Added
     "a foundation, or a civic/residents' association (e.g. a
     neighbourhood association)" to the explicit entity list (both
     places it appears in this category's definition) -- a GENERAL
     addition to the category's own scope, not a one-off patch naming
     this specific association.

  3. Politician vs Interest Group: "the Greens"/"les Verts" criticizing
     (job21121556's Albert Rosti row) was being predicted as Interest
     Group -- the 2026-11-26b party-vs-interest-group fix evidently
     doesn't fire reliably when the party is referred to by a short,
     topic-sounding nickname rather than its formal name, since nothing
     in that fix's wording addressed nicknames specifically. Maxime:
     yes, "les Verts" refers to the Green party, not an ecological
     interest group -- make that an explicit rule. Added a sentence to
     BOTH Politician's and Interest Group's definitions (symmetric, same
     precedent as 2026-11-26b) naming "the Greens"/"les Verts" and a few
     other common Swiss party nicknames (the Socialists, UDC/SVP, PLR/
     FDP, Le Centre, Vert'liberaux/pvl) explicitly, as a GENERAL
     "nickname still means the party" rule rather than special-casing
     just the Greens.

These three do not touch the recurring Politician/Administrative-Unit-of-
the-State confusion (job21121556's largest remaining error bucket, 8/17)
-- that one traces to genuinely structurally ambiguous executive-member
cases (Froidevaux, the Chavannes-pres-Renens Syndic, the Finanzdelegation)
and at least one likely GOLD-labelling inconsistency (the Finanzdelegation
is labelled 'Administrative Unit of the State' in one row and 'Politician'
in another for what reads as the same kind of source), not a prompt-
wording gap -- left untouched per Maxime's own read of that analysis.

scripts/step4b_source_category_logit.py picks this module instead of
src/step4b_logit_prompts.py only when --backend litellm is used; every
other backend (vllm/transformers) is completely unaffected, byte-for-byte,
by anything in this file. Unlike step3/step4a, this step's entire
machinery (summary generation + 6-category logit probe + validate-and-
correct) lives in ONE prompt module in the original design, so the whole
file is duplicated here rather than splitting a "shared validation" file
out separately.
"""
from __future__ import annotations

import pandas as pd

from src.step4b_prompts import CATEGORY_PHRASE  # reuse the exact same phrasing

CATEGORIES = [
    "Administrative Unit of the State",
    "Politician",
    "Interest Group",
    "Civil Servant",
    "General Public",
    "Other",
]

CATEGORY_DEFINITION = {
    "Administrative Unit of the State": (
        "The source is a state administrative body -- a federal, cantonal, or communal "
        "department, office, agency, court, or regulatory/oversight body -- OR a named "
        "individual whose statement the article explicitly anchors to that body: the "
        "administrative unit's own name is given alongside (or instead of) the person's "
        "name as what they are speaking for -- e.g. 'the director of the cantonal "
        "building department said...', 'the federal office for X stated...'. Attaching "
        "the administrative unit's own name to the statement is what signals an "
        "official, representational capacity -- not the person's job title alone, and "
        "not simply the fact that they work for the state. If the same kind of state "
        "employee is speaking with no administrative unit named as backing their "
        "statement -- just their own personal account or opinion -- that is 'Civil "
        "Servant' instead, not this category."
    ),
    "Politician": (
        "The source is a Swiss political office-holder -- elected or appointed at the "
        "federal, cantonal, or communal level (a councillor, a member of parliament, a "
        "party leader, a cantonal or communal executive member) -- or a political party "
        "itself, speaking in a political capacity. A personal political opinion and an "
        "official party statement both count equally here -- there is no need to "
        "determine whether the politician is speaking personally or on behalf of their "
        "party; either way, this is Politician. Often signalled by a party name or "
        "abbreviation given in parentheses right after the person's name. A "
        "parliamentary group or faction (the party's own elected members sitting "
        "together in parliament) IS the party itself here, not a separate "
        "organization -- someone named by a role within it (e.g. 'party leader' or "
        "'faction president') is still Politician, even though a political party is, "
        "in the everyday sense, itself a group representing a particular interest: in "
        "THIS taxonomy a party is never 'Interest Group', which is reserved for "
        "non-political organizations instead. This also applies when the party is "
        "named by a short, topic-sounding nickname rather than its full name -- e.g. "
        "'the Greens'/'les Verts' (the Green party) still means the party itself, "
        "still Politician, not an ecological Interest Group (see that category's own "
        "definition for the full list of common Swiss party nicknames this covers). A "
        "FORMER politician commenting on "
        "public affairs still qualifies -- their political identity does not disappear "
        "on retirement. Restricted to SWISS political office-holders and parties: a "
        "foreign politician or party does not qualify here (see 'Other'). Excludes "
        "business executives or company representatives, even when commenting on "
        "government policy (see 'Interest Group')."
    ),
    "Interest Group": (
        "The source is an organized, non-political entity representing a particular, "
        "non-state interest -- a private company, a business or professional "
        "association, a union, an NGO, a lobby group, a think tank, a media outlet, a "
        "foundation, or a civic/residents' association (e.g. a neighbourhood "
        "association), or an organized advocacy/protest group, even an informal one, "
        "as long as it acts as a collective. IMPORTANT: a political party is, in the "
        "everyday sense of the term, also a group representing a particular interest "
        "-- but in this taxonomy it is NEVER Interest Group. Any Swiss political "
        "party, and anyone speaking as part of one (including a parliamentary group/"
        "faction, which is simply that party's own elected members, and anyone named "
        "by a role within it, e.g. 'party leader' or 'faction president'), is "
        "'Politician' instead, however partisan or interest-driven the statement is "
        "-- Interest Group is reserved for NON-political organizations: companies, "
        "business/trade bodies, unions, NGOs, associations, foundations, and the "
        "like. This includes a party commonly referred to by a short, topic-sounding "
        "nickname rather than its full name -- e.g. 'the Greens'/'les Verts' (the "
        "Green party, Switzerland's 'Parti ecologiste suisse'/'Gruene Partei der "
        "Schweiz'), 'the Socialists'/'les Socialistes' (the Social Democratic Party), "
        "or similarly for the UDC/SVP, the PLR/FDP, Le Centre, or the Vert'liberaux/"
        "pvl -- these nicknames still name the political party itself, not a loose "
        "topic-based interest group, so they are 'Politician' here, never Interest "
        "Group, even though a name like 'the Greens' sounds like it could describe an "
        "ecological advocacy group in the abstract. This category also makes NO "
        "distinction between the entity itself "
        "making a statement and an individual explicitly speaking for it: a company's "
        "owner or CEO personally criticizing a regulation that affects his business is "
        "still Interest Group, exactly as if the company itself had issued the "
        "statement, because he is speaking as the person in charge of that entity. "
        "There is no need to determine whether the entity itself or a specific "
        "individual representing it is speaking -- both count equally as Interest "
        "Group."
    ),
    "Civil Servant": (
        "The source is an individual currently employed by the state -- an official, "
        "auditor, diplomat, police officer, postal worker, etc. -- giving their own "
        "personal account or opinion, with NO administrative unit's name attached to "
        "the statement as its authority. The test against 'Administrative Unit of the "
        "State' is exactly that: when the statement is backed by the name of the "
        "office, department, or agency the person belongs to, it is Administrative "
        "Unit of the State instead; when the person is just speaking for themselves, "
        "with no institutional name invoked, it is Civil Servant. This also covers an "
        "unnamed or anonymous individual identified only by where they currently work "
        "(e.g. 'sources inside Fedpol say...', 'an employee who asked not to be "
        "named...'), as long as they are reporting their own first-hand experience or "
        "opinion rather than an official position attributed to the institution "
        "itself -- not being personally named does not change this: what matters is "
        "that they speak for themselves about their own experience, not on the "
        "institution's behalf. A retired or former state employee with no current "
        "state role does not qualify here (see 'General Public')."
    ),
    "General Public": (
        "The source is an ordinary individual speaking only for themselves -- "
        "regardless of their profession or job title -- with no institution, "
        "administration, party, or organized interest behind their statement: a "
        "resident, a survey respondent, an independent expert, researcher, or "
        "commentator with no organizational affiliation, or a formerly state-employed "
        "person (other than a former politician) with no current institutional role. "
        "Also covers an abstract or aggregate reference to ordinary people -- 'public "
        "opinion', 'many citizens', a group of residents -- with no named organization "
        "behind it."
    ),
    "Other": (
        "A prior analysis step already established that this criticism comes from "
        "someone other than the journalist -- so this category is never used just "
        "because the source seems hard to pin down. Use it ONLY when (a) the article "
        "genuinely does not name or identify who is voicing the criticism (e.g. 'X has "
        "faced mounting criticism in recent weeks' with nobody named), or (b) the "
        "source IS clearly named but is a foreign government or political actor "
        "outside Switzerland's own public institutions (a foreign head of state or "
        "minister) -- a case the other five categories, being Swiss-specific, cannot "
        "otherwise capture. If the article names ANY specific Swiss person, group, or "
        "outlet, one of the other five categories applies -- pick whichever fits best "
        "instead of defaulting here."
    ),
}

# Two short, self-contained illustrative scenarios per category -- these
# serve as this category's own "Yes" examples, and as every OTHER
# category's "No" example (reused verbatim: "here is a case that is
# clearly NOT this category, it's actually {other_category}"). Kept short
# on purpose (a sentence or two, not a full fake article) since they are
# few-shot illustrations of the underlying test, not training data.
CANONICAL_EXAMPLES = {
    "Administrative Unit of the State": [
        "A federal audit office publishes a report concluding that a government agency "
        "mismanaged public funds.",
        "A cantonal office is quoted together with its director's name -- e.g. 'the "
        "director of the cantonal building department said...' -- the office's own name "
        "being attached to the statement is what signals an official, representational "
        "capacity, not a personal one.",
    ],
    "Politician": [
        "A member of the national parliament says in a TV interview that, in her own "
        "opinion, the government's new policy is a mistake.",
        "A politician's name is followed by a party abbreviation in parentheses, or a "
        "political party issues its own statement criticizing a decision.",
    ],
    "Interest Group": [
        "A spokesperson for a national business association criticizes a proposed "
        "regulation on behalf of its member companies.",
        "A company's owner or CEO personally criticizes a regulation that affects his "
        "own business -- he is speaking in his own words, but because he is speaking as "
        "the person in charge of that company, this still counts as Interest Group, "
        "exactly as if the company itself had issued the statement.",
    ],
    "Civil Servant": [
        "A currently serving police officer, in a personal interview, says she is "
        "frustrated with outdated equipment -- speaking only for herself, with no job "
        "title or administrative unit attached to lend her statement any official "
        "weight.",
        "A postal worker describes poor working conditions in his own words, as an "
        "individual employee, not as someone making a statement on behalf of the postal "
        "service.",
    ],
    "General Public": [
        "A local resident tells a journalist that construction noise near her home has "
        "made daily life unbearable.",
        "The article says public opinion has turned against a policy, or that 'many "
        "citizens' are unhappy, without pointing to any specific organized group.",
    ],
    "Other": [
        "The article states that a company 'has faced mounting criticism in recent weeks' "
        "without naming who is criticizing it.",
        "A foreign head of state or minister is named as the one voicing the criticism -- "
        "a real, identified source, but outside the Swiss public-institution framework "
        "these categories are built for.",
    ],
}

# The two candidate letters read off the model's next-token logits. Must
# each be a SINGLE token for the target tokenizer in context -- verified
# at runtime by src/client.py's TransformersClient.logprob_many (and can
# be checked standalone with scripts/check_step4b_logit_tokens.py before
# committing any GPU time).
LETTERS = ("A", "B")

QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. You are given \
information about a newspaper article in which "{keyword}" is criticized, and asked \
whether the criticism comes from a specific type of source.

{context_label}:
{article_text}

CATEGORY TO EVALUATE: {category_phrase}
Definition: {category_definition}

Example (answer: Yes): {yes_example_1}
Example (answer: Yes): {yes_example_2}
Example (answer: No, this is {no_cat_1} instead): {no_example_1}
Example (answer: No, this is {no_cat_2} instead): {no_example_2}
Example (answer: No, this is {no_cat_3} instead): {no_example_3}
Example (answer: No, this is {no_cat_4} instead): {no_example_4}
Example (answer: No, this is {no_cat_5} instead): {no_example_5}

Question: based on the {context_label} above, is the criticism of "{keyword}" genuinely \
coming from {category_phrase}?
Answer with exactly one letter and nothing else:
{letter_yes} = Yes
{letter_no} = No\
"""

CALIBRATION_KEYWORD = "the subject"
CALIBRATION_ARTICLE = "N/A"
DEFAULT_CONTEXT_LABEL = "ARTICLE"


def _build_prompt(
    category: str, keyword: str, article_text: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """order=1 -> A=Yes/B=No ; order=2 -> A=No/B=Yes (see LETTERS).
    context_label names whatever `article_text` actually is -- "ARTICLE"
    for the original direct-article flow, "SUMMARY OF THE CRITICISM" for
    the 2026-10-26 summarize-then-probe flow (see build_summary_prompt)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_yes, letter_no = LETTERS if order == 1 else (LETTERS[1], LETTERS[0])

    other_cats = [c for c in CATEGORIES if c != category]
    no_examples = [CANONICAL_EXAMPLES[c][0] for c in other_cats]

    return QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase=CATEGORY_PHRASE[category],
        category_definition=CATEGORY_DEFINITION[category],
        yes_example_1=CANONICAL_EXAMPLES[category][0],
        yes_example_2=CANONICAL_EXAMPLES[category][1],
        no_cat_1=other_cats[0], no_example_1=no_examples[0],
        no_cat_2=other_cats[1], no_example_2=no_examples[1],
        no_cat_3=other_cats[2], no_example_3=no_examples[2],
        no_cat_4=other_cats[3], no_example_4=no_examples[3],
        no_cat_5=other_cats[4], no_example_5=no_examples[4],
        letter_yes=letter_yes, letter_no=letter_no,
    )


def build_question_prompt(
    row: pd.Series, text_col: str, category: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The real, article- (or summary-) grounded question for one row and
    one category, in one of the two letter orderings. `text_col` is
    whatever column holds the text to reason from -- the raw article for
    the original flow, or `source_summary` for the summarize-then-probe
    flow; pass the matching `context_label` so the prompt's own wording
    doesn't call a summary an "ARTICLE"."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_prompt(category, keyword, article_text, order, context_label=context_label)


def build_calibration_prompt(
    category: str, order: int, context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The SAME question with the article/summary and target neutralised
    -- used ONCE per category (not per row) to measure that category's
    own intrinsic yes-bias, independent of any article content. See
    src/step4b_logit_utils.calibrate. Pass the same `context_label` the
    real questions use, so the calibration prompt matches them exactly
    except for the row-specific content."""
    return _build_prompt(category, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order, context_label=context_label)


def letters_for_order(order: int) -> "tuple[str, str]":
    """Returns (letter_yes, letter_no) for the given order, mirroring
    _build_prompt's own logic -- used by the pipeline script to know
    which letter's probability is P(yes) for a given order."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    return LETTERS if order == 1 else (LETTERS[1], LETTERS[0])


# ---------------------------------------------------------------------------
# Pre-pass (2026-10-26 "summarize-then-probe" addition): a plain GENERATION
# step, unlike everything above. Reads the full article once and produces a
# short, factual summary of the criticism and its source for pass 2's 6
# binary probes to read instead of the raw article -- see the module
# docstring for why. Explicitly told NOT to classify the source itself, so
# it doesn't anchor pass 2 on a premature label the way the old pass 1's
# draft category did.
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the article \
below, "{keyword}" is criticized.

ARTICLE:
{article_text}

A prior analysis step already established that this criticism is attributed, in the \
article, to some other voice the article is reporting on -- not the newspaper's own \
reporting or commentary.

In one short, self-contained paragraph, explain in your own words what exactly \
"{keyword}" is being criticized for, and who is voicing that criticism and on what \
statement, quote, or source in the article this relies. Describe this as factually and \
specifically as you can, keeping names, titles, and organizations exactly as the article \
gives them.

Do not classify the source into any category yourself -- that is a separate step. If the \
article does not make clear who is speaking, say so rather than guessing. Keep the answer \
tight and self-contained, so it does not risk being cut off before it is finished.\
"""


def build_summary_prompt(row: pd.Series, text_col: str) -> str:
    """The pre-pass prompt: reads the FULL ARTICLE (text_col should be the
    raw article column here, never source_summary itself) and asks for a
    short factual summary -- see SUMMARY_TEMPLATE. The caller runs this
    through plain generation (src/client.py's chat_many, e.g. via
    src/runner.py's run_llm_dataframe), not logprob_many -- there is no
    fixed answer set to read logits for here."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return SUMMARY_TEMPLATE.format(keyword=keyword, article_text=article_text)


# ---------------------------------------------------------------------------
# Pass 3 (2026-10-27 "validate-and-correct" addition): a second
# LOGIT-PROBING step -- same machinery as pass 2 (one forward pass,
# order-swap averaging, context-free calibration), reading the SAME
# summary, but asking ONE holistic question instead of 6 independent
# ones: "pass 2 concluded the source is category Y -- true or false?".
# See the module docstring's 2026-10-27 paragraph for why this is not
# just pass 2 repeated, and src/step4b_source_category_logit.py's
# compute_validation_calibration / pass-3 loop for how it's used.
# ---------------------------------------------------------------------------

# Pass-3-ONLY reminder, keyed by claimed_category, empty ("") for any
# category not listed. Historically (see the module docstring's 2026-10-28b
# entry) this restated clauses pass 3 was shown to under-apply even though
# pass 2 read the identical CATEGORY_DEFINITION text. The 2026-11-26 rewrite
# folded all of those clauses (business executive speaking personally,
# informal collectives, media outlets) directly into CATEGORY_DEFINITION's
# "Interest Group" entry above, so there is nothing left to restate here --
# left empty on purpose rather than removed, so a future pass-3-only
# reminder has an obvious place to go.
VALIDATION_REMINDER = {}

VALIDATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the {context_label} \
below, "{keyword}" is criticized.

{context_label}:
{article_text}

A prior automated analysis concluded that the source of this criticism belongs to the \
following category:

CLAIMED CATEGORY: {category_phrase}
Definition: {category_definition}{validation_reminder}

Example (answer: True): {yes_example_1}
Example (answer: True): {yes_example_2}
Example (answer: False -- the source is actually {no_cat_1} instead): {no_example_1}
Example (answer: False -- the source is actually {no_cat_2} instead): {no_example_2}
Example (answer: False -- the source is actually {no_cat_3} instead): {no_example_3}
Example (answer: False -- the source is actually {no_cat_4} instead): {no_example_4}
Example (answer: False -- the source is actually {no_cat_5} instead): {no_example_5}

Question: based ONLY on the {context_label} above, is this conclusion correct -- does the \
criticism of "{keyword}" genuinely come from {category_phrase}?
Answer with exactly one letter and nothing else:
{letter_true} = True
{letter_false} = False\
"""


def _build_validation_prompt(
    claimed_category: str, keyword: str, article_text: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """order=1 -> A=True/B=False ; order=2 -> A=False/B=True. Reuses
    LETTERS/letters_for_order verbatim -- same letter-bias cancellation
    as pass 2, just relabelled True/False instead of Yes/No.

    2026-10-28: now also includes the SAME CANONICAL_EXAMPLES (2 "yes"
    illustrations of claimed_category + 5 contrastive "no, this is
    actually {other category}" illustrations) that pass 2's own
    _build_prompt has always used -- see the module docstring's
    2026-10-28 changelog entry for why: a row-by-row read of job66621913's
    pass-3 errors found that in most cases the summary text ALREADY
    contained everything needed to apply a boundary rule we'd already
    written (e.g. "Trump is a foreign head of state" -> Other; a named
    CEO speaking personally -> Interest Group per its business-executive
    clause), yet pass 3 -- asking the SAME information a second time,
    holistically -- applied these rules less reliably than pass 2 did on
    the very first read. The one structural difference between the two
    prompts was these examples: pass 2 had them, pass 3 didn't, despite
    both reading identical CATEGORY_DEFINITION text. This is a prompt
    fix, not an information-source fix -- still reads source_summary,
    not the raw article.

    2026-10-28b: also inserts VALIDATION_REMINDER[claimed_category] (empty
    string for any category not in that dict) right after the definition
    -- see the module docstring's 2026-10-28b entry for the specific,
    evidence-driven clauses this reinforces for 'Interest Group'."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_true, letter_false = letters_for_order(order)

    other_cats = [c for c in CATEGORIES if c != claimed_category]
    no_examples = [CANONICAL_EXAMPLES[c][0] for c in other_cats]

    return VALIDATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase=CATEGORY_PHRASE[claimed_category],
        category_definition=CATEGORY_DEFINITION[claimed_category],
        validation_reminder=VALIDATION_REMINDER.get(claimed_category, ""),
        yes_example_1=CANONICAL_EXAMPLES[claimed_category][0],
        yes_example_2=CANONICAL_EXAMPLES[claimed_category][1],
        no_cat_1=other_cats[0], no_example_1=no_examples[0],
        no_cat_2=other_cats[1], no_example_2=no_examples[1],
        no_cat_3=other_cats[2], no_example_3=no_examples[2],
        no_cat_4=other_cats[3], no_example_4=no_examples[3],
        no_cat_5=other_cats[4], no_example_5=no_examples[4],
        letter_true=letter_true, letter_false=letter_false,
    )


def build_validation_prompt(
    row: pd.Series, text_col: str, claimed_category: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The real, row-grounded validation question: does pass 2's own
    chosen category for this row hold up under one holistic re-read of
    the same summary? `claimed_category` is whatever pass 2 wrote to
    `source_category` for this row -- the caller reads it off the row
    and passes it in, this function doesn't look it up itself."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_validation_prompt(claimed_category, keyword, article_text, order, context_label=context_label)


def build_validation_calibration_prompt(
    claimed_category: str, order: int, context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """Same empty-summary neutralisation as build_calibration_prompt, for
    pass 3's own question shape -- measures, once per claimed category
    (not per row), how often THIS validation question's wording defaults
    to True independent of any real content."""
    return _build_validation_prompt(claimed_category, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order, context_label=context_label)


# ---------------------------------------------------------------------------
# Disambiguation pass (2026-11-26, Maxime's idea, --backend litellm only):
# for rows where pass 2's own winning margin is very small -- a genuine
# toss-up between its top two categories, not noise (see
# scripts/step4b_source_category_logit.py's margin-by-correctness
# diagnostics: correct rows had a 0.55 median margin in job21084261, wrong
# rows 0.03 -- a clean separation that makes "low margin" a reliable
# trigger) -- ask ONE direct, head-to-head question naming BOTH candidates
# explicitly, instead of trusting six independent absolute yes/no reads in
# isolation.
#
# Different in kind from pass 3 above (validate-and-correct, currently
# disabled for litellm via --skip_validation): pass 3 asks "is pass 2's
# own choice true or false?", one category at a time, against its own
# definition alone -- exactly the frame that let Froidevaux (DDPS/VBS),
# the Chavannes-pres-Renens Syndic, and the Finanzdelegation (job21084261)
# slip past pass 2 with a near-zero margin: each one plausibly satisfies
# BOTH Administrative Unit of the State's and Politician's definitions in
# isolation, so neither a single absolute yes/no read nor a single
# true/false re-read has anything to force a choice between the two. This
# question does: it names both candidates side by side and asks which one
# fits BETTER, a comparison those cases cannot sidestep.
#
# Same logit-probing machinery as pass 2/3 -- one forward pass, order-swap
# averaging (letters_for_order's existing (yes,no) pair reused verbatim:
# "cat1" plays "yes"'s role, "cat2" plays "no"'s role, so the exact same
# letter-swap bias cancellation applies unchanged) -- but calibrated PER
# (cat1, cat2) PAIR rather than per category, computed lazily: only for
# pairs that actually occur among the rows selected this run, once each
# (scripts/step4b_source_category_logit.py caches this per job). Each
# category's own FIRST canonical example only is shown (not both + 5
# contrastive ones like pass 2/3): this question already names and defines
# the two real candidates directly, so the long contrastive-examples block
# that helps pass 2 rule out five OTHER categories has no role to play
# here. An inconclusive read is fail-safe, same precedent as pass 3: the
# row's current choice is kept, never swapped on a signal that isn't
# trusted.
# ---------------------------------------------------------------------------

DISAMBIGUATION_QUESTION_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs. In the {context_label} \
below, "{keyword}" is criticized.

{context_label}:
{article_text}

A prior automated analysis found this to be a genuinely close call between two possible \
categories for the source of this criticism:

CATEGORY {letter_cat1}: {category_phrase_cat1}
Definition: {category_definition_cat1}
Example: {example_cat1}

CATEGORY {letter_cat2}: {category_phrase_cat2}
Definition: {category_definition_cat2}
Example: {example_cat2}

Question: based on the {context_label} above, which of these two categories better describes \
the actual source of the criticism of "{keyword}"?
Answer with exactly one letter and nothing else:
{letter_cat1} = {category_phrase_cat1}
{letter_cat2} = {category_phrase_cat2}\
"""


def _build_disambiguation_prompt(
    cat1: str, cat2: str, keyword: str, article_text: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """order=1 -> A=cat1/B=cat2 ; order=2 -> A=cat2/B=cat1 (letters_for_order
    reused verbatim -- see the module comment above for why this cancels
    position bias exactly like pass 2/3's own order-swap)."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    letter_cat1, letter_cat2 = letters_for_order(order)
    return DISAMBIGUATION_QUESTION_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        context_label=context_label,
        category_phrase_cat1=CATEGORY_PHRASE[cat1],
        category_definition_cat1=CATEGORY_DEFINITION[cat1],
        example_cat1=CANONICAL_EXAMPLES[cat1][0],
        category_phrase_cat2=CATEGORY_PHRASE[cat2],
        category_definition_cat2=CATEGORY_DEFINITION[cat2],
        example_cat2=CANONICAL_EXAMPLES[cat2][0],
        letter_cat1=letter_cat1, letter_cat2=letter_cat2,
    )


def build_disambiguation_prompt(
    row: pd.Series, text_col: str, cat1: str, cat2: str, order: int,
    context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """The real, row-grounded head-to-head question: the row's own pass-2
    winner (cat1) against its own runner-up (cat2), for a row whose margin
    was small enough to flag as a genuine toss-up. The caller reads both
    off the row's own pass-2 output and passes them in -- this function
    doesn't look them up itself."""
    alias = row.get("matched_alias", None)
    keyword = str(alias if pd.notna(alias) and str(alias).strip() else row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return _build_disambiguation_prompt(cat1, cat2, keyword, article_text, order, context_label=context_label)


def build_disambiguation_calibration_prompt(
    cat1: str, cat2: str, order: int, context_label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    """Same empty-summary neutralisation as build_calibration_prompt /
    build_validation_calibration_prompt, for this question's own shape --
    measures, once per (cat1, cat2) pair actually encountered (not per
    row), how often THIS pair's head-to-head wording defaults to cat1
    independent of any real content."""
    return _build_disambiguation_prompt(cat1, cat2, CALIBRATION_KEYWORD, CALIBRATION_ARTICLE, order, context_label=context_label)
