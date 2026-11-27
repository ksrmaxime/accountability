"""
scripts/step4b_source_category_logit.py

STEP 4b of the accountability pipeline — logit-probing redesign (2026-10-26+).

Replaces the generation-based 3-pass design (scripts/step4b_source_category.py,
kept as-is and untouched for later comparison) with a different
architecture proposed by the user, based on a method that worked well on
a separate task: instead of asking the model to GENERATE a single
6-way category label (risking it defaulting to one or two over-used
categories, and anchoring on whichever actor it names first when several
are present), ask 6 INDEPENDENT binary yes/no questions -- one per
category -- and read the answer directly off the model's next-token
logits, never by generating text.

For each row and each of the 6 categories, this produces a calibrated
probability that "the criticism of {keyword} comes from {category}". The
final answer is simply the category with the highest calibrated
probability.

Three ideas make the logit-probing itself work (full math in
src/step4b_logit_utils.py, full prompt design in src/step4b_logit_prompts.py):

  1. One forward pass, not generation -- src/client.py's
     TransformersClient.logprob_many reads the raw logits at the
     answer-letter position directly, no decode loop.
  2. Order-swap averaging -- every question is asked twice (A=Yes/B=No,
     then A=No/B=Yes) and the two P(yes) readings averaged, to cancel any
     intrinsic model preference for one letter over the other.
  3. Context-free calibration (Zhao et al. 2021) -- each category
     question's own intrinsic yes-bias is measured ONCE (not per row) on
     an "empty" variant of the same question, then removed from every
     row's reading in logit space before picking the argmax.

2026-10-26 "summarize-then-probe" addition: the first version of this
script (job66603355, direct-article logit-probing on the full 800-row
dataset) came back at 49.3% accuracy on the gold population -- flat
relative to the old 3-pass design's 52.9%, with a very low mean winning
margin (0.088) suggesting the 6 binary questions were reading too much
noise out of the full, often long and multi-topic article. This adds a
PRE-PASS: a plain GENERATION step (see src/step4b_logit_prompts.py's
SUMMARY_TEMPLATE/build_summary_prompt) that reads the full article once
and produces a short, factual summary of the criticism and its source --
explicitly told NOT to classify the source itself, to avoid anchoring
pass 2 on a premature label. Pass 2's 6 binary probes then read THIS
SUMMARY (column `source_summary`) instead of the raw article -- a
shorter, hopefully cleaner signal for the same calibrated-logit
machinery to work from. Everything about HOW pass 2 reads logits is
unchanged; only WHAT TEXT it reads changed.

This still eliminates the 3-pass design's draft/retry/force/verify/
reclassify machinery for the classification step itself: a forward pass
over valid candidate tokens is always well-defined. The new summary
pre-pass reintroduces one generation step (and therefore one new failure
mode: a row whose summary generation comes back empty) -- handled the
same way the rest of this pipeline always has: that row is simply never
finalized (source_category stays honestly NA), flagged via
source_category_incomplete.

2026-10-27 "validate-and-correct" addition: see src/step4b_logit_prompts.py's
module docstring for the full diagnosis (job66608678's 69-row population,
62.3% accuracy, +1 clean fix / -3 regressions vs the pre-boundary-sharpening
run). Adds PASS 3: for every row pass 2 resolved, re-reads the SAME summary
with ONE holistic "pass 2 concluded category Y -- true or false?" logit
question (same machinery as pass 2: one forward pass, order-swap, per-
category calibration -- see compute_validation_calibration below). Where
pass 3 says false, the row falls back to pass 2's own RUNNER-UP category
(2nd-highest calibrated probability, captured during pass 2's own
aggregation loop) rather than a 3rd fresh guess. An inconclusive pass-3
read (GPU OOM etc.) is fail-safe: pass 2's own choice is kept, not swapped
on a signal that isn't trusted.

2026-11-25 "p0 floor" fix -- --backend litellm ONLY. Diagnosed from comparing
two real job logs: job20620441 (local 8B, --backend transformers) printed
p0 well spread across categories (Administrative Unit 0.2344, Politician
0.4537, Interest Group 0.3689, Civil Servant 0.3641, General Public 0.2227,
Other 0.7877), while job21022122 (--backend litellm, Apertus-v1.5-70B)
printed p0 = 0.0000 for ALL SIX categories on the exact same calibration
design (the "N/A" empty-summary variant). The 70B is simply far more
confidently negative than the 8B on content-free input. The consequence:
src/step4b_logit_utils.calibrate(p, p0) = sigmoid(logit(p) - logit(p0)), so
a p0 this close to 0 makes logit(p0) an extreme negative number and the
correction pushes almost ANY non-negligible raw p (0.1 or 0.9, genuinely
different signals) up to ~1.0 alike -- destroying the probe's ability to
tell a moderate reading from a strong one. This is the direct, confirmed
cause of the Administrative Unit / Politician (and Interest Group / General
Public) ties-at-ceiling found in job21022503's confusion-matrix analysis.
Fix: _floor_p0_litellm() clips every p0[cat] (and p0_val[cat] for pass 3)
into [0.05, 0.95] right after compute_calibration()/compute_validation_
calibration() return, before calibrate() ever sees them -- gated strictly
behind args.backend == "litellm" below. src/step4b_logit_utils.py's
calibrate() function itself is NOT modified (it is shared, backend-
agnostic, used by every step); the production 8B/local-GPU path calls it
exactly as before, with unfloored p0, byte-for-byte unaffected. The 0.05
floor sits comfortably below the 8B's own historical p0 range (0.22-0.79),
so even if this floor were ever accidentally exercised on an 8B-like p0 it
would not change anything -- it only clamps the pathological near-0 (or
near-1) case the 70B produces. This keeps the exact same logit-probing
architecture and the exact same calibration formula requested by the user
-- nothing about the method changes, only a numerical-stability clip on one
of its inputs.

2026-11-26 checkpoint-resume fix -- --backend litellm ONLY. job21084261's
source_category_p_civil_servant came back null for ALL 97 SOURCED rows
(the other five categories each had at most a handful of scattered nulls,
0-5/97 -- ordinary occasional per-row logit-read misses). An all-rows
failure, identical regardless of row content, pointed at the ONE-TIME
compute_calibration() call for that category rather than anything per-row
(p0[cat] is NaN -> every row's corrected[cat] is forced to NaN a few lines
below, see the aggregation loop). The immediate cause was most likely
src/step4b_logit_prompts_remote.py's Civil Servant definition containing
an embedded question mark right before the template's own real "Question:"
section (now rewritten as a declarative sentence -- see that file's own
2026-11-26c changelog entry), but investigating this surfaced a separate,
real robustness gap worth closing regardless of whether that was the exact
trigger: the per-row logit checkpoint (logit_checkpoint_path) and the
pass-3 validation checkpoint (validation_checkpoint_path) only persist
across runs when a run is killed or crashes before reaching the cleanup
at the end of main() (SLURM time limit, a dropped network connection,
etc.) -- on a clean run they are always deleted (see the end of main()).
When one DOES persist and a later run resumes from it, the resume loop
only checked "is this key present in the checkpoint", never "is the value
in the checkpoint actually usable" -- a (row, category, order) entry whose
every candidate-letter logit came back NaN (a transient read failure that
exhausted its retries, see src/client_remote.py's _post_with_retry) would
be loaded into `done`/`val_done` exactly like a real, successful read, and
therefore never retried, however many times the script is resumed
afterwards. Fixed by discarding (not loading) any checkpoint row whose
logits are NaN for every candidate letter, so it falls back into
`todo_indices`/`val_todo` and gets a fresh attempt -- see the two
`[resume]` blocks below. Gated strictly behind args.backend == "litellm":
the local 8B/GPU path's resume behaviour is completely unaffected, byte-
for-byte, exactly like every other litellm-only fix in this file.

2026-11-26 disambiguation pass addition -- --backend litellm ONLY, Maxime's
idea. job21084261's confusion matrix (post the "clean rewrite" prompt
redesign, see src/step4b_logit_prompts_remote.py's 2026-11-26 changelog
entry) still showed several near-zero-margin errors between Administrative
Unit of the State and Politician (Froidevaux/DDPS-VBS, the Chavannes-pres-
Renens Syndic, the Finanzdelegation) where BOTH categories' absolute
definitions plausibly apply to the same elected/appointed executive member
-- a case pass 2's six INDEPENDENT yes/no probes, and pass 3's single
true/false re-read, have no way to force a choice on, because neither ever
asks "which of these two is the better fit" directly. Adds a new pass,
gated by --disambiguation_margin_threshold (0.0 default = disabled; margins
are never negative, so no row ever qualifies at the default): for every row
whose pass-2 (or pass-3-adjusted) margin falls below this threshold, ask
ONE direct head-to-head question naming pass 2's own winner and runner-up
explicitly side by side -- see
src/step4b_logit_prompts_remote.py's DISAMBIGUATION_QUESTION_TEMPLATE for
the full rationale and why it is a different question in kind from pass 3,
not a repeat of it. Same logit-probing machinery as pass 2/3 (one forward
pass, order-swap averaging) but calibrated per (cat1, cat2) PAIR, computed
lazily -- once per unique pair actually encountered among the selected
rows, not all 15 possible pairs up front. An inconclusive read is
fail-safe, same precedent as pass 3: the row's current choice is kept,
never swapped on an untrusted signal. No local-GPU equivalent exists (the
production src/step4b_logit_prompts.py is never touched), so main() raises
if this is requested with any backend other than litellm.

2026-11-27 "one-sided top-20" NaN-tolerant averaging fix -- --backend litellm
ONLY. job21084261 and job21085911 BOTH showed source_category_p_civil_servant
null for ALL 97 SOURCED rows, with job21085911 confirmed to still fail this
way AFTER the 2026-11-26c question-mark fix to that category's definition --
disproving that theory. scripts/diag_step4b_calibration.py (a new standalone,
hand-run diagnostic, see that file) was used to inspect the model's RAW
response to each category's real calibration prompt via
RemoteAPIClient.raw_request(), bypassing logprob_many's silent
letter-missing->NaN conversion. Finding: the model answers cleanly with a
single confident letter token every time (no refusal, no reasoning
preamble, finish_reason="length" as expected for max_tokens=1) -- so there
is nothing wrong with any prompt's wording. The real cause is structural:
on the content-free "N/A" calibration input, the model is sometimes SO
one-sided (e.g. logprob(B) = -0.001 with 'A' not even appearing in the
returned top_logprobs(20), the OpenAI-spec-capped maximum -- see
src/client_remote.py's own module docstring, point 2) that ONE of the two
letter orderings has its "yes" letter fall out of the returned top-20
window entirely. This is not unique to Civil Servant -- the live diagnostic
run on 2026-11-27 reproduced it for Interest Group and General Public too,
at that moment -- but Civil Servant's definition (shortest of the 6, built
entirely around an exclusion -- "NO administrative unit's name attached")
seems to trigger the fully-confident-negative response on an empty article
unusually reliably across repeated runs. Either way, the general bug is in
how a single missing-letter reading is handled upstream of the model,
not in any one category's wording: src/step4b_logit_utils.
average_yes_over_orderings(p1, p2) = 0.5*(p1+p2) is a plain mean, so ONE
NaN input (letter missing from top-20 -- itself an entirely expected,
correctly-reported "this probability is near zero" signal, see
src/client_remote.py's _extract_candidate_logprobs docstring) poisons the
WHOLE average, even though the OTHER ordering's reading is perfectly
valid and usable on its own. Downstream, this one-time, per-category p0
used by EVERY row (compute_calibration/compute_validation_calibration) or
per-pair p0_pairs used by every disambiguated row sharing that pair
(compute_calibration's sibling in the disambiguation block below)
amplifies a single unlucky calibration call into every row for that
category/pair losing the ability to ever be assigned that category, for
the whole run -- exactly job21084261/job21085911's symptom. The SAME
contagious-NaN issue, at smaller scale, also affects any individual row's
own per-category/per-claim/per-pair reading (compute'd inline in the pass
2 / pass 3 / disambiguation aggregation loops), contributing to the
handful of scattered per-row NaNs already seen on other categories.
Fix: _average_yes_over_orderings_litellm_tolerant() below, used in place
of the plain src/step4b_logit_utils.average_yes_over_orderings EVERYWHERE
in this file, gated behind args.backend == "litellm" via the same
active_build_*-style indirection already used for the prompt builders
(see active_average_yes_over_orderings below) -- a general fix at the
averaging step itself, not a per-category wording patch, so it applies
equally to whichever category or pair happens to hit this on a given run.
When both orderings are valid, behaviour is byte-identical to the shared
function (same formula); when exactly one is NaN, uses the other,
perfectly valid reading alone instead of discarding the whole measurement;
only when BOTH orderings are NaN (a genuine, total failure) does the
result stay NaN, same as before. src/step4b_logit_utils.py itself is NOT
modified (shared, backend-agnostic, used by every step); the production
8B/local-GPU path keeps calling the plain function exactly as before,
byte-for-byte unaffected.

Output columns:
  source_summary                     -> the pre-pass's factual summary of
                                          the criticism and its source (free
                                          text) | NA if generation failed
  source_category                    -> one of 6 categories | NA (FINAL,
                                          possibly pass 2's runner-up if
                                          pass 3 rejected pass 2's choice)
  source_category_before_validation  -> pass 2's own choice, BEFORE pass 3
                                          -- kept for audit even when pass 3
                                          accepts it unchanged
  source_category_runnerup           -> pass 2's 2nd-highest-probability
                                          category (the pass-3 fallback
                                          target)
  source_category_validation_p_true  -> pass 3's calibrated P(true) that
                                          pass 2's own choice is correct
  source_category_validated          -> "TRUE" | "FALSE" | NA (pass 3's
                                          verdict on pass 2's choice; NA if
                                          pass 3's read was inconclusive)
  source_category_p_<category>       -> calibrated probability for each of
                                          the 6 categories from PASS 2 (QA /
                                          diagnostics; not updated by pass 3)
  source_category_margin             -> pass 2's winning category minus its
                                          runner-up (a confidence signal
                                          this design gives "for free")
  source_category_incomplete         -> "TRUE" if the summary pre-pass, ANY
                                          pass-2 logit read, the pass-3
                                          validation read, OR the
                                          disambiguation read failed, else NA
  source_category_disambiguation_applied   -> "TRUE" if this row's pass-2
                                          margin was below
                                          --disambiguation_margin_threshold
                                          and it went through the head-to-
                                          head disambiguation pass, else NA
                                          (--backend litellm only; see that
                                          flag's own help text)
  source_category_before_disambiguation    -> the category this row held
                                          right before the disambiguation
                                          pass ran (pass 2's own choice, or
                                          pass 3's if that swapped it) --
                                          only set when applied is "TRUE"
  source_category_disambiguation_p_original -> the disambiguation pass's
                                          calibrated P(the ORIGINAL -- pre-
                                          disambiguation -- choice is the
                                          better fit); NA if inconclusive
                                          (kept, not swapped) or not applied

Usage
-----
  python scripts/step4b_source_category_logit.py \\
      --input       data/output/step4a_merged/results.parquet \\
      --output_base data/output/step4b_logit \\
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \\
      --dtype bf16 \\
      --summary_batch_size 8 --summary_max_new_tokens 220 \\
      --batch_size 32 --max_input_tokens 16384 \\
      --validation_batch_size 32 --validation_checkpoint_every 200
"""
from __future__ import annotations

import math
import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from src.client import build_client
from src.runner import RunConfig, run_llm_dataframe
from src.step4b_config import build_mask
from src.step4b_logit_prompts import (
    CATEGORIES,
    build_question_prompt,
    build_calibration_prompt,
    build_summary_prompt,
    build_validation_prompt,
    build_validation_calibration_prompt,
    letters_for_order,
)
# 2026-11-24: --backend litellm (remote API models via the UNIL/CSCS proxy,
# e.g. the Apertus-70B family) uses its OWN prompt module
# (src/step4b_logit_prompts_remote.py), selected right after argparse below
# -- same pattern as scripts/step3_criticism_detection.py, see
# src/step3_prompts_remote.py's module docstring for the full rationale and
# src/step4b_logit_prompts_remote.py's own docstring for this file's version
# of it. src/step4b_logit_prompts.py (the production vllm/transformers, 8B,
# local-GPU path) is completely unaffected, byte-for-byte, by anything
# imported here. CATEGORIES and letters_for_order are NOT re-imported from
# the remote module: CATEGORIES is the fixed set of 6 category labels (not
# prompt wording -- identical in both modules, and PROB_COLS below is built
# from it at import time, before --backend is even known), and
# letters_for_order is a trivial order/letter-index helper, not prompt text.
# Only the actual wording-bearing prompt builders are backend-conditional.
from src.step4b_logit_prompts_remote import (
    build_question_prompt as build_question_prompt_remote,
    build_calibration_prompt as build_calibration_prompt_remote,
    build_summary_prompt as build_summary_prompt_remote,
    build_validation_prompt as build_validation_prompt_remote,
    build_validation_calibration_prompt as build_validation_calibration_prompt_remote,
    # 2026-11-26: disambiguation pass, --backend litellm ONLY -- no local-GPU
    # equivalent exists (src/step4b_logit_prompts.py is never touched), so
    # these are referenced directly inside the backend=="litellm" branch
    # below, with no "active_*" indirection and no fallback to a production
    # builder, unlike the five imports above.
    build_disambiguation_prompt as build_disambiguation_prompt_remote,
    build_disambiguation_calibration_prompt as build_disambiguation_calibration_prompt_remote,
)
from src.step4b_logit_utils import (
    softmax_restricted,
    average_yes_over_orderings,
    calibrate,
    pick_argmax_category,
)

CANDIDATE_LETTERS_DEFAULT = ["A", "B"]

# Pass 2 reads the pre-pass's summary, never the raw article -- see the
# module docstring's "summarize-then-probe" section. This is the only
# prompt-wording change needed on top of the original direct-article
# design (src/step4b_logit_prompts._build_prompt's context_label param).
SUMMARY_CONTEXT_LABEL = "SUMMARY OF THE CRITICISM"
SUMMARY_TEXT_COL = "source_summary"


def _slug(category: str) -> str:
    return category.lower().replace(" ", "_")


# 2026-11-25 "p0 floor" fix -- --backend litellm ONLY, see this module's
# docstring changelog entry above for the full diagnosis and justification.
_LITELLM_P0_FLOOR = 0.05


def _floor_p0_litellm(p0: dict) -> dict:
    """--backend litellm ONLY. Clips every p0[cat] into
    [_LITELLM_P0_FLOOR, 1 - _LITELLM_P0_FLOOR] before it is ever handed to
    src/step4b_logit_utils.calibrate(), so a near-0 (or near-1) empty-prompt
    reading from a very confident model can no longer blow up calibrate()'s
    logit-space correction term. calibrate() itself is NOT touched -- this
    only pre-clips one of its two inputs, and only on this call path. A
    genuine NaN (failed calibration read) is passed through unchanged, not
    floored, so it is still correctly treated as missing downstream."""
    floored = {}
    for cat, v in p0.items():
        floored[cat] = v if pd.isna(v) else min(max(v, _LITELLM_P0_FLOOR), 1.0 - _LITELLM_P0_FLOOR)
    return floored


# 2026-11-27 "one-sided top-20" NaN-tolerant averaging fix -- --backend
# litellm ONLY, see this module's docstring changelog entry above for the
# full diagnosis (scripts/diag_step4b_calibration.py's findings) and
# justification.
def _average_yes_over_orderings_litellm_tolerant(p1: float, p2: float) -> float:
    """Same intent as src/step4b_logit_utils.average_yes_over_orderings
    (cancel a model's intrinsic A/B letter preference by averaging both
    letter-to-meaning orderings), but tolerant of a single missing
    reading: when the remote server's top_logprobs(20) cap means one
    ordering's "yes" letter fell out of the returned top-20 (NaN) because
    the model was ALREADY extremely one-sided on that ordering, use the
    other, valid ordering's reading alone rather than discarding the
    measurement entirely via contagious-NaN propagation. If BOTH
    orderings are NaN (a genuine, total calibration failure), stays NaN,
    unchanged -- that case is still correctly treated as missing
    downstream (see compute_calibration/compute_validation_calibration's
    own docstrings and the pass 2/3/disambiguation aggregation loops).
    When both inputs ARE valid, this is byte-identical to the shared
    average_yes_over_orderings (same 0.5*(p1+p2) formula) -- nothing about
    the calibration math changes, only how a single missing reading is
    handled."""
    if pd.notna(p1) and pd.notna(p2):
        return average_yes_over_orderings(p1, p2)
    if pd.notna(p1):
        return p1
    if pd.notna(p2):
        return p2
    return float("nan")


PROB_COLS = [f"source_category_p_{_slug(c)}" for c in CATEGORIES]
ALL_OUTPUT_COLS = (
    [
        SUMMARY_TEXT_COL,
        "source_category",
        "source_category_margin",
        "source_category_runnerup",
        "source_category_before_validation",
        "source_category_validation_p_true",
        "source_category_validated",
        "source_category_incomplete",
        # 2026-11-26 disambiguation pass (--backend litellm only, opt-in via
        # --disambiguation_margin_threshold) -- see the module docstring.
        "source_category_disambiguation_applied",
        "source_category_before_disambiguation",
        "source_category_disambiguation_p_original",
    ]
    + PROB_COLS
)


def parse_summary_output(raw: str) -> dict:
    """The pre-pass is plain generation (free text), not logit-probing --
    whatever comes back IS the summary, no label to extract. An empty
    response is treated as a failure (NA), same "never finalize on
    nothing" policy as the rest of this pipeline."""
    text = (raw or "").strip()
    return {SUMMARY_TEXT_COL: text if text else pd.NA}


def compute_calibration(client, candidate_letters, max_input_tokens, context_label: str,
                         build_calibration_prompt_fn=build_calibration_prompt,
                         average_fn=average_yes_over_orderings) -> dict:
    """p0(category): this category's question's own intrinsic yes-bias,
    measured ONCE (order-averaged) on the 'empty' variant of the question
    -- see src/step4b_logit_prompts.build_calibration_prompt. Independent
    of every row, so this never needs to be re-run per row, and is cheap
    enough (6 categories x 2 orders = 12 forward passes) to simply
    recompute at the start of every SLURM array task rather than share
    across tasks.

    2026-11-24: build_calibration_prompt_fn defaults to the production
    (vllm/transformers) prompt builder, byte-for-byte unaffected; main()
    passes the remote one explicitly when --backend litellm is used -- see
    the src/step4b_logit_prompts_remote import above.

    2026-11-27: average_fn defaults to the shared, backend-agnostic
    average_yes_over_orderings, byte-for-byte unaffected; main() passes
    _average_yes_over_orderings_litellm_tolerant explicitly when --backend
    litellm is used -- see this module's own 2026-11-27 changelog entry."""
    prompts = []
    meta = []  # (category, order)
    for cat in CATEGORIES:
        for order in (1, 2):
            prompts.append(build_calibration_prompt_fn(cat, order, context_label=context_label))
            meta.append((cat, order))

    raw = client.logprob_many(
        system_prompt="",
        user_prompts=prompts,
        candidate_letters=candidate_letters,
        max_input_tokens=max_input_tokens,
    )

    p_yes_by_cat_order = {}
    for (cat, order), logits_dict in zip(meta, raw):
        probs = softmax_restricted(logits_dict)
        letter_yes, _letter_no = letters_for_order(order)
        p_yes_by_cat_order[(cat, order)] = probs.get(letter_yes, float("nan"))

    p0 = {}
    for cat in CATEGORIES:
        p0[cat] = average_fn(
            p_yes_by_cat_order[(cat, 1)], p_yes_by_cat_order[(cat, 2)]
        )
    return p0


def compute_validation_calibration(client, candidate_letters, max_input_tokens, context_label: str,
                                    build_validation_calibration_prompt_fn=build_validation_calibration_prompt,
                                    average_fn=average_yes_over_orderings) -> dict:
    """p0_val(claimed_category): pass 3's own question has a different
    shape than pass 2's (one holistic true/false claim instead of 6
    independent yes/no ones), so it gets its own intrinsic yes("true")-
    bias measurement, one per possible claimed category, on the same
    empty-summary variant -- see
    src/step4b_logit_prompts.build_validation_calibration_prompt. Same
    cost as compute_calibration (6 categories x 2 orders = 12 forward
    passes).

    2026-11-24: build_validation_calibration_prompt_fn defaults to the
    production (vllm/transformers) prompt builder, byte-for-byte
    unaffected; main() passes the remote one explicitly when --backend
    litellm is used -- see the src/step4b_logit_prompts_remote import
    above.

    2026-11-27: average_fn defaults to the shared, backend-agnostic
    average_yes_over_orderings, byte-for-byte unaffected; main() passes
    _average_yes_over_orderings_litellm_tolerant explicitly when --backend
    litellm is used -- see this module's own 2026-11-27 changelog entry."""
    prompts = []
    meta = []  # (claimed_category, order)
    for cat in CATEGORIES:
        for order in (1, 2):
            prompts.append(build_validation_calibration_prompt_fn(cat, order, context_label=context_label))
            meta.append((cat, order))

    raw = client.logprob_many(
        system_prompt="",
        user_prompts=prompts,
        candidate_letters=candidate_letters,
        max_input_tokens=max_input_tokens,
    )

    p_true_by_cat_order = {}
    for (cat, order), logits_dict in zip(meta, raw):
        probs = softmax_restricted(logits_dict)
        letter_true, _letter_false = letters_for_order(order)
        p_true_by_cat_order[(cat, order)] = probs.get(letter_true, float("nan"))

    p0_val = {}
    for cat in CATEGORIES:
        p0_val[cat] = average_fn(
            p_true_by_cat_order[(cat, 1)], p_true_by_cat_order[(cat, 2)]
        )
    return p0_val


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True, help="Step 4a merged output: parquet or csv")
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="text", help="Raw article column (read by the summary pre-pass only)")
    ap.add_argument("--n_rows",       type=int, default=0)

    # --- Model ---
    # 2026-11-24: --backend "litellm" added alongside "vllm"/"transformers" --
    # sends requests to an OpenAI-compatible gateway (src/client_remote.py)
    # instead of loading a model locally, e.g. swiss-ai/Apertus-v1.5-70B
    # hosted at CSCS via UNIL's litellm proxy on a plain CPU allocation. For
    # this backend, --model_path doubles as the remote model identifier
    # rather than a local filesystem path, and --dtype/--trust_remote_code
    # are simply ignored. See src/client.py's build_client() and
    # src/client_remote.py's module docstring. Same change as
    # scripts/step3_criticism_detection.py. Both the summary pre-pass
    # (generation) and passes 2/3 (logit-probing, via client.logprob_many)
    # work unchanged through this client for either backend.
    ap.add_argument("--model_path",        required=True,
                    help="Local model path (vllm/transformers backends) or remote model "
                         "identifier, e.g. 'swiss-ai/Apertus-v1.5-70B' (litellm backend).")
    ap.add_argument("--dtype",             default="bf16", choices=["bf16", "fp16", "auto"],
                    help="Ignored for --backend litellm.")
    ap.add_argument("--backend",           default="transformers",
                    choices=["vllm", "transformers", "litellm"])
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Ignored for --backend litellm.")

    # --- Remote API (--backend litellm only) ---
    ap.add_argument("--api_base",    default="https://litellm.dcsr.unil.ch/v1",
                    help="Base URL for the OpenAI-compatible gateway (--backend litellm only).")
    ap.add_argument("--api_key_env", default="UNIL_LITELLM_API_KEY",
                    help="Name of the env var holding the API key (--backend litellm only). "
                         "Never hardcode the key itself -- export it before launching.")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Concurrent HTTP requests (--backend litellm only) -- replaces "
                         "--batch_size/--summary_batch_size's role as the parallelism knob for "
                         "this backend, since there is no local GPU batch to size.")

    # --- Pre-pass: summarize (plain generation, reads the full article) ---
    ap.add_argument("--summary_batch_size",      type=int,   default=8)
    ap.add_argument("--summary_temperature",     type=float, default=0.0)
    ap.add_argument("--summary_max_new_tokens",  type=int,   default=220)
    ap.add_argument("--summary_checkpoint_every", type=int,  default=10,
                    help="Save a checkpoint every N summary batches.")

    # --- Pass 2 (logit-probing, no generation): no temperature /
    # max_new_tokens here -- a single forward pass reads next-token
    # logits, nothing is sampled and no output length to bound.
    # batch_size is a count of individual (row, category, order) PROMPTS
    # per forward-pass call, not of rows -- each row with a usable
    # summary produces 6 categories x 2 orders = 12 prompts. ---
    ap.add_argument("--batch_size",        type=int,   default=32)
    ap.add_argument("--max_input_tokens",  type=int,   default=16384)
    ap.add_argument("--candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Yes/No. Verify these "
                         "with scripts/check_step4b_logit_tokens.py before relying on them.")
    ap.add_argument("--checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N flat (row,category,order) prompts resolved.")

    # --- Pass 3 (validate-and-correct, logit-probing, no generation):
    # one holistic true/false re-read per row pass 2 resolved, reusing
    # --candidate_letters and --max_input_tokens from pass 2 above.
    # batch_size here counts (row, order) prompts -- 2 per validated row. ---
    ap.add_argument("--validation_batch_size", type=int, default=32)
    ap.add_argument("--validation_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) validation prompts resolved.")
    ap.add_argument("--skip_validation", action="store_true",
                    help="2026-10-29: disable pass 3 entirely -- no validation forward passes are run "
                    "(saves the GPU time too, not just the output), and source_category stays exactly "
                    "pass 2's own choice for every row. Added after job66672510's corrected-gold "
                    "evaluation showed pass 3 net-negative (-1) with no category it reliably improved. "
                    "Kept as a flag rather than deleting pass 3's code, in case a future fix (summary "
                    "quality, full-article re-read, etc.) makes it worth re-enabling.")

    # --- Disambiguation pass (2026-11-26, --backend litellm ONLY, opt-in):
    # one direct head-to-head question between pass 2's own top-2 categories,
    # for rows whose pass-2 margin is below the threshold below. See the
    # module docstring's 2026-11-26 "disambiguation pass addition" entry and
    # src/step4b_logit_prompts_remote.py's DISAMBIGUATION_QUESTION_TEMPLATE
    # module comment for the full rationale. batch_size/checkpoint_every
    # mirror pass 3's own (count (row, order) prompts, 2 per disambiguated
    # row). ---
    ap.add_argument("--disambiguation_margin_threshold", type=float, default=0.0,
                    help="Run the head-to-head disambiguation pass on every row whose pass-2 "
                         "winning margin is below this value -- a genuine toss-up between its "
                         "top two categories, not noise. 0.0 (the default) disables this pass "
                         "entirely: margins are never negative, so no row ever qualifies. Pass "
                         "a positive value (e.g. 0.15) to opt in. --backend litellm ONLY -- "
                         "raises if used with --backend vllm/transformers, since there is no "
                         "local-GPU prompt builder for this (src/step4b_logit_prompts.py is "
                         "never touched).")
    ap.add_argument("--disambiguation_batch_size", type=int, default=32)
    ap.add_argument("--disambiguation_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) disambiguation prompts resolved.")

    # --- Internal ---
    ap.add_argument("--job_id",    default=None)
    ap.add_argument("--task_id",   type=int, default=None,
                    help="Array task index (0-indexed). If omitted, read from SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Total number of array tasks. If omitted, read from SLURM_ARRAY_TASK_COUNT.")

    args = ap.parse_args()
    candidate_letters = [s.strip() for s in args.candidate_letters.split(",") if s.strip()]
    if len(candidate_letters) != 2:
        raise ValueError(f"--candidate_letters must give exactly 2 letters, got {candidate_letters!r}")
    if args.disambiguation_margin_threshold > 0 and args.backend != "litellm":
        raise ValueError(
            "--disambiguation_margin_threshold > 0 is only supported with --backend litellm "
            "-- no local-GPU prompt builder exists for this pass, and src/step4b_logit_prompts.py "
            "(the production 8B/local-GPU path) is never touched."
        )

    # 2026-11-24: pick the prompt module for the summary pre-pass, pass 2's
    # question prompts, and pass 3's validation prompts based on --backend --
    # see the src/step4b_logit_prompts_remote import above. vllm/transformers
    # (the production 8B/local-GPU path) keep using src/step4b_logit_prompts.py
    # exactly as before, byte-for-byte unaffected. CATEGORIES and
    # letters_for_order are shared/unconditional -- see that import's own
    # comment. Same pattern as scripts/step3_criticism_detection.py.
    if args.backend == "litellm":
        active_build_summary_prompt = build_summary_prompt_remote
        active_build_question_prompt = build_question_prompt_remote
        active_build_calibration_prompt = build_calibration_prompt_remote
        active_build_validation_prompt = build_validation_prompt_remote
        active_build_validation_calibration_prompt = build_validation_calibration_prompt_remote
        # 2026-11-27 "one-sided top-20" NaN-tolerant averaging fix -- see
        # this module's own 2026-11-27 changelog entry above.
        active_average_yes_over_orderings = _average_yes_over_orderings_litellm_tolerant
    else:
        active_build_summary_prompt = build_summary_prompt
        active_build_question_prompt = build_question_prompt
        active_build_calibration_prompt = build_calibration_prompt
        active_build_validation_prompt = build_validation_prompt
        active_build_validation_calibration_prompt = build_validation_calibration_prompt
        active_average_yes_over_orderings = average_yes_over_orderings

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    # --- Load step 4a output (ALL rows) ---
    df = (pd.read_parquet(args.input)
          if args.input.endswith(".parquet")
          else pd.read_csv(args.input, low_memory=False))

    if args.n_rows > 0:
        df = df.head(args.n_rows).copy()
        print(f"[n_rows] Subsetting to first {args.n_rows} rows")

    if task_id is not None and num_tasks is not None:
        chunk_size = math.ceil(len(df) / num_tasks)
        start = task_id * chunk_size
        end   = min(start + chunk_size, len(df))
        print(
            f"[pipeline] Array task {task_id}/{num_tasks} — "
            f"rows {start}:{end} ({end - start} rows)",
            flush=True,
        )
        df = df.iloc[start:end].copy()

    df = df.reset_index(drop=True)
    eligible_mask = build_mask(df, text_col=args.text_col)
    eligible_idx_all = df.index[eligible_mask].tolist()
    print(f"[pipeline] {len(df):,} rows in chunk — {len(eligible_idx_all):,} SOURCED rows "
          f"eligible for the summary pre-pass (rest keep source_category = NA)")

    summary_checkpoint_path        = args.output_base + (f"_task{task_id}" if task_id is not None else "") + "_summary_checkpoint.parquet"
    logit_checkpoint_path          = args.output_base + (f"_task{task_id}" if task_id is not None else "") + "_checkpoint.parquet"
    validation_checkpoint_path     = args.output_base + (f"_task{task_id}" if task_id is not None else "") + "_validation_checkpoint.parquet"
    disambiguation_checkpoint_path = args.output_base + (f"_task{task_id}" if task_id is not None else "") + "_disambiguation_checkpoint.parquet"

    for col in ALL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="object")

    # --- LLM client ---
    client = build_client(args)

    # =========================================================================
    # 2026-10-28 -- diagnostic logging, pas une etape du pipeline.
    #
    # Contexte : job66621913 (2026-10-27) et job66666813 (2026-10-28), meme
    # code, meme GPU physique (UUID identique confirme), meme input, et un
    # calcul de calibration (p0) dont la forme de batch est fixe (12 prompts,
    # contenu invariant "N/A"/"the subject") -- ont quand meme produit des p0
    # differents. Les deux pistes testees et ELIMINEES avec preuve a l'appui :
    #   - un warning torch "operation non-deterministe, fallback silencieux"
    #     (absent des .err des deux jobs) ;
    #   - une calibration relue depuis un vieux checkpoint perime (p0 n'est
    #     JAMAIS checkpointe sur disque -- toujours recalculee a neuf, cf. le
    #     bloc [calibration] quelques lignes plus bas).
    # Il reste deux pistes NI confirmees NI exclues, simplement parce
    # qu'aucune des deux n'a ete loggee au moment de ces deux runs :
    #   1. une derive de version torch/transformers (ou du driver CUDA) entre
    #      les deux jours -- rien ne les pingue dans requirements.txt ;
    #   2. device_map="auto" placant une couche sur CPU au lieu du GPU selon
    #      la memoire GPU libre au moment du chargement (pas flagge par
    #      torch.use_deterministic_algorithms, donc invisible dans les .err).
    # Ce bloc ne coute rien (pas de job separe, juste quelques lignes au
    # demarrage de chaque tache) et logge les 3 informations qui manquaient,
    # pour qu'on puisse trancher si ca se reproduit.
    # =========================================================================
    print("=== DIAGNOSTIC ENV (2026-10-28) ===", flush=True)
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        print(smi.stdout, flush=True)
        if smi.returncode != 0:
            print(f"[diag] nvidia-smi returncode={smi.returncode} stderr={smi.stderr}", flush=True)
    except Exception as e:
        print(f"[diag] nvidia-smi a echoue: {e}", flush=True)

    try:
        import torch as _torch
        import transformers as _transformers
        print(f"[diag] torch={_torch.__version__} | transformers={_transformers.__version__} "
              f"| torch.version.cuda={_torch.version.cuda}", flush=True)
    except Exception as e:
        print(f"[diag] lecture versions torch/transformers a echoue: {e}", flush=True)

    try:
        device_map = getattr(client.model, "hf_device_map", None)
        if device_map is not None:
            non_gpu = {k: v for k, v in device_map.items() if "cuda" not in str(v)}
            print(f"[diag] device_map: {device_map}", flush=True)
            if non_gpu:
                print(f"[diag] ATTENTION: {len(non_gpu)} entree(s) du device_map ne sont PAS sur "
                      f"un GPU (split CPU/GPU possible, affecte la reproductibilite): {non_gpu}",
                      flush=True)
            else:
                print("[diag] device_map: tout est sur GPU, pas de split CPU/GPU detecte", flush=True)
        else:
            print("[diag] client.model n'a pas d'attribut hf_device_map", flush=True)
    except Exception as e:
        print(f"[diag] lecture device_map a echoue: {e}", flush=True)
    print("=== FIN DIAGNOSTIC ENV ===", flush=True)
    # =========================================================================

    # --- Pre-pass: summarize (plain generation). Reuses the same
    # run_llm_dataframe runner as the old generation-based design -- it
    # already handles batching, checkpointing and resuming by row id. ---
    print(f"[summary] Generating factual summaries for {len(eligible_idx_all):,} eligible rows...", flush=True)
    summary_cfg = RunConfig(
        id_col="__index__",
        text_col=args.text_col,
        batch_size=args.summary_batch_size,
        temperature=args.summary_temperature,
        max_new_tokens=args.summary_max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    df = run_llm_dataframe(
        df=df,
        cfg=summary_cfg,
        client=client,
        system_prompt="",
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: active_build_summary_prompt(row, col),
        parse_fn=parse_summary_output,
        output_cols=[SUMMARY_TEXT_COL],
        skip_if_already_filled=SUMMARY_TEXT_COL,
        checkpoint_path=summary_checkpoint_path,
        checkpoint_every=args.summary_checkpoint_every,
    )

    eligible_idx = [i for i in eligible_idx_all if pd.notna(df.at[i, SUMMARY_TEXT_COL])]
    n_summary_failed = len(eligible_idx_all) - len(eligible_idx)
    if n_summary_failed:
        print(f"[summary] WARNING: {n_summary_failed:,} eligible rows got no usable summary "
              f"(empty generation) -- these never reach pass 2, source_category stays NA.", flush=True)
        for i in eligible_idx_all:
            if pd.isna(df.at[i, SUMMARY_TEXT_COL]):
                df.at[i, "source_category_incomplete"] = "TRUE"

    print(f"[pipeline] {len(eligible_idx):,}/{len(eligible_idx_all):,} eligible rows have a "
          f"usable summary and proceed to pass 2 (logit-probing)", flush=True)

    # --- One-time (per task) calibration on the SAME context_label pass 2
    # will use, so the calibration prompt matches the real ones exactly
    # except for row-specific content. ---
    print("[calibration] Computing p0(category) on the empty-summary variant...", flush=True)
    p0 = compute_calibration(client, candidate_letters, args.max_input_tokens, context_label=SUMMARY_CONTEXT_LABEL,
                              build_calibration_prompt_fn=active_build_calibration_prompt,
                              average_fn=active_average_yes_over_orderings)
    if args.backend == "litellm":
        # 2026-11-25 "p0 floor" fix -- see module docstring. Local 8B path
        # (the else-branch that never runs this) is completely unaffected.
        p0_raw = dict(p0)
        p0 = _floor_p0_litellm(p0)
        for cat in CATEGORIES:
            if pd.notna(p0_raw[cat]) and abs(p0[cat] - p0_raw[cat]) > 1e-9:
                print(f"  p0[{cat}] = {p0_raw[cat]:.4f} -> floored to {p0[cat]:.4f} "
                      f"(2026-11-25 p0-floor fix, --backend litellm only)")
            else:
                print(f"  p0[{cat}] = {p0[cat]:.4f}")
    else:
        for cat in CATEGORIES:
            print(f"  p0[{cat}] = {p0[cat]:.4f}")

    # --- Build the flat list of (row_idx, category, order) prompts for
    # every row with a usable summary. Reads source_summary, NOT the raw
    # article -- see SUMMARY_CONTEXT_LABEL / SUMMARY_TEXT_COL above.
    # Resumable via a long-format checkpoint keyed on (row_idx, category, order). ---
    flat_keys = []     # (row_idx, category, order)
    flat_prompts = []
    for row_idx in eligible_idx:
        row = df.loc[row_idx]
        for cat in CATEGORIES:
            for order in (1, 2):
                flat_keys.append((row_idx, cat, order))
                flat_prompts.append(
                    active_build_question_prompt(row, SUMMARY_TEXT_COL, cat, order, context_label=SUMMARY_CONTEXT_LABEL)
                )

    print(f"[pipeline] {len(flat_prompts):,} total (row, category, order) prompts "
          f"({len(eligible_idx):,} rows x {len(CATEGORIES)} categories x 2 orders)")

    # --- Resume from checkpoint if present (long format: one row per
    # flat prompt, with its raw logits for each candidate letter). ---
    done = {}  # (row_idx, cat, order) -> {letter: logit}
    if Path(logit_checkpoint_path).exists():
        ckpt = pd.read_parquet(logit_checkpoint_path)
        n_skipped_nan = 0
        for _, r in ckpt.iterrows():
            key = (int(r["row_idx"]), str(r["category"]), int(r["order"]))
            logits_dict = {letter: r[f"logit_{letter}"] for letter in candidate_letters}
            if args.backend == "litellm" and all(pd.isna(v) for v in logits_dict.values()):
                # 2026-11-26 checkpoint-resume fix -- see module docstring.
                # A previously-failed read (every candidate letter's logit
                # NaN) must never be treated as "done": discard it here so it
                # falls back into todo_indices and gets retried. litellm-only
                # -- the local 8B path's resume behaviour is unaffected.
                n_skipped_nan += 1
                continue
            done[key] = logits_dict
        print(f"[resume] Loaded {len(done):,} already-resolved prompts from {logit_checkpoint_path}"
              + (f" ({n_skipped_nan:,} previously-failed all-NaN entries discarded, will retry)"
                 if n_skipped_nan else ""), flush=True)

    todo_indices = [i for i, key in enumerate(flat_keys) if key not in done]
    print(f"[pipeline] {len(todo_indices):,} prompts remaining to compute "
          f"({len(flat_prompts) - len(todo_indices):,} already done)")

    n_since_checkpoint = 0
    for start in range(0, len(todo_indices), args.batch_size):
        batch_flat_idx = todo_indices[start:start + args.batch_size]
        batch_prompts = [flat_prompts[i] for i in batch_flat_idx]

        raw = client.logprob_many(
            system_prompt="",
            user_prompts=batch_prompts,
            candidate_letters=candidate_letters,
            max_input_tokens=args.max_input_tokens,
        )
        for flat_idx, logits_dict in zip(batch_flat_idx, raw):
            done[flat_keys[flat_idx]] = logits_dict

        n_since_checkpoint += len(batch_flat_idx)
        n_done_total = len(flat_prompts) - (len(todo_indices) - start - len(batch_flat_idx))
        print(f"[progress] {n_done_total:,}/{len(flat_prompts):,} prompts resolved", flush=True)

        if n_since_checkpoint >= args.checkpoint_every:
            _save_checkpoint(logit_checkpoint_path, done, candidate_letters)
            n_since_checkpoint = 0

    if todo_indices:
        _save_checkpoint(logit_checkpoint_path, done, candidate_letters)

    # --- Aggregate: for every row with a usable summary, for every
    # category, average P(yes) over the 2 orders, calibrate against p0,
    # then argmax. ---
    letter_yes_by_order = {order: letters_for_order(order)[0] for order in (1, 2)}

    for row_idx in eligible_idx:
        corrected = {}
        any_nan = False
        for cat in CATEGORIES:
            p_yes_by_order = {}
            for order in (1, 2):
                logits_dict = done.get((row_idx, cat, order))
                if logits_dict is None:
                    p_yes_by_order[order] = float("nan")
                    any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                p_yes_by_order[order] = probs.get(letter_yes_by_order[order], float("nan"))
                if pd.isna(p_yes_by_order[order]):
                    any_nan = True

            p = active_average_yes_over_orderings(p_yes_by_order[1], p_yes_by_order[2])
            if pd.isna(p) or pd.isna(p0[cat]):
                corrected[cat] = float("nan")
                any_nan = True
            else:
                corrected[cat] = calibrate(p, p0[cat])

        for cat, col in zip(CATEGORIES, PROB_COLS):
            df.at[row_idx, col] = corrected[cat]

        valid = {c: v for c, v in corrected.items() if not pd.isna(v)}
        best_cat, best_p = pick_argmax_category(valid)
        if best_cat is None:
            df.at[row_idx, "source_category"] = pd.NA
            df.at[row_idx, "source_category_margin"] = pd.NA
            df.at[row_idx, "source_category_runnerup"] = pd.NA
        else:
            ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
            runner_up_cat = ranked[1][0] if len(ranked) > 1 else pd.NA
            margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            df.at[row_idx, "source_category"] = best_cat
            df.at[row_idx, "source_category_margin"] = margin
            df.at[row_idx, "source_category_runnerup"] = runner_up_cat
            df.at[row_idx, "source_category_before_validation"] = best_cat
        if any_nan:
            df.at[row_idx, "source_category_incomplete"] = "TRUE"

    n_incomplete_pass2 = int((df["source_category_incomplete"] == "TRUE").sum())
    if n_incomplete_pass2:
        print(f"[warning] {n_incomplete_pass2:,} rows flagged source_category_incomplete "
              f"after pass 1+2 (failed summary pre-pass and/or a failed logit read).", flush=True)

    # ---------------------------------------------------------------------
    # Pass 3 (2026-10-27 "validate-and-correct" addition): one holistic
    # TRUE/FALSE re-read of pass 2's own chosen category, per row pass 2
    # actually resolved. Same logit-probing machinery as pass 2 (one
    # forward pass, order-swap averaging, per-claimed-category
    # calibration) but a different question shape -- see
    # src/step4b_logit_prompts.py's VALIDATION_QUESTION_TEMPLATE and the
    # module docstring above. Where pass 3 says FALSE, the row falls
    # back to pass 2's own runner-up category (captured just above)
    # rather than a fresh guess. An inconclusive pass-3 read is fail-safe
    # -- pass 2's own choice is kept, never swapped on a signal that
    # isn't trusted.
    # ---------------------------------------------------------------------
    if args.skip_validation:
        print("[validation] --skip_validation set -- pass 3 disabled (2026-10-29: "
              "net-negative on job66672510's corrected-gold eval, see src/step4b_logit_prompts.py "
              "module docstring). Keeping pass 2's own choice as the final source_category "
              "for every row, no validation forward passes run.", flush=True)
        n_val_incomplete = 0
    else:
        validated_idx = [i for i in eligible_idx if pd.notna(df.at[i, "source_category"])]
        print(f"[validation] Re-checking pass 2's chosen category for {len(validated_idx):,} "
              f"resolved rows...", flush=True)

        print("[validation] Computing p0_val(claimed category) on the empty-summary variant...", flush=True)
        p0_val = compute_validation_calibration(client, candidate_letters, args.max_input_tokens, context_label=SUMMARY_CONTEXT_LABEL,
                                                 build_validation_calibration_prompt_fn=active_build_validation_calibration_prompt,
                                                 average_fn=active_average_yes_over_orderings)
        if args.backend == "litellm":
            # 2026-11-25 "p0 floor" fix -- see module docstring. Local 8B path
            # (the else-branch that never runs this) is completely unaffected.
            p0_val_raw = dict(p0_val)
            p0_val = _floor_p0_litellm(p0_val)
            for cat in CATEGORIES:
                if pd.notna(p0_val_raw[cat]) and abs(p0_val[cat] - p0_val_raw[cat]) > 1e-9:
                    print(f"  p0_val[{cat}] = {p0_val_raw[cat]:.4f} -> floored to {p0_val[cat]:.4f} "
                          f"(2026-11-25 p0-floor fix, --backend litellm only)")
                else:
                    print(f"  p0_val[{cat}] = {p0_val[cat]:.4f}")
        else:
            for cat in CATEGORIES:
                print(f"  p0_val[{cat}] = {p0_val[cat]:.4f}")

        val_flat_keys = []    # (row_idx, order)
        val_flat_prompts = []
        for row_idx in validated_idx:
            row = df.loc[row_idx]
            claimed_category = df.at[row_idx, "source_category"]
            for order in (1, 2):
                val_flat_keys.append((row_idx, order))
                val_flat_prompts.append(
                    active_build_validation_prompt(row, SUMMARY_TEXT_COL, claimed_category, order, context_label=SUMMARY_CONTEXT_LABEL)
                )

        print(f"[validation] {len(val_flat_prompts):,} total (row, order) validation prompts "
              f"({len(validated_idx):,} rows x 2 orders)")

        val_done = {}  # (row_idx, order) -> {letter: logit}
        if Path(validation_checkpoint_path).exists():
            ckpt = pd.read_parquet(validation_checkpoint_path)
            n_skipped_nan = 0
            for _, r in ckpt.iterrows():
                vkey = (int(r["row_idx"]), int(r["order"]))
                logits_dict = {letter: r[f"logit_{letter}"] for letter in candidate_letters}
                if args.backend == "litellm" and all(pd.isna(v) for v in logits_dict.values()):
                    # 2026-11-26 checkpoint-resume fix -- see module docstring
                    # and the matching pass-2 resume block above.
                    n_skipped_nan += 1
                    continue
                val_done[vkey] = logits_dict
            print(f"[resume] Loaded {len(val_done):,} already-resolved validation prompts from "
                  f"{validation_checkpoint_path}"
                  + (f" ({n_skipped_nan:,} previously-failed all-NaN entries discarded, will retry)"
                     if n_skipped_nan else ""), flush=True)

        val_todo = [i for i, vkey in enumerate(val_flat_keys) if vkey not in val_done]
        print(f"[validation] {len(val_todo):,} validation prompts remaining "
              f"({len(val_flat_prompts) - len(val_todo):,} already done)")

        n_since_val_checkpoint = 0
        for start in range(0, len(val_todo), args.validation_batch_size):
            batch_flat_idx = val_todo[start:start + args.validation_batch_size]
            batch_prompts = [val_flat_prompts[i] for i in batch_flat_idx]

            raw = client.logprob_many(
                system_prompt="",
                user_prompts=batch_prompts,
                candidate_letters=candidate_letters,
                max_input_tokens=args.max_input_tokens,
            )
            for flat_idx, logits_dict in zip(batch_flat_idx, raw):
                val_done[val_flat_keys[flat_idx]] = logits_dict

            n_since_val_checkpoint += len(batch_flat_idx)
            n_done_total = len(val_flat_prompts) - (len(val_todo) - start - len(batch_flat_idx))
            print(f"[validation progress] {n_done_total:,}/{len(val_flat_prompts):,} prompts resolved", flush=True)

            if n_since_val_checkpoint >= args.validation_checkpoint_every:
                _save_validation_checkpoint(validation_checkpoint_path, val_done, candidate_letters)
                n_since_val_checkpoint = 0

        if val_todo:
            _save_validation_checkpoint(validation_checkpoint_path, val_done, candidate_letters)

        n_rejected = 0
        n_val_incomplete = 0
        for row_idx in validated_idx:
            claimed_category = df.at[row_idx, "source_category"]
            p_true_by_order = {}
            val_any_nan = False
            for order in (1, 2):
                logits_dict = val_done.get((row_idx, order))
                if logits_dict is None:
                    p_true_by_order[order] = float("nan")
                    val_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_true, _letter_false = letters_for_order(order)
                p_true_by_order[order] = probs.get(letter_true, float("nan"))
                if pd.isna(p_true_by_order[order]):
                    val_any_nan = True

            p_true = active_average_yes_over_orderings(p_true_by_order[1], p_true_by_order[2])
            if pd.isna(p_true) or pd.isna(p0_val.get(claimed_category)):
                val_any_nan = True

            if val_any_nan:
                # Inconclusive read -- fail safe: keep pass 2's own choice,
                # don't swap on a signal we don't trust.
                n_val_incomplete += 1
                df.at[row_idx, "source_category_incomplete"] = "TRUE"
                continue

            p_true_corrected = calibrate(p_true, p0_val[claimed_category])
            df.at[row_idx, "source_category_validation_p_true"] = p_true_corrected

            if p_true_corrected >= 0.5:
                df.at[row_idx, "source_category_validated"] = "TRUE"
            else:
                df.at[row_idx, "source_category_validated"] = "FALSE"
                runner_up = df.at[row_idx, "source_category_runnerup"]
                if pd.notna(runner_up):
                    df.at[row_idx, "source_category"] = runner_up
                    n_rejected += 1
                # else: no runner-up available (shouldn't normally happen with
                # 6 categories) -- keep pass 2's own choice rather than leave
                # the row with no answer at all.

        print(f"[validation] {n_rejected:,}/{len(validated_idx):,} rows rejected by pass 3 and "
              f"swapped to pass 2's runner-up category "
              f"({n_val_incomplete:,} validation reads inconclusive, pass 2's choice kept as-is)",
              flush=True)

    # ---------------------------------------------------------------------
    # Disambiguation pass (2026-11-26, --backend litellm ONLY, opt-in via
    # --disambiguation_margin_threshold > 0, Maxime's idea): for every
    # resolved row whose CURRENT margin (pass 2's own, or pass 3's if that
    # ran and swapped it) is below the threshold -- a genuine toss-up, not
    # noise -- ask ONE direct head-to-head question naming its top two
    # categories explicitly, instead of trusting six independent absolute
    # reads in isolation. See the module docstring's 2026-11-26
    # "disambiguation pass addition" entry and
    # src/step4b_logit_prompts_remote.py's DISAMBIGUATION_QUESTION_TEMPLATE
    # module comment for the full rationale and how this differs from pass
    # 3 above. An inconclusive read is fail-safe, same precedent as pass 3:
    # the row's current choice is kept, never swapped on an untrusted
    # signal.
    # ---------------------------------------------------------------------
    if args.backend == "litellm" and args.disambiguation_margin_threshold > 0:
        disamb_idx = [
            i for i in eligible_idx
            if pd.notna(df.at[i, "source_category"])
            and pd.notna(df.at[i, "source_category_runnerup"])
            and pd.notna(df.at[i, "source_category_margin"])
            and float(df.at[i, "source_category_margin"]) < args.disambiguation_margin_threshold
        ]
        print(f"[disambiguation] {len(disamb_idx):,}/{len(eligible_idx):,} resolved rows have a "
              f"margin below {args.disambiguation_margin_threshold} -- re-checking these with a "
              f"direct head-to-head question...", flush=True)

        # Each row's own (cat1, cat2) pair -- cat1/cat2 = the row's own
        # current winner/runner-up. Canonicalised to a SORTED tuple purely
        # for calibration caching (so (X, Y) and (Y, X) share one
        # calibration), independent of which one is actually "ahead" for
        # any given row.
        row_pair = {}
        for i in disamb_idx:
            cat_a = df.at[i, "source_category"]
            cat_b = df.at[i, "source_category_runnerup"]
            row_pair[i] = tuple(sorted((cat_a, cat_b)))

        unique_pairs = sorted(set(row_pair.values()))
        print(f"[disambiguation] {len(unique_pairs):,} unique category pair(s) among these rows", flush=True)

        # --- Lazy, once-per-pair calibration on the empty-summary variant
        # -- only the pairs that actually occur, never all 15 possible
        # pairs up front. ---
        p0_pairs = {}
        if unique_pairs:
            calib_prompts, calib_meta = [], []  # meta: (pair, order)
            for pair in unique_pairs:
                for order in (1, 2):
                    calib_prompts.append(
                        build_disambiguation_calibration_prompt_remote(pair[0], pair[1], order, context_label=SUMMARY_CONTEXT_LABEL)
                    )
                    calib_meta.append((pair, order))
            print(f"[disambiguation] Computing p0(pair) on the empty-summary variant "
                  f"({len(calib_prompts):,} forward passes)...", flush=True)
            calib_raw = client.logprob_many(
                system_prompt="",
                user_prompts=calib_prompts,
                candidate_letters=candidate_letters,
                max_input_tokens=args.max_input_tokens,
            )
            p_cat1_by_pair_order = {}
            for (pair, order), logits_dict in zip(calib_meta, calib_raw):
                probs = softmax_restricted(logits_dict)
                letter_cat1, _letter_cat2 = letters_for_order(order)
                p_cat1_by_pair_order[(pair, order)] = probs.get(letter_cat1, float("nan"))
            for pair in unique_pairs:
                p0_pairs[pair] = active_average_yes_over_orderings(
                    p_cat1_by_pair_order[(pair, 1)], p_cat1_by_pair_order[(pair, 2)]
                )
                print(f"  p0[{pair[0]} vs {pair[1]}] = {p0_pairs[pair]:.4f}", flush=True)

        # --- Flat (row_idx, order) prompts, one per disambiguated row x 2 orders. ---
        dis_flat_keys = []    # (row_idx, order)
        dis_flat_prompts = []
        for i in disamb_idx:
            pair = row_pair[i]
            row = df.loc[i]
            for order in (1, 2):
                dis_flat_keys.append((i, order))
                dis_flat_prompts.append(
                    build_disambiguation_prompt_remote(row, SUMMARY_TEXT_COL, pair[0], pair[1], order, context_label=SUMMARY_CONTEXT_LABEL)
                )

        print(f"[disambiguation] {len(dis_flat_prompts):,} total (row, order) prompts "
              f"({len(disamb_idx):,} rows x 2 orders)", flush=True)

        dis_done = {}  # (row_idx, order) -> {letter: logit}
        if Path(disambiguation_checkpoint_path).exists():
            ckpt = pd.read_parquet(disambiguation_checkpoint_path)
            n_skipped_nan = 0
            for _, r in ckpt.iterrows():
                dkey = (int(r["row_idx"]), int(r["order"]))
                logits_dict = {letter: r[f"logit_{letter}"] for letter in candidate_letters}
                if all(pd.isna(v) for v in logits_dict.values()):
                    # 2026-11-26 checkpoint-resume fix -- see module
                    # docstring and the matching pass-2/pass-3 resume blocks
                    # above (this pass is litellm-only already, so no extra
                    # backend check is needed here).
                    n_skipped_nan += 1
                    continue
                dis_done[dkey] = logits_dict
            print(f"[resume] Loaded {len(dis_done):,} already-resolved disambiguation prompts from "
                  f"{disambiguation_checkpoint_path}"
                  + (f" ({n_skipped_nan:,} previously-failed all-NaN entries discarded, will retry)"
                     if n_skipped_nan else ""), flush=True)

        dis_todo = [i for i, dkey in enumerate(dis_flat_keys) if dkey not in dis_done]
        print(f"[disambiguation] {len(dis_todo):,} prompts remaining "
              f"({len(dis_flat_prompts) - len(dis_todo):,} already done)", flush=True)

        n_since_dis_checkpoint = 0
        for start in range(0, len(dis_todo), args.disambiguation_batch_size):
            batch_flat_idx = dis_todo[start:start + args.disambiguation_batch_size]
            batch_prompts = [dis_flat_prompts[i] for i in batch_flat_idx]

            raw = client.logprob_many(
                system_prompt="",
                user_prompts=batch_prompts,
                candidate_letters=candidate_letters,
                max_input_tokens=args.max_input_tokens,
            )
            for flat_idx, logits_dict in zip(batch_flat_idx, raw):
                dis_done[dis_flat_keys[flat_idx]] = logits_dict

            n_since_dis_checkpoint += len(batch_flat_idx)
            n_done_total = len(dis_flat_prompts) - (len(dis_todo) - start - len(batch_flat_idx))
            print(f"[disambiguation progress] {n_done_total:,}/{len(dis_flat_prompts):,} prompts resolved", flush=True)

            if n_since_dis_checkpoint >= args.disambiguation_checkpoint_every:
                _save_validation_checkpoint(disambiguation_checkpoint_path, dis_done, candidate_letters)
                n_since_dis_checkpoint = 0

        if dis_todo:
            _save_validation_checkpoint(disambiguation_checkpoint_path, dis_done, candidate_letters)

        n_swapped = 0
        n_dis_incomplete = 0
        for i in disamb_idx:
            pair = row_pair[i]
            original_cat = df.at[i, "source_category"]
            p_cat1_by_order = {}
            dis_any_nan = False
            for order in (1, 2):
                logits_dict = dis_done.get((i, order))
                if logits_dict is None:
                    p_cat1_by_order[order] = float("nan")
                    dis_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_cat1, _letter_cat2 = letters_for_order(order)
                p_cat1_by_order[order] = probs.get(letter_cat1, float("nan"))
                if pd.isna(p_cat1_by_order[order]):
                    dis_any_nan = True

            p_cat1 = active_average_yes_over_orderings(p_cat1_by_order[1], p_cat1_by_order[2])
            if pd.isna(p_cat1) or pd.isna(p0_pairs.get(pair)):
                dis_any_nan = True

            df.at[i, "source_category_disambiguation_applied"] = "TRUE"
            df.at[i, "source_category_before_disambiguation"] = original_cat

            if dis_any_nan:
                # Inconclusive read -- fail-safe: keep the current choice,
                # don't swap on a signal we don't trust (same precedent as
                # pass 3 above).
                n_dis_incomplete += 1
                df.at[i, "source_category_incomplete"] = "TRUE"
                continue

            p_cat1_corrected = calibrate(p_cat1, p0_pairs[pair])
            winner = pair[0] if p_cat1_corrected >= 0.5 else pair[1]
            p_original = p_cat1_corrected if original_cat == pair[0] else (1.0 - p_cat1_corrected)
            df.at[i, "source_category_disambiguation_p_original"] = p_original

            if winner != original_cat:
                df.at[i, "source_category"] = winner
                n_swapped += 1

        print(f"[disambiguation] {n_swapped:,}/{len(disamb_idx):,} rows swapped by the head-to-head "
              f"pass ({n_dis_incomplete:,} reads inconclusive, original choice kept)", flush=True)

    n_incomplete = int((df["source_category_incomplete"] == "TRUE").sum())
    if n_incomplete > n_incomplete_pass2:
        print(f"[warning] {n_incomplete:,} rows flagged source_category_incomplete in total "
              f"(+{n_incomplete - n_incomplete_pass2:,} from inconclusive pass-3 reads).", flush=True)

    # --- Save ALL rows ---
    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or args.job_id
        or "nojobid"
    )
    if task_id is not None:
        base = f"{args.output_base}_task{task_id}_job{job_id}"
    else:
        base = f"{args.output_base}_job{job_id}"

    parquet_path = base + ".parquet"
    csv_path     = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    counts = df["source_category"].value_counts(dropna=True)
    na_count = int(df["source_category"].isna().sum())
    avg_margin = df["source_category_margin"].dropna().astype(float).mean() if df["source_category_margin"].notna().any() else float("nan")
    n_validated_false = int((df["source_category_validated"] == "FALSE").sum())
    n_disambiguated = int((df["source_category_disambiguation_applied"] == "TRUE").sum())
    n_disambiguation_swapped = int((
        (df["source_category_disambiguation_applied"] == "TRUE")
        & df["source_category_before_disambiguation"].notna()
        & (df["source_category"] != df["source_category_before_disambiguation"])
    ).sum()) if n_disambiguated else 0
    print(f"Saved: {parquet_path} | {len(df):,} rows total | NA: {na_count:,} | "
          f"incomplete: {n_incomplete:,} | avg winning margin (pass 2): {avg_margin:.4f} | "
          f"pass-3 rejections (swapped to runner-up): {n_validated_false:,} | "
          f"disambiguation: {n_disambiguation_swapped:,}/{n_disambiguated:,} rows swapped")
    print(counts.to_string())

    for p in (summary_checkpoint_path, logit_checkpoint_path, validation_checkpoint_path,
              disambiguation_checkpoint_path):
        if Path(p).exists():
            Path(p).unlink()
            print(f"[checkpoint] Deleted {p}")

    # 2026-11-24: usage telemetry for the remote (--backend litellm) client --
    # pure observability, unrelated to this run's actual results. The local
    # TransformersClient has no print_stats(), so this is a no-op for the
    # GPU path -- see src/client_remote.py's module docstring. Same addition
    # as scripts/step3_criticism_detection.py.
    if hasattr(client, "print_stats"):
        client.print_stats()

    return 0


def _save_checkpoint(checkpoint_path, done, candidate_letters):
    rows = []
    for (row_idx, cat, order), logits_dict in done.items():
        r = {"row_idx": row_idx, "category": cat, "order": order}
        for letter in candidate_letters:
            r[f"logit_{letter}"] = logits_dict.get(letter, float("nan"))
        rows.append(r)
    ckpt_df = pd.DataFrame(rows)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt_df.to_parquet(checkpoint_path, index=False)
    print(f"[checkpoint] Saved {len(ckpt_df):,} resolved prompts -> {checkpoint_path}", flush=True)


def _save_validation_checkpoint(checkpoint_path, done, candidate_letters):
    rows = []
    for (row_idx, order), logits_dict in done.items():
        r = {"row_idx": row_idx, "order": order}
        for letter in candidate_letters:
            r[f"logit_{letter}"] = logits_dict.get(letter, float("nan"))
        rows.append(r)
    ckpt_df = pd.DataFrame(rows)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt_df.to_parquet(checkpoint_path, index=False)
    print(f"[checkpoint] Saved {len(ckpt_df):,} resolved validation prompts -> {checkpoint_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
