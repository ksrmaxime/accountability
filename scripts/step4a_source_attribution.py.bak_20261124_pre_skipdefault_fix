"""
scripts/step4a_source_attribution.py

STEP 4a of the accountability pipeline — first half of source detection.

For every (article, target) row step 3 confirmed as a criticism, asks the
LLM a single yes/no-style question: is the criticism attributed to a
specific external actor (quoted, named, or clearly referenced), or is it
just the journalist's own narrative framing with no such actor?

Rows step 3 did NOT flag as criticism (keyword_answer != "YES") are left
untouched in the output — this script only ever reads keyword_answer, it
never overwrites it, and `source_attribution` stays NA for those rows.

Two-pass design (added after the justification-first run showed the label
sometimes contradicting its own justification — see src/verify_utils.py).

2026-10-13: pass 2 removed (that version of it). A draft-vs-final
comparison on job65746412 (800-row gold) found it purely harmful: of the
7 rows it touched where pass 1 was already correct, all 7 were flipped to
a WRONG final answer (0 corrections) -- draft accuracy 72.2% vs final
66.9% on that run. The mechanism: that pass 2 read only pass 1's own
justification and fired its SOURCED rule whenever that text named the
journalist by name -- see VERIFY_USER_TEMPLATE's own changelog in
src/step4a_prompts.py. Pass 1's own draft label was written straight to
the final column, with no second LLM call, from 2026-10-13 until the
2026-11-17 redesign below.

2026-11-16 "validate-on-justification" addition (REMOVED 2026-11-17, see
below): job20479169 (800-row gold, the then-current best step3 lineage)
showed step4a's 95 YES-gated rows scoring BELOW their own majority-class
baseline (74.7% vs 82.1%), with a heavily directional error pattern -- 20
FN (gold=SOURCED, Apertus=JOURNALIST) vs only 4 FP. A separate calibrated-
logit validation pass (src/step4a_logit_prompts.py) was added to re-check
pass 1's JOURNALIST verdicts, mirroring step3's own validation pass.
Tested in BOTH of its configured modes on real runs: reading only pass 1's
justification (job20588894) found essentially no separation between true
SOURCED and true JOURNALIST (mean p_sourced 0.375 vs 0.363); reading the
full article a second time (job20606446) found no real separation either
(0.506 vs 0.552, if anything inverted) while actively making precision
worse (FP 4->16 for only 10 FN fixed, net accuracy DROPPING 75.8%->73.7%).
Diagnosis (see src/step4a_prompts.py's own changelog for the full
reasoning): a single calibrated logit read, with no generated reasoning,
cannot force the genuine re-search through a long article that these
missed cases actually need -- a read of GOLD's own justification for the
FN rows showed the missed outside voice is usually an explicitly NAMED
actor (a politician, a court, an interest group) that pass 1's
justification simply never mentions having looked for. REMOVED entirely
in favour of the generation-based redesign immediately below -- see
src/step4a_logit_prompts.py's own module docstring for the full record of
this negative result (kept there for reference, no longer imported here).

2026-11-17 pass 2 redesign ("forced re-search", ARTICLE-grounded): adds
back a pass 2, but a GENERATION pass (reasoning allowed, like pass 1 is),
not the logit-probing of the version just removed above, and reads the
FULL ORIGINAL ARTICLE again -- not pass 1's own justification, which was
the fatal blind spot of the 2026-10-13-removed pass 2 (it could not tell
a named journalist byline from a real outside source, because it never
saw the article to check). The actual fix for the diagnosed retrieval
failure is VERIFY_USER_TEMPLATE's explicit instruction (src/step4a_prompts.py)
to enumerate every named actor mentioned near the criticism before
answering -- forcing the same kind of deliberate search pass 1's own
justification-before-answer design relies on, but specifically re-
triggered on rows pass 1 already got wrong once. Scoped ONLY to rows pass
1 said JOURNALIST (the same MIRROR-of-step3 scoping the removed
validation pass used, for the same reason: the measured error is one-
directional, FN-heavy, so this pass can only ever flip JOURNALIST->SOURCED,
never the reverse -- it cannot create a new error on a row pass 1 already
got right as SOURCED). Forced rows (source_attribution_forced == "TRUE")
skip pass 2 entirely, same convention as the removed validation pass --
there was no real pass 1 for those, nothing to re-check. Deliberately NO
retry on a pass-2 parse failure (unlike pass 1's own retry ladder): the
fail-safe default is to keep pass 1's JOURNALIST either way, the affected
population is small (~60-95/800 rows), and the goal here is a scoped,
simple fix to test, not another retry ladder. Toggle with --skip_pass2
(see sbatch_step4a_array.sh's SKIP_PASS2). Confirmed on job20620441: 95-row
YES-gated population, accuracy 75.8% (pass 1 alone) -> 83.2% (final),
FIRST TIME above the 82.1% majority baseline for this step.

2026-11-18 pass 3 ("pass2 coherence check", logit-probing): reading
job20620441's 32 pass-2-touched rows line by line found roughly half of
pass 2's 27 flips are self-contradictory -- the model's OWN written
reasoning concludes JOURNALIST-style things ("no outside voice is
credited", "framed as the author's own opinion", even explicitly naming
the article's byline journalist as "the actor") and then the bare final
word it commits to is SOURCED anyway. Precisely: 11/27 flips are "clean"
(genuinely name a real outside voice, 82% precision -- the forced-
enumeration idea working as intended); the other 16 are this self-
contradiction, a near-exact coin flip (8 right / 8 wrong) that happens to
net to zero on this one run only because the touched population is
majority-SOURCED -- noise, not a result. Two fixes were tried against
this run's real justification texts before picking one: deterministic
Python rules (bag-of-phrases: 18/32 rows AMBIGUOUS, the model routinely
names a candidate actor AND negates their relevance in the same
paragraph, which plain keyword counting cannot resolve; "last occurring
phrase wins", mirroring src/verify_utils.py's own _find_label: resolves
the ambiguity but changes 17/32 decisions, only 9 of which are real
corrections against 8 regressions -- net +1/32, indistinguishable from
noise). Replaced by a narrow THIRD logit-probing pass
(src/step4a_pass2_coherence_logit_prompts.py) that reads ONLY pass 2's
own justification text (never the article, never pass 1's justification)
and asks a bounded reading-comprehension question -- "what does THIS
PARAGRAPH itself conclude?" -- rather than re-deciding the real-world
question the way the removed 2026-11-16 validation pass tried and failed
to. See that file's own module docstring for the full rationale. Only
OVERRIDES pass 2's stated answer when the calibrated reading clearly
disagrees (--coherence_margin around the 0.5 boundary); an unclear or
agreeing read leaves pass 2's own answer untouched. Toggle with
--skip_coherence. Not yet confirmed on a real run.

2026-11-24 "pass1 coherence check" (logit-probing, --backend litellm ONLY):
job20934804 (the second full remote-chain run on GOLD800, after the
2026-11-24 "verdict-last" fix to src/step4a_prompts_remote.py eliminated
the earlier verdict-first parsing bug) still sits below the classic 8B
baseline. Reading the remaining errors found a NEW self-contradiction,
specific to pass 1 this time (not pass 2, which pass 3 above already
covers): 18/113 rows (16%) with both a draft label and a justification
have a justification whose own plain-language reasoning clearly argues
JOURNALIST ("no outside voice is credited", "framed as the author's own
analysis") while the stored source_attribution_draft is the opposite
(SOURCED) -- checked specifically against the 2026-11-24 parsing bug and
ruled out (zero rows open on the wrong bare word; this is a *different*
mechanism, the model simply not inserting a real line break before the
final required word, so the true last token doesn't reliably reflect the
preceding reasoning). Maxime's suggestion, implemented here: the exact
same remedy already proven for pass 2's own version of this problem
(src/step4a_pass2_coherence_logit_prompts.py / pass 3 above) -- a narrow
third-party reading-comprehension check, via calibrated logit-probing,
that reads ONLY pass 1's own justification text (never the article) and
reports what that paragraph itself concludes -- applied one stage
earlier. See src/step4a_pass1_coherence_logit_prompts.py's own module
docstring for the full diagnosis and rationale. Runs BEFORE pass 2,
directly on source_attribution_draft (not after the fill into
source_attribution), so a correction from SOURCED->JOURNALIST becomes a
legitimate pass 2 candidate and a correction from JOURNALIST->SOURCED is
correctly excluded from pass 2 -- the ambiguity is already resolved.
Scoped to every row with a genuine pass 1 (parsed draft label AND a real,
non-bare justification); forced one-word rows have no justification and
are naturally excluded. Only OVERRIDES the draft when the calibrated
reading clearly disagrees (--pass1_coherence_margin around the 0.5
boundary); an unclear or agreeing read leaves pass 1's own draft label
untouched -- same fail-safe convention as pass 3. Toggle with
--skip_pass1_coherence. **Gated STRICTLY to --backend litellm** (see the
call site below): this problem was diagnosed only on the remote/70B path,
and the production local-GPU path (--backend vllm/transformers,
src/step4a_prompts.py) stays completely unaffected, byte-for-byte --
this check is simply skipped (n_p1coh_checked = 0) for any other backend,
regardless of --skip_pass1_coherence. Not yet confirmed on a real run.

Output columns: source_attribution                   -> "SOURCED" | "JOURNALIST" | NA  (FINAL --
                                                          pass 1, possibly corrected by the pass-1
                                                          coherence check, possibly flipped by
                                                          pass 2, possibly overridden by pass 3)
                source_attribution_draft              -> "SOURCED" | "JOURNALIST" | NA  (pass 1,
                                                          possibly corrected in place by the
                                                          pass-1 coherence check -- see
                                                          source_attribution_draft_before_pass1_coherence
                                                          for the original value)
                source_attribution_justification      -> free text | NA                 (pass 1)
                source_attribution_draft_before_pass1_coherence -> "SOURCED" | "JOURNALIST" | NA
                                                          (pass 1's own draft verdict, before the
                                                          pass-1 coherence check potentially
                                                          corrected it -- only set for rows that
                                                          check looked at; --backend litellm only)
                source_attribution_pass1_coherence_p_sourced -> calibrated P(pass 1's own
                                                          justification concludes SOURCED), or NA
                                                          (only set for rows the pass-1 coherence
                                                          check looked at; --backend litellm only)
                source_attribution_pass1_coherence_overridden -> "TRUE" | "FALSE" | NA ("TRUE" =
                                                          the pass-1 coherence check disagreed
                                                          clearly enough to correct
                                                          source_attribution_draft; "FALSE" =
                                                          checked and agreed, or was inconclusive;
                                                          NA = never looked at this row)
                source_attribution_before_pass2        -> "JOURNALIST" | NA  (pass 1's own verdict,
                                                          before pass 2 touched it -- only set for
                                                          rows pass 2 actually looked at)
                source_attribution_pass2_justification -> free text | NA (pass 2's own re-search,
                                                          only set for rows pass 2 looked at)
                source_attribution_pass2_answer        -> "SOURCED" | "JOURNALIST" | NA (pass 2's
                                                          own stated verdict, BEFORE pass 3's
                                                          coherence check; NA = pass 2 ran but its
                                                          output didn't parse)
                source_attribution_pass2_coherence_p_sourced -> calibrated P(pass 2's own
                                                          justification concludes SOURCED), or NA
                                                          (only set for rows pass 3 looked at)
                source_attribution_pass2_overridden    -> "TRUE" | "FALSE" | NA ("TRUE" = pass 3
                                                          disagreed clearly enough to override
                                                          pass 2's stated answer; "FALSE" = pass 3
                                                          checked and agreed, or was inconclusive;
                                                          NA = pass 3 never looked at this row)

Usage
-----
  python scripts/step4a_source_attribution.py \\
      --input       data/output/step3_merged/results.parquet \\
      --output_base data/output/step4a \\
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \\
      --dtype bf16 --batch_size 4 --temperature 0.0 \\
      --max_new_tokens 140 --max_input_tokens 16384 \\
      --verify_batch_size 8 --verify_max_new_tokens 250 --verify_max_input_tokens 16384 \\
      --coherence_batch_size 32
"""
from __future__ import annotations

import math
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from src.client import build_client
from src.runner import run_llm_dataframe, RunConfig
from src.step4a_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
    FORCE_SYSTEM_PROMPT, build_force_prompt,
)
# 2026-11-24: --backend litellm (remote API models via the UNIL/CSCS proxy,
# e.g. the Apertus-70B family) uses its OWN prompt module
# (src/step4a_prompts_remote.py), selected right after argparse below -- same
# pattern as scripts/step3_criticism_detection.py, see
# src/step3_prompts_remote.py's module docstring for the full rationale and
# src/step4a_prompts_remote.py's own docstring for this file's version of it.
# src/step4a_prompts.py (the production vllm/transformers, 8B, local-GPU
# path) is completely unaffected, byte-for-byte, by anything imported here.
# Pass 2 ("forced re-search") is an active part of this step's pipeline
# (unlike step3/step5/step6's pass 2), so its remote counterpart is also
# aliased below, alongside pass 1's.
from src.step4a_prompts_remote import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_REMOTE, build_user_prompt as build_user_prompt_remote,
    VERIFY_SYSTEM_PROMPT as VERIFY_SYSTEM_PROMPT_REMOTE, build_verify_prompt as build_verify_prompt_remote,
    FORCE_SYSTEM_PROMPT as FORCE_SYSTEM_PROMPT_REMOTE, build_force_prompt as build_force_prompt_remote,
)
from src.step4a_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled
from src.step4a_pass2_coherence_logit_prompts import (
    build_coherence_prompt, build_coherence_calibration_prompt, letters_for_order,
)
# 2026-11-24 "pass1 coherence check" -- see
# src/step4a_pass1_coherence_logit_prompts.py and this file's own module
# docstring. Aliased to avoid colliding with pass 3's own imports of the
# same function names just above. Backend-gated at the call site below
# (args.backend == "litellm" only) -- the module itself is backend-
# agnostic, same convention as the --backend litellm prompt-module split
# above.
from src.step4a_pass1_coherence_logit_prompts import (
    build_coherence_prompt as build_pass1_coherence_prompt,
    build_coherence_calibration_prompt as build_pass1_coherence_calibration_prompt,
    letters_for_order as pass1_letters_for_order,
)
from src.step4b_logit_utils import softmax_restricted, average_yes_over_orderings, calibrate

DRAFT_COLS = ["source_attribution_draft", "source_attribution_justification"]
FINAL_COLS = ["source_attribution"]
# Flag column: "TRUE" when a row's final answer came from the last-resort
# forced one-word pass rather than a genuine two-pass verification -- see
# the retry/force block in main(). NA for every normally-answered row.
FLAG_COLS = ["source_attribution_forced"]
# 2026-11-17 "forced re-search" pass 2 -- see src/step4a_prompts.py's
# VERIFY_SYSTEM_PROMPT/build_verify_prompt and this file's own module
# docstring.
PASS2_COLS = [
    "source_attribution_before_pass2",
    "source_attribution_pass2_justification",
    "source_attribution_pass2_answer",
]
# 2026-11-18 "pass2 coherence check" pass 3 -- see
# src/step4a_pass2_coherence_logit_prompts.py and this file's own module
# docstring. source_attribution_pass2_answer (above) stays pass 2's own
# RAW stated answer, untouched by this pass -- the coherence check's
# verdict lives only in these two columns plus the final source_attribution.
PASS2_COHERENCE_COLS = [
    "source_attribution_pass2_coherence_p_sourced",
    "source_attribution_pass2_overridden",
]
# 2026-11-24 "pass1 coherence check" -- see
# src/step4a_pass1_coherence_logit_prompts.py and this file's own module
# docstring. source_attribution_draft_before_pass1_coherence preserves
# pass 1's ORIGINAL draft label before this check potentially corrects it
# in place -- mirrors source_attribution_before_pass2's own role for pass
# 2. --backend litellm only; stays NA for every row on other backends.
PASS1_COHERENCE_COLS = [
    "source_attribution_draft_before_pass1_coherence",
    "source_attribution_pass1_coherence_p_sourced",
    "source_attribution_pass1_coherence_overridden",
]
ALL_COLS = DRAFT_COLS + FINAL_COLS + FLAG_COLS + PASS1_COHERENCE_COLS + PASS2_COLS + PASS2_COHERENCE_COLS
# Columns holding a calibrated FLOAT probability rather than a label/flag --
# see the dtype-initialization comment at their point of use below.
_FLOAT_COLS = {
    "source_attribution_pass1_coherence_p_sourced",
    "source_attribution_pass2_coherence_p_sourced",
}

VERIFY_LABELS = {"SOURCED": "SOURCED", "JOURNALIST": "JOURNALIST"}


def parse_draft_output(raw: str) -> dict:
    """A bare label with no real justification is a full failure, not a
    partial success -- see src/verify_utils.py:parse_draft_labeled."""
    empty = {"source_attribution_draft": pd.NA, "source_attribution_justification": pd.NA}
    label, justification = parse_draft_labeled(raw, VERIFY_LABELS)
    if label is None:
        return empty
    return {"source_attribution_draft": label, "source_attribution_justification": justification}


def parse_pass2_output(raw: str) -> dict:
    """Same justification-before-answer parsing as pass 1 (src/verify_utils.py) --
    a bare label with no reasoning is a parse failure, not a partial success,
    so a lazy one-word reflex can't silently pass as a genuine re-search."""
    empty = {"source_attribution_pass2_justification": pd.NA, "source_attribution_pass2_answer": pd.NA}
    label, justification = parse_draft_labeled(raw, VERIFY_LABELS)
    if label is None:
        return empty
    return {"source_attribution_pass2_justification": justification, "source_attribution_pass2_answer": label}


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True, help="Step 3 merged output: parquet or csv")
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="text")
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
    # scripts/step3_criticism_detection.py.
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
                         "--batch_size's role as the parallelism knob for this backend, since "
                         "there is no local GPU batch to size.")

    # --- Inference: pass 1 (draft, reads the full article) ---
    ap.add_argument("--batch_size",        required=True, type=int)
    ap.add_argument("--temperature",       required=True, type=float)
    ap.add_argument("--max_new_tokens",    required=True, type=int)
    ap.add_argument("--max_input_tokens",  required=True, type=int)

    # --- Pass 1 retries + last-resort force: every row step 3 flagged as
    # criticism is supposed to come out of this step with a real answer
    # (see src/verify_utils.py and the retry/force block in the script).
    # temperature=0.0 retried unchanged would just reproduce the same
    # failure, so each retry raises temperature and token budget a notch;
    # whatever is still unresolved after --draft_max_retries gets one
    # forced bare-word attempt (flagged via source_attribution_forced,
    # never sent through pass 2). ---
    ap.add_argument("--draft_max_retries",   type=int,   default=2,
                    help="Extra pass-1 attempts (beyond the first) for eligible rows still unanswered, each at a higher temperature/token budget.")
    ap.add_argument("--retry_temperature",   type=float, default=0.4,
                    help="Temperature for the first retry attempt; raised by +0.2 per further attempt, capped at 0.9.")

    # --- Inference: pass 2 ("forced re-search", 2026-11-17 redesign --
    # reads the FULL ARTICLE again, not just pass 1's justification, see
    # this file's own module docstring). Needs a much larger input budget
    # than the old justification-only pass 2 did (raised from 512 to
    # 16384 to fit the article) and a larger output budget (raised from 8
    # to 250, for the enumerate-then-answer reasoning). ---
    ap.add_argument("--verify_batch_size",       type=int,   default=8)
    ap.add_argument("--verify_temperature",      type=float, default=0.0)
    ap.add_argument("--verify_max_new_tokens",   type=int,   default=250)
    ap.add_argument("--verify_max_input_tokens", type=int,   default=16384)
    ap.add_argument("--skip_pass2", action="store_true",
                    help="2026-11-17: disable pass 2 entirely -- no second LLM call is made, and "
                    "source_attribution stays exactly pass 1's own (possibly forced) answer for "
                    "every row.")

    # --- Pass 3 ("pass2 coherence check", 2026-11-18, logit-probing): see
    # this file's own module docstring and
    # src/step4a_pass2_coherence_logit_prompts.py. Reads ONLY pass 2's own
    # justification text (short, cheap -- no large max_input_tokens needed
    # here, unlike pass 1/pass 2 above), one forward pass per order, same
    # calibrated-logit machinery as step4b/the removed step4a validation
    # pass. ---
    ap.add_argument("--coherence_candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Sourced/Journalist in the "
                         "pass-2 coherence check. Verify these with "
                         "scripts/check_step4b_logit_tokens.py before relying on them (same "
                         "tokenizer, same letters already checked for step4b/step3/the removed "
                         "step4a validation pass).")
    ap.add_argument("--coherence_batch_size", type=int, default=32)
    ap.add_argument("--coherence_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) coherence-check prompts resolved.")
    ap.add_argument("--coherence_threshold", type=float, default=0.5,
                    help="2026-11-18: decision midpoint for the pass-2 coherence check's "
                    "calibrated P(pass 2's own justification concludes SOURCED). Combined with "
                    "--coherence_margin below to decide whether to override pass 2's stated "
                    "answer -- see that argument's own help text. Starting value is the neutral "
                    "logit midpoint, NOT yet empirically tuned; this run's p_sourced distribution "
                    "is the first real data point for tuning it, same convention as every other "
                    "threshold in this project's history (job20255405 for step3, etc.) -- do not "
                    "grid-search a 'best' value on this same run's own data (circular).")
    ap.add_argument("--coherence_margin", type=float, default=0.15,
                    help="2026-11-18: minimum distance from --coherence_threshold the calibrated "
                    "p_sourced must reach, in the DISAGREEING direction, before pass 3 overrides "
                    "pass 2's stated answer -- e.g. pass 2 said SOURCED, but p_sourced_corrected "
                    "<= threshold-margin: override to JOURNALIST. A read that disagrees only "
                    "weakly (within the margin), or that fails outright (NaN), leaves pass 2's own "
                    "answer untouched -- fail-safe, same convention as every other "
                    "validation/verification pass in this codebase. NOT yet empirically tuned.")
    ap.add_argument("--skip_coherence", action="store_true",
                    help="2026-11-18: disable the pass-2 coherence check entirely -- no third LLM "
                    "call is made, and source_attribution stays exactly pass 2's own (or pass 1's, "
                    "if pass 2 was skipped/didn't touch the row) answer for every row.")

    # --- Pass "1.5" ("pass1 coherence check", 2026-11-24, logit-probing,
    # --backend litellm ONLY): see this file's own module docstring and
    # src/step4a_pass1_coherence_logit_prompts.py. Mirrors the pass-2
    # coherence check's own argument set above, applied one stage earlier,
    # directly to source_attribution_draft, BEFORE pass 2 runs -- see the
    # ordering rationale in src/step4a_pass1_coherence_logit_prompts.py's
    # docstring. Always a no-op (n_p1coh_checked=0) for --backend
    # vllm/transformers, regardless of --skip_pass1_coherence. ---
    ap.add_argument("--pass1_coherence_candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Sourced/Journalist in the "
                         "pass-1 coherence check. Same convention as --coherence_candidate_letters "
                         "above -- verify with scripts/check_step4b_logit_tokens.py before relying "
                         "on them.")
    ap.add_argument("--pass1_coherence_batch_size", type=int, default=32)
    ap.add_argument("--pass1_coherence_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) pass-1 coherence-check "
                         "prompts resolved.")
    ap.add_argument("--pass1_coherence_threshold", type=float, default=0.5,
                    help="2026-11-24: decision midpoint for the pass-1 coherence check's "
                    "calibrated P(pass 1's own justification concludes SOURCED). Combined with "
                    "--pass1_coherence_margin below to decide whether to correct "
                    "source_attribution_draft in place -- see that argument's own help text. "
                    "Starting value is the neutral logit midpoint, NOT yet empirically tuned -- "
                    "same convention as --coherence_threshold above (do not grid-search a 'best' "
                    "value on this same run's own data -- circular).")
    ap.add_argument("--pass1_coherence_margin", type=float, default=0.15,
                    help="2026-11-24: minimum distance from --pass1_coherence_threshold the "
                    "calibrated p_sourced must reach, in the DISAGREEING direction, before this "
                    "check corrects source_attribution_draft -- e.g. draft said SOURCED, but "
                    "p_sourced_corrected <= threshold-margin: correct to JOURNALIST. A read that "
                    "disagrees only weakly (within the margin), or that fails outright (NaN), "
                    "leaves the draft label untouched -- fail-safe, same convention as "
                    "--coherence_margin above. NOT yet empirically tuned.")
    ap.add_argument("--skip_pass1_coherence", action="store_true",
                    help="2026-11-24: disable the pass-1 coherence check entirely -- no extra LLM "
                    "call is made, and source_attribution_draft stays exactly pass 1's own answer "
                    "for every row. Irrelevant (always effectively on) for --backend "
                    "vllm/transformers, which never runs this check regardless of this flag -- see "
                    "this file's own module docstring.")

    # --- Internal ---
    ap.add_argument("--job_id",    default=None)
    ap.add_argument("--task_id",   type=int, default=None,
                    help="Array task index (0-indexed). If omitted, read from SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Total number of array tasks. If omitted, read from SLURM_ARRAY_TASK_COUNT.")

    args = ap.parse_args()
    coherence_candidate_letters = [s.strip() for s in args.coherence_candidate_letters.split(",") if s.strip()]
    if len(coherence_candidate_letters) != 2:
        raise ValueError(f"--coherence_candidate_letters must give exactly 2 letters, got {coherence_candidate_letters!r}")
    pass1_coherence_candidate_letters = [s.strip() for s in args.pass1_coherence_candidate_letters.split(",") if s.strip()]
    if len(pass1_coherence_candidate_letters) != 2:
        raise ValueError(f"--pass1_coherence_candidate_letters must give exactly 2 letters, got {pass1_coherence_candidate_letters!r}")

    # 2026-11-24: pick the prompt module for pass 1/pass 2 based on
    # --backend -- see the src/step4a_prompts_remote import above.
    # vllm/transformers (the production 8B/local-GPU path) keep using
    # src/step4a_prompts.py exactly as before, byte-for-byte unaffected.
    # Pass 3 (src/step4a_pass2_coherence_logit_prompts.py) is a separate,
    # SHARED file used identically regardless of backend -- see that
    # import's own comment above. Same pattern as
    # scripts/step3_criticism_detection.py.
    if args.backend == "litellm":
        active_system_prompt = SYSTEM_PROMPT_REMOTE
        active_build_user_prompt = build_user_prompt_remote
        active_verify_system_prompt = VERIFY_SYSTEM_PROMPT_REMOTE
        active_build_verify_prompt = build_verify_prompt_remote
        active_force_system_prompt = FORCE_SYSTEM_PROMPT_REMOTE
        active_build_force_prompt = build_force_prompt_remote
    else:
        active_system_prompt = SYSTEM_PROMPT
        active_build_user_prompt = build_user_prompt
        active_verify_system_prompt = VERIFY_SYSTEM_PROMPT
        active_build_verify_prompt = build_verify_prompt
        active_force_system_prompt = FORCE_SYSTEM_PROMPT
        active_build_force_prompt = build_force_prompt

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    # --- Load step 3 output (ALL rows: YES, NO, and unevaluated) ---
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

    expected_indices = set(df.index)
    eligible = int(build_mask(df, text_col=args.text_col).sum())
    print(f"[pipeline] {len(df):,} rows in chunk — {eligible:,} confirmed criticism "
          f"rows sent to the LLM (rest keep source_attribution = NA)")

    if task_id is not None:
        checkpoint_path = args.output_base + f"_task{task_id}_checkpoint.parquet"
    else:
        checkpoint_path = args.output_base + "_checkpoint.parquet"

    if Path(checkpoint_path).exists():
        ckpt = pd.read_parquet(checkpoint_path)
        if set(ckpt.index) != expected_indices:
            print(
                f"[resume] Checkpoint index mismatch "
                f"(checkpoint={len(ckpt)} rows, expected={len(expected_indices)}) — ignoring stale checkpoint.",
                flush=True,
            )
            Path(checkpoint_path).unlink()
            df = df.copy()
            for col in ALL_COLS:
                # source_attribution_pass2_coherence_p_sourced and
                # source_attribution_pass1_coherence_p_sourced each hold a
                # FLOAT (the calibrated P(sourced)), not a label -- same
                # dtype issue already hit (and fixed) for the removed
                # validation pass's own p_sourced column: pandas' strict
                # "string" dtype raises TypeError on a float assignment.
                # Every other column here is a label/flag and keeps the
                # "string" dtype.
                dtype = "object" if col in _FLOAT_COLS else "string"
                df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            df = ckpt
    else:
        for col in ALL_COLS:
            dtype = "object" if col in _FLOAT_COLS else "string"
            df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)

    # --- LLM client ---
    client = build_client(args)

    # --- Pass 1: draft (justification + first-guess label, from the article) ---
    draft_cfg = RunConfig(
        id_col="__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    df = run_llm_dataframe(
        df=df,
        cfg=draft_cfg,
        client=client,
        system_prompt=active_system_prompt,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: active_build_user_prompt(row, col),
        parse_fn=parse_draft_output,
        output_cols=DRAFT_COLS,
        skip_if_already_filled="source_attribution_draft",
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
    )

    # --- Pass 1 retries: a temperature=0.0 retry of the identical prompt
    # would just reproduce the identical failure, so each retry raises the
    # temperature (and token budget, in case truncation was the culprit).
    retry_temp = args.retry_temperature
    retry_tokens = args.max_new_tokens
    for attempt in range(1, args.draft_max_retries + 1):
        still_na = build_mask(df, text_col=args.text_col) & df["source_attribution_draft"].isna()
        n_remaining = int(still_na.sum())
        if n_remaining == 0:
            break
        retry_tokens = min(retry_tokens + 60, 300)
        print(
            f"[retry pass1] attempt {attempt}/{args.draft_max_retries}: "
            f"{n_remaining:,} eligible rows still without a draft answer -- "
            f"retrying at temperature={retry_temp:.2f}, max_new_tokens={retry_tokens}",
            flush=True,
        )
        retry_cfg = RunConfig(
            id_col="__index__",
            text_col=args.text_col,
            batch_size=args.batch_size,
            temperature=retry_temp,
            max_new_tokens=retry_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        df = run_llm_dataframe(
            df=df,
            cfg=retry_cfg,
            client=client,
            system_prompt=active_system_prompt,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["source_attribution_draft"].isna(),
            build_prompt_fn=lambda row, col: active_build_user_prompt(row, col),
            parse_fn=parse_draft_output,
            output_cols=DRAFT_COLS,
            skip_if_already_filled="source_attribution_draft",
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
        )
        retry_temp = min(retry_temp + 0.2, 0.9)

    # --- Last resort: a bare one-word forced decision for whatever is still
    # unresolved after every retry. No justification is requested (there is
    # nothing left to try to extract from this model on this row), so the
    # forced label is written straight to source_attribution and the row
    # never goes through pass 2 -- source_attribution_forced=True marks it
    # so it is never mistaken for a genuine two-pass-verified answer.
    still_na = build_mask(df, text_col=args.text_col) & df["source_attribution_draft"].isna()
    n_forced_candidates = int(still_na.sum())
    if n_forced_candidates:
        print(
            f"[force] {n_forced_candidates:,} eligible rows unresolved after "
            f"{args.draft_max_retries} retries -- forcing a bare one-word decision",
            flush=True,
        )

        def parse_force_output(raw: str) -> dict:
            label = parse_label_only(raw, VERIFY_LABELS)
            if pd.isna(label):
                return {}
            return {
                "source_attribution_draft": label,
                "source_attribution": label,
                "source_attribution_forced": "TRUE",
            }

        force_cfg = RunConfig(
            id_col="__index__",
            text_col=args.text_col,
            batch_size=args.batch_size,
            temperature=0.7,
            max_new_tokens=8,
            max_input_tokens=args.max_input_tokens,
        )
        df = run_llm_dataframe(
            df=df,
            cfg=force_cfg,
            client=client,
            system_prompt=active_force_system_prompt,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["source_attribution_draft"].isna(),
            build_prompt_fn=lambda row, col: active_build_force_prompt(row, col),
            parse_fn=parse_force_output,
            output_cols=ALL_COLS,
            skip_if_already_filled="source_attribution_draft",
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
        )

    still_na = build_mask(df, text_col=args.text_col) & df["source_attribution_draft"].isna()
    n_unresolved = int(still_na.sum())
    if n_unresolved:
        print(
            f"[force] WARNING: {n_unresolved:,} eligible rows still have no answer "
            f"even after the forced pass (source_attribution stays honestly NA for these).",
            flush=True,
        )

    # -------------------------------------------------------------------
    # Pass "1.5" ("pass1 coherence check", 2026-11-24, --backend litellm
    # ONLY): see this file's own module docstring and
    # src/step4a_pass1_coherence_logit_prompts.py for the full design
    # rationale. Runs BEFORE pass 2 and before source_attribution is filled
    # from source_attribution_draft, operating directly on the draft label
    # itself -- see that file's own docstring for why this ordering
    # matters (a correction here becomes a legitimate pass 2 candidate, or
    # is correctly excluded from it). Scoped to rows with a genuine pass 1
    # (a parsed draft label AND a real, non-bare justification) -- forced
    # one-word rows have no justification to read and are naturally
    # excluded. Gated STRICTLY to --backend litellm: this problem was
    # diagnosed only on the remote/70B path, and the production local-GPU
    # path (--backend vllm/transformers) must stay completely unaffected.
    # -------------------------------------------------------------------
    if args.backend != "litellm":
        n_p1coh_checked = 0
        n_p1coh_overridden = 0
        n_p1coh_inconclusive = 0
    elif args.skip_pass1_coherence:
        print("[pass1_coherence] --skip_pass1_coherence set -- pass-1 coherence check disabled. "
              "Keeping pass 1's own draft source_attribution for every row, no extra LLM call "
              "made.", flush=True)
        n_p1coh_checked = 0
        n_p1coh_overridden = 0
        n_p1coh_inconclusive = 0
    else:
        p1coh_mask = (
            build_mask(df, text_col=args.text_col)
            & df["source_attribution_draft"].notna()
            & df["source_attribution_justification"].notna()
        )
        p1coh_idx = df.index[p1coh_mask].tolist()
        n_p1coh_checked = len(p1coh_idx)
        print(f"[pass1_coherence] Checking {n_p1coh_checked:,} rows' pass-1 justification for "
              f"self-contradiction against pass 1's own stated draft answer "
              f"(--pass1_coherence_threshold={args.pass1_coherence_threshold}, "
              f"--pass1_coherence_margin={args.pass1_coherence_margin})...", flush=True)

        if task_id is not None:
            p1coherence_checkpoint_path = args.output_base + f"_task{task_id}_pass1coherence_checkpoint.parquet"
        else:
            p1coherence_checkpoint_path = args.output_base + "_pass1coherence_checkpoint.parquet"

        print("[pass1_coherence] Computing p0 (this question's own intrinsic sourced-bias) on "
              "the empty/neutral-input variant...", flush=True)
        p1_calib_prompts = [build_pass1_coherence_calibration_prompt(order) for order in (1, 2)]
        p1_calib_raw = client.logprob_many(
            system_prompt="",
            user_prompts=p1_calib_prompts,
            candidate_letters=pass1_coherence_candidate_letters,
            max_input_tokens=args.verify_max_input_tokens,
        )
        p1_p_sourced_by_order = {}
        for order, logits_dict in zip((1, 2), p1_calib_raw):
            probs = softmax_restricted(logits_dict)
            letter_sourced, _letter_journalist = pass1_letters_for_order(order)
            p1_p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
        p1_p0 = average_yes_over_orderings(p1_p_sourced_by_order[1], p1_p_sourced_by_order[2])
        print(f"  p0 = {p1_p0:.4f}", flush=True)

        p1coh_flat_keys = []    # (row_idx, order)
        p1coh_flat_prompts = []
        for row_idx in p1coh_idx:
            row = df.loc[row_idx]
            for order in (1, 2):
                p1coh_flat_keys.append((row_idx, order))
                p1coh_flat_prompts.append(build_pass1_coherence_prompt(row, order))

        print(f"[pass1_coherence] {len(p1coh_flat_prompts):,} total (row, order) coherence "
              f"prompts ({n_p1coh_checked:,} rows x 2 orders)", flush=True)

        p1coh_done = {}  # (row_idx, order) -> {letter: logit}
        if Path(p1coherence_checkpoint_path).exists():
            ckpt = pd.read_parquet(p1coherence_checkpoint_path)
            for _, r in ckpt.iterrows():
                vkey = (int(r["row_idx"]), int(r["order"]))
                p1coh_done[vkey] = {letter: r[f"logit_{letter}"] for letter in pass1_coherence_candidate_letters}
            print(f"[resume] Loaded {len(p1coh_done):,} already-resolved pass-1 coherence prompts "
                  f"from {p1coherence_checkpoint_path}", flush=True)

        p1coh_todo = [i for i, vkey in enumerate(p1coh_flat_keys) if vkey not in p1coh_done]
        print(f"[pass1_coherence] {len(p1coh_todo):,} coherence prompts remaining "
              f"({len(p1coh_flat_prompts) - len(p1coh_todo):,} already done)", flush=True)

        def _save_p1coherence_checkpoint():
            rows = []
            for (row_idx, order), logits_dict in p1coh_done.items():
                r = {"row_idx": row_idx, "order": order}
                for letter in pass1_coherence_candidate_letters:
                    r[f"logit_{letter}"] = logits_dict.get(letter, float("nan"))
                rows.append(r)
            ckpt_df = pd.DataFrame(rows)
            Path(p1coherence_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            ckpt_df.to_parquet(p1coherence_checkpoint_path, index=False)
            print(f"[checkpoint] Saved {len(ckpt_df):,} resolved pass-1 coherence prompts -> "
                  f"{p1coherence_checkpoint_path}", flush=True)

        n_since_p1coh_checkpoint = 0
        for start in range(0, len(p1coh_todo), args.pass1_coherence_batch_size):
            batch_flat_idx = p1coh_todo[start:start + args.pass1_coherence_batch_size]
            batch_prompts = [p1coh_flat_prompts[i] for i in batch_flat_idx]

            raw = client.logprob_many(
                system_prompt="",
                user_prompts=batch_prompts,
                candidate_letters=pass1_coherence_candidate_letters,
                max_input_tokens=args.verify_max_input_tokens,
            )
            for flat_idx, logits_dict in zip(batch_flat_idx, raw):
                p1coh_done[p1coh_flat_keys[flat_idx]] = logits_dict

            n_since_p1coh_checkpoint += len(batch_flat_idx)
            n_done_total = len(p1coh_flat_prompts) - (len(p1coh_todo) - start - len(batch_flat_idx))
            print(f"[pass1_coherence progress] {n_done_total:,}/{len(p1coh_flat_prompts):,} "
                  f"prompts resolved", flush=True)

            if n_since_p1coh_checkpoint >= args.pass1_coherence_checkpoint_every:
                _save_p1coherence_checkpoint()
                n_since_p1coh_checkpoint = 0

        if p1coh_todo:
            _save_p1coherence_checkpoint()

        n_p1coh_overridden = 0
        n_p1coh_inconclusive = 0
        for row_idx in p1coh_idx:
            p1_p_sourced_by_order = {}
            p1coh_any_nan = False
            for order in (1, 2):
                logits_dict = p1coh_done.get((row_idx, order))
                if logits_dict is None:
                    p1_p_sourced_by_order[order] = float("nan")
                    p1coh_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_sourced, _letter_journalist = pass1_letters_for_order(order)
                p1_p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
                if pd.isna(p1_p_sourced_by_order[order]):
                    p1coh_any_nan = True

            p1_p_sourced = average_yes_over_orderings(p1_p_sourced_by_order[1], p1_p_sourced_by_order[2])
            if pd.isna(p1_p_sourced) or pd.isna(p1_p0):
                p1coh_any_nan = True

            df.at[row_idx, "source_attribution_draft_before_pass1_coherence"] = df.at[row_idx, "source_attribution_draft"]

            if p1coh_any_nan:
                # Inconclusive read -- fail-safe: keep pass 1's own draft
                # answer, don't override on a signal we don't trust.
                n_p1coh_inconclusive += 1
                df.at[row_idx, "source_attribution_pass1_coherence_overridden"] = "FALSE"
                continue

            p1_p_sourced_corrected = calibrate(p1_p_sourced, p1_p0)
            df.at[row_idx, "source_attribution_pass1_coherence_p_sourced"] = p1_p_sourced_corrected

            draft_answer = df.at[row_idx, "source_attribution_draft"]
            low = args.pass1_coherence_threshold - args.pass1_coherence_margin
            high = args.pass1_coherence_threshold + args.pass1_coherence_margin
            overridden = False
            if draft_answer == "SOURCED" and p1_p_sourced_corrected <= low:
                df.at[row_idx, "source_attribution_draft"] = "JOURNALIST"
                overridden = True
            elif draft_answer == "JOURNALIST" and p1_p_sourced_corrected >= high:
                df.at[row_idx, "source_attribution_draft"] = "SOURCED"
                overridden = True

            df.at[row_idx, "source_attribution_pass1_coherence_overridden"] = "TRUE" if overridden else "FALSE"
            if overridden:
                n_p1coh_overridden += 1

        print(f"[pass1_coherence] {n_p1coh_overridden:,}/{n_p1coh_checked:,} rows overridden "
              f"(pass 1's draft answer disagreed clearly with its own justification) "
              f"({n_p1coh_inconclusive:,} reads inconclusive, pass 1's draft answer kept as-is "
              f"for those)", flush=True)

        if Path(p1coherence_checkpoint_path).exists():
            Path(p1coherence_checkpoint_path).unlink()
            print(f"[checkpoint] Deleted {p1coherence_checkpoint_path}")

    # Pass 1's own draft label is the final answer until/unless pass 2
    # (below) flips a JOURNALIST row to SOURCED. `source_attribution_forced`
    # rows already have `source_attribution` set from the force pass above
    # and are left untouched by this fill.
    df["source_attribution"] = df["source_attribution"].where(
        df["source_attribution"].notna(), df["source_attribution_draft"]
    )

    # -------------------------------------------------------------------
    # Pass 2 ("forced re-search", 2026-11-17): see this file's own module
    # docstring and src/step4a_prompts.py's VERIFY_SYSTEM_PROMPT/
    # build_verify_prompt for the full design rationale. Scoped ONLY to
    # rows pass 1 said JOURNALIST (the error this pass targets is FN-
    # heavy, not FP-heavy, so it can only ever flip JOURNALIST->SOURCED),
    # excluding forced rows (a forced row never went through a real pass 1
    # either, there is nothing to re-check). Re-reads the full article.
    # -------------------------------------------------------------------
    if args.skip_pass2:
        print("[pass2] --skip_pass2 set -- pass 2 disabled. Keeping pass 1's own (possibly "
              "forced) source_attribution for every row, no second LLM call made.", flush=True)
        n_pass2_flipped = 0
        n_pass2_unparsed = 0
        n_pass2_candidates = 0
    else:
        # Same .fillna(True) convention as step3/the removed validation pass:
        # a pyarrow-backed "string" NA on source_attribution_forced must read
        # as "not forced", not propagate to NA (which a boolean mask silently
        # treats as False, excluding every non-forced row).
        not_forced = (df["source_attribution_forced"] != "TRUE").fillna(True)
        pass2_mask = (
            build_mask(df, text_col=args.text_col)
            & (df["source_attribution"] == "JOURNALIST")
            & not_forced
        )
        n_pass2_candidates = int(pass2_mask.sum())
        print(f"[pass2] Re-checking pass 1's JOURNALIST verdict for {n_pass2_candidates:,} rows "
              f"against the full article...", flush=True)

        df.loc[pass2_mask, "source_attribution_before_pass2"] = "JOURNALIST"

        if task_id is not None:
            pass2_checkpoint_path = args.output_base + f"_task{task_id}_pass2_checkpoint.parquet"
        else:
            pass2_checkpoint_path = args.output_base + "_pass2_checkpoint.parquet"

        pass2_cfg = RunConfig(
            id_col="__index__",
            text_col=args.text_col,
            batch_size=args.verify_batch_size,
            temperature=args.verify_temperature,
            max_new_tokens=args.verify_max_new_tokens,
            max_input_tokens=args.verify_max_input_tokens,
        )
        df = run_llm_dataframe(
            df=df,
            cfg=pass2_cfg,
            client=client,
            system_prompt=active_verify_system_prompt,
            select_mask_fn=lambda df_, _m=pass2_mask: _m,
            build_prompt_fn=lambda row, col: active_build_verify_prompt(row, col),
            parse_fn=parse_pass2_output,
            output_cols=PASS2_COLS,
            skip_if_already_filled="source_attribution_pass2_answer",
            checkpoint_path=pass2_checkpoint_path,
            checkpoint_every=50,
        )

        flip_mask = pass2_mask & (df["source_attribution_pass2_answer"] == "SOURCED")
        n_pass2_flipped = int(flip_mask.sum())
        df.loc[flip_mask, "source_attribution"] = "SOURCED"

        unparsed_mask = pass2_mask & df["source_attribution_pass2_answer"].isna()
        n_pass2_unparsed = int(unparsed_mask.sum())

        print(f"[pass2] {n_pass2_flipped:,}/{n_pass2_candidates:,} rows flipped "
              f"(JOURNALIST->SOURCED) ({n_pass2_unparsed:,} pass-2 reads didn't parse, pass 1's "
              f"JOURNALIST kept as-is for those -- no retry)", flush=True)

        if Path(pass2_checkpoint_path).exists():
            Path(pass2_checkpoint_path).unlink()
            print(f"[checkpoint] Deleted {pass2_checkpoint_path}")

    # -------------------------------------------------------------------
    # Pass 3 ("pass2 coherence check", 2026-11-18): see this file's own
    # module docstring and src/step4a_pass2_coherence_logit_prompts.py for
    # the full design rationale. Scoped to every row pass 2 produced a
    # parseable answer for (both flips AND confirms -- self-contradiction
    # is possible in either direction in principle, even though only the
    # SOURCED direction was observed on job20620441). Naturally empty if
    # --skip_pass2 was set or pass 2 touched nothing. Reads ONLY pass 2's
    # own justification text, never the article.
    # -------------------------------------------------------------------
    if args.skip_coherence:
        print("[coherence] --skip_coherence set -- pass-2 coherence check disabled. Keeping "
              "pass 2's own (or pass 1's) source_attribution for every row, no third LLM call "
              "made.", flush=True)
        n_coh_checked = 0
        n_coh_overridden = 0
        n_coh_inconclusive = 0
    else:
        coherence_mask = (
            build_mask(df, text_col=args.text_col)
            & df["source_attribution_pass2_answer"].notna()
        )
        coherence_idx = df.index[coherence_mask].tolist()
        n_coh_checked = len(coherence_idx)
        print(f"[coherence] Checking {n_coh_checked:,} rows' pass-2 justification for "
              f"self-contradiction against pass 2's own stated answer "
              f"(--coherence_threshold={args.coherence_threshold}, "
              f"--coherence_margin={args.coherence_margin})...", flush=True)

        if task_id is not None:
            coherence_checkpoint_path = args.output_base + f"_task{task_id}_pass2coherence_checkpoint.parquet"
        else:
            coherence_checkpoint_path = args.output_base + "_pass2coherence_checkpoint.parquet"

        print("[coherence] Computing p0 (this question's own intrinsic sourced-bias) on the "
              "empty/neutral-input variant...", flush=True)
        calib_prompts = [build_coherence_calibration_prompt(order) for order in (1, 2)]
        calib_raw = client.logprob_many(
            system_prompt="",
            user_prompts=calib_prompts,
            candidate_letters=coherence_candidate_letters,
            max_input_tokens=args.verify_max_input_tokens,
        )
        p_sourced_by_order = {}
        for order, logits_dict in zip((1, 2), calib_raw):
            probs = softmax_restricted(logits_dict)
            letter_sourced, _letter_journalist = letters_for_order(order)
            p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
        p0 = average_yes_over_orderings(p_sourced_by_order[1], p_sourced_by_order[2])
        print(f"  p0 = {p0:.4f}", flush=True)

        coh_flat_keys = []    # (row_idx, order)
        coh_flat_prompts = []
        for row_idx in coherence_idx:
            row = df.loc[row_idx]
            for order in (1, 2):
                coh_flat_keys.append((row_idx, order))
                coh_flat_prompts.append(build_coherence_prompt(row, order))

        print(f"[coherence] {len(coh_flat_prompts):,} total (row, order) coherence prompts "
              f"({n_coh_checked:,} rows x 2 orders)", flush=True)

        coh_done = {}  # (row_idx, order) -> {letter: logit}
        if Path(coherence_checkpoint_path).exists():
            ckpt = pd.read_parquet(coherence_checkpoint_path)
            for _, r in ckpt.iterrows():
                vkey = (int(r["row_idx"]), int(r["order"]))
                coh_done[vkey] = {letter: r[f"logit_{letter}"] for letter in coherence_candidate_letters}
            print(f"[resume] Loaded {len(coh_done):,} already-resolved coherence prompts from "
                  f"{coherence_checkpoint_path}", flush=True)

        coh_todo = [i for i, vkey in enumerate(coh_flat_keys) if vkey not in coh_done]
        print(f"[coherence] {len(coh_todo):,} coherence prompts remaining "
              f"({len(coh_flat_prompts) - len(coh_todo):,} already done)", flush=True)

        def _save_coherence_checkpoint():
            rows = []
            for (row_idx, order), logits_dict in coh_done.items():
                r = {"row_idx": row_idx, "order": order}
                for letter in coherence_candidate_letters:
                    r[f"logit_{letter}"] = logits_dict.get(letter, float("nan"))
                rows.append(r)
            ckpt_df = pd.DataFrame(rows)
            Path(coherence_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            ckpt_df.to_parquet(coherence_checkpoint_path, index=False)
            print(f"[checkpoint] Saved {len(ckpt_df):,} resolved coherence prompts -> "
                  f"{coherence_checkpoint_path}", flush=True)

        n_since_coh_checkpoint = 0
        for start in range(0, len(coh_todo), args.coherence_batch_size):
            batch_flat_idx = coh_todo[start:start + args.coherence_batch_size]
            batch_prompts = [coh_flat_prompts[i] for i in batch_flat_idx]

            raw = client.logprob_many(
                system_prompt="",
                user_prompts=batch_prompts,
                candidate_letters=coherence_candidate_letters,
                max_input_tokens=args.verify_max_input_tokens,
            )
            for flat_idx, logits_dict in zip(batch_flat_idx, raw):
                coh_done[coh_flat_keys[flat_idx]] = logits_dict

            n_since_coh_checkpoint += len(batch_flat_idx)
            n_done_total = len(coh_flat_prompts) - (len(coh_todo) - start - len(batch_flat_idx))
            print(f"[coherence progress] {n_done_total:,}/{len(coh_flat_prompts):,} prompts "
                  f"resolved", flush=True)

            if n_since_coh_checkpoint >= args.coherence_checkpoint_every:
                _save_coherence_checkpoint()
                n_since_coh_checkpoint = 0

        if coh_todo:
            _save_coherence_checkpoint()

        n_coh_overridden = 0
        n_coh_inconclusive = 0
        for row_idx in coherence_idx:
            p_sourced_by_order = {}
            coh_any_nan = False
            for order in (1, 2):
                logits_dict = coh_done.get((row_idx, order))
                if logits_dict is None:
                    p_sourced_by_order[order] = float("nan")
                    coh_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_sourced, _letter_journalist = letters_for_order(order)
                p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
                if pd.isna(p_sourced_by_order[order]):
                    coh_any_nan = True

            p_sourced = average_yes_over_orderings(p_sourced_by_order[1], p_sourced_by_order[2])
            if pd.isna(p_sourced) or pd.isna(p0):
                coh_any_nan = True

            if coh_any_nan:
                # Inconclusive read -- fail-safe: keep pass 2's own stated
                # answer, don't override on a signal we don't trust.
                n_coh_inconclusive += 1
                df.at[row_idx, "source_attribution_pass2_overridden"] = "FALSE"
                continue

            p_sourced_corrected = calibrate(p_sourced, p0)
            df.at[row_idx, "source_attribution_pass2_coherence_p_sourced"] = p_sourced_corrected

            pass2_answer = df.at[row_idx, "source_attribution_pass2_answer"]
            low = args.coherence_threshold - args.coherence_margin
            high = args.coherence_threshold + args.coherence_margin
            overridden = False
            if pass2_answer == "SOURCED" and p_sourced_corrected <= low:
                df.at[row_idx, "source_attribution"] = "JOURNALIST"
                overridden = True
            elif pass2_answer == "JOURNALIST" and p_sourced_corrected >= high:
                df.at[row_idx, "source_attribution"] = "SOURCED"
                overridden = True

            df.at[row_idx, "source_attribution_pass2_overridden"] = "TRUE" if overridden else "FALSE"
            if overridden:
                n_coh_overridden += 1

        print(f"[coherence] {n_coh_overridden:,}/{n_coh_checked:,} rows overridden "
              f"(pass 2's stated answer disagreed clearly with its own justification) "
              f"({n_coh_inconclusive:,} reads inconclusive, pass 2's answer kept as-is for "
              f"those)", flush=True)

        if Path(coherence_checkpoint_path).exists():
            Path(coherence_checkpoint_path).unlink()
            print(f"[checkpoint] Deleted {coherence_checkpoint_path}")

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

    sourced    = int((df["source_attribution"] == "SOURCED").sum())
    journalist = int((df["source_attribution"] == "JOURNALIST").sum())
    na_count   = int(df["source_attribution"].isna().sum())
    forced     = int((df["source_attribution_forced"] == "TRUE").sum())
    both = df["source_attribution_draft"].notna() & df["source_attribution"].notna() & df["source_attribution_justification"].notna()
    disagree = int((df.loc[both, "source_attribution_draft"] != df.loc[both, "source_attribution"]).sum())
    print(
        f"Saved: {parquet_path} | {len(df):,} rows total "
        f"(source_attribution: {sourced:,} SOURCED / {journalist:,} JOURNALIST / {na_count:,} NA | "
        f"draft/final disagreement: {disagree:,}/{int(both.sum()):,} | "
        f"forced (no real verification): {forced:,} | "
        f"pass1 coherence: {n_p1coh_overridden:,}/{n_p1coh_checked:,} overridden, "
        f"{n_p1coh_inconclusive:,} inconclusive | "
        f"pass2: {n_pass2_flipped:,}/{n_pass2_candidates:,} flipped, "
        f"{n_pass2_unparsed:,} unparsed | "
        f"pass3 coherence: {n_coh_overridden:,}/{n_coh_checked:,} overridden, "
        f"{n_coh_inconclusive:,} inconclusive)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    # 2026-11-24: usage telemetry for the remote (--backend litellm) client --
    # pure observability, unrelated to this run's actual results. The local
    # TransformersClient has no print_stats(), so this is a no-op for the
    # GPU path -- see src/client_remote.py's module docstring. Same addition
    # as scripts/step3_criticism_detection.py.
    if hasattr(client, "print_stats"):
        client.print_stats()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
