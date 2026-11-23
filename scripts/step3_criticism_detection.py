"""
scripts/step3_criticism_detection.py

STEP 3 of the accountability pipeline — first LLM run: criticism detection.

Renamed from the legacy scripts/run3_pipeline.py. Structurally: the legacy
run3 received a not-yet-exploded, one-row-per-article file and exploded it
into one row per (article, keyword) itself (splitting a pipe-separated
`matched_keywords` column). That exploding — plus alias canonicalisation,
target typing and the councillor in-office filter — now happens once in
step 2, so step 3's input is already one row per (article, target) with a
single `keyword` value per row. This script just runs the LLM over that.

The prompt itself was revised after a real full run showed both a missing
justification and a tendency to flag the target as criticised when it was
actually the one doing the criticising, or only mentioned in passing (see
src/step3_prompts.py for the fix). The model now returns a one-sentence
justification BEFORE the YES/NO label.

Two-pass design (added after the justification-first run showed the label
sometimes contradicting its own justification — see src/verify_utils.py).

2026-10-13: pass 2 removed. Comparing pass-1-draft alone against the final
(pass-2-verified) answer across every recent gold-compared run showed pass 2
contributes little and is sometimes net negative -- on job65556910 (the
best full-800-row run so far), pass 2 touched only 43/800 rows (5.4%), and
of its 30 NO->YES overrides, 24 were wrong, leaving pass-1-alone's own
accuracy (81.6%) and F1 (66.4%) both HIGHER than the full two-pass
pipeline's (80.5% / 65.6%) on that same run. See the 2026-10-13 entry in
src/step3_prompts.py for the full writeup. Pass 1's own draft label is now
written straight to the final column below, with no second LLM call:

  Pass 1 (draft)  — reads the article, writes the justification, then a
    first-guess label right after it. Both are kept: the justification
    under `keyword_justification`, the first-guess label under
    `keyword_answer_draft`. `keyword_answer_draft` is then copied straight
    to `keyword_answer` (see the end of main(), just before Save).
  Pass 2 (verify) — no longer called. VERIFY_SYSTEM_PROMPT/build_verify_prompt
    and the --verify_* CLI args below are kept, unused, in case pass 2 is
    revisited later.

2026-11-02 "validate-on-full-article" addition: a deep manual read of
job66790113's 148 step3 errors (gold corrected) found a heavily FP-skewed
pattern -- 109 false positives against 39 false negatives, concentrated on
named Federal Councillors, with every read FP justification showing the
same root cause (pass 1 treats any negative-valence mention near the named
entity as criticism of it personally, even when fault belongs elsewhere).
Rather than another round of pass-1 prompt tuning (a dozen rounds already
tried and reverted/net-negative, see src/step3_prompts.py), this adds a
NEW, separate validation pass -- see src/step3_logit_prompts.py's module
docstring for the full design rationale. It is deliberately NOT the old
removed pass 2: it re-reads the FULL ARTICLE (genuinely new information,
not pass 1's own justification text), it is scoped ONLY to rows pass 1
said YES to (can only ever flip YES->NO, never create a new false
negative), and it uses step4b's proven logit-probing mechanism (one
forward pass, order-swap, context-free calibration) instead of generation.
Forced rows (keyword_answer_forced == "TRUE") skip this pass entirely,
same convention as the old pass 2. Toggle with --skip_validation (see
sbatch_step3_array.sh's SKIP_VALIDATION).

2026-11-06: added --validation_input (article|justification) after the
article-mode pass measured net-negative on job20002898 -- see
src/step3_logit_prompts.py's module docstring for the full diagnosis and
the justification-mode ablation it motivated.

2026-11-09: added --validation_threshold (default 0.30, was hardcoded to
0.5) after job20255405 (--validation_input=justification) showed a real,
correctly-directed signal (p_true 0.464 mean for gold-YES vs 0.374 for
gold-NO) that a 0.5 cutoff nonetheless over-rejected -- see the
--validation_threshold argparse help just below for the full numbers.
0.30 was chosen as a round value on that same job's 235 candidates, not
validated out-of-sample -- the next run with this default is that check.

2026-11-09 (later same day): --validation_input's default changed from
"article" to "justification". job20310461 accidentally re-ran in "article"
mode (the sbatch call omitted VALIDATION_INPUT=justification, so it
silently fell back to the old default) and reproduced the exact same
near-total-rejection failure as job20002898 (0/237 kept, step3 accuracy
77.7%, below the 78.7% naive-NO baseline) -- a costly way to relearn that
"article" is the broken mode. Making "justification" the default removes
that footgun: a bare `sbatch sbatch_step3_array.sh <input>` with no env
vars now runs the mode that has actually measured net-positive.

2026-11-13: added ADMIN_UNIT_CAUTION (src/step3_logit_prompts.py), an
extra prompt paragraph included whenever a candidate row's target_type
isn't "Federal Councillor", targeting exactly this project's own-
diagnosed FP concentration on bureaucratic/institutional targets (see
that file's module docstring). REVERTED the same day: deployed on
job20496316 and checked with a controlled paired comparison against
job20479169 (same 234 article+keyword pairs that were YES-candidates in
both runs) -- for the 119 of those rows with target_type != "Federal
Councillor" (i.e. that should have received the caution), mean
keyword_answer_validation_p_true went UP, 0.532 -> 0.582, not down (only
39/119, 33%, moved the intended direction). Globally FP rose 57->63 and
the Administrative-Unit over-representation among errors got worse
(1.45x->1.65x), for only a marginal F1 gain (0.594->0.600) bought with a
precision loss (0.625->0.611) -- the wrong trade given this project's
explicit precision-over-recall preference. The caution, though correctly
wired (mock-verified pre-deployment), measurably failed to make the real
model more skeptical on these targets. Fully reverted -- single p0
calibration restored below, ADMIN_UNIT_CAUTION/needs_admin_caution removed
from src/step3_logit_prompts.py.

2026-11-24: --validation_scope (default "yes_only", byte-identical to every
prior run) can be set to "all" to additionally run this SAME validation
question against every non-forced NO row's own keyword_justification, not
just YES rows -- see --validation_scope's own argparse help for the full
rationale (job20870226's p_true overlap finding + the self-contradiction
read) and --validation_threshold_no_to_yes for the (separate, opt-in)
NO->YES promotion cutoff. Nothing in src/step3_logit_prompts.py is touched
or forked for this -- it is purely a scoping change in this script, off by
default.

Output columns: keyword_answer                   -> "YES" | "NO" | NA   (FINAL -- possibly
                                                       flipped to NO by the validation pass, or,
                                                       with --validation_scope=all and
                                                       --validation_threshold_no_to_yes set,
                                                       promoted from NO to YES)
                keyword_answer_draft              -> "YES" | "NO" | NA   (pass 1)
                keyword_justification             -> free text | NA      (pass 1)
                keyword_answer_before_validation  -> "YES" | "NO" | NA  (pass 1's own verdict,
                                                       before the validation pass touched it --
                                                       only set for rows the validation pass
                                                       actually looked at; "NO" only occurs with
                                                       --validation_scope=all)
                keyword_answer_validation_p_true  -> calibrated P(the justification supports a
                                                       YES verdict), or NA -- computed the same
                                                       way regardless of whether pass 1 said YES
                                                       or NO
                keyword_answer_validated          -> "TRUE" | "FALSE" | NA (TRUE iff the FINAL
                                                       keyword_answer for this row is "YES" after
                                                       validation; NA = not eligible for
                                                       validation, or an inconclusive read --
                                                       fail-safe: pass 1's own answer is kept
                                                       either way)

Usage
-----
  python scripts/step3_criticism_detection.py \\
      --input       data/processed/step2_clean.parquet \\
      --output_base data/output/step3 \\
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \\
      --dtype bf16 --batch_size 4 --temperature 0.0 \\
      --max_new_tokens 80 --max_input_tokens 16384 \\
      --verify_batch_size 16 --verify_max_new_tokens 8 --verify_max_input_tokens 512
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
from src.step3_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
    FORCE_SYSTEM_PROMPT, build_force_prompt,
)
# 2026-11-19: --backend litellm (remote API models via the UNIL/CSCS proxy,
# e.g. the Apertus-70B family) uses its OWN prompt module
# (src/step3_prompts_remote.py), selected right after argparse below. This
# keeps any prompt experiment on the remote backend from ever touching the
# src/step3_prompts.py import above, which the production vllm/transformers
# (8B, local GPU) path depends on untouched -- see
# src/step3_prompts_remote.py's module docstring for why and what, if
# anything, differs from it.
from src.step3_prompts_remote import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_REMOTE, build_user_prompt as build_user_prompt_remote,
    FORCE_SYSTEM_PROMPT as FORCE_SYSTEM_PROMPT_REMOTE, build_force_prompt as build_force_prompt_remote,
)
from src.step3_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled
from src.step3_logit_prompts import (
    build_validation_prompt, build_validation_calibration_prompt,
    build_validation_prompt_from_justification, build_validation_calibration_prompt_from_justification,
    letters_for_order,
)
from src.step4b_logit_utils import softmax_restricted, average_yes_over_orderings, calibrate

DRAFT_COLS = ["keyword_answer_draft", "keyword_justification"]
FINAL_COLS = ["keyword_answer"]
# Flag column: "TRUE" when a row's final answer came from the last-resort
# forced one-word pass rather than a genuine two-pass verification -- see
# the retry/force block in main(). NA for every normally-answered row.
FLAG_COLS = ["keyword_answer_forced"]
# 2026-11-02 "validate-on-full-article" pass -- see src/step3_logit_prompts.py
# and this file's own module docstring.
VALIDATION_COLS = [
    "keyword_answer_before_validation",
    "keyword_answer_validation_p_true",
    "keyword_answer_validated",
]
ALL_COLS = DRAFT_COLS + FINAL_COLS + FLAG_COLS + VALIDATION_COLS

VERIFY_LABELS = {"YES": "YES", "NO": "NO"}


def parse_draft_output(raw: str) -> dict:
    """A bare label with no real justification is a full failure, not a
    partial success -- see src/verify_utils.py:parse_draft_labeled."""
    empty = {"keyword_answer_draft": pd.NA, "keyword_justification": pd.NA}
    label, justification = parse_draft_labeled(raw, VERIFY_LABELS)
    if label is None:
        return empty
    return {"keyword_answer_draft": label, "keyword_justification": justification}


def parse_verify_output(raw: str) -> dict:
    return {"keyword_answer": parse_label_only(raw, VERIFY_LABELS)}


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True, help="Step 2 output: parquet or csv")
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="text")
    ap.add_argument("--n_rows",       type=int, default=0)

    # --- Model ---
    # 2026-11-18: --backend "litellm" added alongside "vllm"/"transformers" --
    # sends requests to an OpenAI-compatible gateway (src/client_remote.py)
    # instead of loading a model locally, e.g. to test swiss-ai/Apertus-v1.5-70B
    # hosted at CSCS via UNIL's litellm proxy on a plain CPU allocation. For
    # this backend, --model_path doubles as the remote model identifier (e.g.
    # "swiss-ai/Apertus-v1.5-70B") rather than a local filesystem path, and
    # --dtype/--trust_remote_code are simply ignored. See src/client.py's
    # build_client() and src/client_remote.py's module docstring.
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
                         "there is no local GPU batch to size. No rate-limit guidance was given "
                         "by UNIL/CSCS for this key as of 2026-11-18, so this default errs "
                         "conservative; raise it once you know what the service can take.")

    # --- Inference: pass 1 (draft, reads the full article) ---
    ap.add_argument("--batch_size",        required=True, type=int)
    ap.add_argument("--temperature",       required=True, type=float)
    ap.add_argument("--max_new_tokens",    required=True, type=int)
    ap.add_argument("--max_input_tokens",  required=True, type=int)

    # --- Pass 1 retries + last-resort force: every eligible row is
    # supposed to get an answer here. temperature=0.0 retried unchanged
    # would just reproduce the same failure, so each retry raises
    # temperature and token budget a notch; whatever is still unresolved
    # after --draft_max_retries gets one forced bare-word attempt (flagged
    # via keyword_answer_forced, never sent through pass 2). Mirrors
    # step4a's own retry/force mechanism exactly. ---
    ap.add_argument("--draft_max_retries",   type=int,   default=2,
                    help="Extra pass-1 attempts (beyond the first) for eligible rows still unanswered, each at a higher temperature/token budget.")
    ap.add_argument("--retry_temperature",   type=float, default=0.4,
                    help="Temperature for the first retry attempt; raised by +0.2 per further attempt, capped at 0.9.")

    # --- Inference: pass 2 (verify, reads only the justification) ---
    ap.add_argument("--verify_batch_size",       type=int,   default=16)
    ap.add_argument("--verify_temperature",      type=float, default=0.0)
    ap.add_argument("--verify_max_new_tokens",   type=int,   default=8)
    ap.add_argument("--verify_max_input_tokens", type=int,   default=512)

    # --- Validation pass (2026-11-02 "validate-on-full-article", logit-
    # probing): no temperature/max_new_tokens -- a single forward pass
    # reads next-token logits, nothing is sampled. batch_size counts
    # individual (row, order) PROMPTS, not rows -- each validated row
    # produces 2 prompts (order-swap). Reuses --max_input_tokens from
    # pass 1 above (same article text, same truncation budget needed). ---
    ap.add_argument("--candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Yes/No in the validation "
                         "pass. Verify these with scripts/check_step4b_logit_tokens.py before "
                         "relying on them (same tokenizer, same letters already checked for "
                         "step4b).")
    ap.add_argument("--validation_batch_size", type=int, default=32)
    ap.add_argument("--validation_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) validation prompts resolved.")
    ap.add_argument("--validation_input", default="justification", choices=["article", "justification"],
                    help="2026-11-06: what the validation pass re-checks pass 1's YES verdict "
                    "against. 'article' re-reads the full raw article -- this measured net-"
                    "negative on job20002898 (122 new FN vs only 96 fixed FP; step3 accuracy "
                    "fell BELOW the naive-NO baseline), with raw P(yes) crushed to ~0.17 even "
                    "for true positives despite a non-degenerate calibration baseline "
                    "(p0=0.3932) -- see src/step3_logit_prompts.py's module docstring for the "
                    "full diagnosis. 'justification' (DEFAULT since 2026-11-09, after job20310461 "
                    "accidentally re-ran 'article' mode by omitting this flag and reproduced that "
                    "same near-total-rejection failure almost exactly -- 0/237 kept, accuracy "
                    "77.7%, again below baseline) instead reads only pass 1's own "
                    "keyword_justification (shorter, already-focused on the claimed criticism, "
                    "no real-article noise/distractors to re-locate), with the question reframed "
                    "as an explicit validation request rather than confirm/reject-a-prior-"
                    "conclusion. Not a blind repeat of the pre-2026-10-13 removed pass 2: still "
                    "scoped to YES rows only, still logit-probing with order-swap + context-free "
                    "calibration, not free-text generation on every row. job20255405 confirmed "
                    "this mode gives a real, correctly-directed signal (p_true 0.464 mean for "
                    "gold-YES vs 0.374 for gold-NO) -- see --validation_threshold for the "
                    "cutoff tuning that goes with it.")
    ap.add_argument("--validation_threshold", type=float, default=0.30,
                    help="2026-11-09: decision cutoff on keyword_answer_validation_p_true -- "
                    "p_true >= threshold keeps pass 1's YES, otherwise flips it to NO. Was "
                    "hardcoded to 0.5 (never actually tuned, just the natural logit midpoint). "
                    "job20255405 (--validation_input=justification) showed p_true genuinely "
                    "separates gold-YES from gold-NO rows (mean 0.464 vs 0.374) but BOTH sit "
                    "below 0.5, so a 0.5 cutoff over-rejects: step3 recall collapsed 76.8%->32.7% "
                    "for only a 54.9%->64.7% precision gain (F1 0.640->0.516, net negative vs no "
                    "validation at all). Re-scoring that same job's 235 candidates at threshold=0.30 "
                    "gives accuracy 87.7%/F1 0.671, beating the no-validation baseline (81.6%/0.640) "
                    "-- 48 of 106 original FPs corrected for only 35 TPs lost, a clearly favourable "
                    "trade vs 0.5's wash. Not yet confirmed out-of-sample (0.30 was picked as a "
                    "round, non-cherry-picked value on that same job, not a withheld validation "
                    "set) -- this run is exactly that out-of-sample check. See "
                    "src/step3_logit_prompts.py's module docstring for the full writeup.")
    ap.add_argument("--skip_validation", action="store_true",
                    help="2026-11-02: disable the validation pass entirely -- no validation "
                    "forward passes are run, and keyword_answer stays exactly pass 1's own "
                    "(possibly forced) answer for every row. Kept as a flag from day one, "
                    "having learned from step4b's pass 3 that a validation pass needs to be "
                    "easy to turn off without touching code if it measures net-negative.")
    ap.add_argument("--validation_scope", default="yes_only", choices=["yes_only", "all"],
                    help="2026-11-24: which pass-1 rows the validation pass re-checks. "
                    "'yes_only' (DEFAULT, byte-identical to every prior run) scopes it to rows "
                    "pass 1 said YES to, exactly as designed on 2026-11-02 -- see this file's "
                    "own module docstring and src/step3_logit_prompts.py's for why that scoping "
                    "was a deliberate choice, not an oversight (a check that can only ever turn "
                    "YES into NO cannot create a new false negative on a row pass 1 already got "
                    "right). 'all' additionally re-checks every non-forced NO row's own "
                    "keyword_justification the same way (same question, same logit-probing/"
                    "order-swap/context-free-calibration machinery, same src/step3_logit_prompts.py "
                    "content -- nothing in that file is touched or forked for this), recording "
                    "keyword_answer_validation_p_true for those rows too. Added at Maxime's "
                    "request after job20870226 (--backend litellm, the broadened remote prompt) "
                    "showed the validation pass's own p_true could not separate the 34 true "
                    "positives it wrongly flipped to NO from the 46 false positives it correctly "
                    "flipped (near-identical p_true distributions on the YES side), while a "
                    "qualitative read of the 43 remaining false negatives found a third of them "
                    "wrote a justification that already argues a YES case; 'all' lets the SAME "
                    "strict, narrow classic definition this pass has always used act as a second, "
                    "independent, skeptical read on those NO rows too, rather than treating pass "
                    "1's NO as automatically final. Whether a NO row is actually PROMOTED to YES "
                    "is controlled separately by --validation_threshold_no_to_yes -- by default "
                    "'all' only RECORDS p_true for NO rows without flipping anything, so the "
                    "right cutoff can be chosen by sweeping this run's own p_true against gold "
                    "afterwards, the same way --validation_threshold's 0.30 was originally picked, "
                    "instead of guessing a value upfront.")
    ap.add_argument("--validation_threshold_no_to_yes", type=float, default=None,
                    help="2026-11-24: decision cutoff on keyword_answer_validation_p_true for "
                    "rows pass 1 answered NO (only consulted when --validation_scope=all). "
                    "p_true >= this value promotes the row to YES; otherwise it stays NO. Left "
                    "unset (None) by default -- in that case 'all' scope still computes and saves "
                    "p_true for every NO row (for later offline threshold-sweeping against gold), "
                    "but never promotes anything, so a run with --validation_scope=all and no "
                    "value here changes NOTHING about the final keyword_answer column vs "
                    "--validation_scope=yes_only; it only adds data. Set this only after that "
                    "sweep has picked a value -- do not guess one a priori (see "
                    "--validation_scope's own help for why).")

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

    # 2026-11-19: pick the prompt module for pass 1 (and its force fallback)
    # based on --backend -- see the src/step3_prompts_remote import above.
    # vllm/transformers (the production 8B/local-GPU path) keep using
    # src/step3_prompts.py exactly as before, byte-for-byte unaffected.
    if args.backend == "litellm":
        active_system_prompt = SYSTEM_PROMPT_REMOTE
        active_build_user_prompt = build_user_prompt_remote
        active_force_system_prompt = FORCE_SYSTEM_PROMPT_REMOTE
        active_build_force_prompt = build_force_prompt_remote
    else:
        active_system_prompt = SYSTEM_PROMPT
        active_build_user_prompt = build_user_prompt
        active_force_system_prompt = FORCE_SYSTEM_PROMPT
        active_build_force_prompt = build_force_prompt

    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    # --- Load step 2 output (already one row per article-target) ---
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
    print(f"[pipeline] {len(df):,} (article, target) rows in chunk — "
          f"{eligible:,} eligible for criticism detection", flush=True)

    if task_id is not None:
        checkpoint_path = args.output_base + f"_task{task_id}_checkpoint.parquet"
    else:
        checkpoint_path = args.output_base + "_checkpoint.parquet"

    # --- Load or create the working dataframe (ALL rows kept, not just
    # eligible ones — ineligible rows simply keep keyword_answer = NA) ---
    if Path(checkpoint_path).exists():
        ckpt = pd.read_parquet(checkpoint_path)
        if set(ckpt.index) != expected_indices:
            print(
                f"[resume] Checkpoint index mismatch "
                f"(checkpoint={len(ckpt)} rows, expected={len(expected_indices)}) — ignoring stale checkpoint.",
                flush=True,
            )
            Path(checkpoint_path).unlink()
            working = df.copy()
            for col in ALL_COLS:
                # 2026-11-05: keyword_answer_validation_p_true holds a FLOAT
                # (the calibrated P(yes)), not a label -- pandas' strict
                # "string" dtype raises TypeError on a float assignment
                # (caught in testing before this ever reached production;
                # see the validation-pass block below). Every other column
                # here is a label/flag and keeps the original "string" dtype.
                dtype = "object" if col == "keyword_answer_validation_p_true" else "string"
                working[col] = pd.Series(pd.NA, index=working.index, dtype=dtype)
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            working = ckpt
    else:
        working = df.copy()
        for col in ALL_COLS:
            dtype = "object" if col == "keyword_answer_validation_p_true" else "string"
            working[col] = pd.Series(pd.NA, index=working.index, dtype=dtype)

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
    working = run_llm_dataframe(
        df=working,
        cfg=draft_cfg,
        client=client,
        system_prompt=active_system_prompt,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: active_build_user_prompt(row, col),
        parse_fn=parse_draft_output,
        output_cols=DRAFT_COLS,
        skip_if_already_filled="keyword_answer_draft",
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
    )

    # --- Pass 1 retries: a temperature=0.0 retry of the identical prompt
    # would just reproduce the identical failure, so each retry raises the
    # temperature (and token budget, in case truncation was the culprit).
    retry_temp = args.retry_temperature
    retry_tokens = args.max_new_tokens
    for attempt in range(1, args.draft_max_retries + 1):
        still_na = build_mask(working, text_col=args.text_col) & working["keyword_answer_draft"].isna()
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
        working = run_llm_dataframe(
            df=working,
            cfg=retry_cfg,
            client=client,
            system_prompt=active_system_prompt,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["keyword_answer_draft"].isna(),
            build_prompt_fn=lambda row, col: active_build_user_prompt(row, col),
            parse_fn=parse_draft_output,
            output_cols=DRAFT_COLS,
            skip_if_already_filled="keyword_answer_draft",
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
        )
        retry_temp = min(retry_temp + 0.2, 0.9)

    # --- Last resort: a bare one-word forced decision for whatever is still
    # unresolved after every retry. No justification is requested (there is
    # nothing left to try to extract from this model on this row), so the
    # forced label is written straight to keyword_answer and the row never
    # goes through pass 2 -- keyword_answer_forced=True marks it so it is
    # never mistaken for a genuine two-pass-verified answer.
    still_na = build_mask(working, text_col=args.text_col) & working["keyword_answer_draft"].isna()
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
                "keyword_answer_draft": label,
                "keyword_answer": label,
                "keyword_answer_forced": "TRUE",
            }

        force_cfg = RunConfig(
            id_col="__index__",
            text_col=args.text_col,
            batch_size=args.batch_size,
            temperature=0.7,
            max_new_tokens=8,
            max_input_tokens=args.max_input_tokens,
        )
        working = run_llm_dataframe(
            df=working,
            cfg=force_cfg,
            client=client,
            system_prompt=active_force_system_prompt,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["keyword_answer_draft"].isna(),
            build_prompt_fn=lambda row, col: active_build_force_prompt(row, col),
            parse_fn=parse_force_output,
            output_cols=ALL_COLS,
            skip_if_already_filled="keyword_answer_draft",
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
        )

    still_na = build_mask(working, text_col=args.text_col) & working["keyword_answer_draft"].isna()
    n_unresolved = int(still_na.sum())
    if n_unresolved:
        print(
            f"[force] WARNING: {n_unresolved:,} eligible rows still have no answer "
            f"even after the forced pass (keyword_answer stays honestly NA for these).",
            flush=True,
        )

    # --- Pass 2 removed (2026-10-13): comparing pass-1-draft alone against
    # the final (pass-2-verified) answer across every recent gold-compared
    # run showed pass 2 contributes little and is sometimes net negative --
    # see the 2026-10-13 entry in src/step3_prompts.py for the full
    # evidence (job65556910: pass 2 touched only 43/800 rows, and left
    # pass-1-alone's own accuracy/F1 both HIGHER than the two-pass result).
    # Pass 1's own draft label is now the final answer: no further LLM call.
    # `keyword_answer_forced` rows already have `keyword_answer` set from the
    # force pass above and are left untouched by this fill.
    working["keyword_answer"] = working["keyword_answer"].where(
        working["keyword_answer"].notna(), working["keyword_answer_draft"]
    )

    # -------------------------------------------------------------------
    # Validation pass (2026-11-02 "validate-on-full-article", logit-
    # probing): see this file's own module docstring and
    # src/step3_logit_prompts.py for the full design rationale. Scoped
    # ONLY to rows pass 1 said YES to, excluding forced rows (same
    # convention as the old, removed pass 2 -- a forced row never went
    # through pass 2 either, there is nothing left to try to extract from
    # this model on it). Re-reads the RAW ARTICLE (args.text_col), never
    # pass 1's own justification.
    # -------------------------------------------------------------------
    if args.skip_validation:
        print("[validation] --skip_validation set -- validation pass disabled. Keeping pass "
              "1's own (possibly forced) keyword_answer for every row, no validation forward "
              "passes run.", flush=True)
        n_val_rejected = 0
        n_val_promoted = 0
        n_val_incomplete = 0
        n_val_candidates = 0
    else:
        # 2026-11-05 bug fix: keyword_answer_forced is pandas "string" dtype
        # (nullable/pyarrow-backed), and for the (overwhelming majority of)
        # rows that were NEVER forced it holds NA, not "FALSE". `!= "TRUE"`
        # on a pyarrow-backed string column propagates NA->NA rather than
        # NA->True, and indexing a DataFrame with a boolean mask containing
        # NA silently treats NA as False (observed directly: no exception,
        # just an empty validated_idx) -- so on job66796444/job66813431
        # this excluded EVERY non-forced row, and the validation pass ran
        # with 0 candidates on every task, completely silently (no crash,
        # no visible symptom beyond keyword_answer_before_validation being
        # NA for every single row, caught only by auditing that column
        # after the fact). .fillna(True) makes "no forced flag recorded"
        # correctly mean "not forced", matching every other NA check this
        # script already does (e.g. keyword_answer_draft.isna()).
        not_forced = (working["keyword_answer_forced"] != "TRUE").fillna(True)
        # 2026-11-24: --validation_scope=all additionally includes NO rows --
        # see this file's own module docstring and --validation_scope's
        # argparse help for the full rationale. Default ("yes_only") is
        # byte-identical to every prior run.
        if args.validation_scope == "all":
            answer_mask = working["keyword_answer"].isin(["YES", "NO"])
        else:
            answer_mask = working["keyword_answer"] == "YES"
        validation_mask = (
            build_mask(working, text_col=args.text_col)
            & answer_mask
            & not_forced
        )
        if args.validation_input == "justification":
            # 2026-11-06: this variant has nothing to validate against
            # without a real justification -- defensive guard, should
            # already always be true for any non-forced row (pass 1 never
            # writes YES or NO without one; see parse_draft_output).
            validation_mask = validation_mask & working["keyword_justification"].notna()
        validated_idx = working.index[validation_mask].tolist()
        n_val_candidates = len(validated_idx)
        val_input_desc = (
            "pass 1's own justification" if args.validation_input == "justification"
            else "the full article"
        )
        scope_desc = (
            "pass 1's YES verdict" if args.validation_scope == "yes_only"
            else "BOTH pass 1's YES and NO verdicts"
        )
        print(f"[validation] Re-checking {scope_desc} for {n_val_candidates:,} rows "
              f"against {val_input_desc} (--validation_scope={args.validation_scope}, "
              f"--validation_input={args.validation_input}, "
              f"--validation_threshold={args.validation_threshold}, "
              f"--validation_threshold_no_to_yes={args.validation_threshold_no_to_yes})...", flush=True)

        if task_id is not None:
            validation_checkpoint_path = args.output_base + f"_task{task_id}_step3val_checkpoint.parquet"
        else:
            validation_checkpoint_path = args.output_base + "_step3val_checkpoint.parquet"

        print("[validation] Computing p0 (this question's own intrinsic yes-bias) on the "
              "empty-article variant...", flush=True)
        if args.validation_input == "justification":
            calib_prompts = [build_validation_calibration_prompt_from_justification(order) for order in (1, 2)]
        else:
            calib_prompts = [build_validation_calibration_prompt(order) for order in (1, 2)]
        calib_raw = client.logprob_many(
            system_prompt="",
            user_prompts=calib_prompts,
            candidate_letters=candidate_letters,
            max_input_tokens=args.max_input_tokens,
        )
        p_yes_by_order = {}
        for order, logits_dict in zip((1, 2), calib_raw):
            probs = softmax_restricted(logits_dict)
            letter_yes, _letter_no = letters_for_order(order)
            p_yes_by_order[order] = probs.get(letter_yes, float("nan"))
        p0 = average_yes_over_orderings(p_yes_by_order[1], p_yes_by_order[2])
        print(f"  p0 = {p0:.4f}", flush=True)

        val_flat_keys = []    # (row_idx, order)
        val_flat_prompts = []
        for row_idx in validated_idx:
            row = working.loc[row_idx]
            for order in (1, 2):
                val_flat_keys.append((row_idx, order))
                if args.validation_input == "justification":
                    val_flat_prompts.append(build_validation_prompt_from_justification(row, order))
                else:
                    val_flat_prompts.append(build_validation_prompt(row, args.text_col, order))

        print(f"[validation] {len(val_flat_prompts):,} total (row, order) validation prompts "
              f"({n_val_candidates:,} rows x 2 orders)", flush=True)

        val_done = {}  # (row_idx, order) -> {letter: logit}
        if Path(validation_checkpoint_path).exists():
            ckpt = pd.read_parquet(validation_checkpoint_path)
            for _, r in ckpt.iterrows():
                vkey = (int(r["row_idx"]), int(r["order"]))
                val_done[vkey] = {letter: r[f"logit_{letter}"] for letter in candidate_letters}
            print(f"[resume] Loaded {len(val_done):,} already-resolved validation prompts from "
                  f"{validation_checkpoint_path}", flush=True)

        val_todo = [i for i, vkey in enumerate(val_flat_keys) if vkey not in val_done]
        print(f"[validation] {len(val_todo):,} validation prompts remaining "
              f"({len(val_flat_prompts) - len(val_todo):,} already done)", flush=True)

        def _save_val_checkpoint():
            rows = []
            for (row_idx, order), logits_dict in val_done.items():
                r = {"row_idx": row_idx, "order": order}
                for letter in candidate_letters:
                    r[f"logit_{letter}"] = logits_dict.get(letter, float("nan"))
                rows.append(r)
            ckpt_df = pd.DataFrame(rows)
            Path(validation_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            ckpt_df.to_parquet(validation_checkpoint_path, index=False)
            print(f"[checkpoint] Saved {len(ckpt_df):,} resolved validation prompts -> "
                  f"{validation_checkpoint_path}", flush=True)

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
            print(f"[validation progress] {n_done_total:,}/{len(val_flat_prompts):,} prompts "
                  f"resolved", flush=True)

            if n_since_val_checkpoint >= args.validation_checkpoint_every:
                _save_val_checkpoint()
                n_since_val_checkpoint = 0

        if val_todo:
            _save_val_checkpoint()

        n_val_rejected = 0   # YES -> NO flips (unchanged behaviour)
        n_val_promoted = 0   # 2026-11-24: NO -> YES flips, only possible with
                              # --validation_scope=all AND
                              # --validation_threshold_no_to_yes set
        n_val_incomplete = 0
        for row_idx in validated_idx:
            # 2026-11-24: capture pass 1's OWN answer before this pass can
            # touch it -- "YES" always before, now also "NO" when
            # --validation_scope=all. (Previously hardcoded to "YES" since
            # only YES rows were ever in validated_idx.)
            original_answer = working.at[row_idx, "keyword_answer"]
            working.at[row_idx, "keyword_answer_before_validation"] = original_answer

            p_yes_by_order = {}
            val_any_nan = False
            for order in (1, 2):
                logits_dict = val_done.get((row_idx, order))
                if logits_dict is None:
                    p_yes_by_order[order] = float("nan")
                    val_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_yes, _letter_no = letters_for_order(order)
                p_yes_by_order[order] = probs.get(letter_yes, float("nan"))
                if pd.isna(p_yes_by_order[order]):
                    val_any_nan = True

            p_yes = average_yes_over_orderings(p_yes_by_order[1], p_yes_by_order[2])
            if pd.isna(p_yes) or pd.isna(p0):
                val_any_nan = True

            if val_any_nan:
                # Inconclusive read -- fail-safe: keep pass 1's own answer,
                # don't swap on a signal we don't trust.
                n_val_incomplete += 1
                continue

            p_yes_corrected = calibrate(p_yes, p0)
            working.at[row_idx, "keyword_answer_validation_p_true"] = p_yes_corrected

            if original_answer == "YES":
                # Unchanged from every prior run: p_true >= threshold keeps
                # pass 1's YES, otherwise flips it to NO.
                if p_yes_corrected >= args.validation_threshold:
                    working.at[row_idx, "keyword_answer_validated"] = "TRUE"
                else:
                    working.at[row_idx, "keyword_answer_validated"] = "FALSE"
                    working.at[row_idx, "keyword_answer"] = "NO"
                    n_val_rejected += 1
            else:
                # 2026-11-24: original_answer == "NO", only reached with
                # --validation_scope=all. p_true is ALWAYS recorded above;
                # it is only ACTED on (promoted to YES) if an explicit
                # --validation_threshold_no_to_yes was given -- see that
                # flag's help for why this is opt-in rather than a guessed
                # default.
                if (args.validation_threshold_no_to_yes is not None
                        and p_yes_corrected >= args.validation_threshold_no_to_yes):
                    working.at[row_idx, "keyword_answer_validated"] = "TRUE"
                    working.at[row_idx, "keyword_answer"] = "YES"
                    n_val_promoted += 1
                else:
                    working.at[row_idx, "keyword_answer_validated"] = "FALSE"

        print(f"[validation] {n_val_rejected:,}/{n_val_candidates:,} rows rejected (flipped "
              f"YES->NO); {n_val_promoted:,} rows promoted (flipped NO->YES, only possible "
              f"with --validation_scope=all and --validation_threshold_no_to_yes set) "
              f"({n_val_incomplete:,} validation reads inconclusive, pass 1's own answer "
              f"kept as-is for those)", flush=True)

        if Path(validation_checkpoint_path).exists():
            Path(validation_checkpoint_path).unlink()
            print(f"[checkpoint] Deleted {validation_checkpoint_path}")

    # --- Save ---
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
    working.to_parquet(parquet_path, index=False)
    working.to_csv(csv_path, index=False)

    yes_count  = int((working["keyword_answer"] == "YES").sum())
    no_count   = int((working["keyword_answer"] == "NO").sum())
    na_count   = int(working["keyword_answer"].isna().sum())
    just_count = int(working["keyword_justification"].notna().sum())
    forced     = int((working["keyword_answer_forced"] == "TRUE").sum())
    both = working["keyword_answer_draft"].notna() & working["keyword_answer"].notna()
    disagree = int((working.loc[both, "keyword_answer_draft"] != working.loc[both, "keyword_answer"]).sum())
    print(
        f"Saved: {parquet_path} | {len(working):,} (article, target) rows "
        f"(keyword_answer: {yes_count:,} YES / {no_count:,} NO / {na_count:,} NA | "
        f"keyword_justification: {just_count:,} filled | "
        f"draft/final disagreement: {disagree:,}/{int(both.sum()):,} | "
        f"forced (no real verification): {forced:,} | "
        f"validation: {n_val_rejected:,}/{n_val_candidates:,} rejected, "
        f"{n_val_promoted:,} promoted, "
        f"{n_val_incomplete:,} inconclusive)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    # 2026-11-18: usage telemetry for the remote (--backend litellm) client --
    # pure observability (latency/token counts/throughput for the UNIL/CSCS
    # service itself), unrelated to this run's actual results. The local
    # TransformersClient has no print_stats(), so this is a no-op for the
    # GPU path -- see src/client_remote.py's module docstring.
    if hasattr(client, "print_stats"):
        client.print_stats()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
