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

Output columns: source_attribution                   -> "SOURCED" | "JOURNALIST" | NA  (FINAL --
                                                          pass 1, possibly flipped by pass 2,
                                                          possibly overridden by pass 3)
                source_attribution_draft              -> "SOURCED" | "JOURNALIST" | NA  (pass 1)
                source_attribution_justification      -> free text | NA                 (pass 1)
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

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.step4a_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
    FORCE_SYSTEM_PROMPT, build_force_prompt,
)
from src.step4a_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled
from src.step4a_pass2_coherence_logit_prompts import (
    build_coherence_prompt, build_coherence_calibration_prompt, letters_for_order,
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
ALL_COLS = DRAFT_COLS + FINAL_COLS + FLAG_COLS + PASS2_COLS + PASS2_COHERENCE_COLS

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
    ap.add_argument("--model_path",        required=True)
    ap.add_argument("--dtype",             required=True, choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend",           default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

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
                # source_attribution_pass2_coherence_p_sourced holds a FLOAT
                # (the calibrated P(sourced)), not a label -- same dtype
                # issue already hit (and fixed) for the removed validation
                # pass's own p_sourced column: pandas' strict "string" dtype
                # raises TypeError on a float assignment. Every other column
                # here is a label/flag and keeps the "string" dtype.
                dtype = "object" if col == "source_attribution_pass2_coherence_p_sourced" else "string"
                df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            df = ckpt
    else:
        for col in ALL_COLS:
            dtype = "object" if col == "source_attribution_pass2_coherence_p_sourced" else "string"
            df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)

    # --- LLM client ---
    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
        )
    )

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
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
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
            system_prompt=SYSTEM_PROMPT,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["source_attribution_draft"].isna(),
            build_prompt_fn=lambda row, col: build_user_prompt(row, col),
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
            system_prompt=FORCE_SYSTEM_PROMPT,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["source_attribution_draft"].isna(),
            build_prompt_fn=lambda row, col: build_force_prompt(row, col),
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
            system_prompt=VERIFY_SYSTEM_PROMPT,
            select_mask_fn=lambda df_, _m=pass2_mask: _m,
            build_prompt_fn=lambda row, col: build_verify_prompt(row, col),
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
        f"pass2: {n_pass2_flipped:,}/{n_pass2_candidates:,} flipped, "
        f"{n_pass2_unparsed:,} unparsed | "
        f"pass3 coherence: {n_coh_overridden:,}/{n_coh_checked:,} overridden, "
        f"{n_coh_inconclusive:,} inconclusive)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
