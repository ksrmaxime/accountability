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

2026-10-13: pass 2 removed. A draft-vs-final comparison on job65746412
(800-row gold) found pass 2 purely harmful for this step: of the 7 rows it
touched where pass 1 was already correct, all 7 were flipped to a WRONG
final answer (0 corrections) -- draft accuracy 72.2% vs final 66.9% on
that run. The mechanism: VERIFY_USER_TEMPLATE's SOURCED rule ("if the
explanation names... any specific actor... even briefly") fires whenever
pass 1's justification names the journalist by name (e.g. "attributed to
the journalist, Larissa Rhyn"), which it does as a matter of routine
style -- pass 2, never seeing the article, cannot tell that this named
actor IS the journalist the SOURCED/JOURNALIST distinction exists to
exclude. This is the same 2026-10-07/10-12 "bare named actor counts"
clarification (see VERIFY_USER_TEMPLATE's own changelog in
src/step4a_prompts.py) having overshot in the opposite direction. Pass 1's
own draft label is now written straight to the final column below, with
no second LLM call:

  Pass 1 (draft)  — reads the article, names who (if anyone) is making the
    criticism, then a first-guess SOURCED/JOURNALIST label right after it.
    Both are kept: the justification under `source_attribution_justification`,
    the first-guess label under `source_attribution_draft`.
    `source_attribution_draft` is then copied straight to
    `source_attribution` (see the end of main(), just before Save).
  Pass 2 (verify) — no longer called. VERIFY_SYSTEM_PROMPT/build_verify_prompt
    and the --verify_* CLI args below are kept, unused, in case pass 2 is
    revisited later.

2026-11-16 "validate-on-justification" addition: job20479169 (800-row
gold, the current best step3 lineage) showed step4a's 95 YES-gated rows
scoring BELOW their own majority-class baseline (74.7% vs 82.1%), with a
heavily directional error pattern -- 20 FN (gold=SOURCED, Apertus=
JOURNALIST) vs only 4 FP. This adds a SEPARATE validation pass -- see
src/step4a_logit_prompts.py's module docstring for the full design
rationale. It is deliberately the MIRROR of step3's own validation pass,
not a copy: scoped ONLY to rows pass 1 said JOURNALIST (never SOURCED),
and can only ever flip JOURNALIST->SOURCED, the opposite direction from
step3's YES->NO-only pass, because step4a's own error profile is FN-heavy
rather than FP-heavy. Uses the same calibrated-logit machinery (one
forward pass, order-swap, context-free calibration) as step3/step4b.
Forced rows (source_attribution_forced == "TRUE") skip this pass
entirely, same convention as step3. Toggle with --skip_validation (see
sbatch_step4a_array.sh's SKIP_VALIDATION).

Output columns: source_attribution                 -> "SOURCED" | "JOURNALIST" | NA  (FINAL --
                                                        possibly flipped to SOURCED by the
                                                        validation pass)
                source_attribution_draft            -> "SOURCED" | "JOURNALIST" | NA  (pass 1)
                source_attribution_justification    -> free text | NA                 (pass 1)
                source_attribution_before_validation -> "JOURNALIST" | NA  (pass 1's own verdict,
                                                        before the validation pass touched it --
                                                        only set for rows the validation pass
                                                        actually looked at)
                source_attribution_validation_p_sourced -> calibrated P(an outside voice IS
                                                        credited, i.e. SOURCED is correct), or NA
                source_attribution_validated        -> "TRUE" | "FALSE" | NA ("TRUE" = pass 1's
                                                        JOURNALIST confirmed/kept, "FALSE" =
                                                        rejected and flipped to SOURCED; NA = not
                                                        eligible, or an inconclusive read --
                                                        fail-safe: pass 1's JOURNALIST is kept
                                                        either way)

Usage
-----
  python scripts/step4a_source_attribution.py \\
      --input       data/output/step3_merged/results.parquet \\
      --output_base data/output/step4a \\
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

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.step4a_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
    FORCE_SYSTEM_PROMPT, build_force_prompt,
)
from src.step4a_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled
from src.step4a_logit_prompts import (
    build_validation_prompt, build_validation_calibration_prompt,
    build_validation_prompt_from_justification, build_validation_calibration_prompt_from_justification,
    letters_for_order,
)
from src.step4b_logit_utils import softmax_restricted, average_yes_over_orderings, calibrate

DRAFT_COLS = ["source_attribution_draft", "source_attribution_justification"]
FINAL_COLS = ["source_attribution"]
# Flag column: "TRUE" when a row's final answer came from the last-resort
# forced one-word pass rather than a genuine two-pass verification -- see
# the retry/force block in main(). NA for every normally-answered row.
FLAG_COLS = ["source_attribution_forced"]
# 2026-11-16 "validate-on-justification" pass -- see src/step4a_logit_prompts.py
# and this file's own module docstring.
VALIDATION_COLS = [
    "source_attribution_before_validation",
    "source_attribution_validation_p_sourced",
    "source_attribution_validated",
]
ALL_COLS = DRAFT_COLS + FINAL_COLS + FLAG_COLS + VALIDATION_COLS

VERIFY_LABELS = {"SOURCED": "SOURCED", "JOURNALIST": "JOURNALIST"}


def parse_draft_output(raw: str) -> dict:
    """A bare label with no real justification is a full failure, not a
    partial success -- see src/verify_utils.py:parse_draft_labeled."""
    empty = {"source_attribution_draft": pd.NA, "source_attribution_justification": pd.NA}
    label, justification = parse_draft_labeled(raw, VERIFY_LABELS)
    if label is None:
        return empty
    return {"source_attribution_draft": label, "source_attribution_justification": justification}


def parse_verify_output(raw: str) -> dict:
    return {"source_attribution": parse_label_only(raw, VERIFY_LABELS)}


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

    # --- Inference: pass 2 (verify, reads only the justification) ---
    ap.add_argument("--verify_batch_size",       type=int,   default=16)
    ap.add_argument("--verify_temperature",      type=float, default=0.0)
    ap.add_argument("--verify_max_new_tokens",   type=int,   default=8)
    ap.add_argument("--verify_max_input_tokens", type=int,   default=512)

    # --- Validation pass (2026-11-16 "validate-on-justification", logit-
    # probing): no temperature/max_new_tokens -- a single forward pass
    # reads next-token logits, nothing is sampled. batch_size counts
    # individual (row, order) PROMPTS, not rows -- each validated row
    # produces 2 prompts (order-swap). Reuses --max_input_tokens from
    # pass 1 above (same article text, same truncation budget needed, in
    # case --validation_input=article is used). ---
    ap.add_argument("--candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Sourced/Journalist in the "
                         "validation pass. Verify these with scripts/check_step4b_logit_tokens.py "
                         "before relying on them (same tokenizer, same letters already checked "
                         "for step4b/step3).")
    ap.add_argument("--validation_batch_size", type=int, default=32)
    ap.add_argument("--validation_checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N (row,order) validation prompts resolved.")
    ap.add_argument("--validation_input", default="justification", choices=["article", "justification"],
                    help="2026-11-16: what the validation pass re-checks pass 1's JOURNALIST "
                    "verdict against. 'justification' (DEFAULT) reads only pass 1's own "
                    "source_attribution_justification -- by analogy with step3's own finding that "
                    "re-reading the full article a second time measured net-negative there (see "
                    "src/step3_logit_prompts.py). This is a PRIOR, not yet independently "
                    "confirmed for step4a -- the first run with this default is that check. "
                    "'article' re-reads the full raw article instead, kept available precisely to "
                    "test that prior if justification-mode underperforms here.")
    ap.add_argument("--validation_threshold", type=float, default=0.5,
                    help="2026-11-16: decision cutoff on source_attribution_validation_p_sourced "
                    "-- p_sourced >= threshold flips pass 1's JOURNALIST to SOURCED, otherwise "
                    "JOURNALIST is kept. Starting value is the neutral logit midpoint (0.5), NOT "
                    "yet empirically tuned -- step3's own threshold-tuning history "
                    "(src/step3_criticism_detection.py) showed the natural midpoint can be "
                    "miscalibrated for a specific validation question, so treat this run's "
                    "p_sourced distribution (gold-SOURCED vs gold-JOURNALIST means) as the first "
                    "real data point for tuning this, the same way job20255405 was for step3 -- "
                    "do not grid-search a 'best' threshold on this same run's own data (circular); "
                    "pick a round value motivated by this run and confirm out-of-sample on the "
                    "next one.")
    ap.add_argument("--skip_validation", action="store_true",
                    help="2026-11-16: disable the validation pass entirely -- no validation "
                    "forward passes are run, and source_attribution stays exactly pass 1's own "
                    "(possibly forced) answer for every row.")

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
                # source_attribution_validation_p_sourced holds a FLOAT
                # (the calibrated P(sourced)), not a label -- same dtype
                # issue already fixed for step3's own p_true column (see
                # src/step3_criticism_detection.py's 2026-11-05 entry):
                # pandas' strict "string" dtype raises TypeError on a
                # float assignment. Every other column here is a
                # label/flag and keeps the "string" dtype.
                dtype = "object" if col == "source_attribution_validation_p_sourced" else "string"
                df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            df = ckpt
    else:
        for col in ALL_COLS:
            dtype = "object" if col == "source_attribution_validation_p_sourced" else "string"
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

    # --- Pass 2 removed (2026-10-13): see the docstring above for the full
    # evidence (job65746412: draft accuracy 72.2% vs final 66.9%, 7/7
    # pass-1-correct flips turned wrong, 0 corrected). Pass 1's own draft
    # label is now the final answer: no further LLM call.
    # `source_attribution_forced` rows already have `source_attribution` set
    # from the force pass above and are left untouched by this fill.
    df["source_attribution"] = df["source_attribution"].where(
        df["source_attribution"].notna(), df["source_attribution_draft"]
    )

    # -------------------------------------------------------------------
    # Validation pass (2026-11-16 "validate-on-justification", logit-
    # probing): see this file's own module docstring and
    # src/step4a_logit_prompts.py for the full design rationale. Scoped
    # ONLY to rows pass 1 said JOURNALIST (the MIRROR of step3's own
    # YES-only scoping -- step4a's error is FN-heavy, not FP-heavy),
    # excluding forced rows (a forced row never went through a real
    # pass 1 either, there is nothing to validate). Re-reads
    # source_attribution_justification by default (--validation_input).
    # -------------------------------------------------------------------
    if args.skip_validation:
        print("[validation] --skip_validation set -- validation pass disabled. Keeping pass "
              "1's own (possibly forced) source_attribution for every row, no validation forward "
              "passes run.", flush=True)
        n_val_flipped = 0
        n_val_incomplete = 0
        n_val_candidates = 0
    else:
        # Same .fillna(True) convention as step3 (src/step3_criticism_detection.py's
        # 2026-11-05 fix): a pyarrow-backed "string" NA on source_attribution_forced
        # must read as "not forced", not propagate to NA (which a boolean
        # mask silently treats as False, excluding every non-forced row).
        not_forced = (df["source_attribution_forced"] != "TRUE").fillna(True)
        validation_mask = (
            build_mask(df, text_col=args.text_col)
            & (df["source_attribution"] == "JOURNALIST")
            & not_forced
        )
        if args.validation_input == "justification":
            validation_mask = validation_mask & df["source_attribution_justification"].notna()
        validated_idx = df.index[validation_mask].tolist()
        n_val_candidates = len(validated_idx)
        val_input_desc = (
            "pass 1's own justification" if args.validation_input == "justification"
            else "the full article"
        )
        print(f"[validation] Re-checking pass 1's JOURNALIST verdict for {n_val_candidates:,} rows "
              f"against {val_input_desc} (--validation_input={args.validation_input}, "
              f"--validation_threshold={args.validation_threshold})...", flush=True)

        if task_id is not None:
            validation_checkpoint_path = args.output_base + f"_task{task_id}_step4aval_checkpoint.parquet"
        else:
            validation_checkpoint_path = args.output_base + "_step4aval_checkpoint.parquet"

        print("[validation] Computing p0 (this question's own intrinsic sourced-bias) on the "
              "empty/neutral-input variant...", flush=True)
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
        p_sourced_by_order = {}
        for order, logits_dict in zip((1, 2), calib_raw):
            probs = softmax_restricted(logits_dict)
            letter_sourced, _letter_journalist = letters_for_order(order)
            p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
        p0 = average_yes_over_orderings(p_sourced_by_order[1], p_sourced_by_order[2])
        print(f"  p0 = {p0:.4f}", flush=True)

        val_flat_keys = []    # (row_idx, order)
        val_flat_prompts = []
        for row_idx in validated_idx:
            row = df.loc[row_idx]
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

        n_val_flipped = 0
        n_val_incomplete = 0
        for row_idx in validated_idx:
            df.at[row_idx, "source_attribution_before_validation"] = "JOURNALIST"

            p_sourced_by_order = {}
            val_any_nan = False
            for order in (1, 2):
                logits_dict = val_done.get((row_idx, order))
                if logits_dict is None:
                    p_sourced_by_order[order] = float("nan")
                    val_any_nan = True
                    continue
                probs = softmax_restricted(logits_dict)
                letter_sourced, _letter_journalist = letters_for_order(order)
                p_sourced_by_order[order] = probs.get(letter_sourced, float("nan"))
                if pd.isna(p_sourced_by_order[order]):
                    val_any_nan = True

            p_sourced = average_yes_over_orderings(p_sourced_by_order[1], p_sourced_by_order[2])
            if pd.isna(p_sourced) or pd.isna(p0):
                val_any_nan = True

            if val_any_nan:
                # Inconclusive read -- fail-safe: keep pass 1's own
                # JOURNALIST, don't flip on a signal we don't trust.
                n_val_incomplete += 1
                continue

            p_sourced_corrected = calibrate(p_sourced, p0)
            df.at[row_idx, "source_attribution_validation_p_sourced"] = p_sourced_corrected

            if p_sourced_corrected >= args.validation_threshold:
                df.at[row_idx, "source_attribution_validated"] = "FALSE"
                df.at[row_idx, "source_attribution"] = "SOURCED"
                n_val_flipped += 1
            else:
                df.at[row_idx, "source_attribution_validated"] = "TRUE"

        print(f"[validation] {n_val_flipped:,}/{n_val_candidates:,} rows flipped "
              f"(JOURNALIST->SOURCED) ({n_val_incomplete:,} validation reads inconclusive, pass "
              f"1's JOURNALIST kept as-is for those)", flush=True)

        if Path(validation_checkpoint_path).exists():
            Path(validation_checkpoint_path).unlink()
            print(f"[checkpoint] Deleted {validation_checkpoint_path}")

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
        f"validation: {n_val_flipped:,}/{n_val_candidates:,} flipped, "
        f"{n_val_incomplete:,} inconclusive)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
