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

Output columns: keyword_answer                   -> "YES" | "NO" | NA   (FINAL -- possibly
                                                       flipped to NO by the validation pass)
                keyword_answer_draft              -> "YES" | "NO" | NA   (pass 1)
                keyword_justification             -> free text | NA      (pass 1)
                keyword_answer_before_validation  -> "YES" | NA  (pass 1's own verdict, before
                                                       the validation pass touched it -- only
                                                       set for rows the validation pass actually
                                                       looked at)
                keyword_answer_validation_p_true  -> calibrated P(pass 1's YES is correct), or NA
                keyword_answer_validated          -> "TRUE" | "FALSE" | NA (NA = not eligible for
                                                       validation, or an inconclusive read --
                                                       fail-safe: pass 1's YES is kept either way)

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

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src.step3_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
    FORCE_SYSTEM_PROMPT, build_force_prompt,
)
from src.step3_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled
from src.step3_logit_prompts import (
    build_validation_prompt, build_validation_calibration_prompt, letters_for_order,
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
    ap.add_argument("--model_path",        required=True)
    ap.add_argument("--dtype",             required=True, choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend",           default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

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
    ap.add_argument("--skip_validation", action="store_true",
                    help="2026-11-02: disable the validation pass entirely -- no validation "
                    "forward passes are run, and keyword_answer stays exactly pass 1's own "
                    "(possibly forced) answer for every row. Kept as a flag from day one, "
                    "having learned from step4b's pass 3 that a validation pass needs to be "
                    "easy to turn off without touching code if it measures net-negative.")

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
                working[col] = pd.Series(pd.NA, index=working.index, dtype="string")
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            working = ckpt
    else:
        working = df.copy()
        for col in ALL_COLS:
            working[col] = pd.Series(pd.NA, index=working.index, dtype="string")

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
    working = run_llm_dataframe(
        df=working,
        cfg=draft_cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
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
            system_prompt=SYSTEM_PROMPT,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["keyword_answer_draft"].isna(),
            build_prompt_fn=lambda row, col: build_user_prompt(row, col),
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
            system_prompt=FORCE_SYSTEM_PROMPT,
            select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col) & df_["keyword_answer_draft"].isna(),
            build_prompt_fn=lambda row, col: build_force_prompt(row, col),
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
        n_val_incomplete = 0
        n_val_candidates = 0
    else:
        not_forced = working["keyword_answer_forced"] != "TRUE"
        validated_idx = working.index[
            build_mask(working, text_col=args.text_col)
            & (working["keyword_answer"] == "YES")
            & not_forced
        ].tolist()
        n_val_candidates = len(validated_idx)
        print(f"[validation] Re-checking pass 1's YES verdict for {n_val_candidates:,} rows "
              f"against the full article...", flush=True)

        if task_id is not None:
            validation_checkpoint_path = args.output_base + f"_task{task_id}_step3val_checkpoint.parquet"
        else:
            validation_checkpoint_path = args.output_base + "_step3val_checkpoint.parquet"

        print("[validation] Computing p0 (this question's own intrinsic yes-bias) on the "
              "empty-article variant...", flush=True)
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

        n_val_rejected = 0
        n_val_incomplete = 0
        for row_idx in validated_idx:
            working.at[row_idx, "keyword_answer_before_validation"] = "YES"

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
                # Inconclusive read -- fail-safe: keep pass 1's own YES,
                # don't swap on a signal we don't trust.
                n_val_incomplete += 1
                continue

            p_yes_corrected = calibrate(p_yes, p0)
            working.at[row_idx, "keyword_answer_validation_p_true"] = p_yes_corrected

            if p_yes_corrected >= 0.5:
                working.at[row_idx, "keyword_answer_validated"] = "TRUE"
            else:
                working.at[row_idx, "keyword_answer_validated"] = "FALSE"
                working.at[row_idx, "keyword_answer"] = "NO"
                n_val_rejected += 1

        print(f"[validation] {n_val_rejected:,}/{n_val_candidates:,} rows rejected (flipped "
              f"YES->NO) ({n_val_incomplete:,} validation reads inconclusive, pass 1's YES "
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
        f"{n_val_incomplete:,} inconclusive)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
