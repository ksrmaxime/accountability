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
probability that "the criticism of {keyword} in this article comes from
{category}". The final answer is simply the category with the highest
calibrated probability.

Three ideas make this work (full math in src/step4b_logit_utils.py, full
prompt design in src/step4b_logit_prompts.py):

  1. One forward pass, not generation -- src/client.py's
     TransformersClient.logprob_many reads the raw logits at the
     answer-letter position directly, no decode loop, and can never
     produce a malformed or missing answer the way free-text generation
     with retries/forcing could.
  2. Order-swap averaging -- every question is asked twice (A=Yes/B=No,
     then A=No/B=Yes) and the two P(yes) readings averaged, to cancel any
     intrinsic model preference for one letter over the other.
  3. Context-free calibration (Zhao et al. 2021) -- each category
     question's own intrinsic yes-bias is measured ONCE (not per row) on
     an "empty article" variant of the same question, then removed from
     every row's reading in logit space before picking the argmax.

This eliminates the entire draft/retry/force/verify/reclassify machinery
of the 3-pass design: a forward pass over valid candidate tokens is
always well-defined, so there is no parsing step that can fail and
nothing to retry. The only failure mode is a GPU OOM on an exceptionally
long article, handled the same way chat_many always has (automatic
batch-splitting, see src/client.py's *_with_fallback helpers) -- the rare
row that still fails even alone gets NaN logits, which propagate to
NaN category probabilities and an honestly-NA source_category, flagged
via source_category_incomplete.

Output columns:
  source_category                 -> one of 6 categories | NA (FINAL)
  source_category_p_<category>    -> calibrated probability for each of
                                       the 6 categories (QA / diagnostics)
  source_category_margin          -> calibrated prob of the winning
                                       category minus the runner-up's
                                       (a natural confidence signal this
                                       design gives "for free" -- the
                                       3-pass design had no equivalent)
  source_category_incomplete      -> "TRUE" if any underlying logit read
                                       failed (GPU OOM even alone) for
                                       this row, else NA

Usage
-----
  python scripts/step4b_source_category_logit.py \
      --input       data/output/step4a_merged/results.parquet \
      --output_base data/output/step4b_logit \
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
      --dtype bf16 --batch_size 32 --max_input_tokens 16384
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
from src.step4b_config import build_mask
from src.step4b_logit_prompts import (
    CATEGORIES,
    build_question_prompt,
    build_calibration_prompt,
    letters_for_order,
)
from src.step4b_logit_utils import (
    softmax_restricted,
    average_yes_over_orderings,
    calibrate,
    pick_argmax_category,
)

CANDIDATE_LETTERS_DEFAULT = ["A", "B"]


def _slug(category: str) -> str:
    return category.lower().replace(" ", "_")


PROB_COLS = [f"source_category_p_{_slug(c)}" for c in CATEGORIES]
ALL_OUTPUT_COLS = ["source_category", "source_category_margin", "source_category_incomplete"] + PROB_COLS


def compute_calibration(client, candidate_letters, max_input_tokens) -> dict:
    """p0(category): this category's question's own intrinsic yes-bias,
    measured ONCE (order-averaged) on the 'empty article' variant of the
    question -- see src/step4b_logit_prompts.build_calibration_prompt.
    Independent of every row, so this never needs to be re-run per row,
    and is cheap enough (6 categories x 2 orders = 12 forward passes) to
    simply recompute at the start of every SLURM array task rather than
    share across tasks."""
    prompts = []
    meta = []  # (category, order)
    for cat in CATEGORIES:
        for order in (1, 2):
            prompts.append(build_calibration_prompt(cat, order))
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
        p0[cat] = average_yes_over_orderings(
            p_yes_by_cat_order[(cat, 1)], p_yes_by_cat_order[(cat, 2)]
        )
    return p0


def main() -> int:
    ap = argparse.ArgumentParser()

    # --- I/O ---
    ap.add_argument("--input",        required=True, help="Step 4a merged output: parquet or csv")
    ap.add_argument("--output_base",  required=True)
    ap.add_argument("--text_col",     default="text")
    ap.add_argument("--n_rows",       type=int, default=0)

    # --- Model ---
    ap.add_argument("--model_path",        required=True)
    ap.add_argument("--dtype",             required=True, choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend",           default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    # --- Inference: no temperature / max_new_tokens here -- this design
    # never generates text, it reads next-token logits in a single
    # forward pass, so there is nothing to sample and no output length to
    # bound. batch_size is a count of individual (row, category, order)
    # PROMPTS per forward-pass call, not of rows -- each eligible row
    # produces 6 categories x 2 orders = 12 prompts. ---
    ap.add_argument("--batch_size",        type=int,   default=32)
    ap.add_argument("--max_input_tokens",  type=int,   default=16384)
    ap.add_argument("--candidate_letters", default="A,B",
                    help="Comma-separated single-token candidates for Yes/No. Verify these "
                         "with scripts/check_step4b_logit_tokens.py before relying on them.")

    ap.add_argument("--checkpoint_every", type=int, default=200,
                    help="Save a checkpoint after every N flat (row,category,order) prompts resolved.")

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
    eligible_idx = df.index[eligible_mask].tolist()
    print(f"[pipeline] {len(df):,} rows in chunk — {len(eligible_idx):,} SOURCED rows "
          f"sent to the logit-probing classifier (rest keep source_category = NA)")

    if task_id is not None:
        checkpoint_path = args.output_base + f"_task{task_id}_checkpoint.parquet"
    else:
        checkpoint_path = args.output_base + "_checkpoint.parquet"

    for col in ALL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="object")

    # --- LLM client ---
    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
        )
    )

    # --- One-time (per task) calibration: this category question's own
    # intrinsic yes-bias on an empty article, order-averaged. Cheap (12
    # forward passes total) -- recomputed per task rather than shared,
    # since it's deterministic and the cost is negligible either way. ---
    print("[calibration] Computing p0(category) on the empty-article variant...", flush=True)
    p0 = compute_calibration(client, candidate_letters, args.max_input_tokens)
    for cat in CATEGORIES:
        print(f"  p0[{cat}] = {p0[cat]:.4f}")

    # --- Build the flat list of (row_idx, category, order) prompts for
    # every eligible row. Resumable via a long-format checkpoint keyed on
    # (row_idx, category, order). ---
    flat_keys = []     # (row_idx, category, order)
    flat_prompts = []
    for row_idx in eligible_idx:
        row = df.loc[row_idx]
        for cat in CATEGORIES:
            for order in (1, 2):
                flat_keys.append((row_idx, cat, order))
                flat_prompts.append(build_question_prompt(row, args.text_col, cat, order))

    print(f"[pipeline] {len(flat_prompts):,} total (row, category, order) prompts "
          f"({len(eligible_idx):,} rows x {len(CATEGORIES)} categories x 2 orders)")

    # --- Resume from checkpoint if present (long format: one row per
    # flat prompt, with its raw logits for each candidate letter). ---
    done = {}  # (row_idx, cat, order) -> {letter: logit}
    if Path(checkpoint_path).exists():
        ckpt = pd.read_parquet(checkpoint_path)
        for _, r in ckpt.iterrows():
            key = (int(r["row_idx"]), str(r["category"]), int(r["order"]))
            done[key] = {letter: r[f"logit_{letter}"] for letter in candidate_letters}
        print(f"[resume] Loaded {len(done):,} already-resolved prompts from {checkpoint_path}", flush=True)

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
            _save_checkpoint(checkpoint_path, done, candidate_letters)
            n_since_checkpoint = 0

    if todo_indices:
        _save_checkpoint(checkpoint_path, done, candidate_letters)

    # --- Aggregate: for every eligible row, for every category, average
    # P(yes) over the 2 orders, calibrate against p0, then argmax. ---
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

            p = average_yes_over_orderings(p_yes_by_order[1], p_yes_by_order[2])
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
        else:
            sorted_probs = sorted(valid.values(), reverse=True)
            margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
            df.at[row_idx, "source_category"] = best_cat
            df.at[row_idx, "source_category_margin"] = margin
        df.at[row_idx, "source_category_incomplete"] = "TRUE" if any_nan else pd.NA

    n_incomplete = int((df["source_category_incomplete"] == "TRUE").sum())
    if n_incomplete:
        print(f"[warning] {n_incomplete:,} rows had at least one failed logit read "
              f"(GPU OOM even alone) -- flagged via source_category_incomplete.", flush=True)

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
    print(f"Saved: {parquet_path} | {len(df):,} rows total | NA: {na_count:,} | "
          f"incomplete: {n_incomplete:,} | avg winning margin: {avg_margin:.4f}")
    print(counts.to_string())

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

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


if __name__ == "__main__":
    raise SystemExit(main())
