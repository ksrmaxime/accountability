"""
scripts/step4b_source_category.py

STEP 4b of the accountability pipeline — second half of source detection.

For every row step 4a marked SOURCED (a specific external actor is quoted
or referenced), asks the LLM to classify that actor into one of 6 broad
categories (see src/step4b_prompts.py for the full typology).

Rows step 4a did NOT mark SOURCED (JOURNALIST, or never reached because
step 3 did not confirm criticism) are left untouched — `source_category`
stays NA for those rows.

Two-pass design (added after the justification-first run showed the label
sometimes contradicting its own justification — see src/verify_utils.py):

  Pass 1 (draft)  — reads the article, names who is making the criticism,
    then a first-guess category label right after it. Both are kept: the
    justification under `source_category_justification`, the first-guess
    label under `source_category_draft` (QA only).
  Pass 2 (verify) — reads ONLY `source_category_justification` (no
    article) and classifies it into one of the SAME 6 categories. This is
    a genuine 6-way classification, not a binary one, so the verify system
    prompt repeats the full category legend from src/step4b_prompts.py —
    the second pass gets the same category boundaries to work with, just
    without the article. This is the FINAL label, saved as
    `source_category`.

Output columns: source_category              -> one of 6 categories | NA  (final, pass 2)
                source_category_draft         -> one of 6 categories | NA  (pass 1, QA only)
                source_category_justification -> free text | NA            (pass 1)

Usage
-----
  python scripts/step4b_source_category.py \\
      --input       data/output/step4a_merged/results.parquet \\
      --output_base data/output/step4b \\
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \\
      --dtype bf16 --batch_size 4 --temperature 0.0 \\
      --max_new_tokens 90 --max_input_tokens 16384 \\
      --verify_batch_size 16 --verify_max_new_tokens 12 --verify_max_input_tokens 512
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
from src.step4b_prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    VERIFY_SYSTEM_PROMPT, build_verify_prompt,
)
from src.step4b_config import build_mask
from src.verify_utils import parse_label_only, parse_draft_labeled

CATEGORIES = [
    "Interest Group",
    "Civil Servant",
    "General Public",
    "Politician",
    "Administrative Unit of the State",
    "Other",
]
_CATEGORIES_UPPER = {c.upper(): c for c in CATEGORIES}

DRAFT_COLS = ["source_category_draft", "source_category_justification"]
FINAL_COLS = ["source_category"]
ALL_COLS = DRAFT_COLS + FINAL_COLS

# Pass 2 uses the exact same 6-way mapping as pass 1 — this is a genuine
# multi-choice classification, not a binary one (unlike step 3/4a/5/6).
VERIFY_LABELS = _CATEGORIES_UPPER


def parse_draft_output(raw: str) -> dict:
    """A bare label with no real justification is a full failure, not a
    partial success -- see src/verify_utils.py:parse_draft_labeled."""
    empty = {"source_category_draft": pd.NA, "source_category_justification": pd.NA}
    label, justification = parse_draft_labeled(raw, VERIFY_LABELS)
    if label is None:
        return empty
    return {"source_category_draft": label, "source_category_justification": justification}


def parse_verify_output(raw: str) -> dict:
    return {"source_category": parse_label_only(raw, VERIFY_LABELS)}


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

    # --- Inference: pass 1 (draft, reads the full article) ---
    ap.add_argument("--batch_size",        required=True, type=int)
    ap.add_argument("--temperature",       required=True, type=float)
    ap.add_argument("--max_new_tokens",    required=True, type=int)
    ap.add_argument("--max_input_tokens",  required=True, type=int)

    # --- Inference: pass 2 (verify, reads only the justification) ---
    ap.add_argument("--verify_batch_size",       type=int,   default=16)
    ap.add_argument("--verify_temperature",      type=float, default=0.0)
    ap.add_argument("--verify_max_new_tokens",   type=int,   default=12)
    ap.add_argument("--verify_max_input_tokens", type=int,   default=512)

    # --- Internal ---
    ap.add_argument("--job_id",    default=None)
    ap.add_argument("--task_id",   type=int, default=None,
                    help="Array task index (0-indexed). If omitted, read from SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Total number of array tasks. If omitted, read from SLURM_ARRAY_TASK_COUNT.")

    args = ap.parse_args()

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

    expected_indices = set(df.index)
    eligible = int(build_mask(df, text_col=args.text_col).sum())
    print(f"[pipeline] {len(df):,} rows in chunk — {eligible:,} SOURCED rows "
          f"sent to the LLM (rest keep source_category = NA)")

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
                df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            df = ckpt
    else:
        for col in ALL_COLS:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")

    # --- LLM client ---
    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
        )
    )

    # --- Pass 1: draft (justification + first-guess category, from the article) ---
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
        skip_if_already_filled="source_category_draft",
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
    )

    # --- Pass 2: verify (final category, from the justification alone) ---
    verify_cfg = RunConfig(
        id_col="__index__",
        text_col="source_category_justification",
        batch_size=args.verify_batch_size,
        temperature=args.verify_temperature,
        max_new_tokens=args.verify_max_new_tokens,
        max_input_tokens=args.verify_max_input_tokens,
    )
    df = run_llm_dataframe(
        df=df,
        cfg=verify_cfg,
        client=client,
        system_prompt=VERIFY_SYSTEM_PROMPT,
        select_mask_fn=lambda df_: df_["source_category_draft"].notna() & df_["source_category_justification"].notna(),  # defense in depth: parse_draft_labeled already guarantees these travel together
        build_prompt_fn=lambda row, col: build_verify_prompt(row, col),
        parse_fn=parse_verify_output,
        output_cols=FINAL_COLS,
        skip_if_already_filled="source_category",
        checkpoint_path=checkpoint_path,
        checkpoint_every=100,
    )

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
    both = df["source_category_draft"].notna() & df["source_category"].notna()
    disagree = int((df.loc[both, "source_category_draft"] != df.loc[both, "source_category"]).sum())
    print(f"Saved: {parquet_path} | {len(df):,} rows total | NA: {na_count:,} | "
          f"draft/final disagreement: {disagree:,}/{int(both.sum()):,}")
    print(counts.to_string())

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
