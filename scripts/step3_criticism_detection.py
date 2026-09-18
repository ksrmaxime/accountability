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

Output columns: keyword_answer -> "YES" | "NO" | NA
                keyword_justification -> free text | NA

Usage
-----
  python scripts/step3_criticism_detection.py \\
      --input       data/processed/step2_clean.parquet \\
      --output_base data/output/step3 \\
      --model_path  /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \\
      --dtype bf16 --batch_size 4 --temperature 0.0 \\
      --max_new_tokens 80 --max_input_tokens 16384
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
from src.step3_prompts import SYSTEM_PROMPT, build_user_prompt
from src.step3_config import build_mask


def parse_output(raw: str) -> dict:
    """Parse "<justification sentence>\n...\nYES|NO" (justification first,
    label last — see src/step3_prompts.py). Falls back to checking the
    first line in case the model answers the label first anyway."""
    empty = {"keyword_answer": pd.NA, "keyword_justification": pd.NA}
    if not raw:
        return empty
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return empty

    def _label(word: str):
        word = word.upper()
        if word.startswith("YES"):
            return "YES"
        if word.startswith("NO"):
            return "NO"
        return None

    label = _label(lines[-1])
    if label is not None:
        justification = " ".join(lines[:-1]).strip() or pd.NA
        return {"keyword_answer": label, "keyword_justification": justification}

    label = _label(lines[0])
    if label is not None:
        justification = " ".join(lines[1:]).strip() or pd.NA
        return {"keyword_answer": label, "keyword_justification": justification}

    return empty


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

    # --- Inference ---
    ap.add_argument("--batch_size",        required=True, type=int)
    ap.add_argument("--temperature",       required=True, type=float)
    ap.add_argument("--max_new_tokens",    required=True, type=int)
    ap.add_argument("--max_input_tokens",  required=True, type=int)

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
            for col in ("keyword_answer", "keyword_justification"):
                working[col] = pd.Series(pd.NA, index=working.index, dtype="string")
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            working = ckpt
    else:
        working = df.copy()
        for col in ("keyword_answer", "keyword_justification"):
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

    run_cfg = RunConfig(
        id_col="__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    working = run_llm_dataframe(
        df=working,
        cfg=run_cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
        parse_fn=parse_output,
        output_cols=["keyword_answer", "keyword_justification"],
        skip_if_already_filled="keyword_answer",
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
    )

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

    yes_count = int((working["keyword_answer"] == "YES").sum())
    no_count  = int((working["keyword_answer"] == "NO").sum())
    just_count = int(working["keyword_justification"].notna().sum())
    print(
        f"Saved: {parquet_path} | {len(working):,} (article, target) rows "
        f"(keyword_answer: {yes_count:,} YES / {no_count:,} NO | "
        f"keyword_justification: {just_count:,} filled)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
