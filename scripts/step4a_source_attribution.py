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
sometimes contradicting its own justification — see src/verify_utils.py):

  Pass 1 (draft)  — reads the article, names who (if anyone) is making the
    criticism, then a first-guess SOURCED/JOURNALIST label right after it.
    Both are kept: the justification under `source_attribution_justification`,
    the first-guess label under `source_attribution_draft` (QA only).
  Pass 2 (verify) — reads ONLY `source_attribution_justification` (no
    article) and classifies it into SOURCED/JOURNALIST. This is the FINAL
    label, saved as `source_attribution` — the name step 4b's mask uses.

Output columns: source_attribution              -> "SOURCED" | "JOURNALIST" | NA  (final, pass 2)
                source_attribution_draft         -> "SOURCED" | "JOURNALIST" | NA  (pass 1, QA only)
                source_attribution_justification -> free text | NA                 (pass 1)

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

DRAFT_COLS = ["source_attribution_draft", "source_attribution_justification"]
FINAL_COLS = ["source_attribution"]
# Flag column: "TRUE" when a row's final answer came from the last-resort
# forced one-word pass rather than a genuine two-pass verification -- see
# the retry/force block in main(). NA for every normally-answered row.
FLAG_COLS = ["source_attribution_forced"]
ALL_COLS = DRAFT_COLS + FINAL_COLS + FLAG_COLS

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
    # criticism is supposed to get an answer here. temperature=0.0 retried
    # unchanged would just reproduce the same failure, so each retry raises
    # temperature and token budget a notch; whatever is still unresolved
    # after --draft_max_retries gets one forced bare-word attempt (flagged
    # via source_attribution_forced, never sent through pass 2). ---
    ap.add_argument("--draft_max_retries",   type=int,   default=2,
                    help="Extra pass-1 attempts (beyond the first) for eligible rows still unanswered, each at a higher temperature/token budget.")
    ap.add_argument("--retry_temperature",   type=float, default=0.4,
                    help="Temperature for the first retry attempt; raised by +0.2 per further attempt, capped at 0.9.")

    # --- Inference: pass 2 (verify, reads only the justification) ---
    ap.add_argument("--verify_batch_size",       type=int,   default=16)
    ap.add_argument("--verify_temperature",      type=float, default=0.0)
    ap.add_argument("--verify_max_new_tokens",   type=int,   default=8)
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

    # --- Pass 2: verify (final label, from the justification alone) ---
    verify_cfg = RunConfig(
        id_col="__index__",
        text_col="source_attribution_justification",
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
        select_mask_fn=lambda df_: df_["source_attribution_draft"].notna() & df_["source_attribution_justification"].notna(),  # defense in depth: parse_draft_labeled already guarantees these travel together
        build_prompt_fn=lambda row, col: build_verify_prompt(row, col),
        parse_fn=parse_verify_output,
        output_cols=FINAL_COLS,
        skip_if_already_filled="source_attribution",
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
        f"forced (no real verification): {forced:,})"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
