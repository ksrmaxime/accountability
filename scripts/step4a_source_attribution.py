"""
scripts/step4a_source_attribution.py

STEP 4a of the accountability pipeline — first half of source detection.

For every (article, target) row step 3 confirmed as a criticism, asks the
LLM to first identify (in one sentence) who -- if anyone -- is named as
making the criticism, then commit to a SOURCED / JOURNALIST label. A real
full run with the original one-shot (no reasoning) version of this prompt
produced only 7 SOURCED out of ~13,000 eligible rows -- essentially always
JOURNALIST even when a spot check found named actors quoted directly. See
src/step4a_prompts.py for the fix (justification-first, like every other
step now).

Rows step 3 did NOT flag as criticism (keyword_answer != "YES") are left
untouched in the output — this script only ever reads keyword_answer, it
never overwrites it, and `source_attribution` stays NA for those rows.

Output columns: source_attribution -> "SOURCED" | "JOURNALIST" | NA
                source_attribution_justification -> free text | NA

Usage
-----
  python scripts/step4a_source_attribution.py \\
      --input       data/output/step3_merged/results.parquet \\
      --output_base data/output/step4a \\
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
from src.step4a_prompts import SYSTEM_PROMPT, build_user_prompt
from src.step4a_config import build_mask


def parse_output(raw: str) -> dict:
    """Parse "<who, if anyone>\n...\nSOURCED|JOURNALIST" (justification
    first, label last — see src/step4a_prompts.py). Falls back to checking
    the first line in case the model answers the label first anyway."""
    empty = {"source_attribution": pd.NA, "source_attribution_justification": pd.NA}
    if not raw:
        return empty
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return empty

    def _label(word: str):
        word = word.upper()
        if word.startswith("SOURCED"):
            return "SOURCED"
        if word.startswith("JOURNALIST"):
            return "JOURNALIST"
        return None

    label = _label(lines[-1])
    if label is not None:
        justification = " ".join(lines[:-1]).strip() or pd.NA
        return {"source_attribution": label, "source_attribution_justification": justification}

    label = _label(lines[0])
    if label is not None:
        justification = " ".join(lines[1:]).strip() or pd.NA
        return {"source_attribution": label, "source_attribution_justification": justification}

    return empty


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

    # --- Load step 3 output (ALL rows: YES, NO, and unevaluated) ---
    df = (pd.read_parquet(args.input)
          if args.input.endswith(".parquet")
          else pd.read_csv(args.input, low_memory=False))

    if args.n_rows > 0:
        df = df.head(args.n_rows).copy()
        print(f"[n_rows] Subsetting to first {args.n_rows} rows")

    # --- Chunk across tasks on ALL rows, then report how many are eligible ---
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
            for col in ("source_attribution", "source_attribution_justification"):
                df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            print(f"[resume] Loading checkpoint: {checkpoint_path}", flush=True)
            df = ckpt
    else:
        for col in ("source_attribution", "source_attribution_justification"):
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

    run_cfg = RunConfig(
        id_col="__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    # --- Run LLM on confirmed-criticism rows only; the rest keep NA ---
    df = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=SYSTEM_PROMPT,
        select_mask_fn=lambda df_: build_mask(df_, text_col=args.text_col),
        build_prompt_fn=lambda row, col: build_user_prompt(row, col),
        parse_fn=parse_output,
        output_cols=["source_attribution", "source_attribution_justification"],
        skip_if_already_filled="source_attribution",
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
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

    sourced = int((df["source_attribution"] == "SOURCED").sum())
    journalist = int((df["source_attribution"] == "JOURNALIST").sum())
    just_count = int(df["source_attribution_justification"].notna().sum())
    print(
        f"Saved: {parquet_path} | {len(df):,} rows total "
        f"(source_attribution: {sourced:,} SOURCED / {journalist:,} JOURNALIST | "
        f"source_attribution_justification: {just_count:,} filled)"
    )

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"[checkpoint] Deleted {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
