# src/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    id_col: str
    text_col: str
    batch_size: int = 4          # keep small — full articles are long, padding cost is high
    temperature: float = 0.0
    max_new_tokens: int = 200
    max_input_tokens: int = 7168  # 8192 context - 1024 headroom for output


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def run_llm_dataframe(
    df: pd.DataFrame,
    cfg: RunConfig,
    client,
    system_prompt: str,
    select_mask_fn: Callable[[pd.DataFrame], pd.Series],
    build_prompt_fn: Callable[[pd.Series, str], str],
    parse_fn: Callable[[str], Dict[str, object]],
    output_cols: List[str],
    skip_if_already_filled: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """
    Generic LLM batch runner over a DataFrame.

    - select_mask_fn(df)         -> bool mask of rows to process
    - build_prompt_fn(row, col)  -> user prompt string
    - parse_fn(raw)              -> dict {output_col: value, ...}
    - output_cols                -> columns guaranteed to exist in output df
    - skip_if_already_filled     -> skip rows where df[col] is not NA (resumable runs)
    - checkpoint_path            -> path to save partial results every checkpoint_every batches
    - checkpoint_every           -> save checkpoint every N batches (default 50)
    """
    df = df.copy()
    df = ensure_columns(df, output_cols)

    mask = select_mask_fn(df)
    if mask is None:
        mask = pd.Series([True] * len(df), index=df.index)

    todo = mask.copy()
    if skip_if_already_filled:
        todo = todo & df[skip_if_already_filled].isna()

    idx = df.index[todo].tolist()
    if not idx:
        return df

    for batch_num, start in enumerate(range(0, len(idx), int(cfg.batch_size))):
        batch_idx = idx[start:start + int(cfg.batch_size)]
        batch_rows = df.loc[batch_idx]

        user_prompts = [build_prompt_fn(row, cfg.text_col) for _, row in batch_rows.iterrows()]
        raw = client.chat_many(
            system_prompt=system_prompt,
            user_prompts=user_prompts,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            max_input_tokens=cfg.max_input_tokens,
        )

        for i, row_id in enumerate(batch_idx):
            parsed = parse_fn(raw[i])
            for k, v in parsed.items():
                if k in df.columns:
                    df.at[row_id, k] = v

        if checkpoint_path and (batch_num + 1) % checkpoint_every == 0:
            df.to_parquet(checkpoint_path, index=False)
            filled = int(df[skip_if_already_filled].notna().sum()) if skip_if_already_filled else "?"
            print(f"[checkpoint] batch {batch_num + 1} — {filled}/{len(df)} rows done → {checkpoint_path}", flush=True)

    return df
