"""
scripts/check_step4b_logit_tokens.py

Run this FIRST, before any real logit-probing job for step 4b (see
scripts/step4b_source_category_logit.py), and before spending any real
SLURM/GPU time on it. It answers one question: for THIS model's
tokenizer, are "A" and "B" (or whichever candidate letters you pass) each
a single token right where the assistant's answer would begin -- and are
they actually plausible answer tokens for a multiple-choice-style prompt,
not buried far down the model's own ranking?

Needs the real model loaded (same --model_path/--dtype/--backend as the
main job), so it needs a GPU node, but it is a few seconds of actual
inference -- run it in an interactive session (e.g. `srun --pty
--gres=gpu:1 ...`) rather than a full sbatch job.

What it checks, and why each matters:

1. Single-token check. logprob_many (src/client.py) reads ONE specific
   vocabulary id's logit per candidate letter. If "A" tokenizes into two
   or more pieces for this tokenizer, there is no single id to read, and
   the whole design breaks silently (it would read the logit of the
   FIRST sub-token only, which is not the same thing as "the model's
   probability of answering A"). This must be a hard failure, not a
   warning.

2. Plausibility check. Even if "A"/"B" are single tokens, the model might
   essentially never put meaningful probability mass on them at the
   assistant-turn-start position for THIS chat template (e.g. if Apertus
   was never instruction-tuned on this exact answer-with-a-letter
   pattern). This script builds one real example prompt (same chat
   template/format as the main pipeline) and prints the top-10 most
   likely next tokens with their probabilities, so you can eyeball
   whether "A"/"B" (or close variants, e.g. a leading-space " A") show up
   near the top -- not just that they CAN be read, but that reading them
   means something.

If "A"/"B" fail either check, try a different pair with
--candidate_letters (e.g. "Y,N" or "1,2") and re-run.

Usage
-----
  python scripts/check_step4b_logit_tokens.py \
      --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
      --dtype bf16 --backend transformers \
      --candidate_letters A,B
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.client import TransformersClient, LLMConfig
from src.step4b_logit_prompts import CATEGORIES, build_question_prompt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", required=True, choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--candidate_letters", default="A,B")
    args = ap.parse_args()

    candidate_letters = [s.strip() for s in args.candidate_letters.split(",") if s.strip()]
    if len(candidate_letters) != 2:
        print(f"[FAIL] --candidate_letters must give exactly 2 letters, got {candidate_letters!r}")
        return 1

    print(f"[check] Loading {args.model_path} (backend={args.backend}, dtype={args.dtype})...")
    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
        )
    )

    if args.backend != "transformers":
        print("[FAIL] logprob_many (and this check) only supports backend='transformers'.")
        return 1

    # --- Check 1: single-token encoding for each candidate letter ---
    print(f"\n=== Check 1: single-token encoding ===")
    all_single = True
    for letter in candidate_letters:
        ids = client.tok.encode(letter, add_special_tokens=False)
        ok = len(ids) == 1
        all_single &= ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {letter!r} -> token ids {ids}"
              + ("" if ok else f"  (expected exactly 1 token, got {len(ids)})"))
        # Also check a leading-space variant, since some tokenizers encode
        # "A" differently depending on what precedes it -- informational
        # only (doesn't fail the check), but worth seeing.
        ids_sp = client.tok.encode(" " + letter, add_special_tokens=False)
        print(f"         (informational: {' ' + letter!r} -> token ids {ids_sp})")

    if not all_single:
        print("\n[FAIL] At least one candidate letter is not a single token for this "
              "tokenizer. Try a different --candidate_letters pair and re-run.")
        return 1

    # --- Check 2: plausibility -- build one real example prompt and look
    # at the actual next-token distribution's top 10. ---
    print(f"\n=== Check 2: plausibility (top-10 next-token predictions on a real example prompt) ===")
    example_row = pd.Series({
        "matched_alias": "Exemple Testaverson",
        "keyword": "Exemple Testaverson",
        "text": (
            "Le conseiller national Exemple Testaverson a vivement critique la decision du "
            "gouvernement lors d'une interview televisee, affirmant qu'il s'agit d'une erreur "
            "politique grave."
        ),
    })
    prompt = build_question_prompt(example_row, "text", CATEGORIES[1], order=1)  # "Politician"

    msgs = [{"role": "user", "content": prompt}]
    formatted = client.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = client.tok([formatted], return_tensors="pt", truncation=True, max_length=16384)
    enc = {k: v.to(client.model.device) for k, v in enc.items()}

    import torch
    with torch.inference_mode():
        out = client.model(**enc)
    next_logits = out.logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    top = torch.topk(probs, k=10)

    candidate_ranks = {}
    for rank, (p, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), start=1):
        tok_str = client.tok.decode([idx])
        marker = ""
        for letter in candidate_letters:
            if tok_str.strip() == letter:
                marker = f"  <-- candidate letter {letter!r}"
                candidate_ranks[letter] = rank
        print(f"  #{rank:2d}  p={p:.4f}  token={tok_str!r}{marker}")

    print()
    all_found_top10 = True
    for letter in candidate_letters:
        letter_id = client.tok.encode(letter, add_special_tokens=False)[0]
        p_letter = probs[letter_id].item()
        in_top10 = letter in candidate_ranks
        all_found_top10 &= in_top10
        status = "OK" if in_top10 else "WARNING"
        print(f"  [{status}] P({letter!r}) = {p_letter:.4f}"
              + (f" (rank #{candidate_ranks[letter]} in top-10)" if in_top10
                 else " (NOT in top-10 -- model rarely considers this letter here)"))

    print()
    if all_found_top10:
        print(f"[PASS] Both candidate letters {candidate_letters} are single tokens and "
              f"appear among the model's top-10 most likely next tokens on this example. "
              f"Safe to proceed with scripts/step4b_source_category_logit.py.")
        return 0
    else:
        print(f"[WARNING] Both letters are valid single tokens (Check 1 passed), but at "
              f"least one did not appear in the top-10 on this one example -- the "
              f"restricted-softmax probabilities will still be well-defined (we only "
              f"compare the two candidates against each other, not the full vocabulary), "
              f"but consider trying a different letter pair and comparing P(yes) spread "
              f"across a few more example prompts before committing real GPU time.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
