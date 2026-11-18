# scripts/check_remote_api.py
"""
Standalone smoke test for src/client_remote.py (RemoteAPIClient) -- run
this BY HAND, with a real UNIL litellm API key exported, BEFORE wiring
the remote backend into any real pipeline step (step3, step4a, step4b).

What it checks, and why each one matters:

  1. chat_many works at all (connectivity, auth, model name, basic
     request/response shape) -- this is the easy part, almost every
     OpenAI-compatible gateway gets this right.

  2. logprob_many actually returns usable logprobs for our A/B candidate
     letters -- this is the part that has NEVER been verified against
     litellm.dcsr.unil.ch, and the whole reason this script exists before
     touching step3/step4a/step4b for real. If this fails, logit-probing
     passes (step3's validation pass, step4a's pass3, step4b's category
     classification) simply cannot run against this backend -- only plain
     generation-based passes (step3 pass1, step4a pass1/pass2) could.

  3. Both letter orderings (A=yes/B=no and the reverse) behave sanely --
     the order-swap trick this pipeline relies on everywhere assumes the
     model has no strong built-in preference for the letter "A" itself;
     this script doesn't correct for that (that's src/step4b_logit_utils'
     job on real data), it just prints both so you can eyeball whether
     the swing between orders looks reasonable or wildly broken.

Usage (no sbatch needed -- this is cheap, ~4 API calls):
    export UNIL_LITELLM_API_KEY='...'
    python scripts/check_remote_api.py --model swiss-ai/Apertus-v1.5-70B

Run it for every model you intend to actually use (a smaller/different
model behind the same proxy might render its chat template differently,
or might not support logprobs at all even if Apertus-70B does).
"""
from __future__ import annotations

import argparse
import math
import sys

sys.path.insert(0, ".")  # so `python scripts/check_remote_api.py` finds src/ when run from the repo root

from src.client_remote import RemoteAPIClient, RemoteAPIConfig  # noqa: E402

CHAT_SYSTEM = "You are a helpful assistant. Answer in one short sentence."
CHAT_USER = "What is the capital of Switzerland?"

# Deliberately NOT using this project's real step3/step4a/step4b prompts --
# this script only wants to isolate "does logprobs/top_logprobs plumbing
# work at all", not re-validate any specific prompt's wording.
LOGPROB_SYSTEM = "You are a careful assistant that answers multiple-choice questions."
LOGPROB_USER_TEMPLATE = """\
Is the sky blue on a clear day?
Answer with exactly one letter and nothing else:
{letter_yes} = Yes
{letter_no} = No\
"""


def run_logprob_check(client: RemoteAPIClient, order: int) -> dict:
    letters = ("A", "B") if order == 1 else ("B", "A")
    letter_yes, letter_no = letters
    prompt = LOGPROB_USER_TEMPLATE.format(letter_yes=letter_yes, letter_no=letter_no)
    result = client.logprob_many(LOGPROB_SYSTEM, [prompt], candidate_letters=["A", "B"])[0]
    return {"order": order, "letter_yes": letter_yes, "letter_no": letter_no, "logprobs": result}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="https://litellm.dcsr.unil.ch/v1")
    ap.add_argument("--api_key_env", default="UNIL_LITELLM_API_KEY")
    ap.add_argument("--model", required=True, help="e.g. swiss-ai/Apertus-v1.5-70B")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    client = RemoteAPIClient(
        RemoteAPIConfig(
            api_base=args.api_base,
            model=args.model,
            api_key_env=args.api_key_env,
            concurrency=args.concurrency,
        )
    )

    print(f"=== 1. chat_many -- model={args.model} ===")
    try:
        out = client.chat_many(CHAT_SYSTEM, [CHAT_USER], max_new_tokens=50)
        print(f"  PASS -- response: {out[0]!r}")
        chat_ok = True
    except Exception as exc:  # noqa: BLE001 -- smoke test, want to see any failure clearly
        print(f"  FAIL -- {exc!r}")
        chat_ok = False

    print(f"\n=== 2. logprob_many -- both letter orderings ===")
    logprob_ok = True
    for order in (1, 2):
        try:
            r = run_logprob_check(client, order)
            lp = r["logprobs"]
            print(f"  order={order} (yes={r['letter_yes']}, no={r['letter_no']}): raw logprobs = {lp}")
            for letter, val in lp.items():
                if math.isnan(val):
                    print(f"    FAIL -- candidate letter {letter!r} never appeared in top_logprobs "
                          f"(not enough probability mass, or logprobs unsupported by this model/proxy).")
                    logprob_ok = False
                else:
                    print(f"    OK -- {letter!r}: logprob={val:.4f} (p={math.exp(val):.4f} under full-vocab softmax)")
        except Exception as exc:  # noqa: BLE001
            print(f"  order={order} FAIL -- {exc!r}")
            logprob_ok = False

    print("\n=== SUMMARY ===")
    print(f"  chat_many:    {'PASS' if chat_ok else 'FAIL'}")
    print(f"  logprob_many: {'PASS' if logprob_ok else 'FAIL'}")
    if chat_ok and logprob_ok:
        print("\nBoth checks passed -- safe to try this model on a real pipeline step.")
        return 0
    elif chat_ok and not logprob_ok:
        print("\nchat_many works but logprob_many does NOT -- this model/proxy cannot run any "
              "logit-probing pass (step3 validation, step4a pass3, step4b category classification) "
              "as-is. Plain generation-based passes (step3 pass1, step4a pass1/pass2) would still "
              "work. Do not wire this backend into a logit-probing step until this is fixed or "
              "worked around.")
        return 1
    else:
        print("\nchat_many itself failed -- check API key, --api_base, and --model name before "
              "anything else.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
