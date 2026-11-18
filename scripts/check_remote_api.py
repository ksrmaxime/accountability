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

  4. (2026-11-18, added for "-thinking" model variants, e.g.
     swiss-ai/Apertus-v1.5-70B-thinking) WHETHER logprob_many's core
     assumption actually holds for this model: that the FIRST generated
     token (max_tokens=1) already IS the model's real answer, not the
     start of an internal chain-of-thought block. A model that reasons
     before answering can still return *some* letter somewhere in that
     first token's top_logprobs (so check #2 above could still nominally
     "pass"), while that letter's probability is completely meaningless
     -- it describes the start of a reasoning block, not the model's
     actual confidence in the answer. This check runs automatically for
     any model whose name contains "thinking" (case-insensitive), and can
     be forced on/off with --probe_thinking / --no_probe_thinking for any
     other model you're unsure about. It does two things a normal
     chat/logprob call doesn't: (a) a full generation with a generous
     token budget, to see the model's real output shape (is there a
     separate "reasoning_content" field? a visible <think> block in
     "content" itself?), and (b) the EXACT request shape logprob_many
     uses (max_tokens=1, logprobs=True), but reading the RAW response via
     RemoteAPIClient.raw_request() to see literally what token got
     generated -- a real candidate letter, or something else entirely.

Usage (no sbatch needed -- this is cheap, ~6-8 API calls):
    # .env at the repo root must hold UNIL_LITELLM_API_KEY=... (see
    # .env.example) -- no "export" needed, RemoteAPIClient loads it itself.
    python scripts/check_remote_api.py --model swiss-ai/Apertus-v1.5-70B
    python scripts/check_remote_api.py --model swiss-ai/Apertus-v1.5-70B-thinking

Run it for every model you intend to actually use (a smaller/different
model behind the same proxy might render its chat template differently,
or might not support logprobs at all even if Apertus-70B does; a
"-thinking" variant specifically needs check #4 above before you trust
it with any logit-probing pass).
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


def run_thinking_diagnostic(client: RemoteAPIClient, model: str, probe_tokens: int) -> dict:
    """See module docstring, check #4. Returns a dict with a "verdict" key
    in {"COMPATIBLE", "LIKELY INCOMPATIBLE", "FAIL", "UNKNOWN"}."""
    print(f"\n=== 4. thinking-mode structure probe -- model={model} (probe_tokens={probe_tokens}) ===")
    result: dict = {"verdict": "UNKNOWN"}

    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LOGPROB_SYSTEM},
            {"role": "user", "content": LOGPROB_USER_TEMPLATE.format(letter_yes="A", letter_no="B")},
        ],
        "temperature": 0.0,
    }

    # --- Step A: full generation with a generous token budget -- what does
    # this model's real output actually look like? ---
    payload_full = dict(base_payload, max_tokens=probe_tokens)
    try:
        resp_full = client.raw_request(payload_full, kind="diag_full")
        msg = resp_full["choices"][0]["message"]
        content = msg.get("content") or ""
        extra_keys = [k for k in msg if k not in ("role", "content")]

        print(f"  full generation -- message keys: {list(msg.keys())}")
        if extra_keys:
            print(f"  extra field(s) beyond role/content: {extra_keys}")
            for k in extra_keys:
                v = msg.get(k)
                shown = (v[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
                print(f"    {k!r}: {shown!r}")
        preview = content[:500] + ("..." if len(content) > 500 else "")
        print(f"  content ({len(content)} chars): {preview!r}")

        has_visible_think_block = "<think" in content.lower()
        if has_visible_think_block:
            print("  -> 'content' itself contains what looks like a visible reasoning block "
                  "(e.g. a <think> tag) before the answer.")
        result.update(
            full_gen_ok=True,
            reasoning_field_seen=bool(extra_keys),
            extra_keys=extra_keys,
            has_visible_think_block=has_visible_think_block,
            content_preview=content[:200],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL -- full generation request itself failed: {exc!r}")
        result.update(full_gen_ok=False, verdict="FAIL")
        return result

    # --- Step B: the EXACT request shape logprob_many uses (max_tokens=1,
    # logprobs=True) -- what does the single returned token actually decode
    # to, independent of whether a candidate letter happens to appear
    # somewhere in its top_logprobs? ---
    payload_1tok = dict(base_payload, max_tokens=1, logprobs=True, top_logprobs=20)
    try:
        resp_1tok = client.raw_request(payload_1tok, kind="diag_1tok")
        msg1 = resp_1tok["choices"][0]["message"]
        token0_text = msg1.get("content") or ""
        lp = resp_1tok["choices"][0].get("logprobs") or {}
        lp_content = lp.get("content") or []

        print(f"\n  single-token (max_tokens=1) call -- actual generated token's content: {token0_text!r}")
        if lp_content:
            tok_entry = lp_content[0]
            top = tok_entry.get("top_logprobs") or []
            print(f"  logprobs token field: {tok_entry.get('token')!r} (logprob={tok_entry.get('logprob')})")
            print(f"  top_logprobs (first 10 of {len(top)}): "
                  f"{[(t.get('token'), round(t.get('logprob', 0.0), 3)) for t in top[:10]]}")
        else:
            print("  WARNING -- no 'logprobs.content' in this response at all.")

        is_letter = token0_text.strip() in ("A", "B")
        result.update(single_token_text=token0_text, single_token_is_letter=is_letter)

        if is_letter:
            result["verdict"] = "COMPATIBLE"
            print(f"\n  -> VERDICT: COMPATIBLE. The first generated token IS a direct answer "
                  f"letter ({token0_text!r}). logprob_many's single-token design looks safe to "
                  f"use for this model as currently configured.")
        else:
            result["verdict"] = "LIKELY INCOMPATIBLE"
            print(f"\n  -> VERDICT: LIKELY INCOMPATIBLE. The first generated token is NOT a "
                  f"candidate letter (got {token0_text!r} instead). This strongly suggests the "
                  f"model emits reasoning BEFORE its answer, so max_tokens=1 only captures the "
                  f"start of that reasoning -- any letter found in its top_logprobs would be "
                  f"COINCIDENTAL, not a measure of the model's real answer confidence. Do NOT "
                  f"trust logprob_many for this model as currently configured: step3's validation "
                  f"pass, step4a's pass3, and step4b's category classification would all be "
                  f"silently wrong, not just imprecise. Use SKIP_VALIDATION=1 (step3) / the "
                  f"equivalent for other steps until this is resolved -- e.g. a proxy-side "
                  f"parameter to disable extended thinking for a single call, if one exists; this "
                  f"script does not know of or try one, since none has been confirmed for this "
                  f"proxy.")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL -- single-token logprobs request itself failed: {exc!r}")
        result.update(single_token_is_letter=False, verdict="FAIL")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="https://litellm.dcsr.unil.ch/v1")
    ap.add_argument("--api_key_env", default="UNIL_LITELLM_API_KEY")
    ap.add_argument("--model", required=True, help="e.g. swiss-ai/Apertus-v1.5-70B")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--probe_thinking", dest="probe_thinking", action="store_true", default=None,
        help="Force-run check #4 (thinking-mode structure probe) even if 'thinking' isn't in "
             "--model's name. Default: auto-on when --model's name contains 'thinking' "
             "(case-insensitive), auto-off otherwise.",
    )
    ap.add_argument(
        "--no_probe_thinking", dest="probe_thinking", action="store_false",
        help="Force-skip check #4, even for a model whose name contains 'thinking'.",
    )
    ap.add_argument(
        "--thinking_probe_tokens", type=int, default=1500,
        help="max_tokens for check #4's full-generation step -- needs to be generous enough for "
             "a reasoning model to actually finish its chain-of-thought and reach an answer "
             "(default 1500; raise it if 'content' below looks cut off mid-reasoning).",
    )
    args = ap.parse_args()

    do_thinking_probe = args.probe_thinking
    if do_thinking_probe is None:
        do_thinking_probe = "thinking" in args.model.lower()

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

    thinking_result = None
    if do_thinking_probe:
        thinking_result = run_thinking_diagnostic(client, args.model, args.thinking_probe_tokens)
    else:
        print(f"\n=== 4. thinking-mode structure probe -- SKIPPED "
              f"(model name doesn't contain 'thinking'; pass --probe_thinking to force it) ===")

    client.print_stats()

    print("\n=== SUMMARY ===")
    print(f"  chat_many:    {'PASS' if chat_ok else 'FAIL'}")
    print(f"  logprob_many: {'PASS' if logprob_ok else 'FAIL'}")
    if thinking_result is not None:
        print(f"  thinking-mode structure probe: {thinking_result['verdict']}")

    thinking_blocks_logprobs = thinking_result is not None and thinking_result["verdict"] != "COMPATIBLE"

    if chat_ok and logprob_ok and not thinking_blocks_logprobs:
        print("\nAll checks passed -- safe to try this model on a real pipeline step, "
              "including logit-probing passes.")
        return 0
    elif chat_ok and logprob_ok and thinking_blocks_logprobs:
        print(f"\nchat_many and the basic logprob_many check both passed, BUT the thinking-mode "
              f"structure probe says {thinking_result['verdict']} -- that basic pass is "
              f"misleading for this model (see check #4's output above for why). Plain "
              f"generation-based passes (step3 pass1, step4a pass1/pass2) are fine; do NOT use "
              f"this model for any logit-probing pass until that's resolved.")
        return 1
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
