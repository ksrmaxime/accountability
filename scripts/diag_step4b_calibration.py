# scripts/diag_step4b_calibration.py
"""
2026-11-26 -- standalone diagnostic, run BY HAND (no sbatch needed -- this
is cheap, 6 categories x 2 orders = 12 API calls), to root-cause the
recurring source_category_p_civil_servant == NaN for ALL SOURCED rows bug
seen on both job21084261 and job21085911.

Context: compute_calibration() in scripts/step4b_source_category_logit.py
sends each category's "empty-article" calibration prompt through
client.logprob_many(..., max_tokens=1), which reads ONLY the first
generated token's top_logprobs for candidate letters A/B
(src/client_remote.py's RemoteAPIClient._extract_candidate_logprobs). If a
candidate letter is never in the top-20, that letter's p comes back NaN
silently (by design -- see that method's own docstring: this is the
correct, expected behaviour for a genuinely low-probability letter, NOT an
error). compute_calibration() then averages p(yes) over both letter
orderings into p0[cat]; if EITHER order came back fully NaN for both
letters, p0[cat] itself is NaN, and every row's calibrated probability for
that category is forced to NaN a few lines later in main() (see that
file's own 2026-11-26 checkpoint-resume changelog entry for the full
chain).

The first hypothesis (an embedded question mark in the Civil Servant
definition, confusing the template) was fixed on 2026-11-26 and DID NOT
resolve the problem -- job21085911 (run AFTER that fix was deployed and
verified on-device) still shows source_category_p_civil_servant NaN for
all 97 SOURCED rows, identical to job21084261. So the embedded-question-
mark theory is DISPROVEN; this script exists to find the real cause
instead of guessing again.

This script does NOT use logprob_many (which silently converts "letter
missing from top-20" into NaN, hiding WHY). It uses
RemoteAPIClient.raw_request() instead -- the same diagnostic escape hatch
scripts/check_remote_api.py's --probe_thinking check already uses -- to
print the FULL, unparsed response for each of the 6 real
step4b_logit_prompts_remote.CATEGORY_DEFINITION calibration prompts (both
letter orderings), so you can see directly:
  - what token the model actually generated first (a real "A"/"B", or
    something else -- e.g. the start of a refusal/explanation, or a
    reasoning token)
  - the full top-20 logprobs at that position (is a candidate letter
    present at all, just with very low probability -- or truly absent)
  - finish_reason and the full message content, in case max_tokens=1 is
    silently truncating something that would otherwise read as a letter

Usage:
    python scripts/diag_step4b_calibration.py --model swiss-ai/Apertus-v1.5-70B

Requires a .env at the repo root with UNIL_LITELLM_API_KEY=... set (same
convention as every other remote-backend script in this repo) -- this
script never prints or logs that key.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")  # so `python scripts/diag_step4b_calibration.py` finds src/ when run from the repo root

from src.client_remote import RemoteAPIClient, RemoteAPIConfig  # noqa: E402
from src.step4b_logit_prompts_remote import CATEGORIES, build_calibration_prompt  # noqa: E402


def probe_one(client: RemoteAPIClient, category: str, order: int) -> None:
    prompt = build_calibration_prompt(category, order=order)
    letters = ("A", "B") if order == 1 else ("B", "A")
    payload = {
        "model": client.cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }
    print(f"--- category={category!r} order={order} (A={letters[0]}=yes, B={letters[1]}=no) ---")
    try:
        resp = client.raw_request(payload, kind="diag")
    except Exception as exc:  # noqa: BLE001 -- diagnostic script, want to see any failure clearly
        print(f"  REQUEST FAILED: {exc!r}")
        return

    choice0 = resp.get("choices", [{}])[0]
    msg = choice0.get("message", {}) or {}
    content = msg.get("content") or ""
    finish_reason = choice0.get("finish_reason")
    lp = (choice0.get("logprobs") or {}).get("content") or []

    print(f"  finish_reason: {finish_reason!r}")
    print(f"  message.content (the actual generated token, as text): {content!r}")
    extra_keys = [k for k in msg if k not in ("role", "content")]
    if extra_keys:
        print(f"  extra message field(s) beyond role/content: {extra_keys}")
        for k in extra_keys:
            v = msg.get(k)
            shown = (v[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
            print(f"    {k!r}: {shown!r}")

    if lp:
        tok_entry = lp[0]
        top = tok_entry.get("top_logprobs") or []
        print(f"  logprobs.content[0].token: {tok_entry.get('token')!r} (logprob={tok_entry.get('logprob')})")
        print(f"  top_logprobs (all {len(top)}): "
              f"{[(t.get('token'), round(t.get('logprob', 0.0), 3)) for t in top]}")
        has_a = any(t.get("token", "").strip() == "A" for t in top)
        has_b = any(t.get("token", "").strip() == "B" for t in top)
        print(f"  'A' present in top-20: {has_a} | 'B' present in top-20: {has_b}")
    else:
        print("  WARNING -- no 'logprobs.content' in this response at all.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="https://litellm.dcsr.unil.ch/v1")
    ap.add_argument("--api_key_env", default="UNIL_LITELLM_API_KEY")
    ap.add_argument("--model", required=True, help="e.g. swiss-ai/Apertus-v1.5-70B")
    ap.add_argument(
        "--categories", nargs="*", default=None,
        help="Subset of categories to probe (default: all 6, in CATEGORIES order). "
             "Pass e.g. --categories 'Civil Servant' 'Politician' to probe just those two "
             "(Politician as a control known to calibrate fine).",
    )
    args = ap.parse_args()

    cats = args.categories if args.categories else list(CATEGORIES)
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        print(f"Unknown categor(y/ies) {unknown} -- must be one of {list(CATEGORIES)}")
        return 1

    client = RemoteAPIClient(
        RemoteAPIConfig(
            api_base=args.api_base,
            model=args.model,
            api_key_env=args.api_key_env,
            concurrency=1,  # sequential on purpose -- this is a tiny, one-off diagnostic
        )
    )

    print(f"=== step4b calibration-prompt diagnostic -- model={args.model} ===")
    print(f"Probing {len(cats)} categor(y/ies) x 2 orderings = {2 * len(cats)} calls.\n")
    for cat in cats:
        for order in (1, 2):
            probe_one(client, cat, order)

    client.print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
