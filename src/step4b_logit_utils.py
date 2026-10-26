# src/step4b_logit_utils.py
"""
Pure math helpers for the 2026-10-26+ logit-probing redesign of step 4b:
turning "which of 6 categories does the critic belong to" into 6
independent binary yes/no probability judgments read directly off the
model's next-token logits (no generation), each de-biased for (a) which
letter the model prefers regardless of content, and (b) each question's
own intrinsic yes-bias.

See scripts/step4b_source_category_logit.py for how these are used, and
src/step4b_logit_prompts.py for the prompts themselves. Background: Zhao
et al., "Calibrate Before Use: Improving Few-Shot Performance of Language
Models", ICML 2021 (the context-free calibration step); Zheng et al.,
"Large Language Models Are Not Robust Multiple Choice Selectors", ICLR
2024 (the order-swap averaging step).

Deliberately dependency-free (no numpy/scipy) and fully unit-tested via
this file's own __main__ block (`python src/step4b_logit_utils.py`),
including a numeric check against the worked example in the user's own
reference write-up: p=0.70/p0=0.65 -> ~0.56, p=0.62/p0=0.40 -> ~0.71.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

_EPS = 1e-6


def _clip(p: float) -> float:
    """Keep p strictly inside (0, 1) so logit() never hits +/-inf."""
    return min(max(p, _EPS), 1.0 - _EPS)


def logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Numerically stable logistic function (avoids overflow in exp(x)
    for large positive x)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax_restricted(logits_by_token: Dict[str, float]) -> Dict[str, float]:
    """Softmax over ONLY the given candidate tokens' logits -- never the
    full vocabulary. Numerically stable (subtracts the max logit first).
    A NaN logit (e.g. from a GPU-OOM skip, see src/client.py's
    logprob_many) propagates to NaN for every entry, which downstream
    code treats as "this read failed" rather than a spurious 0/1."""
    if not logits_by_token:
        return {}
    m = max(logits_by_token.values())
    exps = {tok: math.exp(v - m) for tok, v in logits_by_token.items()}
    total = sum(exps.values())
    return {tok: v / total for tok, v in exps.items()}


def average_yes_over_orderings(p_yes_order1: float, p_yes_order2: float) -> float:
    """Cancel a model's intrinsic preference for the letter "A" or "B"
    (independent of content) by averaging the same yes/no question asked
    in both letter-to-meaning assignments (Zheng et al. 2024)."""
    return 0.5 * (p_yes_order1 + p_yes_order2)


def calibrate(p: float, p0: float) -> float:
    """Context-free calibration (Zhao et al. 2021): remove the question's
    own intrinsic yes-bias (measured on an empty/"N/A" article, p0) from
    the observed probability p, by subtracting in logit space.
    Equivalent to: odds_corrected = odds(p) / odds(p0)."""
    return sigmoid(logit(p) - logit(p0))


def pick_argmax_category(corrected_probs: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    """Returns (best_category, best_probability), or (None, None) if
    corrected_probs is empty (e.g. every category's reading was NaN)."""
    if not corrected_probs:
        return None, None
    best_cat = max(corrected_probs, key=corrected_probs.get)
    return best_cat, corrected_probs[best_cat]


if __name__ == "__main__":
    # Sanity check against the worked example in the user's reference doc:
    # X: p=0.70, p0=0.65 -> corrected ~0.56
    # Y: p=0.62, p0=0.40 -> corrected ~0.71
    for name, p, p0, expected in [("X", 0.70, 0.65, 0.56), ("Y", 0.62, 0.40, 0.71)]:
        got = calibrate(p, p0)
        print(f"{name}: p={p} p0={p0} -> corrected={got:.4f} (expected ~{expected})")
        assert abs(got - expected) < 0.01, f"MISMATCH for {name}"

    r = softmax_restricted({"A": 1.23, "B": 1.23})
    assert abs(r["A"] - 0.5) < 1e-9 and abs(r["B"] - 0.5) < 1e-9
    print("softmax_restricted equal-logits OK:", r)

    r2 = softmax_restricted({"A": 20.0, "B": 0.0})
    print("softmax_restricted big-gap OK:", r2)
    assert r2["A"] > 0.999

    avg = average_yes_over_orderings(0.9, 0.3)
    print("average_yes_over_orderings:", avg)
    assert abs(avg - 0.6) < 1e-9

    cat, prob = pick_argmax_category({"Politician": 0.3, "Civil Servant": 0.71, "Other": 0.05})
    print("argmax:", cat, prob)
    assert cat == "Civil Servant"

    print("\nALL CHECKS PASSED")
