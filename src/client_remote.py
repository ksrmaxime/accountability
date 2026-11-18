# src/client_remote.py
"""
RemoteAPIClient -- 2026-11-18, drop-in replacement for TransformersClient
(src/client.py) that sends requests to an OpenAI-compatible LLM gateway
instead of loading a model locally on a CURNAGL GPU.

Context: UNIL gave Maxime personal access to a litellm proxy
(https://litellm.dcsr.unil.ch/v1) that fronts both UNIL-CI-hosted models
(qwen3.8-flash-next, gemma-4-26b-a4b, nemotron-3.5-super-vl) and
CSCS-hosted models (zai-org/GLM-5.3, and several Apertus v1.5 variants,
including swiss-ai/Apertus-v1.5-70B -- a much bigger model than the 8B
this whole pipeline has run on so far, worth testing given the 8B's
observed ceiling). The point of this file is to let any pipeline step
try that 70B over the network, on a plain CPU allocation, with ZERO
changes to the step's own script logic, prompts or parsing -- only
src/client.py's build_client() factory decides whether to construct this
class or the existing TransformersClient, based on --backend.

Same public interface as TransformersClient: chat_many(...) and
logprob_many(...), same argument names, same return shapes. Every
src/*_prompts.py and src/verify_utils.py / src/step*_logit_utils.py
caller works completely unchanged.

What's genuinely different from the local backend:

  - No GPU to batch a forward pass on, so "concurrency" here means
    concurrent HTTP requests (a thread pool), not a single batched
    tensor op. --concurrency replaces --batch_size's role as the
    parallelism knob for this backend specifically.

  - Network calls can fail transiently in ways a local forward pass
    never does (rate limits, timeouts, 5xx) -- _post_with_retry handles
    this with exponential backoff + jitter, same "never silently drop a
    row, never retry forever" spirit as src/client.py's OOM-split-and-
    retry for local batches.

  - No local tokenizer, so max_input_tokens is accepted (for call-site
    compatibility with TransformersClient) but not used: truncation is
    left to the remote server's own context-window handling. If a
    prompt is ever actually too long for a given model, the server will
    return an error, which surfaces as a normal retry/fail here rather
    than a silent local truncation -- worth checking for on real runs,
    since this pipeline's articles can exceed many models' context
    windows (see src/client.py's own truncation-side history).

Why logprob_many needed real design work, not just an HTTP wrapper: the
local version returns raw pre-softmax logits for 1-2 single-token
candidate letters, read directly off the model's output tensor in one
forward pass (never a generation loop). An OpenAI-compatible chat
endpoint has no equivalent concept -- the closest thing it exposes is
chat.completions' logprobs/top_logprobs parameters, which return the
FIRST GENERATED TOKEN's top-N (token, log-probability) pairs under the
model's full-vocabulary softmax. Two things to know this is fine for:

  1. Units: downstream code (src/step4b_logit_utils.softmax_restricted)
     computes a 2-way softmax restricted to exactly the 2 candidate
     values it's handed. Whether those 2 values are raw pre-softmax
     logits (local backend) or full-vocabulary log-probabilities
     (OpenAI-style, this backend) makes NO difference to that function's
     output: if logit_A, logit_B are the raw logits and Z is the shared
     softmax normalizer, then logprob_X = logit_X - ln(Z) for X in
     {A,B} -- the -ln(Z) term is IDENTICAL for both candidates (same
     softmax, same forward pass) and cancels exactly in the 2-way ratio
     exp(logit_A)/(exp(logit_A)+exp(logit_B))
       == exp(logprob_A)/(exp(logprob_A)+exp(logprob_B)).
     So this file can hand softmax_restricted raw log-probabilities
     directly, with no conversion -- confirmed against this codebase's
     actual softmax_restricted (plain max-shifted exp/sum, nothing else)
     on 2026-11-18, not just asserted.

  2. Coverage: top_logprobs is capped at 20 by the OpenAI spec (most
     vLLM-based servers, which litellm's CSCS backends almost certainly
     are, mirror this). A candidate letter not present in the returned
     top-20 simply never had enough probability mass to be useful
     anyway for a well-calibrated 2-way question, but this has NOT been
     empirically confirmed against litellm.dcsr.unil.ch as of this
     writing -- scripts/check_remote_api.py checks this directly (both
     candidate letters appearing, with sane-looking logprob values) and
     MUST be run by hand, with a real API key, before trusting
     logprob_many on any real pipeline step. A missing letter returns
     NaN (same fail-safe convention as src/client.py's OOM skip) rather
     than a silently wrong guess.

API key handling: never hardcoded here or in any committed sbatch
script. Read from an environment variable (--api_key_env, default
UNIL_LITELLM_API_KEY) that the operator exports in their own shell
before launching -- raises immediately and clearly if it's unset,
rather than sending an unauthenticated request and failing confusingly
downstream.
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import requests
from dotenv import load_dotenv  # same convention as src/download_src.py's own
                                 # SWISSDOX_API_KEY/SWISSDOX_API_SECRET handling


@dataclass(frozen=True)
class RemoteAPIConfig:
    api_base: str                       # e.g. "https://litellm.dcsr.unil.ch/v1"
    model: str                          # e.g. "swiss-ai/Apertus-v1.5-70B"
    api_key_env: str = "UNIL_LITELLM_API_KEY"
    concurrency: int = 6                # see module docstring -- no rate-limit
                                         # guidance was given for this key as of
                                         # 2026-11-18, so this errs conservative.
    max_retries: int = 5
    request_timeout: float = 120.0
    top_logprobs: int = 20               # OpenAI spec max -- see module docstring.


class RemoteAPIClient:
    def __init__(self, cfg: RemoteAPIConfig):
        self.cfg = cfg
        # Same pattern as src/download_src.py's SWISSDOX_API_KEY handling:
        # load_dotenv() searches the current/parent directories for a .env
        # file and loads it into os.environ (never overwriting a variable
        # already exported in the shell). So a line like
        #   UNIL_LITELLM_API_KEY=sk-...
        # in a gitignored .env at the repo root (see .env.example) is picked
        # up automatically, with zero sbatch-script changes needed -- no
        # "export" in your shell required, same as the existing download step.
        load_dotenv()
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"'{cfg.api_key_env}' is not set. Add it to a .env file at the repo root "
                f"(see .env.example), e.g.:\n"
                f"  {cfg.api_key_env}=sk-...\n"
                f".env is gitignored -- this is a per-machine file, so add this line on "
                f"EVERY machine that needs to run this (CURNAGL and your laptop are separate "
                f"copies; git never carries .env between them). Never hardcode the key in a "
                f"committed script."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._url = cfg.api_base.rstrip("/") + "/chat/completions"

    # ------------------------------------------------------------------
    # Interface identical to TransformersClient.chat_many
    # ------------------------------------------------------------------
    def chat_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        max_input_tokens: int = 7168,  # accepted, unused -- see module docstring
    ) -> List[str]:
        if not user_prompts:
            return []

        def _one(up: str) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": up})
            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_new_tokens),
            }
            resp = self._post_with_retry(payload)
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected chat completion response shape: {resp!r}") from exc

        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(_one, user_prompts))

    # ------------------------------------------------------------------
    # Interface identical to TransformersClient.logprob_many -- see module
    # docstring for why log-probabilities instead of raw logits is exact,
    # not an approximation.
    # ------------------------------------------------------------------
    def logprob_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        candidate_letters: List[str],
        max_input_tokens: int = 7168,  # accepted, unused -- see module docstring
    ) -> List[dict]:
        if not user_prompts:
            return []

        def _one(up: str) -> dict:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": up})
            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1,  # only the first generated token's distribution
                                  # is needed -- same "single forward pass" intent
                                  # as the local backend's logprob_many.
                "logprobs": True,
                "top_logprobs": self.cfg.top_logprobs,
            }
            resp = self._post_with_retry(payload)
            return self._extract_candidate_logprobs(resp, candidate_letters)

        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(_one, user_prompts))

    @staticmethod
    def _extract_candidate_logprobs(resp: dict, candidate_letters: List[str]) -> Dict[str, float]:
        try:
            top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            # The server/proxy didn't return logprobs at all -- fail loudly
            # rather than silently returning all-NaN, which downstream code
            # would treat identically to "letter just wasn't in the top N"
            # (a real, expected case) and this would hide a real config
            # problem (model/proxy doesn't support logprobs on this route).
            raise RuntimeError(
                "No logprobs in the API response for this model/proxy -- "
                "logprob_many cannot work here. Run scripts/check_remote_api.py "
                f"first to confirm support before relying on this. Response was: {resp!r}"
            ) from exc

        by_token = {entry["token"]: entry["logprob"] for entry in top}
        result: Dict[str, float] = {}
        for letter in candidate_letters:
            # Tokenizers sometimes return a leading-space variant (" A"
            # instead of "A") depending on how the chat template renders
            # the generation prompt -- check both before giving up.
            if letter in by_token:
                result[letter] = by_token[letter]
            elif f" {letter}" in by_token:
                result[letter] = by_token[f" {letter}"]
            else:
                result[letter] = float("nan")
        return result

    # ------------------------------------------------------------------
    def _post_with_retry(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._session.post(self._url, json=payload, timeout=self.cfg.request_timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.cfg.max_retries:
                    break
                # Exponential backoff + jitter -- never give up on the first
                # transient failure, never retry forever either.
                sleep_s = min(60.0, float(2 ** attempt)) + random.uniform(0, 1.0)
                print(
                    f"[retry] request failed ({exc}); retrying in {sleep_s:.1f}s "
                    f"(attempt {attempt + 1}/{self.cfg.max_retries})",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Request failed after {self.cfg.max_retries} retries: {last_exc}")
