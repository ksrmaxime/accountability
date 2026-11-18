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
model's full-vocabulary softmax. Three things to know about this:

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
     anyway for a well-calibrated 2-way question. CONFIRMED working
     against swiss-ai/Apertus-v1.5-70B via scripts/check_remote_api.py
     on 2026-11-18 (both letters present in both letter orderings, and
     the model's actual answer was consistent across the order-swap --
     a good sign for the de-biasing trick this pipeline relies on
     everywhere). Still worth re-running that check for any OTHER model
     tried through this client -- a missing letter returns NaN (same
     fail-safe convention as src/client.py's OOM skip) rather than a
     silently wrong guess.

  3. CAVEAT specific to "-thinking" model variants (e.g.
     swiss-ai/Apertus-v1.5-70B-thinking), added 2026-11-18: logprob_many
     requests max_tokens=1 and reads the logprobs of THAT single
     generated token. This is only meaningful if the first token the
     model generates is already part of its real answer. A model that
     always emits an internal chain-of-thought block before its answer
     would instead return, as "the first token", the start of that
     reasoning -- and a candidate letter could still coincidentally show
     up somewhere in that token's top_logprobs (point 2 above would
     still technically "pass"), while being completely meaningless as a
     measure of the model's actual answer probability. This client does
     NOT attempt to detect or work around that on its own (it has no way
     to know, from here, whether a given model/proxy combination
     reasons before answering, or whether the proxy exposes a way to
     turn that off for a single call). See raw_request() below and
     scripts/check_remote_api.py's thinking-mode structure probe, which
     checks this explicitly before any "-thinking" model is trusted with
     a logit-probing pass (step3 validation, step4a pass3, step4b
     category classification).

API key handling: never hardcoded here or in any committed sbatch
script. Loaded via python-dotenv's load_dotenv() from a gitignored .env
file at the repo root (see .env.example) -- same convention as
src/download_src.py's existing SWISSDOX_API_KEY/SWISSDOX_API_SECRET.
Raises immediately and clearly if the key is missing, rather than
sending an unauthenticated request and failing confusingly downstream.

Usage telemetry (2026-11-18, added at Maxime's request -- he is, in his
own words, "beta testing" this UNIL/CSCS service and wants to come away
with real numbers on latency/throughput/token volume, independent of
whatever this specific pipeline run's own results turn out to be): every
request made through this client is counted, by kind ("chat" for
chat_many, "logprob" for logprob_many), tracking request count, retries,
failures, prompt/completion/total tokens (read from the standard OpenAI
"usage" field the server returns), and per-request latency. A short
progress line prints every --progress_every successful requests (so a
long sbatch .out log shows live throughput, not just a result at the
very end), and print_stats() prints a full summary -- called
automatically at the end of every pipeline step that builds its client
via src/client.build_client() (see that function's call sites). None of
this affects the pipeline's actual results in any way -- it is pure
observability, kept entirely separate from chat_many/logprob_many's
return values.
"""
from __future__ import annotations

import os
import random
import threading
import time
from collections import defaultdict
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
    top_logprobs: int = 20              # OpenAI spec max -- see module docstring.
    progress_every: int = 25            # print a running stats line every N
                                         # successful requests (0 disables it,
                                         # print_stats() at the end still works).


_EMPTY_KIND_STATS = {
    "n": 0, "n_retried": 0, "retries": 0, "n_failed": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "latency_s": 0.0,
}


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

        # --- Usage telemetry (see module docstring) ---
        self._stats_lock = threading.Lock()
        self._run_start = time.monotonic()
        self._stats_by_kind: Dict[str, dict] = defaultdict(lambda: dict(_EMPTY_KIND_STATS))
        self._n_done = 0  # successful requests, all kinds combined -- drives --progress_every

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
            resp = self._post_with_retry(payload, kind="chat")
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected chat completion response shape: {resp!r}") from exc

        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(_one, user_prompts))

    # ------------------------------------------------------------------
    # Interface identical to TransformersClient.logprob_many -- see module
    # docstring for why log-probabilities instead of raw logits is exact,
    # not an approximation. See also the module docstring's CAVEAT #3 on
    # "-thinking" model variants before trusting this for one.
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
                                  # as the local backend's logprob_many. See module
                                  # docstring CAVEAT #3: this assumes the first
                                  # generated token is already the real answer,
                                  # which can be false for "-thinking" variants.
                "logprobs": True,
                "top_logprobs": self.cfg.top_logprobs,
            }
            resp = self._post_with_retry(payload, kind="logprob")
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
    def _post_with_retry(self, payload: dict, kind: str = "other") -> dict:
        start = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._session.post(self._url, json=payload, timeout=self.cfg.request_timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                resp = r.json()
                self._record_success(kind, attempt, time.monotonic() - start, resp)
                return resp
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
        with self._stats_lock:
            self._stats_by_kind[kind]["n_failed"] += 1
        raise RuntimeError(f"Request failed after {self.cfg.max_retries} retries: {last_exc}")

    # ------------------------------------------------------------------
    # Diagnostic escape hatch -- 2026-11-18, added for
    # scripts/check_remote_api.py's thinking-mode structure probe (see
    # module docstring CAVEAT #3). Sends a raw, caller-built payload
    # through the exact same retry/stats machinery as chat_many/
    # logprob_many, but returns the FULL, unparsed JSON response instead
    # of extracting just the text or just the candidate logprobs --
    # needed to inspect things chat_many/logprob_many deliberately throw
    # away (e.g. a "reasoning_content" field, or exactly what token a
    # max_tokens=1 call actually produced). Not used by any pipeline
    # step itself; only by diagnostic tooling.
    # ------------------------------------------------------------------
    def raw_request(self, payload: dict, kind: str = "diag") -> dict:
        return self._post_with_retry(payload, kind=kind)

    # ------------------------------------------------------------------
    # Usage telemetry -- see module docstring. Pure observability, never
    # read by chat_many/logprob_many's own return values.
    # ------------------------------------------------------------------
    def _record_success(self, kind: str, attempt: int, latency_s: float, resp: dict) -> None:
        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        tt = usage.get("total_tokens") or (pt + ct)
        with self._stats_lock:
            s = self._stats_by_kind[kind]
            s["n"] += 1
            if attempt > 0:
                s["n_retried"] += 1
            s["retries"] += attempt
            s["latency_s"] += latency_s
            s["prompt_tokens"] += pt
            s["completion_tokens"] += ct
            s["total_tokens"] += tt
            self._n_done += 1
            n_done = self._n_done
        if self.cfg.progress_every and n_done % self.cfg.progress_every == 0:
            self._print_progress(n_done)

    def _print_progress(self, n_done: int) -> None:
        elapsed = time.monotonic() - self._run_start
        with self._stats_lock:
            total_tokens = sum(s["total_tokens"] for s in self._stats_by_kind.values())
        rate = total_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[remote-api progress] {n_done} requests done | {elapsed:.0f}s elapsed | "
            f"{total_tokens:,} tokens so far | ~{rate:.1f} tok/s (wall-clock, "
            f"concurrency={self.cfg.concurrency})",
            flush=True,
        )

    def print_stats(self) -> None:
        """Full usage summary -- call once, typically at the end of a
        pipeline step's main(). Safe to call even if zero requests were
        made (e.g. an empty input). Purely informational: this is beta-test
        telemetry for the UNIL/CSCS service, unrelated to the pipeline's
        actual results."""
        elapsed = time.monotonic() - self._run_start
        with self._stats_lock:
            snapshot = {k: dict(v) for k, v in self._stats_by_kind.items()}

        print("=== RemoteAPIClient usage stats (beta-test telemetry -- unrelated to pipeline results) ===", flush=True)
        print(f"  model={self.cfg.model} api_base={self.cfg.api_base} concurrency={self.cfg.concurrency}", flush=True)
        print(f"  wall-clock elapsed since client creation: {elapsed:.1f}s", flush=True)

        if not snapshot:
            print("  No requests were made.", flush=True)
            return

        grand = dict(_EMPTY_KIND_STATS)
        for kind in sorted(snapshot):
            s = snapshot[kind]
            for k in grand:
                grand[k] += s[k]
            print(
                f"  [{kind}] requests={s['n']} (retried={s['n_retried']}, "
                f"total_retries={s['retries']}, failed={s['n_failed']})",
                flush=True,
            )
            if s["n"]:
                print(
                    f"           tokens: prompt={s['prompt_tokens']:,} "
                    f"completion={s['completion_tokens']:,} total={s['total_tokens']:,} | "
                    f"avg_latency={s['latency_s'] / s['n']:.2f}s/request",
                    flush=True,
                )

        print(
            f"  [ALL KINDS] requests={grand['n']} (retried={grand['n_retried']}, "
            f"total_retries={grand['retries']}, failed={grand['n_failed']})",
            flush=True,
        )
        if grand["n"]:
            print(
                f"              tokens: prompt={grand['prompt_tokens']:,} "
                f"completion={grand['completion_tokens']:,} total={grand['total_tokens']:,} | "
                f"avg_latency={grand['latency_s'] / grand['n']:.2f}s/request",
                flush=True,
            )
        if elapsed > 0 and grand["total_tokens"]:
            print(
                f"  Effective throughput (wall-clock, accounts for concurrency="
                f"{self.cfg.concurrency}): {grand['total_tokens'] / elapsed:.1f} tok/s total, "
                f"{grand['completion_tokens'] / elapsed:.1f} tok/s completion-only",
                flush=True,
            )
        elif grand["n"] and not grand["total_tokens"]:
            print(
                "  No token usage reported by the server (the 'usage' field was missing or "
                "empty in every response) -- cannot compute throughput from this run.",
                flush=True,
            )
