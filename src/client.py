# src/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LLMConfig:
    model_path: str
    dtype: str = "bf16"            # "bf16" | "fp16" | "auto"
    trust_remote_code: bool = True
    max_model_len: int = 8192      # context window (vLLM only)
    backend: str = "vllm"          # "vllm" | "transformers"


class TransformersClient:
    def __init__(self, cfg: LLMConfig):
        if cfg.backend == "vllm":
            self._init_vllm(cfg)
        elif cfg.backend == "transformers":
            self._init_transformers(cfg)
        else:
            raise ValueError(f"Unknown backend: '{cfg.backend}'. Choose 'vllm' or 'transformers'.")

    # ------------------------------------------------------------------
    # vLLM backend (AWQ, GPTQ, BF16 — recommandé sur cluster)
    # ------------------------------------------------------------------
    def _init_vllm(self, cfg: LLMConfig):
        from vllm import LLM, SamplingParams

        self._backend = "vllm"
        self._SamplingParams = SamplingParams

        self.llm = LLM(
            model=cfg.model_path,
            dtype=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            max_model_len=cfg.max_model_len,
        )
        self.tok = self.llm.get_tokenizer()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # ------------------------------------------------------------------
    # HuggingFace Transformers backend (BF16 natif — pour Apertus etc.)
    # ------------------------------------------------------------------
    def _init_transformers(self, cfg: LLMConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._backend = "transformers"
        self.torch = torch

        self.tok = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            dtype=dtype,
            device_map="auto",
        ).eval()

    # ------------------------------------------------------------------
    # Interface commune — identique peu importe le backend
    # ------------------------------------------------------------------
    def chat_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        max_input_tokens: int = 7168,
    ) -> List[str]:
        if not user_prompts:
            return []

        prompts = []
        for up in user_prompts:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": up})
            prompts.append(
                self.tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        if self._backend == "vllm":
            return self._generate_vllm(prompts, temperature, max_new_tokens, max_input_tokens)
        else:
            return self._generate_transformers(prompts, temperature, max_new_tokens, max_input_tokens)

    def _generate_vllm(self, prompts, temperature, max_new_tokens, max_input_tokens):
        truncated = []
        for p in prompts:
            ids = self.tok.encode(p)
            if len(ids) > max_input_tokens:
                ids = ids[:max_input_tokens]
                p = self.tok.decode(ids, skip_special_tokens=False)
            truncated.append(p)

        sampling = self._SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        outputs = self.llm.generate(truncated, sampling)
        return [out.outputs[0].text.strip() for out in outputs]

    def _generate_transformers(self, prompts, temperature, max_new_tokens, max_input_tokens):
        do_sample = float(temperature) > 0.0

        # Sort by token length to minimise padding waste within each batch
        pairs = sorted(enumerate(prompts), key=lambda x: len(self.tok.encode(x[1])))
        sorted_indices = [p[0] for p in pairs]
        sorted_prompts = [p[1] for p in pairs]

        sorted_results = self._generate_with_fallback(
            sorted_prompts, do_sample, temperature, max_new_tokens, max_input_tokens
        )

        # Restore original order
        results = [None] * len(prompts)
        for orig_idx, result in zip(sorted_indices, sorted_results):
            results[orig_idx] = result
        return results

    def _generate_with_fallback(self, prompts, do_sample, temperature, max_new_tokens, max_input_tokens):
        """Try to generate for all prompts at once; on OOM, split in half and retry."""
        if not prompts:
            return []

        try:
            enc = self.tok(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding=True,
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            with self.torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=do_sample,
                    temperature=float(temperature) if do_sample else None,
                    pad_token_id=self.tok.pad_token_id,
                    use_cache=True,
                )

            input_len = enc["input_ids"].shape[1]
            results = [
                self.tok.decode(out[i, input_len:], skip_special_tokens=True).strip()
                for i in range(len(prompts))
            ]
            del enc, out
            self.torch.cuda.empty_cache()
            return results

        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                # Single prompt is too long even alone — skip it
                print(f"[OOM] Single prompt exceeds GPU memory even alone, skipping.", flush=True)
                return [""]
            mid = len(prompts) // 2
            print(f"[OOM] Batch of {len(prompts)} → retrying as {mid} + {len(prompts) - mid}", flush=True)
            left  = self._generate_with_fallback(prompts[:mid],  do_sample, temperature, max_new_tokens, max_input_tokens)
            right = self._generate_with_fallback(prompts[mid:], do_sample, temperature, max_new_tokens, max_input_tokens)
            return left + right