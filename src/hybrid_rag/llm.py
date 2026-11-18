"""Minimal Ollama client (stdlib only, no extra dependency).

Deterministic settings: temperature 0 and a fixed seed, so eval runs are repeatable.
num_ctx is set explicitly because Ollama's default context window is small and an
over-long prompt is truncated without an error; we detect that and fail loudly.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    # Ollama's own breakdown in seconds: load (model into memory), prompt, generation
    breakdown: dict = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResponse: ...


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        timeout: float = 900,
        seed: int = 0,
    ):
        self.name = f"ollama:{model}"
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx
        self.timeout, self.seed = timeout, seed

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema  # Ollama structured output: forces valid JSON
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.load(resp)
        except urllib.error.HTTPError as e:  # must come before URLError (subclass)
            detail = e.read().decode(errors="replace")[:300]
            raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host} ({e.reason}). Is the Ollama app running?"
            ) from e
        seconds = time.perf_counter() - t0

        prompt_tokens = int(body.get("prompt_eval_count", 0))
        if prompt_tokens >= self.num_ctx:
            raise LLMError(
                f"Prompt used {prompt_tokens} tokens, filling num_ctx={self.num_ctx}; "
                "it was probably truncated. Reduce k or chunk size, or raise num_ctx."
            )
        return LLMResponse(
            text=body["message"]["content"],
            prompt_tokens=prompt_tokens,
            output_tokens=int(body.get("eval_count", 0)),
            seconds=seconds,
            breakdown={
                name: round(body[key] / 1e9, 2)
                for name, key in (
                    ("load_s", "load_duration"),
                    ("prompt_s", "prompt_eval_duration"),
                    ("generate_s", "eval_duration"),
                )
                if key in body
            },
        )


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {text[:200]!r}") from e
