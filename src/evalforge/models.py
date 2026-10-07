"""Model backends.

Every backend takes a prompt and returns text. Models are named
"<provider>:<model>", for example "ollama:llama3.2" or "anthropic:claude-haiku-4-5".
Only the standard library is used, so there is nothing extra to install.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


class Model(Protocol):
    name: str

    def generate(self, prompt: str, system: str | None = None) -> str: ...


class ModelError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ModelError(f"{url} returned HTTP {e.code}: {e.read()[:300]!r}") from e
    except urllib.error.URLError as e:
        raise ModelError(f"Could not reach {url}: {e.reason}") from e


@dataclass
class OllamaModel:
    """A model served by a local Ollama install (free, runs on your machine)."""

    model: str
    host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = 0.0
    timeout: float = 300.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post_json(
            f"{self.host.rstrip('/')}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            {},
            self.timeout,
        )
        return data["message"]["content"]


@dataclass
class OpenAICompatibleModel:
    """Any OpenAI-compatible chat endpoint (OpenAI, Groq, LM Studio, vLLM...)."""

    model: str
    base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    temperature: float = 0.0
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.api_key:
            raise ModelError("Set OPENAI_API_KEY to use openai:* models.")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": self.temperature},
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        return data["choices"][0]["message"]["content"]


@dataclass
class AnthropicModel:
    """Claude models through the Anthropic Messages API (needs ANTHROPIC_API_KEY)."""

    model: str
    api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.api_key:
            raise ModelError("Set ANTHROPIC_API_KEY to use anthropic:* models.")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            self.timeout,
        )
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


@dataclass
class FunctionModel:
    """Wraps a Python function. Used for tests and for plugging in custom code."""

    fn: Callable[[str], str]
    label: str = "function"

    @property
    def name(self) -> str:
        return self.label

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.fn(prompt)


def get_model(spec: str) -> Model:
    """Build a model from a "<provider>:<model>" string."""
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Model must look like 'provider:model', got {spec!r}")
    if provider == "ollama":
        return OllamaModel(model)
    if provider == "openai":
        return OpenAICompatibleModel(model)
    if provider == "anthropic":
        return AnthropicModel(model)
    if provider == "echo":
        # Repeats the prompt back. Handy for checking a suite without a real model.
        return FunctionModel(lambda p: p, label=spec)
    raise ValueError(f"Unknown provider {provider!r}. Use ollama, openai, anthropic or echo.")
