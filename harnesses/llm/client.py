"""
LLM client abstraction for local model inference via ollama or OpenAI-compatible APIs.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.2, max_tokens: int = 4096) -> LLMResponse:
        ...


class OllamaClient(LLMClient):
    """Ollama /api/chat with native tool calling."""

    def __init__(self, model: str = "qwen2.5:32b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        text = msg.get("content", "")
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
            ))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=data)


class OpenAICompatClient(LLMClient):
    """OpenAI-compatible /v1/chat/completions (works with vllm, lmstudio, etc.)."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000",
                 api_key: str = "none"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(f"{self.base_url}/v1/chat/completions",
                             json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content", "") or ""
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=data)


def create_client(backend: str = "ollama", model: str = "qwen2.5:32b",
                  base_url: str = "http://localhost:11434", **kwargs) -> LLMClient:
    if backend == "ollama":
        return OllamaClient(model=model, base_url=base_url)
    elif backend in ("openai", "vllm", "lmstudio"):
        return OpenAICompatClient(model=model, base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
