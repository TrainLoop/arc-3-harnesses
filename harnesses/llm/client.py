"""
LLM client abstraction. Supports ollama, OpenAI-compatible, and Anthropic APIs.
"""

import json
import os
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
                 api_key: str = "none", **kwargs):
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


class OpenAIClient(LLMClient):
    """Official OpenAI API client (gpt-4o, gpt-5.2, etc.)."""

    def __init__(self, model: str = "gpt-5.2", api_key: str | None = None, **kwargs):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=16000):
        # Clean messages: strip non-standard fields
        clean_msgs = []
        for m in messages:
            cm = {"role": m["role"], "content": m["content"]}
            if m.get("tool_calls"):
                cm["tool_calls"] = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", tc)
                    cm["tool_calls"].append({
                        "id": fn.get("name", "call"),
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": json.dumps(fn.get("arguments", {}))},
                    })
            if m.get("role") == "tool":
                cm["tool_call_id"] = m.get("tool_use_id", m.get("name", "call"))
            clean_msgs.append(cm)

        kwargs = {
            "model": self.model,
            "messages": clean_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return LLMResponse(text=text, tool_calls=tool_calls, raw={})


class AnthropicClient(LLMClient):
    """Anthropic Claude API with native tool calling."""

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str | None = None):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=16000):
        # Convert OpenAI-style tool defs to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", t)
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        # Separate system message from conversation
        system = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                conv_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_use_id", m.get("name", "tool")),
                        "content": m["content"],
                    }],
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                # Convert tool_calls to Anthropic content blocks
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", tc)
                    content.append({
                        "type": "tool_use",
                        "id": fn.get("name", "tool"),
                        "name": fn["name"],
                        "input": fn.get("arguments", {}),
                    })
                conv_messages.append({"role": "assistant", "content": content})
            else:
                conv_messages.append({"role": m["role"], "content": m["content"]})

        # Merge consecutive same-role messages (Anthropic requires alternating)
        merged = []
        for m in conv_messages:
            if merged and merged[-1]["role"] == m["role"]:
                prev = merged[-1]["content"]
                curr = m["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    merged[-1]["content"] = prev + "\n" + curr
                elif isinstance(prev, list) and isinstance(curr, list):
                    merged[-1]["content"] = prev + curr
                elif isinstance(prev, str) and isinstance(curr, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev}] + curr
                elif isinstance(prev, list) and isinstance(curr, str):
                    merged[-1]["content"] = prev + [{"type": "text", "text": curr}]
            else:
                merged.append(m)

        # Ensure conversation starts with user message
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Begin."})

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": merged,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)

        # Parse response
        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(text=text, tool_calls=tool_calls, raw={})


def create_client(backend: str = "ollama", model: str = "qwen2.5:32b",
                  base_url: str = "http://localhost:11434", **kwargs) -> LLMClient:
    if backend == "ollama":
        return OllamaClient(model=model, base_url=base_url)
    elif backend in ("openai_compat", "vllm", "lmstudio"):
        return OpenAICompatClient(model=model, base_url=base_url, **kwargs)
    elif backend == "openai":
        return OpenAIClient(model=model, **kwargs)
    elif backend == "anthropic":
        return AnthropicClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
