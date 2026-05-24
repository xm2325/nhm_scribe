from __future__ import annotations

import os
from typing import Any

from .logging_utils import get_logger

logger = get_logger(__name__)


def call_llm(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    backend = (config.get("llm", {}).get("backend", "none") or "none").lower()
    if backend == "none":
        return ""
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY is not set; returning empty LLM output.")
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            model = config.get("llm", {}).get("model") or "gpt-4.1-mini"
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("OpenAI backend failed: %s", e)
            return ""
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("ANTHROPIC_API_KEY is not set; returning empty LLM output.")
            return ""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            model = config.get("llm", {}).get("model") or "claude-3-5-haiku-latest"
            system = "Return only valid JSON."
            user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
            resp = client.messages.create(model=model, max_tokens=1000, temperature=0, system=system, messages=[{"role": "user", "content": user_text}])
            return "".join(getattr(block, "text", "") for block in resp.content)
        except Exception as e:
            logger.warning("Anthropic backend failed: %s", e)
            return ""
    if backend == "qwen_local":
        logger.warning("qwen_local is optional and not loaded in the default demo; returning empty output.")
        return ""
    logger.warning("Unknown LLM backend %r; returning empty output.", backend)
    return ""
