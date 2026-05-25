from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .logging_utils import get_logger

logger = get_logger(__name__)


def _chat_completions_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout_seconds: int = 60,
    retries: int = 0,
    retry_backoff_seconds: float = 2.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(max(0, retries) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                logger.warning("Chat completions request hit rate limit on attempt %s; retrying.", attempt + 1)
                time.sleep(retry_backoff_seconds)
                continue
            raise
        except Exception:
            if attempt >= retries:
                raise
            logger.warning("Chat completions request failed on attempt %s; retrying.", attempt + 1)
            time.sleep(retry_backoff_seconds)
    return ""


def call_llm(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    backend = (config.get("llm", {}).get("backend", "none") or "none").lower()
    lcfg = config.get("llm", {})
    if backend == "none":
        return ""
    if backend in {"nvidia", "nvidia_api", "nvidia_nim", "deepseek_nvidia"}:
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        if not key:
            logger.warning("NVIDIA_API_KEY or NGC_API_KEY is not set; returning empty NVIDIA output.")
            return ""
        try:
            return _chat_completions_request(
                base_url=lcfg.get("base_url") or os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
                api_key=key,
                model=lcfg.get("model") or "deepseek-ai/deepseek-v4-pro",
                messages=messages,
                temperature=float(lcfg.get("temperature", 0.0)),
                max_tokens=int(lcfg.get("max_tokens", 1600)),
                timeout_seconds=int(lcfg.get("timeout_seconds", 90)),
                retries=int(lcfg.get("retries", 0)),
                retry_backoff_seconds=float(lcfg.get("retry_backoff_seconds", 2.0)),
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            logger.warning("NVIDIA API backend failed with HTTP %s: %s", e.code, detail[:500])
            return ""
        except Exception as e:
            logger.warning("NVIDIA API backend failed: %s", e)
            return ""
    if backend in {"qwen", "qwen_api", "qwen_dashscope"}:
        key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not key:
            logger.warning("DASHSCOPE_API_KEY or QWEN_API_KEY is not set; returning empty Qwen output.")
            return ""
        try:
            return _chat_completions_request(
                base_url=lcfg.get("base_url") or os.environ.get("QWEN_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                api_key=key,
                model=lcfg.get("model") or "qwen-plus",
                messages=messages,
                temperature=float(lcfg.get("temperature", 0.0)),
                max_tokens=int(lcfg.get("max_tokens", 1200)),
                timeout_seconds=int(lcfg.get("timeout_seconds", 60)),
                retries=int(lcfg.get("retries", 0)),
                retry_backoff_seconds=float(lcfg.get("retry_backoff_seconds", 2.0)),
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            logger.warning("Qwen API backend failed with HTTP %s: %s", e.code, detail[:500])
            return ""
        except Exception as e:
            logger.warning("Qwen API backend failed: %s", e)
            return ""
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY is not set; returning empty LLM output.")
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            model = lcfg.get("model") or "gpt-4.1-mini"
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
            model = lcfg.get("model") or "claude-3-5-haiku-latest"
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
