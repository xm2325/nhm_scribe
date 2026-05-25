from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .logging_utils import get_logger

logger = get_logger(__name__)
_LAST_REQUEST_AT: dict[str, float] = {}


def _provider_http_error(provider: str, code: int) -> str:
    if code in {401, 403}:
        return f"{provider} API backend failed with HTTP {code} authentication_error. The configured API key was rejected."
    if code == 429:
        return f"{provider} API backend failed with HTTP 429 rate_limit. The request was rate limited."
    return f"{provider} API backend failed with HTTP {code} provider_error. See provider dashboard or logs."


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_request_spacing(bucket: str, min_interval_seconds: float) -> None:
    if min_interval_seconds <= 0:
        return
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT.get(bucket, 0.0)
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)
    _LAST_REQUEST_AT[bucket] = time.monotonic()


def _retry_delay_seconds(error: urllib.error.HTTPError | None, attempt: int, retry_backoff_seconds: float) -> float:
    headers = getattr(error, "headers", {}) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    retry_after_seconds = _float_value(retry_after, -1.0)
    if retry_after_seconds >= 0:
        return retry_after_seconds
    return min(retry_backoff_seconds * (attempt + 1), 120.0)


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
) -> dict[str, Any]:
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
            choices = body.get("choices", [])
            choice = choices[0] if choices else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            if not isinstance(message, dict):
                message = {}
            return {
                "content": message.get("content", "") or "",
                "actual_model": body.get("model", ""),
                "finish_reason": choice.get("finish_reason", "") if isinstance(choice, dict) else "",
                "message": message,
                "message_keys": sorted(str(key) for key in message.keys()),
                "reasoning_content": message.get("reasoning_content", "") or "",
                "usage": body.get("usage", {}),
                "error_message": "",
                "response": body,
            }
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                delay = _retry_delay_seconds(e, attempt, retry_backoff_seconds)
                logger.warning("Chat completions request hit rate limit on attempt %s; retrying after %.1fs.", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise
        except Exception:
            if attempt >= retries:
                raise
            delay = _retry_delay_seconds(None, attempt, retry_backoff_seconds)
            logger.warning("Chat completions request failed on attempt %s; retrying after %.1fs.", attempt + 1, delay)
            time.sleep(delay)
    return {"content": "", "actual_model": "", "error_message": "empty_response", "response": {}}


def call_llm_with_metadata(messages: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    backend = (config.get("llm", {}).get("backend", "none") or "none").lower()
    lcfg = config.get("llm", {})
    base: dict[str, Any] = {
        "backend": backend,
        "requested_model": lcfg.get("model_name") or lcfg.get("model") or "",
        "actual_model": "",
        "content": "",
        "error_message": "",
        "endpoint_reachable": False,
        "api_key_present": False,
        "base_url": "",
        "min_interval_seconds": 0.0,
    }
    if backend == "none":
        return base
    if backend in {"deepseek", "deepseek_api"}:
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY_SELF")
        base_url = os.environ.get("DEEPSEEK_BASE_URL") or lcfg.get("base_url") or "https://api.deepseek.com"
        model = os.environ.get("DEEPSEEK_MODEL") or lcfg.get("model_name") or lcfg.get("model") or "deepseek-v4-pro"
        base.update({"requested_model": model, "api_key_present": bool(key), "base_url": base_url})
        if not key:
            msg = "DEEPSEEK_API_KEY or DEEPSEEK_API_KEY_SELF is not set; returning empty DeepSeek output."
            logger.warning(msg)
            base["error_message"] = msg
            return base
        try:
            result = _chat_completions_request(
                base_url=base_url,
                api_key=key,
                model=model,
                messages=messages,
                temperature=float(lcfg.get("temperature", 0.0)),
                max_tokens=int(lcfg.get("max_tokens", 1200)),
                timeout_seconds=int(lcfg.get("timeout_seconds", 120)),
                retries=int(lcfg.get("retries", 0)),
                retry_backoff_seconds=float(lcfg.get("retry_backoff_seconds", 2.0)),
            )
            base.update(result)
            base["endpoint_reachable"] = True
            return base
        except urllib.error.HTTPError as e:
            msg = _provider_http_error("DeepSeek", e.code)
            logger.warning(msg)
            base.update({"error_message": msg, "endpoint_reachable": True})
            return base
        except Exception as e:
            msg = f"DeepSeek API backend failed: {e}"
            logger.warning(msg)
            base["error_message"] = msg
            return base
    if backend in {"nvidia", "nvidia_api", "nvidia_nim", "deepseek_nvidia"}:
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        base_url = os.environ.get("NVIDIA_BASE_URL") or lcfg.get("base_url") or "https://integrate.api.nvidia.com/v1"
        model = os.environ.get("NVIDIA_MODEL") or lcfg.get("model_name") or lcfg.get("model") or "deepseek-ai/deepseek-v4-pro"
        min_interval = _float_value(os.environ.get("NVIDIA_MIN_INTERVAL_SECONDS") or lcfg.get("min_interval_seconds"), 0.0)
        base.update({"requested_model": model, "api_key_present": bool(key), "base_url": base_url, "min_interval_seconds": min_interval})
        if not key:
            msg = "NVIDIA_API_KEY or NGC_API_KEY is not set; returning empty NVIDIA output."
            logger.warning(msg)
            base["error_message"] = msg
            return base
        try:
            _apply_request_spacing(f"{backend}:{base_url}:{model}", min_interval)
            result = _chat_completions_request(
                base_url=base_url,
                api_key=key,
                model=model,
                messages=messages,
                temperature=float(lcfg.get("temperature", 0.0)),
                max_tokens=int(lcfg.get("max_tokens", 1600)),
                timeout_seconds=int(lcfg.get("timeout_seconds", 90)),
                retries=int(lcfg.get("retries", 0)),
                retry_backoff_seconds=float(lcfg.get("retry_backoff_seconds", 2.0)),
            )
            base.update(result)
            base["endpoint_reachable"] = True
            return base
        except urllib.error.HTTPError as e:
            msg = _provider_http_error("NVIDIA", e.code)
            logger.warning(msg)
            base.update({"error_message": msg, "endpoint_reachable": True})
            return base
        except Exception as e:
            msg = f"NVIDIA API backend failed: {e}"
            logger.warning(msg)
            base["error_message"] = msg
            return base
    if backend in {"qwen", "qwen_api", "qwen_dashscope"}:
        key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        base_url = lcfg.get("base_url") or os.environ.get("QWEN_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        model = lcfg.get("model_name") or lcfg.get("model") or "qwen-plus"
        base.update({"requested_model": model, "api_key_present": bool(key), "base_url": base_url})
        if not key:
            msg = "DASHSCOPE_API_KEY or QWEN_API_KEY is not set; returning empty Qwen output."
            logger.warning(msg)
            base["error_message"] = msg
            return base
        try:
            result = _chat_completions_request(
                base_url=base_url,
                api_key=key,
                model=model,
                messages=messages,
                temperature=float(lcfg.get("temperature", 0.0)),
                max_tokens=int(lcfg.get("max_tokens", 1200)),
                timeout_seconds=int(lcfg.get("timeout_seconds", 60)),
                retries=int(lcfg.get("retries", 0)),
                retry_backoff_seconds=float(lcfg.get("retry_backoff_seconds", 2.0)),
            )
            base.update(result)
            base["endpoint_reachable"] = True
            return base
        except urllib.error.HTTPError as e:
            msg = _provider_http_error("Qwen", e.code)
            logger.warning(msg)
            base.update({"error_message": msg, "endpoint_reachable": True})
            return base
        except Exception as e:
            msg = f"Qwen API backend failed: {e}"
            logger.warning(msg)
            base["error_message"] = msg
            return base
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            msg = "OPENAI_API_KEY is not set; returning empty LLM output."
            logger.warning(msg)
            return {**base, "error_message": msg}
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            model = lcfg.get("model") or "gpt-4.1-mini"
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            content = resp.choices[0].message.content or ""
            return {**base, "content": content, "actual_model": model, "requested_model": model, "api_key_present": True, "endpoint_reachable": True}
        except Exception as e:
            msg = f"OpenAI backend failed: {e}"
            logger.warning(msg)
            return {**base, "error_message": msg, "api_key_present": True}
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            msg = "ANTHROPIC_API_KEY is not set; returning empty LLM output."
            logger.warning(msg)
            return {**base, "error_message": msg}
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            model = lcfg.get("model") or "claude-3-5-haiku-latest"
            system = "Return only valid JSON."
            user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
            resp = client.messages.create(model=model, max_tokens=1000, temperature=0, system=system, messages=[{"role": "user", "content": user_text}])
            content = "".join(getattr(block, "text", "") for block in resp.content)
            return {**base, "content": content, "actual_model": model, "requested_model": model, "api_key_present": True, "endpoint_reachable": True}
        except Exception as e:
            msg = f"Anthropic backend failed: {e}"
            logger.warning(msg)
            return {**base, "error_message": msg, "api_key_present": True}
    if backend == "qwen_local":
        logger.warning("qwen_local is optional and not loaded in the default demo; returning empty output.")
        return {**base, "error_message": "qwen_local_optional_not_loaded"}
    logger.warning("Unknown LLM backend %r; returning empty output.", backend)
    return {**base, "error_message": f"unknown_backend:{backend}"}


def call_llm(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    return str(call_llm_with_metadata(messages, config).get("content", ""))
