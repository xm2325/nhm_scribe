from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "qwen_api_preflight.json"
SAMPLE_ARCHIVE = ROOT / "app_data" / "hespi_v10_ocr_visual_report.zip"
SAMPLE_MEMBER = (
    "hespi_v10_ocr_visual_report/assets/crops/"
    "L.2489324_01_collector_00_03_collector.jpg"
)


def provider_error(error: urllib.error.HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    return f"HTTP {error.code}: {detail[:1200]}"


def request_json(
    url: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def response_summary(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices", [])
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content", "")
    return {
        "actual_model": body.get("model", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        "message_keys": sorted(message),
        "usage": body.get("usage", {}),
    }


def main() -> None:
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
    model = os.environ.get("QWEN_MODEL", "qwen3.7-plus")
    if not api_key or not base_url:
        raise SystemExit("QWEN_API_KEY and QWEN_BASE_URL are required.")

    diagnostics: dict[str, Any] = {
        "base_url": base_url,
        "requested_model": model,
        "api_key_present": True,
        "models": {"ok": False, "advertised": [], "error": ""},
        "text_chat": {"ok": False, "error": ""},
        "vision_chat": {"ok": False, "error": ""},
    }

    try:
        body = request_json(f"{base_url}/models", api_key)
        diagnostics["models"] = {
            "ok": True,
            "advertised": [
                item.get("id", "")
                for item in body.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ],
            "error": "",
        }
    except urllib.error.HTTPError as error:
        diagnostics["models"]["error"] = provider_error(error)
    except Exception as error:
        diagnostics["models"]["error"] = f"{type(error).__name__}: {error}"

    text_payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Return only {"status":"ok"}.'}],
        "temperature": 0,
        "max_tokens": 100,
    }
    try:
        body = request_json(f"{base_url}/chat/completions", api_key, payload=text_payload)
        diagnostics["text_chat"] = {"ok": True, "error": "", **response_summary(body)}
    except urllib.error.HTTPError as error:
        diagnostics["text_chat"]["error"] = provider_error(error)
    except Exception as error:
        diagnostics["text_chat"]["error"] = f"{type(error).__name__}: {error}"

    if diagnostics["text_chat"]["ok"]:
        with zipfile.ZipFile(SAMPLE_ARCHIVE) as archive:
            image_bytes = archive.read(SAMPLE_MEMBER)
        image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        vision_payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            'Transcribe the visible text in this image. Return only JSON: '
                            '{"text":"..."}'
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 300,
        }
        try:
            body = request_json(
                f"{base_url}/chat/completions",
                api_key,
                payload=vision_payload,
                timeout=180,
            )
            summary = response_summary(body)
            diagnostics["vision_chat"] = {
                "ok": bool(str(summary.get("content", "")).strip()),
                "error": "" if str(summary.get("content", "")).strip() else "empty_content",
                "sample_member": SAMPLE_MEMBER,
                **summary,
            }
        except urllib.error.HTTPError as error:
            diagnostics["vision_chat"]["error"] = provider_error(error)
        except Exception as error:
            diagnostics["vision_chat"]["error"] = f"{type(error).__name__}: {error}"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if not diagnostics["text_chat"]["ok"] or not diagnostics["vision_chat"]["ok"]:
        raise SystemExit("Qwen text or vision preflight failed.")


if __name__ == "__main__":
    main()
