import urllib.error

from herbarium_scribe.llm_backends import call_llm, call_llm_with_metadata, _chat_completions_request, _provider_http_error


def test_qwen_api_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    cfg = {"llm": {"backend": "qwen_api", "model": "qwen-plus"}}
    assert call_llm([{"role": "user", "content": "hello"}], cfg) == ""


def test_nvidia_api_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    cfg = {"llm": {"backend": "nvidia_api", "model": "deepseek-ai/deepseek-v4-pro"}}
    assert call_llm([{"role": "user", "content": "hello"}], cfg) == ""


def test_deepseek_api_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_SELF", raising=False)
    cfg = {"llm": {"backend": "deepseek_api", "model_name": "deepseek-v4-pro"}}
    assert call_llm([{"role": "user", "content": "hello"}], cfg) == ""


def test_deepseek_self_key_env_is_used(monkeypatch):
    def fake_request(**kwargs):
        return {"content": "{}", "actual_model": kwargs["model"], "error_message": "", "response": {}}

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_SELF", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setattr("herbarium_scribe.llm_backends._chat_completions_request", fake_request)
    cfg = {"llm": {"backend": "deepseek_api", "model_name": "config/model"}}
    out = call_llm_with_metadata([{"role": "user", "content": "hello"}], cfg)
    assert out["api_key_present"] is True
    assert out["requested_model"] == "deepseek-v4-pro"
    assert out["actual_model"] == "deepseek-v4-pro"


def test_nvidia_model_env_overrides_config(monkeypatch):
    def fake_request(**kwargs):
        return {"content": "{}", "actual_model": kwargs["model"], "error_message": "", "response": {}}

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "env/model")
    monkeypatch.setattr("herbarium_scribe.llm_backends._chat_completions_request", fake_request)
    cfg = {"llm": {"backend": "nvidia_api", "model_name": "config/model"}}
    out = call_llm_with_metadata([{"role": "user", "content": "hello"}], cfg)
    assert out["requested_model"] == "env/model"
    assert out["actual_model"] == "env/model"


def test_nvidia_min_interval_env_is_reported(monkeypatch):
    def fake_request(**kwargs):
        return {"content": "{}", "actual_model": kwargs["model"], "error_message": "", "response": {}}

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "env/rate-limit-model")
    monkeypatch.setenv("NVIDIA_MIN_INTERVAL_SECONDS", "2.5")
    monkeypatch.setattr("herbarium_scribe.llm_backends._chat_completions_request", fake_request)
    cfg = {"llm": {"backend": "nvidia_api", "model_name": "config/model"}}
    out = call_llm_with_metadata([{"role": "user", "content": "hello"}], cfg)
    assert out["min_interval_seconds"] == 2.5


def test_provider_http_error_does_not_include_key_fragments():
    msg = _provider_http_error("NVIDIA", 401)
    assert "authentication_error" in msg
    assert "nvapi" not in msg.lower()


def test_chat_completions_retries_429_with_retry_after(monkeypatch):
    calls = []
    sleeps = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"model":"test-model","choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {"Retry-After": "3"}, None)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    result = _chat_completions_request(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        retries=1,
        retry_backoff_seconds=60,
    )

    assert len(calls) == 2
    assert sleeps == [3.0]
    assert result["content"] == "{}"
