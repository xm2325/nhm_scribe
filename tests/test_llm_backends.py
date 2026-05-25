from herbarium_scribe.llm_backends import call_llm, call_llm_with_metadata, _provider_http_error


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
    cfg = {"llm": {"backend": "deepseek_api", "model_name": "deepseek-v4-pro"}}
    assert call_llm([{"role": "user", "content": "hello"}], cfg) == ""


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


def test_provider_http_error_does_not_include_key_fragments():
    msg = _provider_http_error("NVIDIA", 401)
    assert "authentication_error" in msg
    assert "nvapi" not in msg.lower()
