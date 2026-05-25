from herbarium_scribe.llm_backends import call_llm


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
