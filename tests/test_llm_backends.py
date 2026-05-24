from herbarium_scribe.llm_backends import call_llm


def test_qwen_api_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    cfg = {"llm": {"backend": "qwen_api", "model": "qwen-plus"}}
    assert call_llm([{"role": "user", "content": "hello"}], cfg) == ""
