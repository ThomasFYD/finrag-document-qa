import httpx
import pytest

from finrag.remote import DeepInfraClient


def test_no_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="local .env"):
        DeepInfraClient(model="test-model")


def test_retry_usage_cost_and_model_validation(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-only-not-a-real-key")
    calls = []

    def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    transport = httpx.MockTransport(handle)
    client = DeepInfraClient("model", {"input_usd_per_million": 1, "output_usd_per_million": 2},
        client=httpx.Client(base_url="https://example.invalid/v1/openai", transport=transport), sleeper=lambda _: None)
    client.validate_model()
    result = client.complete([{"role": "user", "content": "hello"}])
    assert len(calls) == 2 and result.estimated_cost_usd == pytest.approx(0.00014)
    assert "text" not in result.telemetry()
    client.close()


def test_timeout_not_retried_and_unknown_price_not_zero(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-only")
    calls = []

    def handle(request):
        calls.append(1)
        raise httpx.ReadTimeout("test timeout")
    c = DeepInfraClient("model", client=httpx.Client(base_url="https://example.invalid", transport=httpx.MockTransport(handle)))
    with pytest.raises(RuntimeError, match="not retried automatically"):
        c.complete([])
    assert len(calls) == 1
    c.close()

    c = DeepInfraClient("model", client=httpx.Client(base_url="https://example.invalid", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}))))
    assert c.complete([]).estimated_cost_usd is None
    c.close()
