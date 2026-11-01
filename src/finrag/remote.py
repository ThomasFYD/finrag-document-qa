"""Optional DeepInfra adapter. Nothing in this module calls the network on import."""
from dataclasses import dataclass, asdict
import os
import time

import httpx

BASE_URL = "https://api.deepinfra.com/v1/openai"


@dataclass
class Completion:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None
    finish_reason: str

    def telemetry(self):
        return {k: v for k, v in asdict(self).items() if k != "text"}


class DeepInfraClient:
    def __init__(self, model=None, pricing=None, client=None, sleeper=time.sleep):
        key = os.getenv("DEEPINFRA_API_KEY", "").strip()
        self.model = model or os.getenv("DEEPINFRA_MODEL", "").strip()
        if not key or not self.model:
            raise ValueError("Set DEEPINFRA_API_KEY and DEEPINFRA_MODEL in your local .env file first.")
        self.pricing = pricing or {}
        self.client = client or httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(60, connect=10),
            headers={"Authorization": f"Bearer {key}"}, follow_redirects=False)
        self.sleeper = sleeper

    def close(self):
        self.client.close()

    def validate_model(self):
        response = self.client.get("/models")
        if response.status_code != 200:
            raise RuntimeError(f"DeepInfra model-catalog check failed (HTTP {response.status_code}).")
        ids = {x["id"] for x in response.json().get("data", [])}
        if self.model not in ids:
            raise ValueError(f"Model {self.model} is not in the current DeepInfra catalog. Update configuration.")

    def complete(self, messages, max_tokens=700):
        start = time.perf_counter()
        for attempt in range(3):
            try:
                response = self.client.post("/chat/completions", json={"model": self.model, "messages": messages,
                    "temperature": 0, "max_tokens": max_tokens, "stream": False})
            except httpx.TimeoutException as exc:
                # A timed-out generation might still be billed; do not blindly retry it.
                raise RuntimeError("DeepInfra request timed out; not retried automatically to avoid duplicate billing.") from exc
            except httpx.TransportError as exc:
                raise RuntimeError("DeepInfra network request failed. Check the connection.") from exc
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    self.sleeper(2 ** attempt)
                    continue
            if response.status_code != 200:
                raise RuntimeError(f"DeepInfra returned HTTP {response.status_code}. Check credentials, balance, and model configuration.")
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            input_rate, output_rate = self.pricing.get("input_usd_per_million"), self.pricing.get("output_usd_per_million")
            cost = None
            if all(v is not None for v in [prompt, completion, input_rate, output_rate]):
                cost = (prompt * input_rate + completion * output_rate) / 1_000_000
            return Completion(choice["message"].get("content") or "", self.model,
                (time.perf_counter() - start) * 1000, prompt, completion, cost, choice.get("finish_reason", "unknown"))
        raise RuntimeError("DeepInfra request failed.")
