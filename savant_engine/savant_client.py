import requests
from typing import Any, Dict, Optional


class SavantClient:
    """
    Cliente ligero para la Savant RRF Φ12.0 API en Hugging Face Spaces.

    Uso básico:
        client = SavantClient()
        result = client.evaluate(
            prompt="...",
            answer="..."
        )
    """

    def __init__(self, base_url: str = "https://antonypamo-apisavant.hf.space", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=json_body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check health of the API."""
        import json
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    def evaluate(self, **payload: Any) -> Dict[str, Any]:
        """
        Wrapper genérico sobre POST /evaluate.

        Ejemplos de uso (según cómo definas EvaluateRequest):

            client.evaluate(
                prompt="What is the main function of the heart?",
                answer="The heart pumps blood..."
            )

            client.evaluate(
                mode="quality",
                prompt="...",
                answer="..."
            )

            client.evaluate(
                mode="rerank",
                query="...",
                candidates=[...]
            )
        """
        return self._post("/evaluate", payload)
