"""Tiny HTTP client the Streamlit UI uses. The UI never imports the pipeline: it only
talks to the API, so it works against a local server or one running next to a GPU."""

from __future__ import annotations

import httpx


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 900):  # CPU answers can take minutes
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = httpx.request(method, self.base_url + path, timeout=self.timeout, **kw)
        except httpx.HTTPError as e:
            raise ApiError(
                f"Cannot reach the API at {self.base_url} ({e.__class__.__name__})."
            ) from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        return r.json()

    def health(self) -> bool:
        try:
            return self._call("GET", "/healthz").get("status") == "ok"
        except ApiError:
            return False

    def documents(self) -> dict:
        return self._call("GET", "/v1/documents")

    def ask(self, question: str, mode: str, k: int, verify: bool) -> dict:
        payload = {"question": question, "mode": mode, "k": k, "verify": verify}
        return self._call("POST", "/v1/ask", json=payload)

    def search(self, question: str, modes: list[str], k: int) -> dict:
        return self._call("POST", "/v1/search", json={"question": question, "modes": modes, "k": k})

    def ingest(self, documents: list[dict]) -> dict:
        return self._call("POST", "/v1/ingest", json={"documents": documents})
