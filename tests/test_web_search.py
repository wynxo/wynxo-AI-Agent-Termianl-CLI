"""The web_search tool: no-key DuckDuckGo lookup, tested offline."""

from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.tools.web import WebSearch


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def _page() -> str:
    return """
    <html><body>
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=x">Example Docs</a>
      <a class="result__snippet">The current API reference.</a>
    </div>
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fblog&amp;rut=y">Example Blog</a>
      <a class="result__snippet">What changed in the latest release.</a>
    </div>
    </body></html>
    """


class _FakeClient:
    def __init__(self, text: str):
        self._text = text
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self.calls.append((url, params))
        return _FakeResponse(self._text)


def test_parses_results_and_unwraps_redirects(monkeypatch):
    fake = _FakeClient(_page())

    def fake_client(**kwargs):
        return fake

    monkeypatch.setattr("wynxo.tools.web.httpx.Client", fake_client)
    result = asyncio.run(WebSearch(Path("."), None, None).invoke(
        {"query": "example docs", "max_results": 1}))
    assert result.ok
    assert "https://example.com/docs" in result.output
    assert "Example Docs" in result.output
    assert "The current API reference." in result.output
    # The second result is beyond max_results and must not appear.
    assert "example.org" not in result.output
    # And the request actually went out with the query.
    assert fake.calls and fake.calls[0][1] == {"q": "example docs"}

    # Raising the limit brings the second result in.
    result = asyncio.run(WebSearch(Path("."), None, None).invoke(
        {"query": "example docs", "max_results": 5}))
    assert "example.org" in result.output


def test_no_results_is_a_gentle_success(monkeypatch):
    monkeypatch.setattr(
        "wynxo.tools.web.httpx.Client",
        lambda **kw: _FakeClient("<html><body>nothing here</body></html>"))
    result = asyncio.run(WebSearch(Path("."), None, None).invoke(
        {"query": "zzz" , "max_results": 5}))
    assert result.ok
    assert "No results" in result.output


def test_transport_failure_is_an_error(monkeypatch):
    import httpx

    def boom(**kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("wynxo.tools.web.httpx.Client", boom)
    result = asyncio.run(WebSearch(Path("."), None, None).invoke(
        {"query": "x", "max_results": 1}))
    assert not result.ok
    assert "web search failed" in result.output
