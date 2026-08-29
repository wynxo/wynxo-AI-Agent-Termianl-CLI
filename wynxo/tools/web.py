"""Web search: current, external facts the model does not know.

The model knows what it was trained on. Anything newer, or anything from
outside the workspace -- a library's current API, a breaking change, the
latest release -- is a guess until it is looked up. This tool is the
lookup, and it needs no API key: results come from DuckDuckGo's public
HTML endpoint, so it works on a fresh install with nothing configured.
"""

from __future__ import annotations

import asyncio
import html
import re

import httpx

from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_RESULTS = 8

_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/124.0 Safari/537.36")


class WebSearchInput(Schema):
    query = Field(str, "The search query: a specific question or a few "
                       "keywords, the way you would type them into a "
                       "search engine.")
    max_results = Field(int, "How many results to return.", default=5,
                        ge=1, le=MAX_RESULTS)


class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web for current information. Use it when the answer "
        "might have changed since the model was trained: library versions, "
        "APIs, breaking changes, releases, and facts about the outside "
        "world. Returns result titles, URLs and snippets."
    )
    Input = WebSearchInput
    concurrency_safe = True

    async def run(self, args: WebSearchInput) -> ToolResult:
        try:
            results = await asyncio.to_thread(
                self._search, args.query, args.max_results)
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"web search failed: {exc}")
        if not results:
            return ToolResult.success(
                f"No results for {args.query!r}. Try different keywords.")
        body = "\n\n".join(
            f"{i + 1}. {title}\n   {url}\n   {snippet}"
            for i, (title, url, snippet) in enumerate(results)
        )
        return ToolResult.success(
            body, display=f"web search -> {len(results)} results")

    def _search(self, query: str, limit: int) -> list[tuple[str, str, str]]:
        """One synchronous round trip: fetch the page, pull the result rows.

        DuckDuckGo's HTML endpoint returns a stable, parseable structure
        (a result is an ``a.result__a`` anchor with a ``.result__snippet``
        sibling). Parsed with regexes rather than a parser dependency to
        keep this tool dependency-free.
        """
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=15.0,
        ) as client:
            response = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
            response.raise_for_status()
            page = response.text

        out: list[tuple[str, str, str]] = []
        # Each result is one block; parse the anchor and snippet from the
        # block together so the two never drift apart.
        for block in re.split(r'<div[^>]*class="[^"]*result[^"]*"', page):
            anchor = re.search(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                block, flags=re.DOTALL)
            if not anchor:
                continue
            url, title = anchor.group(1), anchor.group(2)
            title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            if not title:
                continue
            snippet_match = re.search(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                block, flags=re.DOTALL)
            snippet = (html.unescape(re.sub(r"<[^>]+>", "",
                                            snippet_match.group(1))).strip()
                       if snippet_match else "")
            # DDG wraps its redirect URLs; the real target is the uddg param.
            match = re.search(r"uddg=([^&]+)", url)
            if match:
                from urllib.parse import unquote

                url = unquote(match.group(1))
            out.append((title, url, snippet))
            if len(out) >= limit:
                break
        return out
