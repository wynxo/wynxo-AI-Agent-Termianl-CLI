"""Credentials in the shapes they actually appear in.

The shield's job is that nothing it masks reaches the model or the log. What
it must not do is over-mask: a redacted word in the middle of code the model
is trying to read costs more than it saves, so every rule here is checked
from both sides -- the credential is gone, and the sentence that merely
mentions one is untouched.
"""

from __future__ import annotations

import pytest


class TestHTTPAuthorizationHeaders:
    """A credential after a scheme word.

    Neither of the other rules reaches it. The assignment rule cannot: its
    value pattern forbids spaces, so a quoted value beginning with a scheme
    word ends at the space and only the word itself is examined. The token
    rule cannot either, unless the credential happens to be a JWT or carry a
    vendor prefix -- and an opaque session token carries neither. So a header
    holding a live credential went to the model, and into the log, untouched.
    """

    def clean(self, text: str):
        from pathlib import Path

        from wynxo.secrets import Shield

        return Shield(Path("."), ignore=None).clean(text)

    @pytest.mark.parametrize("text", [
        "Authorization: Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZA==",
        'authorization = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"',
        'headers = {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz12"}',
        'curl -H "Authorization: Bearer 0123456789abcdef0123456789abcdef"',
        "Token 0123456789abcdef0123456789abcdef",
        "ApiKey 9f8c2b1e4d6a8f0c2e4b6d8a0f2c4e6b",
    ])
    def test_the_credential_is_masked(self, text):
        cleaned, hits = self.clean(text)
        assert hits >= 1
        assert "[redacted by wynxo]" in cleaned

    @pytest.mark.parametrize("text", [
        "Authorization: Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZA==",
        'authorization = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"',
    ])
    def test_the_scheme_word_survives(self, text):
        """The code still has to read. Masking the whole header would take
        the shape of the line with it."""
        cleaned, _ = self.clean(text)
        assert ("Basic" in cleaned) or ("Bearer" in cleaned)

    @pytest.mark.parametrize("text", [
        "Authorization: Bearer {{token}}",
        "Authorization: Bearer <YOUR_TOKEN>",
        "Authorization: Bearer ${API_TOKEN}",
        'set -x AUTH "Bearer abc123"',
    ])
    def test_a_placeholder_is_left_alone(self, text):
        """Masking these only makes the snippet harder to read."""
        cleaned, hits = self.clean(text)
        assert hits == 0
        assert cleaned == text

    @pytest.mark.parametrize("text", [
        "the bearer of this news should be thanked",
        "Basic arithmetic is important for everyone here",
        "a basic implementation of the algorithm follows",
        "token = self.token",
    ])
    def test_ordinary_english_is_not_a_credential(self, text):
        cleaned, hits = self.clean(text)
        assert hits == 0
        assert cleaned == text

    def test_masking_stays_fast_on_one_long_line(self):
        """Every read_file and every tool result goes through here, and a
        rule that backtracks turns a minified bundle into a stall."""
        import time

        blob = "abcdef0123456789+/=" * 5_000
        started = time.perf_counter()
        self.clean(blob)
        assert time.perf_counter() - started < 2.0
