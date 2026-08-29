"""The edit-analysis helpers used by the file tools: diff_stats and
replacement_ratio."""

from __future__ import annotations

from wynxo.editing import diff_stats, replacement_ratio


class TestDiffStats:
    def test_empty_diff_changes_nothing(self):
        stats = diff_stats("")
        assert stats.additions == 0
        assert stats.deletions == 0
        assert not stats.changed

    def test_counts_added_and_removed_lines(self):
        stats = diff_stats("-old\n+new\n context\n")
        assert stats.additions == 1
        assert stats.deletions == 1
        assert stats.changed

    def test_hunk_headers_are_not_counted_as_lines(self):
        # "+++ b/x" and "--- a/x" start with + and - but are not changes.
        stats = diff_stats("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n+kept\n-removed\n")
        assert stats.additions == 1
        assert stats.deletions == 1

    def test_unchanged_diff_is_not_changed(self):
        stats = diff_stats(" context\n")
        assert not stats.changed


class TestReplacementRatio:
    def test_identical_content_is_zero(self):
        assert replacement_ratio("a\nb\nc\n", "a\nb\nc\n") == 0.0

    def test_identical_without_trailing_newline_is_zero(self):
        # The old byte-length denominator counted the newlines it then
        # never credited back, so a small unchanged file scored 0.5 and
        # tripped the large_rewrite flag on a no-op edit.
        assert replacement_ratio("a\nb", "a\nb") == 0.0

    def test_identical_line_counts_as_common(self):
        # One changed line out of three: roughly 1/3 replaced, never 1.
        ratio = replacement_ratio("a\nb\nc\n", "a\nX\nc\n")
        assert 0.0 < ratio < 1.0

    def test_fully_different_is_one(self):
        assert replacement_ratio("aaa", "bbb") == 1.0

    def test_ratio_stays_bounded(self):
        ratio = replacement_ratio("x", "y" * 100)
        assert 0.0 <= ratio <= 1.0

    def test_empty_before_is_one(self):
        # Nothing in common with a file that has content: fully replaced.
        assert replacement_ratio("", "hello") == 1.0
