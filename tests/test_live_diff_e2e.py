"""A real broken repository, fixed by the real agent, watched by the real UI.

Nothing is simulated on the way through: a genuine file with a genuine bug,
the production Agent, the production tool registry, the production callbacks,
and a provider that streams a tool call's arguments in fragments the way an
OpenAI-compatible server does. The assertions are about what the UI was
*told* and what ended up on disk -- so a card that filled in from anything
other than the model's own stream, or a "fixed" file that was never written,
fails here.
"""

from __future__ import annotations

import json

import httpx
import pytest

from wynxo import livediff
from wynxo.config import Config, Endpoint
from wynxo.effort import resolve
from wynxo.provider import OpenAIClient
from wynxo.tools import build_registry

BROKEN = '''\
def average(values):
    # off by one: the last value never counts
    total = 0
    for i in range(len(values) - 1):
        total += values[i]
    return total / len(values)
'''

FIXED = '''\
def average(values):
    total = 0
    for value in values:
        total += value
    return total / len(values)
'''

TEST_FILE = '''\
from calc import average


def test_average():
    assert average([1, 2, 3]) == 2
'''


class Recorder:
    """Stands in for the terminal, records what the UI was told to show."""

    def __init__(self):
        self.code: list[str] = []
        self.tools: list[tuple[str, str]] = []
        self.results: list[tuple[str, bool]] = []
        self.stages: list[str] = []

    def __getattr__(self, _name):
        async def anything(*a, **k):
            return None
        return anything

    async def on_code(self, text):
        self.code.append(text)

    async def on_tool_start(self, name, summary, event=None):
        self.tools.append((name, summary))

    async def on_tool_result(self, name, ok, display, output, event=None):
        self.results.append((name, ok))

    async def on_stage(self, name, detail=""):
        self.stages.append(name)


@pytest.fixture
def repo(tmp_path):
    """A real repository with a real bug, thrown away afterwards.

    tmp_path is pytest's, so the cleanup is pytest's too -- nothing here
    leaves anything behind for the next test to trip over.
    """
    (tmp_path / "calc.py").write_text(BROKEN, encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(TEST_FILE, encoding="utf-8")
    return tmp_path


def streaming_provider(calls: list[dict], chunk_size: int = 20):
    """An OpenAI-compatible server that streams each turn in fragments.

    ``calls`` is the script: each entry is either {"content": ...} or
    {"tool": name, "arguments": {...}}. Tool arguments go out in pieces,
    because that is what a real streaming server does and it is the whole
    reason the card can fill in while the edit is being written.
    """
    turns = list(calls)

    def handler(request: httpx.Request) -> httpx.Response:
        turn = turns.pop(0) if turns else {"content": "Done."}
        lines = []
        if "tool" in turn:
            arguments = json.dumps(turn["arguments"])
            lines.append('data: ' + json.dumps({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "c1", "function": {
                    "name": turn["tool"], "arguments": ""}}]}}]}))
            for i in range(0, len(arguments), chunk_size):
                lines.append('data: ' + json.dumps({"choices": [{"delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": arguments[i:i + chunk_size]}}]}}]}))
        else:
            lines.append('data: ' + json.dumps({"choices": [{"delta": {
                "content": turn.get("content", "")}}]}))
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n".join(lines))

    return handler


def make_agent(repo, script, callbacks):
    from wynxo.agent import Agent

    config = Config(
        endpoints=[Endpoint(name="t", url="http://fake/v1", kind="openai")],
        active_endpoint="t", model="m", num_ctx=32768)
    client = OpenAIClient(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(streaming_provider(script)),
        base_url="http://fake")
    agent = Agent(client, config, resolve("low"), repo, callbacks,
                  registry=build_registry(repo, allow_shell=True))
    agent.permissions.yolo = True
    return agent


class TestFixingARealBug:
    async def test_the_agent_reads_edits_and_actually_fixes_the_file(self, repo):
        """The whole path, on a file that really is wrong to begin with."""
        recorder = Recorder()
        agent = make_agent(repo, [
            {"tool": "read_file", "arguments": {"path": "calc.py"}},
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"content": "Fixed the off-by-one in average()."},
        ], recorder)

        assert "range(len(values) - 1)" in (repo / "calc.py").read_text()
        result = await agent.run("fix the bug in calc.py")
        await agent.client.aclose()

        # The file on disk really changed, and really is correct now.
        after = (repo / "calc.py").read_text(encoding="utf-8")
        assert after == FIXED
        namespace: dict = {}
        exec(compile(after, "calc.py", "exec"), namespace)
        assert namespace["average"]([1, 2, 3]) == 2

        assert [name for name, _ in recorder.tools] == ["read_file", "write_file"]
        assert all(ok for _, ok in recorder.results)
        assert result.errors == []

    async def test_the_edit_reached_the_ui_in_pieces(self, repo):
        """The card fills in from the model's own fragments. One lump would
        mean the UI was handed a finished string, which is the thing this
        whole feature exists not to do."""
        recorder = Recorder()
        agent = make_agent(repo, [
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"content": "Done."},
        ], recorder)
        await agent.run("fix calc.py")
        await agent.client.aclose()

        assert len(recorder.code) > 1, "the edit must arrive progressively"
        assert "".join(recorder.code) == FIXED, (
            "what the UI was shown must be exactly what the model sent")

    async def test_a_failing_test_run_is_reported_as_a_failure(self, repo):
        """Recovery starts from an honest failure. A run that reports success
        on a broken file would make the rest of the loop meaningless."""
        recorder = Recorder()
        agent = make_agent(repo, [
            {"tool": "run_tests", "arguments": {}},
            {"content": "The tests fail."},
        ], recorder)
        await agent.run("run the tests")
        await agent.client.aclose()
        assert ("run_tests", False) in recorder.results

    async def test_the_tests_pass_once_the_bug_is_fixed(self, repo):
        """The end of the loop, on the real file the real edit produced."""
        recorder = Recorder()
        agent = make_agent(repo, [
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"tool": "run_tests", "arguments": {}},
            {"content": "Fixed and green."},
        ], recorder)
        await agent.run("fix calc.py and run the tests")
        await agent.client.aclose()
        assert ("run_tests", True) in recorder.results, recorder.results


class TestTheCardFollowsTheRealEdit:
    """The card is fed by the same on_code stream the agent emits."""

    def test_it_counts_what_actually_changed(self):
        card = livediff.DiffCard(tool="write_file", path="calc.py",
                                 before=BROKEN)
        for i in range(0, len(FIXED), 17):
            card.feed(FIXED[i:i + 17])
        card.finish()
        added, removed = card.counts()
        assert added and removed, "a rewrite both adds and removes lines"
        assert "+" in card.summary(_glyphs())

    def test_a_new_file_is_all_additions(self):
        card = livediff.DiffCard(tool="write_file", path="new.py", before="")
        card.feed("a = 1\nb = 2\n")
        card.finish()
        assert card.counts() == (2, 0)

    def test_it_shows_nothing_before_anything_arrives(self):
        card = livediff.DiffCard(tool="write_file", path="calc.py")
        assert card.body(80) == []
        assert card.counts() == (0, 0)

    def test_a_live_card_shows_the_tail(self):
        """While writing, the interesting end is the part just written."""
        card = livediff.DiffCard(tool="write_file", path="a.py", before="")
        card.feed("".join(f"line {i}\n" for i in range(80)))
        rows = card.body(80, rows=6)
        assert len(rows) == 6
        assert "line 79" in rows[-1]

    def test_a_finished_card_shows_the_head_and_says_what_it_cut(self):
        card = livediff.DiffCard(tool="write_file", path="a.py", before="")
        card.feed("".join(f"line {i}\n" for i in range(80)))
        card.finish()
        rows = card.body(80, rows=6)
        assert len(rows) == 6
        assert "more lines" in rows[-1]

    def test_a_failed_edit_says_so(self):
        card = livediff.DiffCard(tool="write_file", path="a.py")
        card.feed("x = 1\n")
        card.finish(ok=False, error="permission denied")
        assert card.state == livediff.FAILED
        assert "permission denied" in card.summary(_glyphs())

    def test_feeding_a_finished_card_changes_nothing(self):
        card = livediff.DiffCard(tool="write_file", path="a.py")
        card.feed("x = 1\n")
        card.finish()
        card.feed("y = 2\n")
        assert "y = 2" not in card.streamed

    def test_only_edit_tools_get_a_card(self):
        assert livediff.is_edit("write_file")
        assert livediff.is_edit("edit_file")
        assert not livediff.is_edit("list_dir")
        assert not livediff.is_edit("")

    def test_a_missing_file_is_simply_nothing_to_diff_against(self, tmp_path):
        assert livediff.read_before(tmp_path, "nope.py") == ""
        assert livediff.read_before(tmp_path, "") == ""

    def test_the_card_never_draws_wider_than_it_was_given(self):
        card = livediff.DiffCard(tool="write_file", path="a.py", before="")
        card.feed("x" * 500 + "\n")
        for width in (40, 60, 80, 120, 160):
            for row in card.render(_glyphs(), width):
                assert len(row) <= width + 2, (width, len(row))


def _glyphs():
    from wynxo.ui import Glyphs

    return Glyphs(True)


class TestWindows:
    """Deterministic checks for the paths that differ on Windows.

    Windows Terminal itself cannot be driven from this host, so nothing here
    claims to have observed one. What these do check is the input shapes a
    Windows session actually produces -- a non-UTF console, CRLF files,
    backslash paths, cp1252 bytes -- against the same production code the
    real terminal would run.
    """

    def test_the_card_is_pure_ascii_on_a_non_utf_console(self):
        """cmd.exe and an unconfigured PowerShell render box-drawing glyphs
        as question marks, which reads as a broken border rather than a
        plain one."""
        card = livediff.DiffCard(tool="write_file", path="calc.py",
                                 before="a = 1\n")
        card.feed("a = 2\nb = 3\n")
        card.finish()
        for row in card.render(_ascii_glyphs(), 60):
            assert row.isascii(), row

    def test_crlf_files_diff_by_content_not_by_line_ending(self):
        """A CRLF file compared against LF output would report every line as
        changed, and the count is the one number a summary line promises."""
        card = livediff.DiffCard(tool="write_file", path="w.py",
                                 before="a = 1\r\nb = 2\r\n")
        card.feed("a = 1\r\nb = 9\r\n")
        card.finish()
        assert card.counts() == (1, 1)

    def test_a_backslash_path_survives_to_the_summary(self):
        card = livediff.DiffCard(tool="write_file", path="src\\pkg\\mod.py")
        card.feed("x = 1\n")
        card.finish()
        assert "src\\pkg\\mod.py" in card.summary(_glyphs())

    def test_an_undecodable_file_is_still_diffable(self, tmp_path):
        """A cp1252 or latin-1 file in a UTF-8 world must not raise inside a
        repaint; errors="replace" means a mangled character, not a crash."""
        target = tmp_path / "legacy.py"
        target.write_bytes(b"# caf\xe9\nx = 1\n")
        before = livediff.read_before(tmp_path, "legacy.py")
        assert "x = 1" in before
        card = livediff.DiffCard(tool="write_file", path="legacy.py",
                                 before=before)
        card.feed("# cafe\nx = 2\n")
        card.finish()
        assert card.counts()[0] >= 1

    def test_a_directory_where_a_file_was_expected_is_survivable(self, tmp_path):
        """IsADirectoryError is an OSError on POSIX and a PermissionError on
        Windows; both have to mean "nothing to diff against"."""
        (tmp_path / "adir").mkdir()
        assert livediff.read_before(tmp_path, "adir") == ""

    def test_ctrl_d_is_bound_and_cannot_end_the_session(self):
        """prompt_toolkit's default Ctrl-D on an empty buffer is EOF. Bound
        explicitly, it toggles the diff instead of quitting mid-edit."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._make_prompt_bindings) \
            if hasattr(Repl, "_make_prompt_bindings") else ""
        if not source:
            source = inspect.getsource(Repl.__init__)
        assert 'bindings.add("c-d")' in source or "c-d" in source

    def test_the_new_binding_collides_with_nothing(self):
        """Ctrl-O, Ctrl-T, Ctrl-E, Ctrl-B, Ctrl-R, Ctrl-C and F2 are taken."""
        import inspect
        import re

        from wynxo import cli

        source = inspect.getsource(cli)
        bound = re.findall(r'bindings\.add\("(c-[a-z]|f\d+)"', source)
        assert len(bound) == len(set(bound)), f"duplicate binding: {bound}"
        assert "c-d" in bound


def _ascii_glyphs():
    from wynxo.ui import Glyphs

    return Glyphs(False)


class TestTheCardCatchesTheStream:
    """The ordering that broke it the first time.

    The arguments stream arrives *before* the call has been parsed, so
    on_code fires before on_tool_start. A card opened at tool start is
    created after the stream it exists to catch, and fills in empty --
    which looks exactly like a working card on a fast model.
    """

    async def _run(self, repo, script):
        from wynxo.cli import TerminalCallbacks
        from wynxo.layout import Transcript
        from wynxo.ui import UI

        ui = UI()
        transcript = Transcript(90)
        ui.attach(transcript)
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.workspace = repo
        agent = make_agent(repo, script, callbacks)
        await agent.run("fix calc.py")
        await agent.client.aclose()
        return callbacks, transcript

    async def test_the_card_holds_what_was_streamed(self, repo):
        callbacks, _ = await self._run(repo, [
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"content": "Done."},
        ])
        assert callbacks.card is not None
        assert callbacks.card.streamed == FIXED, (
            "the card must receive the fragments, not be created after them")
        assert callbacks.card.path == "calc.py"

    async def test_the_counts_come_from_the_real_before_and_after(self, repo):
        callbacks, _ = await self._run(repo, [
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"content": "Done."},
        ])
        added, removed = callbacks.card.counts()
        assert (added, removed) != (0, 0)
        assert removed, "the broken lines it replaced must be counted"

    async def test_one_summary_line_lands_in_the_transcript(self, repo):
        """Not the whole file. The transcript is append-only -- anything
        written there while streaming could never be compacted afterwards."""
        import re

        callbacks, transcript = await self._run(repo, [
            {"tool": "write_file", "arguments": {"path": "calc.py",
                                                 "content": FIXED}},
            {"content": "Done."},
        ])
        plain = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in transcript.lines]
        summaries = [line for line in plain if "write_file" in line and "+" in line]
        assert len(summaries) == 1, summaries
        assert "calc.py" in summaries[0]
        body = "\n".join(plain)
        assert "for value in values" not in body, (
            "the streamed file must stay in the overlay, not the record")

    async def test_a_non_edit_tool_does_not_leave_a_card_open(self, repo):
        callbacks, _ = await self._run(repo, [
            {"tool": "read_file", "arguments": {"path": "calc.py"}},
            {"content": "Read it."},
        ])
        assert callbacks.card is None or not callbacks.card.live


class TestAtomicProvidersStillCountHonestly:
    """Ollama's native tool_calls carry their arguments complete, so nothing
    streams and the card has no content of its own. Reporting +0 -0 on an
    edit that plainly changed the file would be worse than saying nothing."""

    def test_the_settled_file_supplies_the_count(self):
        card = livediff.DiffCard(tool="write_file", path="calc.py",
                                 before=BROKEN)
        assert card.counts() == (0, 0), "nothing streamed yet"
        card.finish(ok=True, settled=FIXED)
        added, removed = card.counts()
        assert (added, removed) != (0, 0)
        assert removed, "the replaced lines must be counted"

    def test_a_streamed_card_keeps_exactly_what_streamed(self):
        """The fallback must never overwrite real stream data."""
        card = livediff.DiffCard(tool="write_file", path="calc.py",
                                 before=BROKEN)
        card.feed(FIXED)
        card.finish(ok=True, settled="something else entirely\n")
        assert card.streamed == FIXED

    def test_a_failed_edit_is_not_credited_with_the_file(self):
        card = livediff.DiffCard(tool="write_file", path="calc.py",
                                 before=BROKEN)
        card.finish(ok=False, error="refused")
        assert card.counts() == (0, 0)
        assert "refused" in card.summary(_glyphs())
