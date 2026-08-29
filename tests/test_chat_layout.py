"""The fullscreen chat layout is gone.

wynxo always runs the scrolling prompt: output goes to the terminal's real
scrollback, the mouse is never captured, so scrolling, drag-select and copy
are the terminal's own. The tests here guard that the layout really is
gone -- no config knob, no CLI flag, no reachable code path -- and that
everything that used to be routed through it still works in the classic
prompt.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from wynxo.cli import CommandCompleter

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class TestNoLayoutOptionRemains:
    def test_there_is_no_layout_option_anymore(self):
        """The fullscreen chat layout is gone: wynxo always runs the
        scrolling prompt, so no config knob and no CLI flag may exist for
        switching layouts."""
        from wynxo.config import Config

        assert not hasattr(Config(), "chat_layout")

    def test_the_layout_flags_no_longer_exist(self):
        from wynxo.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--chat"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--classic"])

    def test_no_running_code_reaches_the_layout(self):
        """Nothing in the package may import the deleted module -- an
        import in cli.py would resurrect the whole subsystem."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        banned = ("from .tui import", "from . import tui",
                  "import tui", "from wynxo.tui import", "import wynxo.tui")
        offenders = []
        for path in (root / "wynxo").rglob("*.py"):
            if "site-packages" in str(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                if any(stripped.startswith(b) or stripped.endswith(b)
                       for b in banned):
                    offenders.append(f"{path}: {line}")
        assert not offenders, "the chat layout module is gone; nothing may import it:\n" + "\n".join(offenders)


class TestTheClassicPath:
    """The scrolling prompt is the one and only mode; a command-line prompt
    runs its turn directly, no queueing into a layout."""

    def _repl(self):

        from wynxo.cli import Repl

        repl = Repl.__new__(Repl)
        repl.turn_calls = []

        async def _connect():
            return True

        async def turn(text):
            repl.turn_calls.append(text)

        async def _loop():
            return 0

        repl._connect, repl.turn, repl._loop = _connect, turn, _loop
        return repl

    def test_the_prompt_is_answered_directly(self):
        repl = self._repl()
        assert asyncio.run(repl.start_with("add retries")) == 0
        assert repl.turn_calls == ["add retries"]

    def test_the_real_constructor_builds_a_lazy_prompt(self, tmp_path):
        """The classic PromptSession opens the Windows console in its
        constructor, so it must be built lazily, on first real use."""
        from wynxo.cli import Repl
        from wynxo.config import Config
        from wynxo.ui import UI

        repl = Repl(Config(), tmp_path, UI())
        assert repl.prompt_session is not None
        assert type(repl.prompt_session).__name__ == "_LazyPromptSession"

    def test_the_watcher_always_runs_during_a_turn(self):
        """With the layout gone there is nothing else reading the terminal,
        so the key watcher must start unconditionally."""
        import inspect

        from wynxo.cli import Repl

        source = (inspect.getsource(Repl.turn)
                  + inspect.getsource(Repl._turn_locked))
        assert "watcher.start()" in source, "the classic REPL needs its key watcher"
        assert "self.chat" not in source, (
            "no chat-layout guard may remain in the turn path")


class TestCommandSuggestions:
    def test_typing_a_prefix_offers_the_commands(self):
        from prompt_toolkit.document import Document

        completer = CommandCompleter(lambda: ".")
        found = [c.text for c in completer.get_completions(
            Document("/mo", len("/mo")), None)]
        assert "/model" in found and "/mode" in found

    def test_subcommand_values_are_offered(self):
        from prompt_toolkit.document import Document

        completer = CommandCompleter(lambda: ".")
        found = [c.text for c in completer.get_completions(
            Document("/effort h", len("/effort h")), None)]
        assert found == ["high"]
        found = [c.text for c in completer.get_completions(
            Document("/effort x", len("/effort x")), None)]
        assert found == ["xhigh"]

    def test_models_are_offered_when_a_getter_is_wired(self):
        from prompt_toolkit.document import Document

        completer = CommandCompleter(lambda: ".",
                                     model_names_getter=lambda: ["qwen3:8b", "llama3"])
        found = [c.text for c in completer.get_completions(
            Document("/model qw", len("/model qw")), None)]
        assert found == ["qwen3:8b"]

    def test_dictation_is_a_command_now(self):
        from wynxo.cli import COMMANDS

        assert "/dictate" in COMMANDS
        assert "/layout" not in COMMANDS


class TestJsonOutput:
    """`wynxo -p "..." --json` prints one parseable object, nothing else."""

    def test_the_flag_exists(self):
        from wynxo.cli import build_parser

        args = build_parser().parse_args(["-p", "--json", "hello"])
        assert args.json is True

    def test_run_once_prints_a_single_json_object(self, tmp_path, monkeypatch):
        import json

        import wynxo.cli as cli

        class _Client:
            base_url = "http://x"

            async def ping(self):
                return "0.1.0"

            async def aclose(self):
                pass

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 20
            requests = 1
            tool_calls = 2

        class _Agent:
            permissions = type("P", (), {"mode": None})()
            session = type("S", (), {"usage": _Usage()})()

            def refresh_system_prompt(self):
                pass

            async def detect_capabilities(self):
                pass

            async def run(self, prompt):
                return type("R", (), {"content": "the answer",
                                        "errors": []})()

        monkeypatch.setattr(cli, "make_client", lambda config: _Client())
        monkeypatch.setattr(cli, "Agent", lambda *a, **k: _Agent())
        monkeypatch.setattr(cli, "resolve", lambda *a: None)
        monkeypatch.setattr(cli, "Memory", lambda *a: None)
        monkeypatch.setattr(cli, "resolve_scope", lambda *a: None)
        class _Callbacks:
            def _end_stream(self):
                pass

        monkeypatch.setattr(cli, "TerminalCallbacks", lambda *a, **k: _Callbacks())
        from wynxo.ui import UI

        ui = UI()
        out = {}

        def fake_print(obj):
            out.update(json.loads(obj))

        monkeypatch.setattr("builtins.print", fake_print)

        async def go():
            return await cli.run_once(
                type("C", (), {"model": "qwen3:8b", "stream": True,
                                "effort": "medium"})(),
                tmp_path, ui, "hi", json_output=True)

        import asyncio

        code = asyncio.run(go())
        assert code == 0
        assert out["ok"] is True
        assert out["content"] == "the answer"
        assert out["errors"] == []
        assert out["usage"]["completion_tokens"] == 20
        assert out["model"] == "qwen3:8b"
        # Nothing else reached stdout: the console was swapped for a sink.
        assert out.keys() == {"ok", "content", "errors", "model", "usage"}


class TestCopyCommand:
    """/copy -- the clipboard escape hatch, from clean session messages."""

    def _repl(self, messages):
        import types

        from wynxo.cli import Repl

        repl = Repl.__new__(Repl)
        repl.agent = types.SimpleNamespace(
            session=types.SimpleNamespace(messages=messages))
        ui = type("UI", (), {
            "info": lambda self, m: setattr(self, "message", m),
            "success": lambda self, m: setattr(self, "message", m),
            "error": lambda self, m: setattr(self, "message", m),
        })()
        repl.ui = ui
        return repl

    def _copy(self, repl, args, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "wynxo.platforms.copy_to_clipboard",
            lambda text: (captured.update(text=text) or True))
        repl.cmd_copy(args)
        return captured.get("text", None)

    def test_copy_builds_the_conversation_from_session_messages(self, monkeypatch):
        repl = self._repl([
            {"role": "user", "content": "what does auth.py do?"},
            {"role": "assistant", "content": "It guards the tokens."},
            {"role": "user", "content": "/layout"},
            {"role": "assistant", "content": "Layout report."},
        ])
        text = self._copy(repl, [], monkeypatch)
        assert text == (
            "> what does auth.py do?\n\nIt guards the tokens.\n\nLayout report.")

    def test_copy_last_takes_only_the_last_answer(self, monkeypatch):
        repl = self._repl([
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second answer"},
        ])
        assert self._copy(repl, ["last"], monkeypatch) == "second answer"

    def test_copy_with_nothing_to_say_does_not_touch_the_clipboard(self, monkeypatch):
        repl = self._repl([])
        text = self._copy(repl, [], monkeypatch)
        assert text is None
        assert repl.ui.message == "nothing to copy yet"
