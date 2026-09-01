"""What the keys at the prompt do, and what they must not take away.

Ctrl-D was bound unconditionally to the diff toggle. That is the one
universal way out of a terminal program -- bash, python, psql, every REPL
-- and taking it away left ``except EOFError`` in the prompt loop
unreachable from the keyboard: the only ways out were /quit and two Ctrl-Cs,
neither of which a person tries first.

It was unconditional for a reason that has since gone. These bindings were
also handed to the old full-screen layout, whose application ran for the
whole *turn*, so an unfiltered Ctrl-D there really would have quit
mid-edit. The prompt only runs when nothing else does now, and mid-turn the
key watcher binds Ctrl-D to the same toggle.
"""

from __future__ import annotations


from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from wynxo.cli import LIVE_KEYS, Repl


class TestCtrlDLeavesTheSession:
    async def _prompt(self, keys: str):
        """Run one real prompt, feeding it ``keys``. Returns what happened."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.application.current import get_app

        toggled = []
        bindings = KeyBindings()

        @bindings.add("c-d", filter=Condition(
            lambda: bool(get_app().current_buffer.text)))
        def _(event):
            toggled.append(1)

        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(input=pipe, output=DummyOutput(),
                                    key_bindings=bindings)
            try:
                return ("text", await session.prompt_async()), toggled
            except EOFError:
                return ("eof", None), toggled

    async def test_an_empty_composer_ends_the_session(self):
        (kind, _), toggled = await self._prompt("\x04")
        assert kind == "eof", "Ctrl-D on an empty prompt did not mean EOF"
        assert toggled == []

    async def test_a_typed_line_toggles_the_diff_instead(self):
        (kind, text), toggled = await self._prompt("hello\x04\n")
        assert toggled == [1], "the diff toggle was lost"
        assert (kind, text) == ("text", "hello"), (
            "Ctrl-D ate the line it was typed on")

    async def test_the_filter_is_on_the_real_binding(self):
        """The behaviour above is only worth anything if cli's binding is
        the one carrying the filter."""
        import inspect

        source = inspect.getsource(Repl.__init__)
        block = source.split('bindings.add("c-d"', 1)[1].split("def ")[0]
        assert "filter=" in block and "current_buffer.text" in block


class TestTheAdvertisedKeysStillExist:
    def test_ctrl_d_is_still_a_mid_turn_key(self):
        """It is advertised in the activity bar during every turn, where the
        watcher -- not prompt_toolkit -- is the reader."""
        assert "ctrl+d" in LIVE_KEYS

    def test_the_turn_binds_every_key_it_advertises(self):
        import inspect

        source = inspect.getsource(Repl._turn_locked)
        watcher = source.split("KeyWatcher(")[1].split("on_key=")[0]
        for key in LIVE_KEYS:
            assert f'"{key}"' in watcher, f"{key} is advertised but never bound"
