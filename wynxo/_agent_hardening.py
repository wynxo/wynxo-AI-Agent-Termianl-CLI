from __future__ import annotations


async def _stream_test_output(callback, line: str) -> None:
    """Forward automatic verification output without creating orphaned coroutines."""
    try:
        result = callback("shell", line)
        if result is not None:
            await result
    except Exception:
        pass


def install() -> None:
    from .agent import Agent

    original_turn = Agent.turn
    if not getattr(original_turn, "_wynxo_turn_hardening", False):
        async def turn(self, text):
            self._warned_over_window = False
            return await original_turn(self, text)

        turn._wynxo_turn_hardening = True
        Agent.turn = turn

    original_verify = Agent._verify_with_tests
    if getattr(original_verify, "_wynxo_stream_tests", False):
        return

    async def verify_with_tests(self):
        shell = self.tools.get("shell")
        if shell is None:
            return await original_verify(self)

        previous = shell.on_output

        async def forward(line):
            await _stream_test_output(self.cb.on_tool_output, line)

        shell.on_output = forward
        try:
            return await original_verify(self)
        finally:
            shell.on_output = previous

    verify_with_tests._wynxo_stream_tests = True
    Agent._verify_with_tests = verify_with_tests


install()
