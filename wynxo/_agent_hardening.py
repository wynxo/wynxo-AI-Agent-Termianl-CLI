from __future__ import annotations


def install() -> None:
    from .agent import Agent

    original_turn = Agent.turn
    if not getattr(original_turn, "_wynxo_turn_hardening", False):
        async def turn(self, text):
            # The agent documents this warning as per-turn state, but the
            # original field was only reset in __init__, so later overflowing
            # turns stayed silent forever.
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
        shell.on_output = lambda line: self.cb.on_tool_output("shell", line)
        try:
            return await original_verify(self)
        finally:
            shell.on_output = previous

    verify_with_tests._wynxo_stream_tests = True
    Agent._verify_with_tests = verify_with_tests


install()
