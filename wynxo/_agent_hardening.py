from __future__ import annotations


async def _stream_test_output(callback, line: str) -> None:
    """Forward automatic verification output without orphaned coroutines."""
    try:
        result = callback("shell", line)
        if result is not None:
            await result
    except Exception:
        pass


def install() -> None:
    from .agent import Agent

    original_run = getattr(Agent, "run", None)
    if original_run is not None and not getattr(original_run, "_wynxo_run_hardening", False):
        async def run(self, request):
            self._warned_over_window = False
            return await original_run(self, request)

        run._wynxo_run_hardening = True
        Agent.run = run

    original_verify = getattr(Agent, "_verify_with_tests", None)
    if original_verify is None or getattr(original_verify, "_wynxo_stream_tests", False):
        return

    async def verify_with_tests(self):
        # Preserve the core function's early exits (disabled verification,
        # plan mode, no file changes, no runner, shell disabled). Looking up
        # the shell before those guards made a disabled setting appear to run.
        if not self.config.verify_with_tests or getattr(self.permissions, "mode", None) is not None and getattr(self.permissions.mode, "value", self.permissions.mode) == "plan":
            return await original_verify(self)
        if not self.checkpoints.changes_since(self._turn_mark):
            return await original_verify(self)
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
