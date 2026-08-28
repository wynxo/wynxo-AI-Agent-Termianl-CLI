from __future__ import annotations


def redact_shell_text(text: str) -> str:
    from .secrets import redact
    return redact(text)[0]


def install() -> None:
    from .tools.shell import Shell

    original = Shell._stream
    if getattr(original, "_wynxo_shell_redact", False):
        return

    async def stream(self, process, timeout):
        previous = self.on_output

        async def safe_output(line: str):
            if previous is None:
                return
            await previous(redact_shell_text(line))

        self.on_output = safe_output if previous is not None else None
        try:
            output, timed_out = await original(self, process, timeout)
        finally:
            self.on_output = previous
        return redact_shell_text(output), timed_out

    stream._wynxo_shell_redact = True
    Shell._stream = stream


install()
