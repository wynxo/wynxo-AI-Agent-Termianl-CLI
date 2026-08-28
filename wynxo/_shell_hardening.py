from __future__ import annotations


def install() -> None:
    from .tools import shell

    original = shell._signal_group
    if getattr(original, "_wynxo_graceful_windows", False):
        return

    def signal_group(process, terminate: bool) -> None:
        if process.pid is None:
            return
        if shell.os.name == "nt":
            try:
                # First attempt a normal tree termination. The original code
                # used /F even for the graceful phase, so a Ctrl-C/timeout
                # immediately hard-killed pytest/npm children and skipped
                # their cleanup. _terminate() already escalates to force-kill
                # after its grace period.
                command = ["taskkill", "/T", "/PID", str(process.pid)]
                if not terminate:
                    command.insert(1, "/F")
                shell.subprocess.run(command, capture_output=True, timeout=10)
                return
            except (shell.OSError, shell.subprocess.SubprocessError):
                pass
        original(process, terminate)

    signal_group._wynxo_graceful_windows = True
    shell._signal_group = signal_group


install()
