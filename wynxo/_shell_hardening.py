from __future__ import annotations

import re


def _windows_refusal(line: str) -> str:
    """Detect destructive PowerShell/cmd operations missed by Unix parsing."""
    text = line.strip()
    low = text.lower()
    if re.search(r"\b(?:stop-computer|restart-computer)\b", low):
        return "shutting the machine down"
    if re.search(r"\b(?:clear-disk|initialize-disk|remove-partition)\b", low):
        return "destroying disk/partition state"
    if re.search(r"\bformat-volume\b", low) and re.search(r"(?:-driveletter\s+[a-z]|-path\s+[a-z]:\\?)", low):
        return "formatting a drive"
    if re.search(r"\b(?:remove-item|ri|del|erase|rd|rmdir)\b", low):
        # Catch drive roots and core system roots, without banning ordinary
        # project cleanup such as `Remove-Item build -Recurse`.
        roots = re.findall(r"(?<![\w])([a-z]:\\(?:[\s\"']|$)|[a-z]:\\(?:(?:windows|users|program files)(?:\\|$)))", low)
        if roots:
            return "deleting a whole drive or system root"
    if re.search(r"\b(?:diskpart|bcdedit)\b", low):
        return "modifying low-level Windows system state"
    return ""


def install() -> None:
    from .tools import shell

    original_refusal = shell.hard_refusal
    if not getattr(original_refusal, "_wynxo_windows_hardened", False):
        def hard_refusal(line: str) -> str:
            if shell.os.name == "nt":
                if reason := _windows_refusal(line):
                    return reason
            return original_refusal(line)

        hard_refusal._wynxo_windows_hardened = True
        shell.hard_refusal = hard_refusal

    original_signal = shell._signal_group
    if not getattr(original_signal, "_wynxo_graceful_windows", False):
        def signal_group(process, terminate: bool) -> None:
            if process.pid is None:
                return
            if shell.os.name == "nt":
                try:
                    command = ["taskkill", "/T", "/PID", str(process.pid)]
                    if not terminate:
                        command.insert(1, "/F")
                    shell.subprocess.run(command, capture_output=True, timeout=10)
                    return
                except (shell.OSError, shell.subprocess.SubprocessError):
                    pass
            original_signal(process, terminate)

        signal_group._wynxo_graceful_windows = True
        shell._signal_group = signal_group


install()
