from __future__ import annotations

from wynxo._shell_hardening import _windows_refusal


def test_windows_system_commands_are_blocked() -> None:
    assert _windows_refusal("Stop-Computer -Force")
    assert _windows_refusal("Restart-Computer")
    assert _windows_refusal("Format-Volume -DriveLetter C")
    assert _windows_refusal("Clear-Disk -Number 0 -RemoveData")
    assert _windows_refusal("Remove-Partition -DiskNumber 0 -PartitionNumber 1")
    assert _windows_refusal("diskpart /s wipe.txt")
    assert _windows_refusal("bcdedit /delete {current}")


def test_windows_project_cleanup_is_not_hard_blocked() -> None:
    assert _windows_refusal("Remove-Item -Recurse -Force .\\build") == ""
    assert _windows_refusal("Remove-Item .\\tmp.txt") == ""
