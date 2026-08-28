"""Windows-friendly launcher that avoids generated console-wrapper executables.

This module is intentionally tiny and delegates to the real bootstrap. It is
used by the supported ``wynxo.cmd`` shim, which invokes the venv interpreter
rather than launching pip's generated ``wynxo.exe``.
"""

from __future__ import annotations

from .bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main() or 0)
