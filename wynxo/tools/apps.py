"""Safe, platform-aware launching of common desktop applications."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..schema import Field, Schema
from .base import Tool, ToolResult

APPLICATIONS = ("calculator", "notepad", "browser", "terminal", "explorer", "vscode")


class OpenApplicationInput(Schema):
    application = Field(str, "Allowlisted application: calculator, notepad, browser, terminal, explorer, or vscode.", choices=APPLICATIONS)


class OpenApplication(Tool):
    name = "open_application"
    description = (
        "Open one common desktop application. Use this for explicit requests "
        "such as 'open calculator'; do not use repository tools first. The "
        "application identifier must be one of the allowlisted names."
    )
    Input = OpenApplicationInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: OpenApplicationInput) -> ToolResult:
        command = self._command(args.application)
        if not command:
            return ToolResult.failure(f"{args.application} is not available on {sys.platform}.")
        try:
            subprocess.Popen(command, cwd=str(self.workspace), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=os.name != "nt")
        except OSError as exc:
            return ToolResult.failure(f"Could not open {args.application}: {exc}", application=args.application)
        return ToolResult.success(f"Opened {args.application}.", display=f"opened {args.application}", application=args.application)

    @staticmethod
    def _command(application: str) -> list[str] | None:
        code_bin = shutil.which("code")
        if sys.platform == "win32":
            return {
                "calculator": ["calc.exe"],
                "notepad": ["notepad.exe"],
                "explorer": ["explorer.exe"],
                "browser": ["cmd.exe", "/c", "start", "", "https://www.google.com"],
                "terminal": [os.environ.get("COMSPEC", "cmd.exe")],
                "vscode": [code_bin, "--wait"] if code_bin else None,
            }.get(application)
        if sys.platform == "darwin":
            return {
                "calculator": ["open", "-a", "Calculator"],
                "notepad": ["open", "-a", "TextEdit"],
                "explorer": ["open", str(Path.home())],
                "browser": ["open", "https://www.google.com"],
                "terminal": ["open", "-a", "Terminal"],
                "vscode": [code_bin, "--wait"] if code_bin else None,
            }.get(application)
        browser = shutil.which("xdg-open") or shutil.which("gio")
        terminal = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal")
        return {
            "calculator": [shutil.which("gnome-calculator") or "gnome-calculator"],
            "notepad": [shutil.which("gedit") or "gedit"],
            "explorer": [browser or "xdg-open", str(Path.home())],
            "browser": [browser or "xdg-open", "https://www.google.com"],
            "terminal": [terminal] if terminal else None,
            "vscode": [code_bin, "--wait"] if code_bin else None,
        }.get(application)
