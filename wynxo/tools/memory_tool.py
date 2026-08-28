"""The tool the agent uses to keep its own notes."""

from __future__ import annotations

from pathlib import Path

from ..memory import Memory
from ..schema import Field, Schema
from .base import Tool, ToolResult


class RememberInput(Schema):
    note = Field(str, "One short fact, in a sentence. Write it so it still "
                      "makes sense months from now, with no conversation "
                      "around it.")
    scope = Field(str, "'project' for something about this codebase, 'user' "
                       "for a preference that holds everywhere.",
                  default="project", choices=("project", "user"))
    forget = Field(bool, "Set true to delete remembered entries matching "
                         "`note` instead of adding one.", default=False)


class Remember(Tool):
    name = "remember"
    description = (
        "Write something down so you still know it in a later session. Use it "
        "for durable facts: build and test commands, conventions, decisions and "
        "why, things that bit you, and preferences the user states. "
        "User-scoped memory is never written by this tool: only an explicit "
        "`/memory add user: ...` request may persist personal facts. "
        "Do not use it for what is already in the files, for anything you are "
        "unsure of, or for the details of the current task. "
        "Set forget=true to remove an entry that has stopped being true."
    )
    Input = RememberInput
    mutating = True
    internal = True
    concurrency_safe = False

    def __init__(self, workspace: Path, boundary=None, memory: Memory | None = None,
                 shield=None):
        super().__init__(workspace, boundary, shield)
        self.memory = memory or Memory(workspace)
        self.memory._agent_write = True

    async def run(self, args: RememberInput) -> ToolResult:
        if args.forget:
            count, message = self.memory.forget(args.note, args.scope)
            return ToolResult.success(message, display=message, dropped=count)

        added, message = self.memory.remember(args.note, args.scope, explicit=False)
        if not added:
            # Not an error: the model tried to record something already known,
            # and telling it so is more useful than a failure.
            return ToolResult.success(f"Not added -- {message}", display=message)

        where = "about you" if args.scope == "user" else "about this project"
        return ToolResult.success(
            f"Remembered, {where}: {message}",
            display=f"remembered ({args.scope}): {message[:70]}",
            scope=args.scope,
        )
