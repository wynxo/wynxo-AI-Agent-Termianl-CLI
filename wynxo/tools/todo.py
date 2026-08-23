"""A visible plan the model maintains as it works.

Two jobs. The user gets to see where the agent thinks it is, and the model
gets an external memory that survives context compaction -- which is what
keeps a long ``max``-effort run from forgetting step three of five.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

Status = Literal["pending", "in_progress", "done"]
MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}


class TodoItem(BaseModel):
    task: str = Field(description="One concrete step.")
    status: Status = Field("pending")


class TodoInput(BaseModel):
    items: list[TodoItem] = Field(
        description="The complete list, every time. It replaces the previous one."
    )


class TodoWrite(Tool):
    name = "todo_write"
    description = (
        "Record or update your plan as a checklist. Send the whole list each "
        "time -- it replaces the previous one. Use it for anything that takes "
        "more than two or three steps, and mark exactly one item in_progress."
    )
    Input = TodoInput
    concurrency_safe = False

    def __init__(self, workspace):
        super().__init__(workspace)
        self.items: list[TodoItem] = []

    async def run(self, args: TodoInput) -> ToolResult:
        if not args.items:
            self.items = []
            return ToolResult.success("Todo list cleared.")

        active = [i for i in args.items if i.status == "in_progress"]
        self.items = args.items

        rendered = "\n".join(f"{MARKS[i.status]} {i.task}" for i in self.items)
        done = sum(1 for i in self.items if i.status == "done")
        summary = f"{done}/{len(self.items)} done"

        note = ""
        if len(active) > 1:
            note = "\n(Note: more than one item is in_progress. Work on one at a time.)"
        return ToolResult.success(
            f"Plan updated ({summary}):\n{rendered}{note}",
            display=rendered,
            done=done,
            total=len(self.items),
        )

    def render(self) -> str:
        if not self.items:
            return ""
        return "\n".join(f"{MARKS[i.status]} {i.task}" for i in self.items)

    def outstanding(self) -> list[str]:
        return [i.task for i in self.items if i.status != "done"]
