from __future__ import annotations


def install() -> None:
    from .ui import ActivityBar

    original = ActivityBar.set_lead
    if getattr(original, "_wynxo_live_code", False):
        return

    def set_lead(self, line):
        self.lead = line
        if line is not None and line.plain:
            self.activity = "writing code"
            self.detail = f"{len(line.plain)} chars"
        self.refresh()

    set_lead._wynxo_live_code = True
    ActivityBar.set_lead = set_lead


install()
