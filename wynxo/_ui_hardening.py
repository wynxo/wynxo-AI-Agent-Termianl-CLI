from __future__ import annotations


def install() -> None:
    from .ui import ActivityBar, CodeStreamer

    original_set_lead = ActivityBar.set_lead
    if not getattr(original_set_lead, "_wynxo_live_code", False):
        def set_lead(self, line):
            self.lead = line
            # A partial streamed line is proof that generation is moving.
            # Surface that fact in the pinned status strip without adding a
            # new scrollback line for every character.
            if line is not None and line.plain:
                self.activity = "writing code"
                self.detail = f"{len(line.plain)} chars"
            self.refresh()

        set_lead._wynxo_live_code = True
        ActivityBar.set_lead = set_lead

    original_write = CodeStreamer._write
    if getattr(original_write, "_wynxo_live_code_write", False):
        return

    def write(self, text):
        original_write(self, text)
        if self.ui.bar is not None and self.literal and self.line.plain:
            # Keep the current file-generation byte/character count visible
            # while completed lines are committed into the transcript.
            self.ui.bar.activity = "writing code"
            self.ui.bar.detail = f"{len(self.line.plain)} chars on current line"
            self.ui.bar.refresh()

    write._wynxo_live_code_write = True
    CodeStreamer._write = write


install()
