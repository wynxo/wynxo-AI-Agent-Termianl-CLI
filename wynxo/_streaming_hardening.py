from __future__ import annotations


def install() -> None:
    from .ui import CodeStreamer

    original = CodeStreamer.finish
    if getattr(original, "_wynxo_partial_fix", False):
        return

    def finish(self):
        # CodeStreamer keeps a line fragment in `partial` while a fenced code
        # block is streaming. The original finish() only flushed `buffer`, so
        # the final line vanished if the model ended without a newline.
        if getattr(self, "partial", ""):
            self.buffer = self.partial + self.buffer
            self.partial = ""
        return original(self)

    finish._wynxo_partial_fix = True
    CodeStreamer.finish = finish


install()
