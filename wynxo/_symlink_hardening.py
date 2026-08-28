from __future__ import annotations


def install() -> None:
    from .checkpoints import Checkpoints
    from .tools.files import _atomic_write_bytes
    import wynxo.tools.files as files

    original_write = _atomic_write_bytes
    if not getattr(original_write, "_wynxo_symlink_safe", False):
        def atomic_write_bytes(path, data):
            target = path.resolve() if path.is_symlink() else path
            return original_write(target, data)
        atomic_write_bytes._wynxo_symlink_safe = True
        files._atomic_write_bytes = atomic_write_bytes

    original_capture = Checkpoints.capture
    if not getattr(original_capture, "_wynxo_symlink_safe", False):
        def capture(self, path, tool, label=""):
            target = path.resolve() if path.is_symlink() else path
            return original_capture(self, target, tool, label)
        capture._wynxo_symlink_safe = True
        Checkpoints.capture = capture


install()
