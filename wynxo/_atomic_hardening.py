from __future__ import annotations

import os
import tempfile


def _replace_bytes(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".wynxo-tmp", dir=str(path.parent))
    temporary = path.parent / os.path.basename(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def install() -> None:
    from . import config
    import wynxo.tools.files as files

    original_text = config.atomic_write
    if not getattr(original_text, "_wynxo_unique_temp", False):
        def atomic_write(path, text):
            encoded = text.encode("utf-8")
            _replace_bytes(path, encoded)
        atomic_write._wynxo_unique_temp = True
        config.atomic_write = atomic_write

    original_bytes = files._atomic_write_bytes
    if not getattr(original_bytes, "_wynxo_unique_temp", False):
        def atomic_write_bytes(path, data):
            mode = None
            try:
                if path.exists() and not path.is_symlink():
                    mode = path.stat().st_mode & 0o777
            except OSError:
                pass
            if mode is not None:
                # Apply the mode to a temporary file before replacement.
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".wynxo-tmp", dir=str(path.parent))
                temporary = path.parent / os.path.basename(raw)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        temporary.chmod(mode)
                    except OSError:
                        pass
                    os.replace(temporary, path)
                finally:
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                return
            _replace_bytes(path, data)
        atomic_write_bytes._wynxo_unique_temp = True
        files._atomic_write_bytes = atomic_write_bytes


install()
