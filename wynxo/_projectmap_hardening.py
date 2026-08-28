from __future__ import annotations


def install() -> None:
    from . import projectmap

    projectmap.SKIP_DIRS.discard(".vscode")
    projectmap.SKIP_DIRS.discard(".devcontainer")
    projectmap.SKIP_DIRS.discard(".github")

    original_walk = projectmap.walk
    if getattr(original_walk, "_wynxo_config_dirs", False):
        return

    def walk(root, limit=projectmap.MAX_FILES):
        allowed_hidden = {".github", ".vscode", ".devcontainer", ".husky", ".config"}
        found = []
        stack = [root]
        while stack and len(found) < limit:
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.name.startswith(".") and entry.name not in allowed_hidden:
                    continue
                if entry.is_dir():
                    if entry.name not in projectmap.SKIP_DIRS:
                        stack.append(entry)
                elif entry.suffix in projectmap.LANGUAGES:
                    found.append(entry)
                    if len(found) >= limit:
                        break
        return sorted(found)

    walk._wynxo_config_dirs = True
    projectmap.walk = walk

    original_load = projectmap.load
    if getattr(original_load, "_wynxo_max_age", False):
        return

    def load(root, max_age=0.0):
        import time
        path = projectmap.cache_path(root)
        try:
            cached = path.read_text(encoding="utf-8")
            stamp = path.stat().st_mtime
        except OSError:
            cached, stamp = "", 0.0
        fresh_enough = bool(cached) and (max_age <= 0 or time.time() - stamp <= max_age)
        if fresh_enough and stamp >= projectmap.newest_source(root):
            return cached
        fresh = projectmap.build(root)
        if fresh:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(fresh, encoding="utf-8")
            except OSError:
                pass
        return fresh

    load._wynxo_max_age = True
    projectmap.load = load


install()
