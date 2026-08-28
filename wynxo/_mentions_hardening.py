from __future__ import annotations


def install() -> None:
    from . import mentions

    for name in (".vscode", ".devcontainer", ".github", ".husky"):
        mentions.SKIP_DIRS.discard(name)

    original = mentions.candidates
    if getattr(original, "_wynxo_project_config", False):
        return

    def candidates(workspace, prefix="", limit=200):
        prefix = prefix.replace("\\", "/")
        head, _, tail = prefix.rpartition("/")
        base = (workspace / head) if head else workspace
        try:
            base = base.resolve()
            if not base.is_dir() or not mentions._within(base, workspace):
                return []
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []
        out = []
        for entry in entries:
            if entry.is_symlink():
                # Don't offer links whose target may leave the project scope.
                continue
            if entry.name.startswith(".") and not tail.startswith("."):
                continue
            if entry.is_dir() and entry.name in mentions.SKIP_DIRS:
                continue
            if tail and not entry.name.lower().startswith(tail.lower()):
                continue
            shown = f"{head}/{entry.name}" if head else entry.name
            out.append(shown + "/" if entry.is_dir() else shown)
            if len(out) >= limit:
                break
        return out

    candidates._wynxo_project_config = True
    mentions.candidates = candidates


install()
