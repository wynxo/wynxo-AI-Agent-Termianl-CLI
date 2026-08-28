from __future__ import annotations


def install() -> None:
    from .tools.search import Glob, Grep
    from .tools.files import IGNORED

    original_collect = Glob._collect
    if not getattr(original_collect, "_wynxo_hidden_config", False):
        def collect(self, root, pattern):
            out = []
            import fnmatch
            for path in root.rglob("*"):
                if len(out) > 200 * 4:
                    break
                if not path.is_file():
                    continue
                if any(part in IGNORED for part in path.parts):
                    continue
                rel = self.relative(path)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                    out.append(rel)
            out.sort(key=lambda r: -(root.joinpath(r).stat().st_mtime if (root / r).exists() else 0))
            return out
        collect._wynxo_hidden_config = True
        Glob._collect = collect

    original_candidates = Grep._candidates
    if not getattr(original_candidates, "_wynxo_hidden_config", False):
        def candidates(self, root, glob):
            import fnmatch
            out = []
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in IGNORED for part in path.parts):
                    continue
                if glob and not (fnmatch.fnmatch(path.name, glob)
                                 or fnmatch.fnmatch(self.relative(path), glob)):
                    continue
                out.append(path)
            return out
        candidates._wynxo_hidden_config = True
        Grep._candidates = candidates


install()
