"""Fail if anything wynxo depends on ships compiled code.

Termux has no Rust or C toolchain and no prebuilt wheels for most things
that need one, so a dependency with a native extension is the difference
between wynxo installing on someone's phone and not. pydantic is the
specific one this project turned down: pydantic-core is Rust, and there is
no Android wheel for it.

Two decisions worth stating. It walks wynxo's own dependency closure rather
than everything installed, because the environment this runs in may well
have other things in it and their extensions are not wynxo's problem. And it
looks for extension modules rather than matching known-bad names, so a new
dependency that brings compiled code in is caught without anyone having
remembered to add it to a list.
"""

from __future__ import annotations

import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution, requires
from pathlib import Path

ROOT = "wynxo"

SUFFIXES = tuple(dict.fromkeys(
    s for s in (sysconfig.get_config_var("EXT_SUFFIX"), ".so", ".pyd", ".dll")
    if s
))


def _name_of(requirement: str) -> str | None:
    """The bare package name from a requirement line.

    Extras markers are dropped along with anything conditional: a dependency
    that only installs under `extra == "dev"` is not shipped to a phone.
    """
    text = requirement.strip()
    if ";" in text:
        head, marker = text.split(";", 1)
        if "extra" in marker:
            return None
        text = head
    for stop in ("[", "(", "<", ">", "=", "!", "~", " "):
        text = text.split(stop, 1)[0]
    return text.strip().lower() or None


def closure(root: str) -> set[str]:
    """Every distribution `root` pulls in, transitively."""
    seen: set[str] = set()
    queue = [root.lower()]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            needed = requires(name) or []
        except PackageNotFoundError:
            continue
        for line in needed:
            if (dependency := _name_of(line)) and dependency not in seen:
                queue.append(dependency)
    return seen


def compiled_files(name: str) -> list[str]:
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return []
    return [str(f) for f in (dist.files or [])
            if Path(str(f)).name.endswith(SUFFIXES)]


def main() -> int:
    offenders: dict[str, list[str]] = {}
    for name in sorted(closure(ROOT) - {ROOT}):
        if hits := compiled_files(name):
            offenders[name] = hits[:3]

    if not offenders:
        print(f"{ROOT} and its dependencies are pure python.")
        return 0

    print("These dependencies ship compiled code, so wynxo would not "
          "install on Termux:\n")
    for name, files in sorted(offenders.items()):
        print(f"  {name}")
        for file in files:
            print(f"      {file}")
    print("\nEither drop the dependency or find a pure-python alternative.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
