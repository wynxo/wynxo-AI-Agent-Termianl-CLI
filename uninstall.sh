#!/usr/bin/env sh
# Uninstall wynxo (Linux, macOS, Termux).
#
#   ./uninstall.sh              interactive
#   ./uninstall.sh --yes        accept every prompt
#   ./uninstall.sh --dry-run    list what would go, change nothing
#
# All it does is find a Python 3.10+ and hand over to uninstall.py.

set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            exec "$candidate" "$DIR/uninstall.py" "$@"
        fi
    fi
done

printf '  Python 3.10 or newer is required and was not found.\n'
printf '  wynxo lives in these places; remove them by hand:\n'
printf '    ~/.wynxo-src\n'
printf '    ~/.local/bin/wynxo\n'
printf '    ~/.config/wynxo  and  ~/.local/share/wynxo\n'
printf '    the "# added by wynxo installer" line in your shell profile\n'
exit 1
