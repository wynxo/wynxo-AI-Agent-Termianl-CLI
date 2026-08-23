#!/usr/bin/env sh
# One-command setup for wynxo (Linux, macOS, Termux).
#
#   ./install.sh              interactive
#   ./install.sh --yes        accept every recommendation
#   ./install.sh --no-ollama  just install wynxo
#
# All it does is find a Python 3.10+ and hand over to install.py.

set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            exec "$candidate" "$DIR/install.py" "$@"
        fi
    fi
done

echo "wynxo needs Python 3.10 or newer, and none was found."
echo
if [ -n "${PREFIX:-}" ] && [ -d "/data/data/com.termux" ]; then
    echo "  Termux:  pkg install python"
elif [ "$(uname -s)" = "Darwin" ]; then
    echo "  macOS:   brew install python"
else
    echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv"
    echo "  Fedora:         sudo dnf install python3"
    echo "  Arch:           sudo pacman -S python"
fi
exit 1
