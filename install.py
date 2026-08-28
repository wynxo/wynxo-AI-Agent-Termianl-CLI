"""Installer for wynxo.

    python3 install.py

Installs the agent and nothing else: a virtualenv, the package, and a `wynxo`
command on your PATH. It does not touch Ollama, does not download models and
does not decide anything about them -- wynxo asks the server what it has and
you pick, which is the only way that stays right as your models change.

    --yes      accept the recommended answer to every prompt
    --no-link  do not put a `wynxo` command on PATH
    --no-ollama legacy no-op retained for old wrappers
    --venv DIR virtualenv location (default: .venv)
"""

# NOTE: This compatibility flag is retained because older Windows wrappers
# still pass it. The installer has never acted on it in this version.
