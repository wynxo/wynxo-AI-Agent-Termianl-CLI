"""Configuration: where Ollama lives, which model, which effort level.

Config is resolved from, in increasing order of priority:
  1. built-in defaults
  2. the user config file (platform-appropriate location)
  3. a project-local ``.wynxo.json``
  4. environment variables (``WYNXO_*``, plus ``OLLAMA_HOST``)
  5. command-line flags
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_ENDPOINT = "http://localhost:11434"

# Ollama's default context is small enough (2048 on many builds) that an agent
# silently loses its history with no error at all. This is the single most
# common reason a local agent "goes stupid" halfway through a task.
MIN_USABLE_CONTEXT = 16_384
DEFAULT_CONTEXT = 32_768


def config_dir() -> Path:
    """Per-platform config directory. Windows, macOS and Linux all differ."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "wynxo"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "wynxo"
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "wynxo"


def data_dir() -> Path:
    """Where sessions and logs are kept."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "wynxo"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "wynxo"
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "wynxo"


def config_path() -> Path:
    return config_dir() / "config.json"


class Endpoint(BaseModel):
    """One Ollama server. Users often have more than one -- a laptop for
    quick things and a homelab box with the big GPU."""

    name: str = "local"
    url: str = DEFAULT_ENDPOINT
    api_key: str | None = None
    """Only needed when the server sits behind a reverse proxy that
    requires auth. Sent as ``Authorization: Bearer ...``."""

    @field_validator("url")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return normalise_url(v)


class Config(BaseModel):
    endpoints: list[Endpoint] = Field(default_factory=lambda: [Endpoint()])
    active_endpoint: str = "local"
    model: str = DEFAULT_MODEL
    effort: str = "medium"

    num_ctx: int = DEFAULT_CONTEXT
    keep_alive: str = "30m"
    """Passed to Ollama so the model is not unloaded between turns. A reload
    of a 30B costs many seconds and makes the agent feel broken."""

    request_timeout: float = 600.0
    """Local generation on CPU can be genuinely slow; do not be stingy."""

    auto_approve: list[str] = Field(default_factory=list)
    """Tool names that never prompt for permission, e.g. ``["read_file"]``."""

    allow_shell: bool = True
    theme: str = "dark"
    show_thinking: bool = True
    stream: bool = True

    def endpoint(self) -> Endpoint:
        for ep in self.endpoints:
            if ep.name == self.active_endpoint:
                return ep
        if self.endpoints:
            return self.endpoints[0]
        return Endpoint()

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)  # may hold an api key; no-op on Windows
        except OSError:
            pass
        return path


def normalise_url(raw: str) -> str:
    """Accept the many shapes a person types a server address in.

    ``homelab:11434``, ``http://homelab``, ``192.168.1.5``, and a trailing
    ``/v1`` or ``/api`` all land on the same base URL.
    """
    v = raw.strip().rstrip("/")
    if not v:
        return DEFAULT_ENDPOINT
    if not v.startswith(("http://", "https://")):
        v = "http://" + v
    for suffix in ("/v1", "/api"):
        if v.endswith(suffix):
            v = v[: -len(suffix)]
    # Bare http host with no port: assume Ollama's default rather than :80.
    # https is left alone -- that shape means a reverse proxy on 443, not a
    # directly exposed Ollama.
    if v.startswith("http://"):
        host = v[len("http://"):].split("/", 1)[0]
        has_port = "]" in host if host.startswith("[") else ":" in host
        if not has_port:
            v = v.replace(host, f"{host}:11434", 1)
    return v.rstrip("/")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load(project_dir: Path | None = None) -> Config:
    """Assemble config from every layer."""
    data: dict[str, Any] = {}
    data.update(_read_json(config_path()))

    project_dir = project_dir or Path.cwd()
    project_cfg = project_dir / ".wynxo.json"
    if project_cfg.exists():
        data.update(_read_json(project_cfg))

    # Environment overrides. OLLAMA_HOST is respected because anyone already
    # running Ollama remotely will have it set.
    env_url = os.environ.get("WYNXO_ENDPOINT") or os.environ.get("OLLAMA_HOST")
    if env_url:
        data["endpoints"] = [{"name": "env", "url": normalise_url(env_url)}]
        data["active_endpoint"] = "env"
    if v := os.environ.get("WYNXO_MODEL"):
        data["model"] = v
    if v := os.environ.get("WYNXO_EFFORT"):
        data["effort"] = v
    if v := os.environ.get("WYNXO_NUM_CTX"):
        try:
            data["num_ctx"] = int(v)
        except ValueError:
            pass

    try:
        return Config.model_validate(data)
    except Exception:
        # A corrupt config file should never be fatal -- fall back to defaults
        # rather than leaving the user with an agent that will not start.
        return Config()


def is_configured() -> bool:
    return config_path().exists()
