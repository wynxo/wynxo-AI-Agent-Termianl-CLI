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
from pathlib import Path

from . import platforms
from typing import Any

from .schema import Field, Schema

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# Ollama's default context is small enough (2048 on many builds) that an agent
# silently loses its history with no error at all. This is the single most
# common reason a local agent "goes stupid" halfway through a task.
MIN_USABLE_CONTEXT = 16_384
DEFAULT_CONTEXT = 32_768


def config_dir() -> Path:
    return platforms.config_dir()


def data_dir() -> Path:
    return platforms.data_dir()


def config_path() -> Path:
    return config_dir() / "config.json"


class Endpoint(Schema):
    """One Ollama server.

    Either this machine (``http://127.0.0.1:11434``) or another box on the
    LAN by IP (``http://192.168.1.50:11434``). Most people end up with both:
    the laptop they type on, and the machine with the GPU."""

    name = Field(str, "Short name you refer to this server by.", default="local")
    url = Field(str, "Base URL.", default=DEFAULT_ENDPOINT, transform=lambda v: normalise_url(v))
    api_key = Field(str, "Bearer token, when the server sits behind a proxy that "
                         "requires auth.", default=None)


class Config(Schema):
    endpoints = Field(list, "Known Ollama servers.", item_type=Endpoint,
                      default_factory=lambda: [Endpoint()])
    active_endpoint = Field(str, "Which endpoint to use.", default="local")
    model = Field(str, "Model tag.", default=DEFAULT_MODEL)
    effort = Field(str, "Default effort level.", default="medium")

    num_ctx = Field(int, "Context window sent with every request.", default=DEFAULT_CONTEXT)
    keep_alive = Field(str, "How long Ollama keeps the model resident. A reload of "
                            "a 30B costs many seconds and makes the agent feel broken.",
                       default="30m")
    request_timeout = Field(float, "Seconds to wait for a response. Local generation "
                                   "on CPU is genuinely slow; do not be stingy.",
                            default=600.0)
    auto_approve = Field(list, "Tool names that never prompt for permission.",
                         item_type=str, default_factory=list)
    allow_shell = Field(bool, "Whether the shell tool is available.", default=True)
    theme = Field(str, "Colour palette: purple, midnight, ember or plain.",
                  default="purple")
    clear_on_start = Field(bool, "Clear the terminal when wynxo opens.", default=True)
    log = Field(bool, "Write a session transcript for debugging.", default=True)
    voice = Field(str, "How the agent talks: plain, warm, mentor or blunt.",
                  default="plain", choices=("plain", "warm", "mentor", "blunt", "kawaii"))
    pet = Field(bool, "Show the companion face in the status bar.", default=True)
    pet_name = Field(str, "What to call it.", default="wyn")
    animations = Field(bool, "Animate the face and the startup.", default=True)
    show_thinking = Field(bool, "Display the model's reasoning. It always "
                                "thinks; this only controls whether you see it.",
                          default=False)
    stream = Field(bool, "Stream responses as they are written.", default=True)

    # -- talker / coder ----------------------------------------------------
    talker = Field(str, "A small, fast model that does the talking while the "
                        "main model codes. Empty means one model does both.",
                   default="")
    coder = Field(str, "Model that does the actual work when a talker is set. "
                       "Empty means whatever `model` is.", default="")

    # -- speech ------------------------------------------------------------
    speak = Field(bool, "Read answers out loud. You type; she talks.",
                  default=False)
    speech_engine = Field(str, "Which synthesiser: auto, say, powershell, "
                               "espeak-ng, termux, piper, flite, spd-say.",
                          default="auto")
    speech_voice = Field(str, "Engine-specific voice name. Empty picks a "
                              "female default where the engine has one.",
                         default="")
    speech_rate = Field(int, "Speaking rate. 0 leaves the engine's default; "
                             "the scale differs per engine.", default=0)
    speech_model = Field(str, "Path to a piper .onnx voice model.", default="")

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
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)  # may hold an api key; no-op on Windows
        except OSError:
            pass
        return path


def normalise_url(raw: str) -> str:
    """Accept the many shapes a person types a server address in.

    ``192.168.1.50``, ``192.168.1.50:11434``, ``http://192.168.1.50`` and a
    trailing ``/v1`` or ``/api`` all land on the same base URL.
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
        return Config.validate(data)
    except Exception:
        # A corrupt config file should never be fatal -- fall back to defaults
        # rather than leaving the user with an agent that will not start.
        return Config()


def is_configured() -> bool:
    return config_path().exists()
