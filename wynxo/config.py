"""Configuration: where Ollama lives, which model, which effort level.

Config is resolved from, in increasing order of priority:
  1. built-in defaults
  2. the user config file (platform-appropriate location)
  3. a project-local ``.wynxo.json``
  4. environment variables (``WYNXO_*``, plus ``OLLAMA_HOST``)
  5. command-line flags
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from . import platforms
from typing import Any

from .schema import Field, Schema, ValidationError

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# Ollama's default context is small enough (2048 on many builds) that an agent
# silently loses its history with no error at all. This is the single most
# common reason a local agent "goes stupid" halfway through a task.
MIN_USABLE_CONTEXT = 16_384
DEFAULT_CONTEXT = 32_768

# Not a recommendation -- MIN_USABLE_CONTEXT above is that, and /doctor says
# so. These are the bounds outside which the number is not a context window
# at all. A config holding num_ctx: -5 used to load and be sent to Ollama.
MIN_CONTEXT = 512
MAX_CONTEXT = 8_388_608


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
    kind = Field(str, "What protocol this server speaks. auto means native Ollama "
                      "(its richer /api); set 'openai' for any OpenAI-compatible "
                      "/v1 server -- a real OpenAI account, a self-hosted "
                      "gateway, or Ollama's own OpenAI shim.",
                 default="auto", choices=("auto", "ollama", "openai"))


class Config(Schema):
    endpoints = Field(list, "Known Ollama servers.", item_type=Endpoint,
                      default_factory=lambda: [Endpoint()])
    active_endpoint = Field(str, "Which endpoint to use.", default="local")
    model = Field(str, "Model tag.", default=DEFAULT_MODEL)
    effort = Field(str, "Default effort level.", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max", "ultra"))

    num_ctx = Field(int, "Context window sent with every request.",
                    default=DEFAULT_CONTEXT, ge=MIN_CONTEXT, le=MAX_CONTEXT)
    keep_alive = Field(str, "How long Ollama keeps the model resident. A reload of "
                            "a 30B costs many seconds and makes the agent feel broken.",
                       default="30m")
    request_timeout = Field(float, "Seconds to wait for a response. Local generation "
                                   "on CPU is genuinely slow; do not be stingy.",
                            default=600.0, ge=1.0, le=86400.0)
    """Bounded like every other number here, which this one was not: 0, a
    negative, and NaN all loaded and went straight to httpx, where a zero
    timeout fails every request the instant it is made and the user is told
    to raise a `request_timeout` they had just set. A day is past any real
    wait and still finite, and the bounds reject NaN for free -- every
    comparison against it is false."""
    auto_approve = Field(list, "Tool names that never prompt for permission.",
                         item_type=str, default_factory=list)
    allow_shell = Field(bool, "Whether the shell tool is available.", default=True)
    verify_with_tests = Field(bool, "After a turn that changed files, run the "
                                    "project's own tests and give the agent "
                                    "any failures to fix. The one check in the "
                                    "loop that does not come from the model.",
                              default=True)
    protect_secrets = Field(bool, "Keep credentials out of the model's context "
                                  "and out of the session log. .env files and "
                                  "private keys are refused; keys found inside "
                                  "ordinary files are masked.", default=True)
    theme = Field(str, "Colour palette: purple, sakura, kawaii, midnight, ember, catboy, plain or minimal (reduced motion).",
                   default="purple",
                   choices=("purple", "sakura", "kawaii", "midnight", "ember", "catboy", "plain", "minimal"))
    clear_on_start = Field(bool, "Clear the terminal when wynxo opens.", default=True)
    logo = Field(str, "Which start-up logo to show, or 'off' for none.",
                 default="wordmark")
    """`wyn` is a photograph converted to ASCII: sixty-nine rows of @ and #
    that fill more than half a 30-row terminal, scroll with the conversation
    rather than away from it, and read as corrupted output rather than as a
    logo. The wordmark says the same thing in eleven rows and is legible.
    `wyn` is still there for anyone who wants it."""
    log = Field(bool, "Write a session transcript for debugging.", default=True)
    voice = Field(str, "How the agent talks: plain, warm, mentor, blunt, "
                       "kawaii or mommy.",
                  default="mommy",
                  choices=("plain", "warm", "mentor", "blunt", "kawaii", "mommy"))
    pet = Field(bool, "Show the companion face in the status bar.", default=True)
    pet_name = Field(str, "What to call it.", default="wyn")
    animations = Field(bool, "Animate the face and the startup.", default=True)
    show_thinking = Field(bool, "Display the model's reasoning. It always "
                                "thinks; this only controls whether you see it.",
                          default=False)
    stream = Field(bool, "Stream responses as they are written.", default=True)
    max_tool_iterations = Field(int, "Maximum model/tool iterations per request.", default=40, ge=1, le=1000)
    max_tool_result_chars = Field(int, "Maximum tool-result characters retained in model context.", default=12000, ge=1000, le=200000)
    max_command_output_chars = Field(int, "Maximum command output characters retained.", default=30000, ge=1000, le=1000000)
    max_action_repeats = Field(int, "Stop the tool loop after the same action repeats this many times without any progress event between repeats.", default=3, ge=1, le=50)

    # -- talker / coder ----------------------------------------------------
    talker = Field(str, "A small, fast model that does the talking while the "
                        "main model codes. Empty means one model does both.",
                   default="")

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
                             "the scale differs per engine.", default=0,
                        ge=0, le=1000)
    """The other number that carried no bounds. It is passed to a
    synthesiser's own command line, and the scales differ per engine, so
    the range is wide -- wide enough to hold every engine's, narrow enough
    that a value that is plainly not a rate is refused here rather than by
    the synthesiser."""
    speech_model = Field(str, "Path to a piper .onnx voice model.", default="")
    stt_enabled = Field(bool, "Enable microphone speech recognition (Ctrl-R).", default=True)
    stt_backend = Field(str, "Speech recognition backend: auto, offline (faster-whisper) "
                              "or online (SpeechRecognition).", default="auto",
                        choices=("auto", "offline", "online"))
    stt_device = Field((str, int), "Microphone device name or index, or empty for the default.", default="")
    stt_language = Field(str, "Speech recognition language, for example en-US.", default="")
    stt_silence_timeout = Field(float, "Seconds of silence before recording stops.", default=1.25, ge=0.2, le=10.0)
    stt_max_duration = Field(float, "Maximum speech recording duration in seconds.", default=30.0, ge=1.0, le=600.0)
    stt_transcription_timeout = Field(float, "Maximum transcription duration in seconds.", default=60.0, ge=1.0, le=600.0)

    def endpoint(self) -> Endpoint:
        for ep in self.endpoints:
            if ep.name == self.active_endpoint:
                return ep
        if self.endpoints:
            return self.endpoints[0]
        return Endpoint()

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        payload = self.to_dict()
        atomic_write(path, json.dumps(payload, indent=2) + "\n")
        try:
            path.chmod(0o600)  # may hold an api key; no-op on Windows
        except OSError:
            pass
        return path


def protocol_of(raw: str) -> str:
    """The protocol a typed address is asking for, or "" for the default.

    ``normalise_url`` strips a trailing ``/v1`` so every shape of the same
    address lands on one base URL -- which meant the one part of what
    somebody typed that said *which API they meant* was thrown away. The
    only way to reach an OpenAI-compatible server was to hand-edit the
    config file, so pointing WYNXO_ENDPOINT at llama.cpp's server, LM
    Studio, vLLM or a real OpenAI account silently spoke Ollama's own /api
    at it and reported an empty answer -- which reads as a broken model
    rather than as the wrong protocol.

    Only ``/v1`` is read this way. ``/api`` is Ollama's own prefix and means
    the default, and no suffix at all means the default too, so nobody's
    existing address changes meaning.
    """
    return "openai" if raw.strip().rstrip("/").endswith("/v1") else ""


def normalise_url(raw: str) -> str:
    """Accept the many shapes a person types a server address in.

    ``192.168.1.50``, ``192.168.1.50:11434``, ``http://192.168.1.50`` and a
    trailing ``/v1`` or ``/api`` all land on the same base URL.
    """
    v = raw.strip().rstrip("/")
    if not v:
        return DEFAULT_ENDPOINT
    if not v.startswith(("http://", "https://")):
        # Brackets are required when an IPv6 literal appears in a URL. People
        # normally type ``::1`` rather than the URL spelling ``[::1]``.
        if v.count(":") >= 2 and not v.startswith("["):
            v = f"[{v}]"
        v = "http://" + v
    for suffix in ("/v1", "/api"):
        if v.endswith(suffix):
            v = v[: -len(suffix)]

    # ``0.0.0.0`` and ``::`` mean "listen on every interface" to a server;
    # they are not destinations a client can connect to. OLLAMA_HOST is used
    # for both jobs, so accepting its bind-address form keeps a local Ollama
    # installation usable without requiring users to maintain a second
    # environment variable for clients.
    scheme, rest = v.split("://", 1)
    authority, slash, tail = rest.partition("/")
    if authority == "0.0.0.0" or authority.startswith("0.0.0.0:"):
        authority = "127.0.0.1" + authority[len("0.0.0.0"):]
    elif authority == "[::]" or authority.startswith("[::]:"):
        authority = "[::1]" + authority[len("[::]"):]
    v = f"{scheme}://{authority}" + (f"/{tail}" if slash else "")
    # Bare http host with no port: assume Ollama's default rather than :80.
    # https is left alone -- that shape means a reverse proxy on 443, not a
    # directly exposed Ollama.
    if v.startswith("http://"):
        host = v[len("http://"):].split("/", 1)[0]
        if host.startswith("["):
            closing = host.find("]")
            has_port = closing >= 0 and host[closing + 1:].startswith(":")
        else:
            has_port = ":" in host
        if not has_port:
            v = v.replace(host, f"{host}:11434", 1)
    return v.rstrip("/")


LOAD_PROBLEMS: list[str] = []
"""Config files that exist but could not be read, for the caller to report.

Falling back to defaults silently is how a settings file with one bad
character costs somebody their endpoint list without ever saying so.
"""


def atomic_write(path: Path, text: str) -> None:
    """Write a file that is either the old contents or the new, never half.

    write_text truncates first and writes second, so anything that stops the
    process in between -- Ctrl-C, a full disk, a container going away --
    leaves a half-written file. For a settings file that means the endpoint
    list, the model, the theme and the rest silently back to defaults on the
    next start, which is exactly what happened when this was tested.

    The temporary file is in the same directory so the replace is a rename
    within one filesystem, which is atomic on both platforms this targets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _read_json(path: Path) -> dict[str, Any]:
    """One config layer, or nothing.

    Anything that is not a JSON object is nothing: `dict.update` accepts a
    list of pairs, so a file containing `[[1, 2]]` would otherwise merge a
    key of `1` into the config, and one containing `5` or `"text"` would
    raise out of load() before the fallback that exists to stop exactly
    that.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        LOAD_PROBLEMS.append(f"{path} could not be read ({exc.strerror or exc})")
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        LOAD_PROBLEMS.append(
            f"{path} is not valid JSON ({exc}); this run uses the defaults")
        return {}
    if not isinstance(data, dict):
        LOAD_PROBLEMS.append(f"{path} does not hold settings; using the defaults")
        return {}
    return data


def load(project_dir: Path | None = None) -> Config:
    """Assemble config from every layer."""
    LOAD_PROBLEMS.clear()
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
        endpoint = {"name": "env", "url": normalise_url(env_url)}
        if kind := protocol_of(env_url):
            endpoint["kind"] = kind
        data["endpoints"] = [endpoint]
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

    return _validate_forgivingly(data)


def _validate_forgivingly(data: dict[str, Any]) -> Config:
    """Build a Config, dropping only the settings that are actually wrong.

    A single bad value used to cost the whole file. num_ctx: -5, or an
    effort level named in a version that no longer has it, and every other
    setting went with it: model, endpoints, theme, the lot -- silently, and
    the next save wrote the defaults over what had been there.

    So drop the offending keys and keep the rest, saying which went and
    what it fell back to. Bounded by the number of fields: each pass either
    removes at least one key or stops.
    """
    data = dict(data)
    for _ in range(len(Config._fields) + 1):
        try:
            return Config.validate(data)
        except ValidationError as exc:
            # Extract top-level keys from error locations, including nested ones.
            # "endpoints[0].url" should mark "endpoints" for deletion.
            bad = set()
            for loc, _msg in exc.error_list:
                if loc in data:
                    bad.add(loc)
                else:
                    # Extract top-level key from nested paths like "endpoints[0].url"
                    top_level = loc.split("[")[0].split(".")[0]
                    if top_level in data:
                        bad.add(top_level)
            
            if not bad:
                break
            for loc in bad:
                message = next(msg for l, msg in exc.error_list 
                              if l == loc or l.startswith(loc + "[") or l.startswith(loc + "."))
                LOAD_PROBLEMS.append(
                    f"{loc}={data[loc]!r} in your settings is not usable "
                    f"({message}); using the default instead")
                del data[loc]
        except Exception:
            break
    # Nothing salvageable, or a shape that is not settings at all.
    return Config()


def is_configured() -> bool:
    return config_path().exists()
