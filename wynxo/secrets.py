"""Keeping credentials out of the model's context, and out of the logs.

wynxo reads the user's real files and sends them to a model. That model is
often on another machine -- the whole point of `--endpoint 192.168.1.50:11434`
is the box with the GPU -- and sometimes behind a proxy with an API key. So
"it is all local anyway" is not true, and a `.env` read into context is a
credential on the network.

The logs are the quieter half of the same problem. Every tool result is
written to a jsonl transcript and kept for twenty sessions, so reading a
`.env` once leaves the keys sitting in plaintext on disk long after the
session is forgotten -- which also breaks the promise that wynxo can be
removed without leaving marks.

Two different problems, so two different answers:

* A file that exists to hold credentials -- `.env`, `id_rsa`, `*.pem` -- is
  refused outright. The model almost never needs it, and half a secret is
  still a secret.
* A credential sitting inside a file that is otherwise ordinary code is
  redacted in place. Refusing the whole file would make the agent useless on
  a config module that happens to have one hardcoded key in it.

The detection is deliberately conservative about entropy. "Long random
looking string" also describes a hash, a minified bundle, a lockfile
checksum and a base64 icon, and an agent that redacts those is an agent that
cannot read its own project. So a value is only redacted when something
*names* it a secret, or when it carries a known credential prefix.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

IGNORE_FILE = ".wynxoignore"

# Files whose entire purpose is to hold credentials.
SECRET_NAMES = {
    ".env", ".envrc", ".netrc", "_netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.json", ".htpasswd", ".pgpass",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "identity",
    "secrets.json", "secrets.yml", "secrets.yaml", ".secrets",
    "service-account.json", "serviceaccount.json", ".dockercfg",
}

SECRET_PATTERNS = (
    ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "*.ppk", "*_rsa", "*_ed25519", "*.kdbx", "*credentials*.json",
    "*.asc", "*.gpg", "id_*.pub",
)

# A directory whose contents are keys regardless of what they are called.
SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube"}

# Words that, as a component of a setting's name, make the value beside it a
# credential. Matched against the name's parts rather than as a substring:
# "tokenizer", "monkey" and "keyboard" all contain a secret word and none of
# them is one.
_SECRET_WORDS = {
    "key", "keys", "secret", "secrets", "token", "tokens", "password",
    "passwd", "pwd", "passphrase", "credential", "credentials", "auth",
    "authorization", "apikey", "accesskey", "privatekey", "secretkey",
    "clientsecret", "sessionkey", "dsn",
}

# A name that is only ever a lookup, never a value.
_SAFE_WORDS = {"public", "publickey", "pub", "fingerprint", "id", "name",
               "path", "file", "url", "type", "algorithm", "expiry", "length"}

_ASSIGNMENT = re.compile(
    r"""(?x)
    ([A-Za-z_][A-Za-z0-9_.\-]{0,60})      # the name
    [ \t]*[:=][ \t]*                      # ... and its value, same line
    (["']?)([^\s"',;\\`]{6,})\2
    """
)
# Spaces and tabs rather than \s: a name and its value are on one line. With
# \s the separator could span a newline, so `for key in keys:` followed by
# `marker = ...` read as the setting "keys" holding the value "marker", and
# the next line of ordinary code was masked.
#
# Backslashes and backticks end a value for the same reason: they are never
# part of one, and including them made "export TOKEN=keepme\n" inside a test
# fixture and `token=self.token` inside a doc comment look like credentials.


def _names_a_secret(name: str) -> bool:
    """Whether this setting's name says its value is a credential."""
    parts = [p for p in re.split(r"[_\-.]+|(?<=[a-z0-9])(?=[A-Z])", name) if p]
    lowered = [p.lower() for p in parts]
    if any(p in _SAFE_WORDS for p in lowered):
        # PUBLIC_KEY and KEY_PATH are not the secret itself.
        return False
    return any(p in _SECRET_WORDS for p in lowered)


# Prefixes that identify a credential on their own, wherever they appear.
_KNOWN_TOKEN = re.compile(
    r"""(?x)
    \b(
        sk-[A-Za-z0-9_-]{16,}            # OpenAI-style
      | sk_(?:live|test)_[A-Za-z0-9]{16,}  # Stripe
      | rk_(?:live|test)_[A-Za-z0-9]{16,}
      | gh[pousr]_[A-Za-z0-9]{16,}       # GitHub
      | github_pat_[A-Za-z0-9_]{20,}
      | xox[baprs]-[A-Za-z0-9-]{10,}     # Slack
      | AKIA[0-9A-Z]{16}                 # AWS access key id
      | ASIA[0-9A-Z]{16}
      | AIza[0-9A-Za-z_-]{30,}           # Google
      | ya29\.[0-9A-Za-z_-]{20,}
      | glpat-[0-9A-Za-z_-]{16,}         # GitLab
      | npm_[A-Za-z0-9]{30,}
      | dop_v1_[a-f0-9]{60,}             # DigitalOcean
      | hf_[A-Za-z0-9]{30,}              # Hugging Face
      | eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+  # JWT
    )
    """
)

# A password embedded in a connection string. Its own rule because the value
# sits inside a URL rather than beside a name, so neither the assignment rule
# nor the token prefixes reach it -- and DSNs are one of the most common ways
# a real credential ends up committed.
# The scheme is bounded rather than left open. Unbounded, [\w+.-]* eats to
# the end of the text at every single position, fails to find "://", and
# gives the characters back one at a time -- quadratic. On a file with one
# long line (a minified bundle, a lockfile, a base64 blob) that is not a
# subtlety: masking 80k characters took 28 seconds, and every read_file and
# every tool result goes through here. Bounded, the same text takes 0.02s.
# 31 is far past any real scheme; the longest in the wild are custom
# reverse-DNS app schemes, and those are nowhere near it.
_URL_CRED = re.compile(r"([a-zA-Z][\w+.-]{0,31}://[^\s:/@]+):([^\s:/@]+)@")

# The body has to actually span lines. A real key does; the one-line string
# literal that *builds* the replacement ("BEGIN...\\n{MASK}\\n...END") does
# not, and without this wynxo could not read this very file.
_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[^\n]*\n.*?"
    r"-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

MASK = "[redacted by wynxo]"

# Values that match a rule but are obviously not real, so redacting them only
# makes the code harder to read.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:(?:your|my|the|some|test|fake|dummy|example|sample|insert)"
    r"[_\-]\S*|placeholder\S*|changeme\S*|xxx+|<.*>|\{\{.*\}\}|"
    r"\$\{.*\}|none|null|nil|true|false|empty|todo|tbd|\.\.\.|\*+)$"
)


# An unquoted value made only of letters, underscores and dots is usually a
# name being referred to rather than a credential: `tokens=self.tokens`,
# `key_bindings=bindings`, `protect_secrets = enabled`. Masking those
# corrupts the code the model is trying to read.
#
# "Only letters" alone is too generous, though -- `client_secret:
# GOCSPXabcdefghijklmnop` is also only letters. So the value has to be
# shaped like something a person would type as a name: either it carries
# identifier punctuation (a dot or an underscore), or it is a short single
# word in one of the casings identifiers actually use. A 22-character run
# of letters with `GOCSPX` welded to the front is none of those.
_REFERENCE = re.compile(r"^[A-Za-z_][A-Za-z_]*(?:\.[A-Za-z_][A-Za-z_]*)*$")

# The longest single word worth believing is a variable name. Past this a
# bare run of letters is far likelier to be a key than an identifier.
_LONGEST_NAME = 16

_SHOUTING = re.compile(r"[A-Z]{2}")


def _looks_like_a_reference(value: str) -> bool:
    value = value.strip()
    if not _REFERENCE.match(value):
        return False
    if "." in value or "_" in value:
        # self.tokens, key_bindings -- punctuation no secret carries.
        return True
    if len(value) > _LONGEST_NAME:
        return False
    # tokens, ENABLED, keyBindings: lower, upper or camel. Two capitals in a
    # row inside a mixed-case word is not a casing anyone writes by hand.
    return (value.islower() or value.isupper()
            or not _SHOUTING.search(value))


def _is_placeholder(value: str) -> bool:
    stripped = value.strip().strip("\"'")
    if not stripped or _PLACEHOLDER.match(stripped):
        return True
    # Code, not a literal. A real credential is letters, digits and a little
    # punctuation; brackets and operators mean this is an expression that
    # *fetches* the secret, and masking it hides how the program works.
    if any(ch in stripped for ch in "()[]{}<>$@ +"):
        return True
    # os.environ["API_KEY"] and process.env.API_KEY are lookups, not values.
    return stripped.startswith(("os.", "process.", "env.", "ENV[", "$"))


def is_secret_file(path: Path) -> bool:
    """Whether this file exists to hold credentials."""
    name = path.name
    lowered = name.lower()
    if lowered in SECRET_NAMES:
        return True
    if any(part in SECRET_DIRS for part in path.parts[:-1]):
        return True
    # .env.local yes; .env.example no -- samples are meant to be read, and
    # they are how a model learns which variables a project expects.
    if lowered.startswith(".env") and not _is_sample(lowered):
        return True
    for pattern in SECRET_PATTERNS:
        if fnmatch.fnmatch(lowered, pattern):
            return not _is_sample(lowered)
    return False


def _is_sample(lowered: str) -> bool:
    return any(marker in lowered for marker in
               ("example", "sample", "template", "dist", ".pub"))


def redact(text: str) -> tuple[str, int]:
    """Mask credentials inside otherwise ordinary text.

    Returns the text and how many values were masked, so the caller can tell
    the user something was withheld rather than silently altering what they
    asked to see.
    """
    if not text:
        return text, 0

    count = 0

    def mask_pem(_match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"-----BEGIN PRIVATE KEY-----\n{MASK}\n-----END PRIVATE KEY-----"

    text = _PEM.sub(mask_pem, text)

    def mask_url(match: re.Match) -> str:
        nonlocal count
        if _is_placeholder(match.group(2)):
            return match.group(0)
        count += 1
        return f"{match.group(1)}:{MASK}@"

    text = _URL_CRED.sub(mask_url, text)

    def mask_named(match: re.Match) -> str:
        nonlocal count
        name, quote, value = match.group(1), match.group(2), match.group(3)
        if not _names_a_secret(name) or _is_placeholder(value):
            return match.group(0)
        if not quote and _looks_like_a_reference(value):
            return match.group(0)
        count += 1
        # Keep the name and the shape of the line: the model still needs to
        # know the setting exists and where it is read.
        return match.group(0).replace(value, MASK, 1)

    text = _ASSIGNMENT.sub(mask_named, text)

    def mask_token(match: re.Match) -> str:
        nonlocal count
        count += 1
        return MASK

    text = _KNOWN_TOKEN.sub(mask_token, text)
    return text, count


def refusal(display_path: str) -> str:
    """What the model is told instead of the file's contents."""
    return (
        f"{display_path} holds credentials, so wynxo did not read it. "
        "Work from the variable names if you need them -- an .env.example, "
        "the code that reads os.environ, or the deployment config will have "
        "them. Ask the user to paste any value you genuinely need.\n\n"
        "If this file is not actually secret, the user can allow it with "
        "/secrets allow <path>."
    )


class Ignore:
    """The project's ``.wynxoignore``, in gitignore-ish glob form.

    Only the part of gitignore people actually use: one glob per line, `#`
    comments, and a leading `!` to put something back. Implementing the full
    specification would be a lot of code for a file most projects will not
    have.
    """

    def __init__(self, patterns: list[str] | None = None):
        self.patterns: list[tuple[str, bool]] = []
        for line in patterns or []:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            self.patterns.append((line.lstrip("!").strip("/"), negated))

    @classmethod
    def load(cls, root: Path) -> "Ignore":
        try:
            text = (root / IGNORE_FILE).read_text(encoding="utf-8",
                                                  errors="replace")
        except OSError:
            return cls([])
        return cls(text.splitlines())

    def matches(self, relative: str) -> bool:
        relative = _normalise(relative)
        hit = False
        for pattern, negated in self.patterns:
            if self._one(pattern, relative):
                hit = not negated
        return hit

    @staticmethod
    def _one(pattern: str, relative: str) -> bool:
        if fnmatch.fnmatch(relative, pattern):
            return True
        # A bare name or a directory matches anywhere in the tree, the way
        # `build/` in a gitignore does.
        if fnmatch.fnmatch(relative, f"*/{pattern}"):
            return True
        return any(fnmatch.fnmatch(part, pattern)
                   for part in relative.split("/"))

    def __bool__(self) -> bool:
        return bool(self.patterns)


def _normalise(relative: str) -> str:
    """A path in the one form the comparisons expect.

    Note the explicit "./" prefix removal. `lstrip("./")` strips *characters*
    rather than a prefix, so it turns ".env" into "env" and quietly breaks
    every dotfile -- which is most of what this module is about.
    """
    text = str(relative).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


class Shield:
    """The policy: what the model may see, and what it may not.

    Two settings rather than three. A middle mode that redacted a `.env`
    instead of refusing it sounds useful but is not: the interesting half of
    a credentials file *is* the credentials, and handing the model a page of
    masks costs context to say nothing. So the file is refused, and the
    per-path allowance exists for the case where the user knows better.
    """

    def __init__(self, root: Path, enabled: bool = True,
                 ignore: Ignore | None = None,
                 allowed: set[str] | None = None):
        # Resolved, because the paths it is compared against are: an
        # unresolved root made every allow() silently fail to match under a
        # symlinked temp dir or a /home that is really /var/home.
        try:
            self.root = Path(root).resolve()
        except (OSError, RuntimeError):
            self.root = Path(root)
        self.enabled = enabled
        self.ignore = ignore if ignore is not None else Ignore.load(self.root)
        self.allowed = set(allowed or ())

    def allow(self, relative: str) -> None:
        self.allowed.add(_normalise(relative))

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return path.name

    def blocks(self, path: Path) -> str:
        """Why this file may not be read, or "" if it may."""
        if not self.enabled:
            return ""
        relative = self._relative(path)
        if relative in self.allowed:
            return ""
        if self.ignore.matches(relative):
            return (f"{relative} is excluded by {IGNORE_FILE}, so wynxo did "
                    "not read it.")
        if is_secret_file(Path(relative)) or is_secret_file(path):
            return refusal(relative)
        return ""

    def clean(self, text: str) -> tuple[str, int]:
        """Mask any credentials inside text that is otherwise fine to show."""
        if not self.enabled:
            return text, 0
        return redact(text)

    @classmethod
    def off(cls, root: Path = Path(".")) -> "Shield":
        return cls(root, enabled=False, ignore=Ignore([]))
