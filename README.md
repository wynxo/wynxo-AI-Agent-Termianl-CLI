# wynxo

A terminal AI coding agent for local models, served by Ollama. Claude Code's
shape — a REPL that reads your files, edits them, runs your tests and asks
before it writes — running entirely on hardware you own.

Runs on **Linux, macOS, Windows and Termux**. Pure Python, no compiled
dependencies, no toolchain required.

```
╭──────────────────────────────────────────────────╮
│  wynxo  a local coding agent                     │
│                                                  │
│  model    qwen3-coder:30b                        │
│  server   http://homelab:11434                   │
│  effort   high                                   │
│  project  ~/code/myproject                       │
╰──────────────────────────────────────────────────╯

high > add a retry to the upload path

  → planning
  ● grep  upload  in *.py
    ✓ 3 matches
  ● read_file  src/transfer.py
    ✓ read src/transfer.py (140 lines)

  edit  src/transfer.py
╭──────────────────────────────────────────────╮
│ @@ -88,6 +88,12 @@                           │
│ -    return client.put(url, data)            │
│ +    for attempt in range(3):                │
│ +        try:                                │
│ +            return client.put(url, data)    │
│ +        except TransientError:              │
│ +            time.sleep(2 ** attempt)        │
╰──────────────────────────────────────────────╯
  [y] yes  [a] always  [n] no  [q] stop
```

---

## Contents

1. [Install](#1-install)
2. [Set up Ollama](#2-set-up-ollama)
3. [First run](#3-first-run)
4. [Check it works](#4-check-it-works)
5. [Using it](#5-using-it)
6. [Effort levels](#6-effort-levels)
7. [Tools and permissions](#7-tools-and-permissions)
8. [Commands](#8-commands)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)
11. [How it works](#11-how-it-works)

---

## 1. Install

Python 3.10 or newer. Nothing here compiles, so there is no build toolchain
to install on any platform.

### Linux and macOS

```bash
git clone https://github.com/wynxo/wynxo-AI-Agent-Termianl-CLI
cd wynxo-AI-Agent-Termianl-CLI
pip install -e .
```

A virtualenv, if you prefer to keep things separate:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows

PowerShell:

```powershell
git clone https://github.com/wynxo/wynxo-AI-Agent-Termianl-CLI
cd wynxo-AI-Agent-Termianl-CLI
py -m pip install -e .
```

Use Windows Terminal rather than the old console host — colours and box
drawing will look wrong otherwise. wynxo runs commands through PowerShell,
so tell it PowerShell syntax when you ask it to run something.

### Termux (Android)

```bash
pkg update && pkg install python git
git clone https://github.com/wynxo/wynxo-AI-Agent-Termianl-CLI
cd wynxo-AI-Agent-Termianl-CLI
pip install -e .
```

That is the whole thing — **no `pkg install rust`, no build step.** Most
Python agents need a Rust toolchain on Termux because they depend on
`pydantic`, whose `pydantic-core` is a Rust extension with no Android wheel
on PyPI. Building it on a phone takes a long time and often runs out of
memory. wynxo has its own small schema layer instead, so every dependency is
pure Python and installs in seconds.

wynxo detects Termux and adjusts: config under the app-private home, `$TMPDIR`
instead of `/tmp` (Termux has no `/tmp`), your shell from `$PREFIX/bin`, and a
stacked layout on narrow screens instead of tables that would wrap into
confetti.

> Running a 30B model *on the phone itself* is not realistic. The normal
> setup is Ollama on a desktop or homelab box and wynxo in Termux talking to
> it over Wi-Fi. See [step 2](#2-set-up-ollama).

### Check the install

```bash
wynxo --version
```

If `wynxo` is not found, your Python scripts directory is not on `PATH`. Use
`python -m wynxo` instead, which always works.

---

## 2. Set up Ollama

wynxo does not run models; Ollama does. You need Ollama running somewhere
wynxo can reach.

### Install Ollama

- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **macOS**: download from [ollama.com](https://ollama.com), or `brew install ollama`
- **Windows**: installer from [ollama.com](https://ollama.com)
- **Termux**: don't. Point wynxo at another machine instead.

### Pull a model

```bash
ollama pull qwen3-coder:30b
```

| model | size | why |
|---|---|---|
| `qwen3-coder:30b` | ~18GB | 30B MoE, ~3B active. Tool-tuned, fast. **Start here.** |
| `qwen3:32b` | ~20GB | Dense 32B. Stronger reasoning, noticeably slower. |
| `qwen3:30b-a3b` | ~18GB | General-purpose MoE. Good for chat as well as code. |
| `devstral:24b` | ~14GB | Built for agent loops. Excellent tool discipline. |
| `gpt-oss:20b` | ~13GB | Has a true native reasoning dial. |
| `qwen3:14b` | ~9GB | Fits comfortably in 12GB VRAM. |
| `qwen3:8b` | ~5GB | Runs on almost anything, CPU included. |

The one property that matters is **tool calling**. A model that will not
reliably emit a tool call cannot drive an agent loop, however good its prose
is. Step 4 checks this directly.

### If Ollama is on another machine

This catches everyone exactly once: **Ollama only listens on loopback by
default.** From another device it is simply unreachable until you change that.

On the machine running Ollama:

```bash
# Linux / macOS
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# persist it under systemd
sudo systemctl edit ollama
  [Service]
  Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

```powershell
# Windows, then restart Ollama
[Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0:11434','User')
```

Then find that machine's LAN address (`ip addr`, or `ipconfig` on Windows) —
something like `192.168.1.50`. That is what you give wynxo.

From Termux, the phone must be on the **same Wi-Fi**, not mobile data.

### Worth setting on the server

```bash
OLLAMA_FLASH_ATTENTION=1      # faster attention
OLLAMA_KV_CACHE_TYPE=q8_0     # ~half the KV memory, negligible quality cost
OLLAMA_KEEP_ALIVE=-1          # never unload; a 30B reload costs many seconds
```

---

## 3. First run

```bash
wynxo
```

Four questions, all changeable later. Only the first really matters.

**Where does Ollama serve?** wynxo probes `localhost`, `ollama`, `homelab`,
`nas`, `host.docker.internal` and the other usual names in parallel, and shows
what answered:

```
Where does Ollama serve?
Your own machine, or a box on your network. Either is fine.

  Found:
    1  http://localhost:11434    v0.12.0 · this machine
    2  http://homelab:11434      v0.12.0 · network
    m  enter a different address

  choose [1-2 or m]:
```

Nothing found? Type an address. All of these work — wynxo normalises them:

```
localhost          192.168.1.50        homelab:11434
10.0.0.4:8080      https://ollama.mydomain.com
```

**Which model?** It lists what that server actually has, so there is nothing
to remember or spell correctly. It can pull one for you if the list is empty.

**Default effort level?** `medium` is the right answer to start.
See [section 6](#6-effort-levels).

**Context window?** Press enter for 32768. This matters more than it sounds —
see [the context trap](#the-context-trap).

### Skipping the wizard

Already know your setup? Pass it and the wizard never appears:

```bash
wynxo --endpoint 192.168.1.50 --model qwen3-coder:30b
```

Or set it in the environment:

```bash
export WYNXO_ENDPOINT=192.168.1.50
export WYNXO_MODEL=qwen3-coder:30b
wynxo
```

---

## 4. Check it works

```bash
wynxo --doctor
```

Every assumption wynxo makes, checked one at a time, with a concrete fix for
each failure:

```
  ✓ server reachable      ollama 0.12.0 at http://homelab:11434
  ✓ model installed       qwen3-coder:30b  18.6GB  30.5B Q4_K_M
  ✓ model capabilities    completion, tools, thinking
  ✓ context window        32768 tokens
      model's native window: 262144
  ✓ generation            streamed 14 chunk(s): 'OK. one, two, three...'
      312 tokens in 8.2s
      38.0 tok/s (includes model load)
  ✓ thinking mode         think levels accepted, reasoning returned
  ✓ tool calling          the model calls tools through Ollama's native tools field

  Everything checks out. You are good to go.
```

The last check is the important one. It sends a real tool definition and sees
whether the model actually calls it. A model can pass everything else and
still be useless in an agent loop — this is how you find out in ten seconds
instead of after an hour of confusing behaviour.

Run it again any time from inside wynxo with `/doctor`.

### No Ollama yet? Try wynxo anyway

A stand-in server ships with the repo. It speaks Ollama's real wire protocol
but has no model behind it — so the agent loop, the tools, the permission
prompts, the diffs and the effort machinery all run for real, against real
files.

```bash
python scripts/fake_ollama.py &
wynxo --endpoint localhost:11435 --doctor
wynxo --endpoint localhost:11435
```

Good for seeing the interface, or for developing wynxo without a GPU. Point
at a real server whenever you have one; nothing else changes.

---

## 5. Using it

Start it in the project you want to work on:

```bash
cd ~/code/myproject
wynxo
```

Then just talk to it.

```
medium > what does src/auth.py do?

medium > add a test for the token refresh path

medium > the CI failure on main — find it and fix it
```

It reads files, greps, runs commands, and edits — asking before anything that
writes. Reference a file and it will go read it.

### One-shot mode

```bash
wynxo -p "what does this project do"
wynxo -p -e high "add a test for the retry path"
```

`-p` answers and exits, without prompting for permission (there is nobody
there to answer). Good for scripts and CI.

### Piping

```bash
git diff | wynxo -p "review this"
cat error.log | wynxo -p "what is failing here?"
```

### Start with a prompt, then keep going

```bash
wynxo "fix the failing tests"
```

Runs that immediately, then drops you into the REPL.

### While it is working

- **Ctrl-C** interrupts the current turn. The conversation survives; ask
  something else.
- **Alt-Enter** inserts a newline instead of submitting.
- **Up arrow** walks your history.

### Teaching it about your project

Drop a `WYNXO.md` at the repo root — build commands, conventions, things to
avoid. It is loaded into every system prompt.

```markdown
# Project notes

Run tests with `pytest -q`. Lint with `ruff check .`.
Do not edit anything under `generated/` — it comes from `make codegen`.
Prefer `structlog` over the stdlib logger.
```

`AGENTS.md` and `CLAUDE.md` are picked up too, so an existing one just works.
Or have wynxo write one:

```
medium > /init
```

---

## 6. Effort levels

Effort is a **scheduler policy**, not a single knob passed to the model.

Most local models expose no native reasoning budget at all, so a setting that
only forwarded a number would collapse into two or three real behaviours.
Instead, effort controls how many chances the model gets to be right:

| level    | plan                 | tool iters | verify      | plan consensus | `think`    |
|----------|----------------------|-----------:|-------------|---------------:|------------|
| `low`    | none                 |          6 | none        |              1 | off        |
| `medium` | inline               |         16 | none        |              1 | off        |
| `high`   | separate pass        |         40 | 1 round     |              1 | `"medium"` |
| `xhigh`  | separate pass        |         80 | 2 rounds    |              1 | `"high"`   |
| `max`    | plan + self-critique |        150 | until clean |              2 | `"max"`    |
| `ultra`  | plan + self-critique |        400 | until clean |              3 | `"max"`    |

At `low` the agent reads what it needs, makes the change and stops.

At `max` it plans, attacks its own plan, executes, then reviews its own diff
as a hostile reviewer would — repeating until a review pass turns up nothing.
At the top two levels it drafts the plan several times independently and
reconciles the drafts before acting.

**The same model at `max` genuinely outperforms itself at `low`.** Not because
it thinks harder per token, but because it gets more chances to catch its own
mistakes.

```bash
wynxo -e low "what does auth.py do"           # a question; just answer it
wynxo -e high "add retry logic to the client" # real work
wynxo -e max "refactor the session layer"     # do not get this wrong
```

```
/effort           # show the ladder, and where you are
/effort xhigh     # change gear mid-conversation
```

Rough guide: `low` for questions, `medium` for ordinary edits, `high` when it
touches more than one file, `max` when being wrong is expensive. Higher levels
cost real time on local hardware — `ultra` can run for a very long while.

### The native dial

Ollama's `think` field accepts `"low"`, `"medium"`, `"high"` and `"max"` as
well as a plain boolean, so on a thinking model the effort level drives the
model's own reasoning budget too — the right-hand column above.

The mapping is graduated rather than name-matched: by the time you are at
wynxo's `high` you already get a planning pass and a verification round, which
is more added rigour than the raw dial contributes on its own.

Older Ollama builds only understand the boolean. A rejected string level is
detected and retried once as `think: true`, then remembered — one wasted round
trip per session, not one per request. `--doctor` tells you which form your
server accepts.

---

## 7. Tools and permissions

| tool | writes? | what it does |
|---|---|---|
| `read_file` | | Read with line numbers, offset/limit for big files |
| `write_file` | ✓ | Create or replace a file |
| `edit_file` | ✓ | Exact-match replacement, with a diff |
| `list_dir` | | Tree view, skipping vcs and build noise |
| `glob` | | Find files by name pattern |
| `grep` | | Regex search across the project |
| `shell` | ✓ | Run a command — PowerShell on Windows, your login shell elsewhere |
| `todo_write` | | The visible plan, which also survives compaction |

**Reads are free. Writes ask**, and show you the diff before you answer:

```
  edit  src/auth.py
╭────────────────────────────────╮
│ @@ -12,3 +12,3 @@              │
│ -    if user.token:            │
│ +    if user.token is not None:│
╰────────────────────────────────╯
  [y] yes  [a] always  [n] no  [q] stop
```

`a` remembers — per tool, and for `shell` per **exact command**, so approving
`npm test` forever does not also approve `rm -rf build`. Commands that reach
outside the machine (`git push`, `curl`, `ssh`, `sudo`) always ask, even then.

Tools cannot touch anything outside the project directory, and a short list of
unrecoverable commands is refused outright rather than merely prompted for — a
yes/no prompt is exactly the thing people click through on autopilot.

`--yolo` (or `/yolo`) turns the prompts off. Reasonable in a container or a
scratch repo; think twice anywhere else.

### Tool calling, and the Hermes fallback

Well-behaved models expose tool calling through Ollama's native `tools` field.
Many do not — their template never wires it up, so the calls arrive as plain
text and a naive agent sees nothing at all.

wynxo checks the model's capabilities at startup and, when native tools are
missing, describes them in the prompt using the **Hermes**
`<tool_call>{...}</tool_call>` format that Qwen3, Hermes and most tool-tuned
open models were trained on — then parses those blocks back out.

It also repairs what local models actually emit: trailing commas, `True`
instead of `true`, single quotes, raw newlines inside JSON strings, code fences
around the object, and calls truncated halfway by a token limit. When a call is
genuinely unsalvageable it goes back to the model with the broken text quoted
and a specific correction, up to `repair_attempts` times — itself an
effort-level setting.

`/stats` shows which mode you are in.

---

## 8. Commands

```
/help                    everything below
/effort [level]          low | medium | high | xhigh | max | ultra
/model [name]            switch model, or list what the server has
/endpoint ...            list | test | add <url> [name] | use <name>
/ctx [n]                 show or set the context window
/doctor                  check the server and model for problems
/tools                   what the agent can call
/thinking                show or hide the model's reasoning
/plan                    the current todo list
/clear                   fresh conversation
/compact                 summarise now, reclaim context
/stats                   tokens, speed, context use, tool mode
/yolo                    stop asking permission this session
/sessions                recent sessions
/init                    write a WYNXO.md describing this project
/quit
```

### Several servers

A laptop and a homelab box is the normal case:

```
/endpoint add 192.168.1.50 gpu   # add and name one
/endpoint list                   # what you have
/endpoint test                   # which are up right now
/endpoint use gpu                # switch, mid-conversation
```

### Command-line flags

```
wynxo [prompt]
  -p, --print          answer and exit (implies --yolo)
  -e, --effort LEVEL   low|medium|high|xhigh|max|ultra
  -m, --model NAME     model to use
      --endpoint URL   Ollama server
      --ctx N          context window
  -C, --cwd DIR        project directory
      --doctor         run the checks and exit
      --setup          re-run first-time setup
      --no-stream      wait for the full response
      --no-thinking    hide model reasoning
      --yolo           never ask permission
      --version
```

---

## 9. Configuration

| platform | location |
|---|---|
| Linux | `~/.config/wynxo/config.json` |
| macOS | `~/Library/Application Support/wynxo/config.json` |
| Windows | `%APPDATA%\wynxo\config.json` |
| Termux | `$HOME/.config/wynxo/config.json` |

```json
{
  "endpoints": [
    { "name": "local", "url": "http://localhost:11434", "api_key": null },
    { "name": "gpu", "url": "http://192.168.1.50:11434", "api_key": null }
  ],
  "active_endpoint": "gpu",
  "model": "qwen3-coder:30b",
  "effort": "medium",
  "num_ctx": 32768,
  "keep_alive": "30m",
  "request_timeout": 600.0,
  "auto_approve": ["read_file", "grep", "glob"],
  "allow_shell": true,
  "show_thinking": true,
  "stream": true
}
```

Precedence, lowest to highest:

1. built-in defaults
2. the user config file above
3. a project-local `.wynxo.json`
4. environment: `WYNXO_ENDPOINT`, `WYNXO_MODEL`, `WYNXO_EFFORT`,
   `WYNXO_NUM_CTX`, and `OLLAMA_HOST`
5. command-line flags

`api_key` is only needed when the server sits behind a reverse proxy that
requires auth; it is sent as `Authorization: Bearer …`.

---

## 10. Troubleshooting

### "Cannot reach an Ollama server"

Is `ollama serve` running there? If it is on another machine, it must be
started with `OLLAMA_HOST=0.0.0.0:11434` — see
[step 2](#if-ollama-is-on-another-machine). Check the firewall allows 11434.
From Termux, confirm the phone is on Wi-Fi and not mobile data.

```bash
/endpoint test        # inside wynxo
wynxo --doctor
```

### "The model does not have …"

```bash
ollama pull qwen3-coder:30b
```

Or `/model` with no argument to see what the server actually has.

### The agent goes stupid halfway through a task

This is almost always the context window. See below.

### The context trap

**Ollama's default context is often 2048 tokens.** An agent at that size
silently forgets the first half of its task. No error, no warning — it just
gets confused partway through and nothing tells you why.

wynxo sets `num_ctx` explicitly on every request, defaults to 32768, and warns
at startup if your setting is below what an agent needs or above what the
model was trained for.

```
/ctx           # current setting and how full it is
/ctx 65536     # raise it, if you have the VRAM
```

When the conversation approaches the budget, wynxo compacts: the older half is
summarised into working notes, the recent tail is kept verbatim, and the
outstanding todo list is carried across so a long run does not forget step
three of five.

### It never calls tools, it just talks about calling them

Your model cannot do tool calling well. `wynxo --doctor` will say so
explicitly. Switch to `qwen3-coder`, `devstral`, or `gpt-oss`.

### It is very slow

Check `/stats` for tokens per second. If the first request is slow but later
ones are fast, that is model loading — set `OLLAMA_KEEP_ALIVE=-1` on the
server. If everything is slow, the model is too big for your hardware; drop to
`qwen3:14b` or `qwen3:8b`, or use a lower effort level.

### Out of memory on the server

Lower `num_ctx` (`/ctx 16384`), use a smaller quantisation, or set
`OLLAMA_KV_CACHE_TYPE=q8_0` on the server to roughly halve KV cache memory.

### Timeouts on CPU-only machines

Generation genuinely can take minutes. Raise `request_timeout` in the config.

### Colours or boxes look wrong on Windows

Use Windows Terminal rather than the legacy console host.

### `wynxo: command not found`

Your Python scripts directory is not on `PATH`. Use `python -m wynxo`.

---

## 11. How it works

```
your message
    │
    ├─ plan          (high and up: a separate read-only pass)
    ├─ critique      (max and up: the model attacks its own plan)
    │
    ├─ act ──────────┐
    │   model call   │  ← tools sent natively, or described in Hermes format
    │   tool calls   │  ← parsed, repaired if malformed, permission-checked
    │   tool results │  ← truncated to the effort level's budget
    │   └────────────┘  repeat, up to the iteration ceiling
    │
    ├─ verify        (high and up: review the diff, fix, repeat until clean)
    │
    └─ answer
```

Everything in that diagram is switched on or off by the effort policy. It is
one code path; the level decides which stages execute and how many times.

### Layout

```
wynxo/
  agent.py       the loop above
  effort.py      the six policies, as data
  provider.py    Ollama client: streaming, capabilities, error translation
  parsing.py     thinking tags, Hermes tool calls, JSON repair
  schema.py      the pure-Python schema layer that replaces pydantic
  platforms.py   Linux / macOS / Windows / Termux differences
  session.py     history, token accounting, compaction
  permissions.py what needs asking about, and what is remembered
  prompts.py     system prompts and stage prompts
  doctor.py      the pre-flight checks
  wizard.py      first-run setup
  ui.py          rendering, including the narrow-screen layout
  cli.py         REPL and slash commands
  tools/         read, write, edit, list, glob, grep, shell, todo
scripts/
  fake_ollama.py a stand-in server for development without a GPU
```

Each tool is atomic: one schema, one typed result, no shared state, no
knowledge of the agent loop. That is what lets the registry render them for
three transports — native Ollama tools, Hermes prompted calls, and `/tools` —
from a single definition.

### Development

```bash
pip install -e ".[dev]"
pytest
```

170 tests. The agent tests run the real loop, real tools and real files
against a scripted fake Ollama — malformed tool calls, denied permissions,
path escapes, iteration ceilings, older servers that reject string think
levels, and a simulated Termux.

The wire format is checked against Ollama's own `api/types.go` and
`docs/api.md` rather than against assumptions. That is how the `tool_name`
field and the string `think` levels were found; both were wrong here first.

---

## License

MIT
