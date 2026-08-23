# wynxo

A terminal AI coding agent for local models, served by Ollama. Built for
`qwen3-coder:30b` and its siblings, and for the case where the GPU is not in
the laptop you are typing on.

It has real effort levels — `low` through `ultra` — and they change what the
agent actually *does*, not just a number sent to the API.

```
╭──────────────────────────────────────────────────╮
│  wynxo  a local coding agent                     │
│                                                  │
│  model    qwen3-coder:30b                        │
│  server   http://homelab:11434                   │
│  effort   high                                   │
│  project  ~/code/myproject                       │
╰──────────────────────────────────────────────────╯
```

## Install

Works on Linux, macOS and Windows. Python 3.10+.

```bash
git clone https://github.com/wynxo/wynxo-ai-agent-termianl-cli
cd wynxo-ai-agent-termianl-cli
pip install -e .
wynxo
```

On first run it asks four questions. The only one that matters is the first.

## No Ollama yet? Try it anyway

A stand-in server ships with the repo. It speaks Ollama's real wire protocol —
streaming NDJSON, native `tool_calls`, `/api/show` capabilities — but there is
no model behind it. The agent loop, the tools, the permission prompts, the
diffs and the effort machinery all run for real, against real files.

```bash
python scripts/fake_ollama.py &
wynxo --endpoint localhost:11435 --doctor
wynxo --endpoint localhost:11435
```

Point wynxo at a real server whenever you have one; nothing else changes.

## Does it work? Ask it

```bash
wynxo --doctor
```

Every assumption wynxo makes about your server and model, checked one at a
time, with a concrete fix for each failure:

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

The last check is the one that matters: it sends a real tool definition and
sees whether the model actually calls it, natively or as Hermes text. A model
can pass every other check and still be unable to drive an agent loop.

## Where does Ollama serve?

This is the question everyone answers twice — once for the laptop, once for
the box with the real GPU — so wynxo asks it directly and remembers every
answer.

On first run it probes `localhost`, `ollama`, `homelab`, `nas`,
`host.docker.internal` and the other usual names in parallel, and shows you
what it found:

```
Where does Ollama serve?

  Found:
    1  http://localhost:11434    v0.5.4 · this machine
    2  http://homelab:11434      v0.5.4 · network
    m  enter a different address

  choose [1-2 or m]:
```

Or skip the wizard entirely:

```bash
wynxo --endpoint homelab:11434
wynxo --endpoint 192.168.1.50
wynxo --endpoint https://ollama.mydomain.com     # behind a reverse proxy
```

Addresses are normalised, so `homelab`, `homelab:11434`,
`http://homelab:11434/v1` and `http://homelab:11434/api` all work.

Keep several servers and switch between them:

```
/endpoint list                  # what you have
/endpoint add 192.168.1.50 gpu  # add and name one
/endpoint use gpu               # switch
/endpoint test                  # which are actually up right now
```

### The remote-server gotcha

**Ollama only listens on loopback by default.** A server on another machine is
unreachable until you tell it otherwise. This trips up everyone exactly once:

```bash
# Linux/macOS, on the machine running Ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# persist it under systemd
sudo systemctl edit ollama
  [Service]
  Environment="OLLAMA_HOST=0.0.0.0:11434"
```

```powershell
# Windows, then restart Ollama
[Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0:11434','User')
```

wynxo tells you this whenever a connection fails, rather than just saying
"connection refused".

## Effort levels

Effort is a **scheduler policy**, not one knob. Most local models expose no
native reasoning budget at all, so a level that only forwarded a
`reasoning_effort` field would collapse into two or three real settings.

Instead, effort controls how many chances the model gets to be right:

| level    | plan               | tool iters | verify        | plan consensus | `think` |
|----------|--------------------|-----------:|---------------|---------------:|---------|
| `low`    | none               |          6 | none          |              1 | off     |
| `medium` | inline             |         16 | none          |              1 | off     |
| `high`   | separate pass      |         40 | 1 round       |              1 | `"medium"` |
| `xhigh`  | separate pass      |         80 | 2 rounds      |              1 | `"high"` |
| `max`    | plan + self-critique |      150 | until clean   |              2 | `"max"` |
| `ultra`  | plan + self-critique |      400 | until clean   |              3 | `"max"` |

At `low` the agent reads what it needs, makes the change and stops. At `max`
it plans, attacks its own plan, executes, then reviews its own diff as a
hostile reviewer until a review pass turns up nothing — and at the top two
levels it drafts the plan several times independently and reconciles the
drafts before acting.

The same 30B model at `max` genuinely outperforms itself at `low`. Not because
it thinks harder per token, but because it gets more chances to catch its own
mistakes.

```bash
wynxo -e low "what does auth.py do"
wynxo -e max "refactor the session layer to use the new store"
```

```
/effort           # show the ladder, and where you are
/effort xhigh     # change gear mid-conversation
```

### The native dial

Ollama's `think` field takes `"low"`, `"medium"`, `"high"` or `"max"` as well
as a plain boolean, so on a thinking model the effort level drives the model's
own reasoning budget too — the right-hand column above.

The mapping is graduated rather than name-matched. By the time you are at
wynxo's `high` you are already getting a planning pass and a verification
round, which is more added rigour than the raw thinking dial contributes on
its own.

Older Ollama builds only understand the boolean. A rejected string level is
detected and retried once as `think: true`, then remembered, so there is one
wasted round trip per session rather than one per request. `--doctor` tells
you which form your server accepts.

## Models

Nothing is hardcoded — the setup wizard lists what your server actually has.
Recommended, best first:

| model | why |
|---|---|
| `qwen3-coder:30b` | 30B MoE, ~3B active. Tool-tuned, fast, best all-rounder. |
| `qwen3:32b` | Dense 32B. Stronger reasoning, noticeably slower. |
| `qwen3:30b-a3b` | General-purpose MoE. Good for chat as well as code. |
| `devstral:24b` | Built for agent loops. Excellent tool discipline. |
| `gpt-oss:20b` | The one local family with a true native effort dial. |
| `qwen3:14b` | Fits comfortably in 12GB VRAM. |
| `qwen3:8b` | Runs on almost anything, CPU included. |

```
/model              # what the server has
/model qwen3:32b    # switch, mid-conversation
```

### Tool calling, and the Hermes fallback

Well-behaved models expose tool calling through Ollama's native `tools` field.
Many do not — their template never wires it up, so the calls arrive as plain
text and a naive agent sees nothing at all.

wynxo checks the model's capabilities at startup and, when native tools are
missing, describes the tools in the prompt using the **Hermes**
`<tool_call>{...}</tool_call>` format that Qwen3, Hermes and most tool-tuned
open models were trained on — then parses those blocks back out.

It also repairs what local models actually emit: trailing commas, `True`
instead of `true`, single quotes, raw newlines inside JSON strings, code
fences around the object, and calls truncated halfway by a token limit. When
a call is genuinely unsalvageable it goes back to the model with the broken
text quoted and a specific correction, up to `repair_attempts` times — which
is itself an effort-level setting.

This one file is the difference between a local agent that works and one that
mysteriously does nothing.

## The context trap

**Ollama's default context is often 2048 tokens.** An agent at that size
silently forgets the first half of its task. No error, no warning — it just
gets stupid halfway through and nobody can tell you why.

wynxo sets `num_ctx` explicitly on every request, defaults to 32768, and warns
at startup if your setting is below what an agent realistically needs or above
what the model was trained for.

```
/ctx           # current setting and how full it is
/ctx 65536     # raise it, if you have the VRAM
```

When the conversation approaches the budget, wynxo compacts it: the older half
is summarised into working notes, the recent tail is kept verbatim, and the
outstanding todo list is carried across so a long `max` run does not forget
step three of five.

### Server-side tuning

```bash
OLLAMA_FLASH_ATTENTION=1      # faster attention
OLLAMA_KV_CACHE_TYPE=q8_0     # ~half the KV memory, negligible quality cost
OLLAMA_KEEP_ALIVE=-1          # never unload; a 30B reload costs many seconds
OLLAMA_NUM_PARALLEL=1         # unless you want concurrent requests
```

wynxo sends `keep_alive` on every request too, so the model stays resident
between turns without any server config.

## Tools

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

Reads are free. Writes ask first, and show you the diff before you answer:

```
  edit  src/auth.py
╭────────────────────────────────╮
│ @@ -12,3 +12,3 @@              │
│ -    if user.token:            │
│ +    if user.token is not None:│
╰────────────────────────────────╯
  [y] yes  [a] always  [n] no  [q] stop
```

`a` remembers — per tool, and for `shell` per *exact command*, so approving
`npm test` forever does not also approve `rm -rf build`. Commands that reach
outside the machine (`git push`, `curl`, `ssh`, `sudo`) always ask, even then.

Tools cannot touch anything outside the project directory, and a handful of
unrecoverable commands are refused outright rather than merely prompted for —
a yes/no prompt is exactly the thing people click through on autopilot.

`--yolo` turns the prompts off for a sandbox or a throwaway container.

## Commands

```
/help                    everything below
/effort [level]          low | medium | high | xhigh | max | ultra
/model [name]            switch model, or list what the server has
/endpoint ...            list | test | add <url> [name] | use <name>
/ctx [n]                 show or set the context window
/tools                   what the agent can call
/thinking                show or hide the model's reasoning
/plan                    the current todo list
/clear                   fresh conversation
/compact                 summarise now, reclaim context
/stats                   tokens, speed, context use, tool mode
/doctor                  check the server and model for problems
/yolo                    stop asking permission this session
/sessions                recent sessions
/init                    write a WYNXO.md describing this project
/quit
```

Ctrl-C interrupts a running turn without killing the session. Alt-Enter
inserts a newline.

## Non-interactive

```bash
wynxo -p "what does this project do"
wynxo -p -e high "add a test for the retry path"
git diff | wynxo -p "review this"
```

`-p` implies `--yolo`, since there is nobody there to answer a prompt.

Piped stdin is folded into the prompt as context. wynxo gives up on an idle
pipe after a moment rather than blocking — starting it with stdin attached to
a pipe nobody writes to (CI, a supervisor, some editor terminals) would
otherwise hang with no output at all.

## Project instructions

Drop a `WYNXO.md` at the repo root and it is loaded into every system prompt —
build commands, conventions, things to avoid. `AGENTS.md` and `CLAUDE.md` are
also picked up, so an existing one works as-is. `/init` writes one for you.

## Configuration

| | |
|---|---|
| Linux | `~/.config/wynxo/config.json` |
| macOS | `~/Library/Application Support/wynxo/config.json` |
| Windows | `%APPDATA%\wynxo\config.json` |

A project-local `.wynxo.json` overrides it. Environment variables
(`WYNXO_ENDPOINT`, `WYNXO_MODEL`, `WYNXO_EFFORT`, `WYNXO_NUM_CTX`, and
`OLLAMA_HOST`) override that. Command-line flags win.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The agent tests run the real loop, real tools and real files against a
scripted fake Ollama — including malformed tool calls, denied permissions,
path escapes, iteration ceilings, and older servers that reject string think
levels.

The wire format is checked against Ollama's own `api/types.go` and
`docs/api.md` rather than against assumptions. That is how the `tool_name`
field and the string `think` levels were found; both were wrong here first.

## License

MIT
