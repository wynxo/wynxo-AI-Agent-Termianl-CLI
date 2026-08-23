"""Entry point and REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
import select
import signal
import stat
import sys
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from . import __version__
from .agent import Agent, Callbacks, Interrupted
from .config import Config, Endpoint, data_dir, is_configured, load, normalise_url
from .doctor import run_doctor
from .effort import ORDER, resolve
from .permissions import Decision
from .provider import OllamaClient, ProviderError, check_context
from .session import Session
from .ui import ACCENT, MUTED, UI
from .platforms import ollama_server_help as server_help
from .wizard import probe, run_wizard

COMMANDS = {
    "/help": "show this",
    "/effort": "change effort level (low|medium|high|xhigh|max|ultra)",
    "/model": "switch model, or list what the server has",
    "/endpoint": "list | use <name> | add <url> | test -- where Ollama serves",
    "/ctx": "show or set the context window (num_ctx)",
    "/tools": "list the tools the agent can call",
    "/thinking": "show or hide the model's reasoning",
    "/plan": "show the current plan",
    "/clear": "start a fresh conversation",
    "/compact": "summarise the conversation to reclaim context",
    "/stats": "tokens, speed, context use",
    "/doctor": "check the server and model for problems",
    "/yolo": "stop asking permission for this session",
    "/sessions": "list recent sessions",
    "/init": "write a WYNXO.md describing this project",
    "/quit": "exit",
}


class TerminalCallbacks(Callbacks):
    """Wires the agent's events to the terminal."""

    def __init__(self, ui: UI, prompt_session: PromptSession | None = None):
        self.ui = ui
        # None in non-interactive mode, where nothing can be asked anyway.
        self.prompt_session = prompt_session
        self._streaming = False
        self._thinking_open = False
        self._thinking_chars = 0

    def _end_stream(self) -> None:
        """Close whichever transient line is open, so the next block starts clean."""
        if self._thinking_open:
            self.ui.console.print(f" [{MUTED}]({self._thinking_chars} chars)[/]")
            self._thinking_open = False
            self._thinking_chars = 0
        if self._streaming:
            self.ui.console.print()
            self._streaming = False

    async def on_thinking(self, text: str) -> None:
        if not self.ui.show_thinking:
            return
        if self._streaming:
            self._end_stream()
        if not self._thinking_open:
            self.ui.console.print(f"  [{MUTED}]thinking...[/]", end="")
            self._thinking_open = True
        self._thinking_chars += len(text)

    async def on_content(self, text: str) -> None:
        if not self._streaming:
            self._end_stream()
            self.ui.console.print()
            self._streaming = True
        self.ui.stream_chunk(text)

    async def on_stage(self, name: str, detail: str = "") -> None:
        self._end_stream()
        suffix = f" [{MUTED}]{detail}[/]" if detail else ""
        self.ui.console.print(f"  [{ACCENT}]{self.ui.g.arrow}[/] [{MUTED}]{name}[/]{suffix}")

    async def on_tool_start(self, name: str, summary: str) -> None:
        self._end_stream()
        self.ui.tool_start(name, summary)

    async def on_tool_result(self, name: str, ok: bool, display: str, output: str) -> None:
        self.ui.tool_result(name, ok, display, output)

    async def on_todos(self, rendered: str) -> None:
        self.ui.todos(rendered)

    async def on_warning(self, message: str) -> None:
        self._end_stream()
        self.ui.warn(message)

    async def ask_permission(self, name: str, summary: str, preview: str) -> Decision:
        self._end_stream()
        if self.prompt_session is None:
            return Decision.ALLOW
        self.ui.console.print()
        verb = {
            "write_file": "write to",
            "edit_file": "edit",
            "shell": "run",
        }.get(name, name)
        self.ui.console.print(f"  [bold {ACCENT}]{verb}[/] [bold]{summary}[/]")
        if preview:
            self.ui.diff(preview) if preview.lstrip().startswith(("---", "+", "-")) else self.ui.code(preview)

        while True:
            try:
                answer = (await self.prompt_session.prompt_async(
                    HTML('<ansicyan>  [y] yes  [a] always  [n] no  [q] stop: </ansicyan>')
                )).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return Decision.ABORT
            if answer in ("", "y", "yes"):
                return Decision.ALLOW
            if answer in ("a", "always"):
                return Decision.ALLOW_ALWAYS
            if answer in ("n", "no"):
                return Decision.DENY
            if answer in ("q", "quit", "stop"):
                return Decision.ABORT
            self.ui.warn("y, a, n or q.")


class Repl:
    def __init__(self, config: Config, workspace: Path, ui: UI):
        self.config = config
        self.workspace = workspace
        self.ui = ui
        self.client = OllamaClient(config)
        self.policy = resolve(config.effort)

        history_file = data_dir() / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event):
            """Alt-Enter inserts a newline; Enter submits."""
            event.current_buffer.insert_text("\n")

        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(history_file)),
            completer=WordCompleter(list(COMMANDS), sentence=True),
            key_bindings=bindings,
            multiline=False,
        )
        self.callbacks = TerminalCallbacks(ui, self.prompt_session)
        self.agent = Agent(self.client, config, self.policy, workspace, self.callbacks)
        self._task: asyncio.Task | None = None

    async def _connect(self) -> bool:
        """Reach the server, show the banner, adapt to the model."""
        try:
            version = await self.client.ping()
        except ProviderError as exc:
            self.ui.error(str(exc))
            self.ui.console.print(server_help())
            return False

        self.ui.banner(
            self.config.model,
            f"{self.client.base_url} (ollama {version})",
            self.policy.name,
            str(self.workspace),
        )
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        await self.agent.detect_capabilities()
        return True

    async def start(self) -> int:
        if not await self._connect():
            return 1
        return await self._loop()

    async def _loop(self) -> int:
        while True:
            try:
                text = await self.prompt_session.prompt_async(
                    HTML(f'<ansicyan><b>{self.policy.name}</b> &gt; </ansicyan>')
                )
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

            text = text.strip()
            if not text:
                continue

            if text.startswith("/"):
                if await self.command(text) is False:
                    break
                continue

            await self.turn(text)

        await self.client.aclose()
        self.ui.console.print(f"  [{MUTED}]bye[/]")
        return 0

    async def start_with(self, prompt: str) -> int:
        """Run one prompt, then drop into the REPL. `wynxo "fix the tests"`."""
        if not await self._connect():
            return 1
        await self.turn(prompt)
        return await self._loop()

    async def turn(self, text: str) -> None:
        """Run one request, cancellable with Ctrl-C."""
        self._task = asyncio.ensure_future(self.agent.run(text))
        try:
            result = await self._task
        except (asyncio.CancelledError, Interrupted):
            self.ui.console.print()
            self.ui.warn("Interrupted. The conversation is intact; ask me something else.")
            return
        finally:
            self._task = None

        self.callbacks._end_stream()

        if result.errors:
            self.ui.error("\n".join(result.errors))
            return

        if result.content and not self.config.stream:
            self.ui.assistant_markdown(result.content)
        elif result.content:
            self.ui.console.print()

        used = self.agent.session.token_estimate()
        limit = self.policy.context_budget or self.config.num_ctx
        self.ui.stats(
            self.agent.session.usage,
            result.elapsed,
            self.policy.name,
            100 * used / max(1, limit),
        )
        if result.compacted:
            self.ui.info("context was compacted during this turn")

    def interrupt(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    # -- slash commands ----------------------------------------------------

    async def command(self, text: str) -> bool:
        parts = text.split()
        name, args = parts[0].lower(), parts[1:]

        if name in ("/quit", "/exit", "/q"):
            return False

        if name == "/help":
            self.ui.table(
                ["command", "what it does"],
                [(c, d) for c, d in COMMANDS.items()],
                title="commands",
            )
            self.ui.info("Alt-Enter for a newline, Ctrl-C to interrupt a running turn.")
            return True

        if name == "/effort":
            return await self.cmd_effort(args)
        if name == "/model":
            return await self.cmd_model(args)
        if name == "/endpoint":
            return await self.cmd_endpoint(args)
        if name == "/ctx":
            return await self.cmd_ctx(args)

        if name == "/tools":
            self.ui.table(
                ["tool", "writes?", "what it does"],
                [(t.signature(), "yes" if t.mutating else "", t.description[:70])
                 for t in self.agent.tools],
                title=f"{len(self.agent.tools)} tools"
                + ("" if self.agent.native_tools else "  (Hermes prompted mode)"),
            )
            return True

        if name == "/thinking":
            self.ui.show_thinking = not self.ui.show_thinking
            self.config.show_thinking = self.ui.show_thinking
            self.ui.info(f"thinking display {'on' if self.ui.show_thinking else 'off'}")
            return True

        if name == "/plan":
            todo = self.agent.tools.get("todo_write")
            rendered = todo.render() if todo and hasattr(todo, "render") else ""
            self.ui.todos(rendered) if rendered else self.ui.info("no plan yet")
            return True

        if name == "/clear":
            self.agent.session = Session(workspace=self.workspace)
            self.agent.refresh_system_prompt()
            self.ui.info("conversation cleared")
            return True

        if name == "/compact":
            before = self.agent.session.token_estimate()
            with self.ui.status("compacting..."):
                await self.agent._compact()
            after = self.agent.session.token_estimate()
            self.ui.success(f"{before} -> {after} tokens")
            return True

        if name == "/stats":
            return self.cmd_stats()

        if name == "/doctor":
            from .doctor import Doctor
            await Doctor(self.client, self.config, self.ui).run()
            return True

        if name == "/yolo":
            self.agent.permissions.yolo = not self.agent.permissions.yolo
            if self.agent.permissions.yolo:
                self.ui.warn("Permission prompts off. The agent can write files and run commands freely.")
            else:
                self.ui.info("Permission prompts back on.")
            return True

        if name == "/sessions":
            rows = Session.recent()
            if not rows:
                self.ui.info("no saved sessions")
                return True
            self.ui.table(
                ["id", "messages", "started with"],
                [(r["session_id"], r["messages"], r["preview"]) for r in rows],
                title="recent sessions",
            )
            return True

        if name == "/init":
            await self.turn(
                "Look at this project -- its layout, build files, tests, and "
                "conventions -- then write a WYNXO.md at the root that tells a "
                "new agent what it needs to know: what the project is, how to "
                "build and test it, and the conventions to follow. Be concrete "
                "and brief. No filler."
            )
            return True

        self.ui.warn(f"unknown command {name}. /help for the list.")
        return True

    async def cmd_effort(self, args: list[str]) -> bool:
        if not args:
            self.ui.table(
                ["level", "behaviour"],
                [(n + ("  <-" if n == self.policy.name else ""), resolve(n).describe())
                 for n in ORDER],
                title="effort levels",
            )
            return True
        try:
            policy = resolve(args[0])
        except KeyError as exc:
            self.ui.warn(str(exc))
            return True
        self.policy = policy
        self.config.effort = policy.name
        self.agent.set_effort(policy)
        self.ui.success(f"effort: {policy.name} -- {policy.describe()}")
        return True

    async def cmd_model(self, args: list[str]) -> bool:
        if not args:
            try:
                with self.ui.status("listing models..."):
                    models = await self.client.list_models()
            except ProviderError as exc:
                self.ui.error(str(exc))
                return True
            self.ui.table(
                ["model", "size", "params"],
                [(m.name + ("  <-" if m.name == self.config.model else ""),
                  m.human_size(), m.parameter_size) for m in models],
                title=f"models on {self.client.base_url}",
            )
            self.ui.info("switch with /model <name>")
            return True

        target = args[0]
        try:
            models = await self.client.list_models()
        except ProviderError as exc:
            self.ui.error(str(exc))
            return True

        names = [m.name for m in models]
        if target not in names:
            # Accept a bare name when exactly one tag matches.
            matches = [n for n in names if n.split(":")[0] == target]
            if len(matches) == 1:
                target = matches[0]
            else:
                self.ui.warn(f"{target} is not installed on that server.")
                if matches:
                    self.ui.info(f"did you mean: {', '.join(matches)}")
                else:
                    self.ui.info(f"pull it first:  ollama pull {target}")
                return True

        self.config.model = target
        self.agent.config.model = target
        await self.agent.detect_capabilities()
        self.ui.success(f"model: {target}")
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        return True

    async def cmd_endpoint(self, args: list[str]) -> bool:
        action = args[0].lower() if args else "list"

        if action == "list":
            self.ui.table(
                ["name", "url"],
                [(e.name + ("  <-" if e.name == self.config.active_endpoint else ""), e.url)
                 for e in self.config.endpoints],
                title="ollama servers",
            )
            self.ui.info("/endpoint add <url> · /endpoint use <name> · /endpoint test")
            return True

        if action == "test":
            for endpoint in self.config.endpoints:
                with self.ui.status(f"checking {endpoint.url}..."):
                    version = await probe(endpoint.url, timeout=5.0)
                if version:
                    self.ui.success(f"{endpoint.name}: {endpoint.url} (ollama {version})")
                else:
                    self.ui.warn(f"{endpoint.name}: {endpoint.url} unreachable")
            return True

        if action == "add" and len(args) > 1:
            url = normalise_url(args[1])
            name = args[2] if len(args) > 2 else url.split("://")[1].split(":")[0]
            with self.ui.status(f"checking {url}..."):
                version = await probe(url, timeout=6.0)
            if not version:
                self.ui.error(f"Nothing answering at {url}.\n\n{server_help()}")
                return True
            self.config.endpoints = [e for e in self.config.endpoints if e.name != name]
            self.config.endpoints.append(Endpoint(name=name, url=url))
            self.config.save()
            self.ui.success(f"added {name} -> {url} (ollama {version})")
            self.ui.info(f"switch to it with /endpoint use {name}")
            return True

        if action == "use" and len(args) > 1:
            name = args[1]
            if not any(e.name == name for e in self.config.endpoints):
                self.ui.warn(f"no endpoint named {name}. /endpoint list to see them.")
                return True
            self.config.active_endpoint = name
            self.config.save()
            await self.client.aclose()
            self.client = OllamaClient(self.config)
            self.agent.client = self.client
            try:
                version = await self.client.ping()
            except ProviderError as exc:
                self.ui.error(str(exc))
                return True
            self.ui.success(f"now using {name}: {self.client.base_url} (ollama {version})")
            return True

        self.ui.warn("usage: /endpoint [list | test | add <url> [name] | use <name>]")
        return True

    async def cmd_ctx(self, args: list[str]) -> bool:
        if not args:
            used = self.agent.session.token_estimate()
            self.ui.info(
                f"num_ctx={self.config.num_ctx}, roughly {used} tokens in use "
                f"({100 * used / max(1, self.config.num_ctx):.0f}%)"
            )
            return True
        try:
            value = int(args[0])
        except ValueError:
            self.ui.warn("usage: /ctx <number>")
            return True
        self.config.num_ctx = value
        self.agent.config.num_ctx = value
        self.ui.success(f"num_ctx = {value}")
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        return True

    def cmd_stats(self) -> bool:
        usage = self.agent.session.usage
        used = self.agent.session.token_estimate()
        limit = self.policy.context_budget or self.config.num_ctx
        self.ui.table(
            ["", ""],
            [
                ("model", self.config.model),
                ("server", self.client.base_url),
                ("effort", f"{self.policy.name} -- {self.policy.describe()}"),
                ("requests", str(usage.requests)),
                ("tool calls", str(usage.tool_calls)),
                ("prompt tokens", str(usage.prompt_tokens)),
                ("output tokens", str(usage.completion_tokens)),
                ("speed", f"{usage.tokens_per_second():.1f} tok/s"),
                ("context", f"~{used} / {limit} ({100 * used / max(1, limit):.0f}%)"),
                ("compactions", str(self.agent.session.compactions)),
                ("tool mode", "native" if self.agent.native_tools else "hermes (prompted)"),
                ("session", self.agent.session.session_id),
            ],
            title="session",
        )
        return True


def read_piped_stdin(grace: float = 0.25) -> str:
    """Read piped stdin, or return "" -- but never block waiting for a writer.

    A plain ``sys.stdin.read()`` here hangs forever whenever stdin is an open
    pipe that nobody writes to, which is the normal state of affairs under CI,
    process supervisors and editor terminals. The agent then sits there with no
    output and no explanation, which looks exactly like a crash.
    """
    if sys.stdin.isatty():
        return ""

    try:
        if stat.S_ISREG(os.fstat(sys.stdin.fileno()).st_mode):
            # A redirect from a real file always has an EOF coming.
            return sys.stdin.read().strip()
    except (OSError, ValueError):
        return ""

    if hasattr(select, "select"):
        try:
            ready, _, _ = select.select([sys.stdin], [], [], grace)
        except (OSError, ValueError):
            return ""
        if not ready:
            return ""
        try:
            return sys.stdin.read().strip()
        except (OSError, UnicodeDecodeError):
            return ""

    # Windows cannot select on a pipe, so read in a thread we can abandon.
    # A slow producer may miss the window; redirect from a file to be certain.
    result: list[str] = []

    def _read() -> None:
        try:
            result.append(sys.stdin.read())
        except (OSError, UnicodeDecodeError, ValueError):
            pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(grace)
    return result[0].strip() if result else ""


async def run_once(config: Config, workspace: Path, ui: UI, prompt: str) -> int:
    """Non-interactive mode: answer one prompt and exit."""
    client = OllamaClient(config)
    try:
        await client.ping()
    except ProviderError as exc:
        ui.error(str(exc))
        await client.aclose()
        return 1

    callbacks = TerminalCallbacks(ui)
    agent = Agent(client, config, resolve(config.effort), workspace, callbacks)
    # No human to answer prompts, so pre-approve; the caller opted into this
    # by using -p.
    agent.permissions.yolo = True
    await agent.detect_capabilities()

    result = await agent.run(prompt)
    callbacks._end_stream()
    await client.aclose()

    if result.errors:
        ui.error("\n".join(result.errors))
        return 1
    if result.content and not config.stream:
        ui.assistant_markdown(result.content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wynxo",
        description="A terminal coding agent for local models via Ollama.",
    )
    parser.add_argument("prompt", nargs="*", help="run one prompt and exit")
    parser.add_argument("-p", "--print", action="store_true",
                        help="non-interactive: answer the prompt and exit")
    parser.add_argument("-e", "--effort", choices=list(ORDER), help="effort level for this run")
    parser.add_argument("-m", "--model", help="model to use")
    parser.add_argument("--endpoint", help="Ollama URL, e.g. http://homelab:11434")
    parser.add_argument("--ctx", type=int, help="context window size (num_ctx)")
    parser.add_argument("-C", "--cwd", help="project directory (default: here)")
    parser.add_argument("--setup", action="store_true", help="re-run first-time setup")
    parser.add_argument("--doctor", action="store_true",
                        help="check the server and model, and report what will not work")
    parser.add_argument("--no-stream", action="store_true", help="wait for the full response")
    parser.add_argument("--no-thinking", action="store_true", help="hide model reasoning")
    parser.add_argument("--yolo", action="store_true", help="never ask permission")
    parser.add_argument("--version", action="version", version=f"wynxo {__version__}")
    return parser


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    if not workspace.is_dir():
        print(f"No such directory: {workspace}", file=sys.stderr)
        return 1

    ui = UI(show_thinking=not args.no_thinking)

    # An endpoint supplied on the command line or in the environment is enough
    # to run: do not drag the user through setup for something they answered.
    endpoint_supplied = bool(
        args.endpoint or os.environ.get("WYNXO_ENDPOINT") or os.environ.get("OLLAMA_HOST")
    )
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if args.setup or (not is_configured() and not endpoint_supplied):
        if not interactive:
            print(
                "wynxo is not configured yet, and there is no terminal to ask on.\n"
                "Either run `wynxo --setup` interactively, or point it at a server:\n"
                "  wynxo --endpoint homelab:11434 -p \"your prompt\"\n"
                "  WYNXO_ENDPOINT=homelab:11434 wynxo -p \"your prompt\"",
                file=sys.stderr,
            )
            return 1
        try:
            config = await run_wizard(ui)
        except (KeyboardInterrupt, EOFError):
            ui.console.print()
            ui.info("setup cancelled")
            return 1
    else:
        config = load(workspace)

    if args.endpoint:
        url = normalise_url(args.endpoint)
        config.endpoints = [Endpoint(name="cli", url=url)]
        config.active_endpoint = "cli"
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.ctx:
        config.num_ctx = args.ctx
    if args.no_stream:
        config.stream = False
    if args.no_thinking:
        config.show_thinking = False
    ui.show_thinking = config.show_thinking

    if args.doctor:
        return await run_doctor(config, ui)

    prompt = " ".join(args.prompt).strip()

    # Piped input becomes context for the prompt: `git diff | wynxo -p "review"`.
    piped = read_piped_stdin()
    if piped:
        prompt = (
            f"{prompt}\n\n<piped-input>\n{piped}\n</piped-input>"
            if prompt
            else f"Here is some input:\n\n<piped-input>\n{piped}\n</piped-input>"
        )
        args.print = True  # nothing to be interactive with

    if prompt and args.print:
        return await run_once(config, workspace, ui, prompt)

    repl = Repl(config, workspace, ui)
    if args.yolo:
        repl.agent.permissions.yolo = True
        ui.warn("--yolo: the agent will write files and run commands without asking.")

    # Ctrl-C cancels the running turn rather than killing the process.
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":
        # No add_signal_handler on Windows. The handler runs in the main thread,
        # so hop back onto the loop before touching the task.
        signal.signal(
            signal.SIGINT,
            lambda *_: loop.call_soon_threadsafe(repl.interrupt),
        )
    else:
        try:
            loop.add_signal_handler(signal.SIGINT, repl.interrupt)
        except NotImplementedError:
            pass

    with patch_stdout():
        if prompt:
            return await repl.start_with(prompt)
        return await repl.start()


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
