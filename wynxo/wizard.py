"""First-run setup.

The one question that actually matters is where Ollama is. Everyone running
local models ends up with at least two answers -- the laptop they are typing
on, and the box in the basement with the real GPU -- so this asks, probes
what it finds, and remembers every server it has seen.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_MODEL,
    Config,
    Endpoint,
    MIN_USABLE_CONTEXT,
    normalise_url,
)
from .effort import ORDER, resolve
from .discovery import Found, private_subnets, scan_loopback, scan_subnets, verify
from .platforms import ollama_server_help as server_help  # re-exported
from .provider import OllamaClient, ProviderError
from .ui import ACCENT, MUTED, UI

# Models worth recommending, best first, with why.
RECOMMENDED = [
    ("qwen3-coder:30b", "30B MoE, ~3B active. Tool-tuned, fast, best all-rounder here."),
    ("qwen3:32b", "Dense 32B. Stronger reasoning, noticeably slower."),
    ("qwen3:30b-a3b", "General-purpose MoE sibling of qwen3-coder. Good for chat."),
    ("devstral:24b", "Built for agent loops. Excellent tool discipline."),
    ("gpt-oss:20b", "Has a real native reasoning dial (low/medium/high)."),
    ("qwen3:14b", "Fits comfortably in 12GB VRAM."),
    ("qwen3:8b", "Runs on almost anything, including CPU-only."),
]


async def probe(url: str, timeout: float = 2.0) -> str | None:
    """Return the Ollama version at ``url``, or None."""
    return await verify(url, timeout=timeout)


async def ask_endpoint(ui: UI, prompt_session: PromptSession) -> Endpoint:
    """The 'where does Ollama serve?' question."""
    ui.console.print()
    ui.console.print(f"[bold {ACCENT}]Where does Ollama serve?[/]")
    ui.console.print(
        f"[{MUTED}]Your own machine, or a box on your network. Either is fine.[/]"
    )
    ui.console.print()

    found: list[Found] = []
    with ui.status("checking this machine...") as status:
        found = await scan_loopback()

        subnets = private_subnets()
        if subnets:
            names = ", ".join(str(n) for n in subnets[:2])
            def progress(done: int, total: int) -> None:
                status.update(f"scanning {names} for Ollama... {done}/{total}")
            status.update(f"scanning {names} for Ollama...")
            found.extend(await scan_subnets(subnets, on_progress=progress))

    if found:
        ui.console.print(f"  [{ACCENT}]Found:[/]")
        for i, hit in enumerate(found, 1):
            ui.console.print(
                f"    [bold]{i}[/]  {hit.url}  [{MUTED}]v{hit.version} · {hit.where}[/]")
        ui.console.print(f"    [bold]m[/]  [{MUTED}]enter a different address[/]")
        ui.console.print()

        while True:
            answer = (await prompt_session.prompt_async(
                HTML(f'<ansicyan>  choose [1-{len(found)} or m]: </ansicyan>')
            )).strip().lower()
            if answer in ("", "1"):
                url = found[0].url
                break
            if answer == "m":
                url = ""
                break
            if answer.isdigit() and 1 <= int(answer) <= len(found):
                url = found[int(answer) - 1].url
                break
            ui.warn("Pick a number from the list, or m to type an address.")
    else:
        ui.console.print(f"  [{MUTED}]Nothing found.[/]")
        ui.console.print(
            f"  [{MUTED}]If Ollama runs on another machine, that machine must be"
            f"\n  started with OLLAMA_HOST=0.0.0.0:11434 -- by default it listens"
            f"\n  only on its own loopback and nothing on the network can reach it."
            f"\n  Then give its LAN address below.[/]"
        )
        ui.console.print()
        url = ""

    while True:
        if not url:
            ui.console.print(
                f"  [{MUTED}]This machine:   127.0.0.1[/]"
            )
            ui.console.print(
                f"  [{MUTED}]Another box:    192.168.1.50   (or 192.168.1.50:11434)[/]"
            )
            raw = (await prompt_session.prompt_async(
                HTML('<ansicyan>  address: </ansicyan>')
            )).strip()
            if not raw:
                ui.warn("Enter an address, or Ctrl-C to quit.")
                continue
            url = normalise_url(raw)

        with ui.status(f"checking {url} ..."):
            version = await probe(url, timeout=6.0)

        if version:
            ui.success(f"Ollama {version} at {url}")
            break

        ui.error(
            f"Nothing answering at {url}.\n\n"
            "  - Is `ollama serve` running there?\n"
            "  - Another machine: it must start with OLLAMA_HOST=0.0.0.0:11434,\n"
            "    otherwise it only listens on its own loopback.\n"
            "  - Firewall open on 11434?\n"
            "  - Right IP? Run `ip addr` (or `ipconfig`) on that machine."
        )
        retry = (await prompt_session.prompt_async(
            HTML('<ansicyan>  try another address? [Y/n]: </ansicyan>')
        )).strip().lower()
        if retry in ("n", "no"):
            raise SystemExit(1)
        url = ""

    api_key = None
    if url.startswith("https://"):
        ui.console.print(
            f"  [{MUTED}]That is an https endpoint, so it may sit behind auth.[/]"
        )
        api_key = (await prompt_session.prompt_async(
            HTML('<ansicyan>  bearer token (blank if none): </ansicyan>'),
            is_password=True,
        )).strip() or None

    is_local = any(h in url for h in ("127.0.0.1", "localhost", "::1"))
    name = "local" if is_local else url.split("://", 1)[1].split(":")[0].replace(".", "-")
    return Endpoint(name=name, url=url, api_key=api_key)


async def ask_model(ui: UI, prompt_session: PromptSession, config: Config) -> str:
    """Pick a model from what the server actually has installed."""
    ui.console.print()
    ui.console.print(f"[bold {ACCENT}]Which model?[/]")

    installed = []
    async with OllamaClient(config) as client:
        try:
            with ui.status("asking the server what it has..."):
                installed = await client.list_models()
        except ProviderError as exc:
            ui.warn(str(exc))

    if installed:
        ui.console.print()
        ui.console.print(f"  [{MUTED}]Installed on that server:[/]")
        for i, model in enumerate(installed, 1):
            note = ""
            for name, why in RECOMMENDED:
                if model.name.startswith(name.split(":")[0]):
                    note = f"  [{MUTED}]{why}[/]"
                    break
            size = f"[{MUTED}]{model.human_size()}[/]"
            ui.console.print(f"    [bold]{i:2}[/]  {model.name:32} {size}{note}")
        ui.console.print(f"    [bold] p[/]  [{MUTED}]pull a model that is not listed[/]")
        ui.console.print()

        names = [m.name for m in installed]
        default_index = next(
            (i for i, n in enumerate(names, 1) if n.startswith("qwen3-coder")), 1
        )

        while True:
            answer = (await prompt_session.prompt_async(
                HTML(f'<ansicyan>  choose [1-{len(names)} or p, default {default_index}]: </ansicyan>')
            )).strip().lower()
            if answer == "":
                return names[default_index - 1]
            if answer == "p":
                break
            if answer.isdigit() and 1 <= int(answer) <= len(names):
                return names[int(answer) - 1]
            if answer in names:
                return answer
            ui.warn("Pick a number from the list, or p to pull something new.")

    # Nothing installed, or the user chose to pull.
    ui.console.print()
    ui.console.print(f"  [{MUTED}]Worth pulling, best first:[/]")
    for name, why in RECOMMENDED:
        ui.console.print(f"    [bold]{name:20}[/] [{MUTED}]{why}[/]")
    ui.console.print()

    completer = WordCompleter([name for name, _ in RECOMMENDED])
    model = (await prompt_session.prompt_async(
        HTML(f'<ansicyan>  model [{DEFAULT_MODEL}]: </ansicyan>'), completer=completer
    )).strip() or DEFAULT_MODEL

    pull = (await prompt_session.prompt_async(
        HTML(f'<ansicyan>  pull {model} now? [Y/n]: </ansicyan>')
    )).strip().lower()
    if pull not in ("n", "no"):
        config.model = model
        async with OllamaClient(config) as client:
            try:
                with ui.status(f"pulling {model} ...") as status:
                    async for line in client.pull(model):
                        status.update(f"pulling {model}: {line}")
                ui.success(f"pulled {model}")
            except ProviderError as exc:
                ui.warn(f"{exc}\nYou can pull it later with: ollama pull {model}")
    return model


async def ask_effort(ui: UI, prompt_session: PromptSession) -> str:
    ui.console.print()
    ui.console.print(f"[bold {ACCENT}]Default effort level?[/]")
    ui.console.print(
        f"[{MUTED}]How hard the agent works before it answers. Change it any time "
        f"with /effort.[/]"
    )
    ui.console.print()
    for name in ORDER:
        policy = resolve(name)
        ui.console.print(f"    [bold]{name:7}[/] [{MUTED}]{policy.describe()}[/]")
    ui.console.print()

    completer = WordCompleter(list(ORDER))
    while True:
        answer = (await prompt_session.prompt_async(
            HTML('<ansicyan>  effort [medium]: </ansicyan>'), completer=completer
        )).strip().lower() or "medium"
        try:
            return resolve(answer).name
        except KeyError:
            ui.warn(f"Choose one of: {', '.join(ORDER)}")


async def ask_context(ui: UI, prompt_session: PromptSession, config: Config) -> int:
    """Set num_ctx, with the warning that this deserves."""
    ui.console.print()
    ui.console.print(f"[bold {ACCENT}]Context window[/]")
    ui.console.print(
        f"[{MUTED}]Ollama defaults to a very small context (often 2048 or 4096). An "
        f"agent\nsilently forgets the first half of its task at that size, with no "
        f"error.\nBigger costs VRAM. {DEFAULT_CONTEXT} is a sensible default for a "
        f"30B on 24GB.[/]"
    )
    ui.console.print()

    native = 0
    async with OllamaClient(config) as client:
        try:
            info = await client.show(config.model)
            native = info.context_length
        except ProviderError:
            pass
    if native:
        ui.console.print(f"  [{MUTED}]{config.model} was trained for {native} tokens.[/]")

    while True:
        answer = (await prompt_session.prompt_async(
            HTML(f'<ansicyan>  num_ctx [{DEFAULT_CONTEXT}]: </ansicyan>')
        )).strip()
        if not answer:
            return DEFAULT_CONTEXT
        try:
            value = int(answer)
        except ValueError:
            ui.warn("Enter a number.")
            continue
        if value < MIN_USABLE_CONTEXT:
            ui.warn(
                f"{value} is below {MIN_USABLE_CONTEXT}; the agent will lose track "
                "of long tasks. Continuing anyway if that is what you want."
            )
        if native and value > native:
            ui.warn(f"Above {config.model}'s native {native}; quality degrades past that.")
        return value


async def run_wizard(ui: UI) -> Config:
    """Full first-run flow. Returns a saved config."""
    ui.console.print()
    ui.console.print(f"[bold {ACCENT}]  wynxo setup[/]")
    ui.console.print(f"  [{MUTED}]Four questions. Everything is changeable later.[/]")

    prompt_session: PromptSession = PromptSession()
    config = Config()

    endpoint = await ask_endpoint(ui, prompt_session)
    config.endpoints = [endpoint]
    config.active_endpoint = endpoint.name

    config.model = await ask_model(ui, prompt_session, config)
    config.effort = await ask_effort(ui, prompt_session)
    config.num_ctx = await ask_context(ui, prompt_session, config)

    path = config.save()
    ui.console.print()
    ui.success(f"Saved to {path}")
    ui.console.print(
        f"  [{MUTED}]Add more servers later with /endpoint add <url>, "
        f"switch with /endpoint use <name>.[/]"
    )
    ui.console.print()
    return config

__all__ = ["run_wizard", "probe", "server_help", "RECOMMENDED"]
