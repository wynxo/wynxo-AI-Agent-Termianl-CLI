"""Project and Ollama discovery helpers."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

OLLAMA_PORT = 11434
MAX_CONCURRENCY = 128
CONNECT_TIMEOUT = 0.35
VERIFY_TIMEOUT = 2.0
_SKIP = {".git", ".wynxo", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target"}
_MARKERS = {
    "pyproject.toml": "Python", "requirements.txt": "Python", "package.json": "JavaScript/Node",
    "Cargo.toml": "Rust", "go.mod": "Go", "pom.xml": "Java", "build.gradle": "Java", "Makefile": "Make",
}


@dataclass(frozen=True)
class ProjectInfo:
    root: Path
    is_git: bool
    branch: str = ""
    dirty: bool = False
    languages: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    test_frameworks: tuple[str, ...] = ()
    build_systems: tuple[str, ...] = ()

    def summary(self) -> str:
        kind = ", ".join(self.languages) or "unknown project"
        tests = ", ".join(self.test_frameworks) or "no detected test runner"
        git = f"git {self.branch or '?'}" + (" · dirty" if self.dirty else "") if self.is_git else "not a git repository"
        return f"{kind} · {git} · tests: {tests}"


class Discovery:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._cached: ProjectInfo | None = None
        self._fingerprint: tuple[tuple[str, int, int], ...] = ()

    def scan(self, force: bool = False) -> ProjectInfo:
        root = self._find_root()
        fingerprint = self._marker_fingerprint(root)
        if self._cached is not None and not force and fingerprint == self._fingerprint:
            return self._cached
        markers = sorted(p.name for p in root.iterdir() if p.is_file() and p.name in _MARKERS)
        languages = sorted({_MARKERS[m] for m in markers})
        tests: list[str] = []
        builds: list[str] = []
        if "pyproject.toml" in markers or "requirements.txt" in markers:
            tests.append("pytest" if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() else "Python tests")
        if "package.json" in markers:
            tests.append("npm test"); builds.append("npm")
        if "Cargo.toml" in markers:
            tests.append("cargo test"); builds.append("cargo")
        if "go.mod" in markers:
            tests.append("go test"); builds.append("go")
        if "Makefile" in markers: builds.append("make")
        is_git, branch, dirty = _git_info(root)
        self._cached = ProjectInfo(root, is_git, branch, dirty, tuple(languages), tuple(markers), tuple(tests), tuple(builds))
        self._fingerprint = fingerprint
        return self._cached

    def invalidate(self) -> None:
        self._cached = None

    def _find_root(self) -> Path:
        for candidate in (self.workspace, *self.workspace.parents):
            if (candidate / ".git").exists() or any((candidate / marker).exists() for marker in _MARKERS):
                return candidate
        return self.workspace

    def _marker_fingerprint(self, root: Path) -> tuple[tuple[str, int, int], ...]:
        out = []
        for marker in _MARKERS:
            try:
                stat = (root / marker).stat()
                out.append((marker, stat.st_mtime_ns, stat.st_size))
            except OSError:
                out.append((marker, 0, 0))
        return tuple(out)


def _git_info(root: Path) -> tuple[bool, str, bool]:
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True, timeout=3)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False, "", False
    return status.returncode == 0, branch.stdout.strip(), bool(status.stdout.strip()) if status.returncode == 0 else False


@dataclass
class Found:
    url: str
    version: str
    local: bool

    @property
    def where(self) -> str:
        return "this machine" if self.local else "network"


def local_ipv4_addresses() -> list[str]:
    found: set[str] = set()
    for probe in ("8.8.8.8", "1.1.1.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(0.2); sock.connect((probe, 80)); found.add(sock.getsockname()[0])
        except OSError: pass
        finally: sock.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET): found.add(info[4][0])
    except (socket.gaierror, OSError): pass
    return sorted(a for a in found if not a.startswith("127."))


def private_subnets() -> list[ipaddress.IPv4Network]:
    out = []
    for address in local_ipv4_addresses():
        try: ip = ipaddress.IPv4Address(address)
        except ipaddress.AddressValueError: continue
        if ip.is_private and not ip.is_loopback and not ip.is_link_local:
            network = ipaddress.IPv4Network(f"{address}/24", strict=False)
            if network not in out: out.append(network)
    return out


async def _port_open(host: str, port: int, timeout: float) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    except (OSError, asyncio.TimeoutError): return False
    writer.close()
    try: await writer.wait_closed()
    except (OSError, asyncio.TimeoutError): pass
    return True


async def verify(url: str, timeout: float = VERIFY_TIMEOUT) -> str | None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(f"{url}/api/version")
            if response.status_code == 200: return response.json().get("version", "unknown")
    except (httpx.HTTPError, ValueError, KeyError): pass
    return None


async def scan_loopback() -> list[Found]:
    out = []
    for host in ("127.0.0.1", "[::1]"):
        url = f"http://{host}:{OLLAMA_PORT}"
        if version := await verify(url, timeout=1.5):
            out.append(Found(url, version, True)); break
    return out


async def scan_subnets(networks=None, port: int = OLLAMA_PORT, on_progress=None) -> list[Found]:
    networks = networks if networks is not None else private_subnets()
    if not networks: return []
    mine = set(local_ipv4_addresses()); hosts = [str(h) for network in networks[:2] for h in network.hosts() if str(h) not in mine]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY); done = 0
    async def check(host):
        nonlocal done
        async with semaphore: opened = await _port_open(host, port, CONNECT_TIMEOUT)
        done += 1
        if on_progress and done % 16 == 0: on_progress(done, len(hosts))
        return host if opened else None
    candidates = [host for host in await asyncio.gather(*(check(h) for h in hosts)) if host]
    versions = await asyncio.gather(*(verify(f"http://{h}:{port}") for h in candidates))
    return [Found(f"http://{host}:{port}", version, False) for host, version in zip(candidates, versions) if version]


async def discover(on_progress=None, scan_network: bool = True) -> list[Found]:
    found = await scan_loopback()
    if scan_network: found.extend(await scan_subnets(on_progress=on_progress))
    return found
