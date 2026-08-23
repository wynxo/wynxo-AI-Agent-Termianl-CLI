"""Finding Ollama on the network.

Guessing hostnames (``homelab``, ``nas``, ``ollama.local``) is a bad bet: it
depends on the user's DNS or mDNS being set up the way you assumed, and on a
phone there is usually no mDNS at all. Scanning the actual subnet for an open
11434 finds the real machine regardless of what it is called.

A /24 sweep is 254 connections at a 0.3s timeout, run concurrently -- a couple
of seconds in practice.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass

OLLAMA_PORT = 11434
MAX_CONCURRENCY = 128
CONNECT_TIMEOUT = 0.35
VERIFY_TIMEOUT = 2.0


@dataclass
class Found:
    url: str
    version: str
    local: bool

    @property
    def where(self) -> str:
        return "this machine" if self.local else "network"


def local_ipv4_addresses() -> list[str]:
    """This machine's IPv4 addresses on real networks.

    The UDP-connect trick is the only portable way to learn which interface
    would be used for outbound traffic; no packet is actually sent.
    """
    found: set[str] = set()

    for probe in ("8.8.8.8", "1.1.1.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(0.2)
            sock.connect((probe, 80))
            found.add(sock.getsockname()[0])
        except OSError:
            pass
        finally:
            sock.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except (socket.gaierror, OSError):
        pass

    return sorted(a for a in found if not a.startswith("127."))


def private_subnets() -> list[ipaddress.IPv4Network]:
    """The /24 networks this machine sits on, private ranges only.

    Scanning is limited to RFC1918 space: sweeping a public range would be
    both useless and rude.
    """
    networks = []
    for address in local_ipv4_addresses():
        try:
            ip = ipaddress.IPv4Address(address)
        except ipaddress.AddressValueError:
            continue
        if not ip.is_private or ip.is_loopback or ip.is_link_local:
            continue
        network = ipaddress.IPv4Network(f"{address}/24", strict=False)
        if network not in networks:
            networks.append(network)
    return networks


async def _port_open(host: str, port: int, timeout: float) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except (OSError, asyncio.TimeoutError):
        pass
    return True


async def verify(url: str, timeout: float = VERIFY_TIMEOUT) -> str | None:
    """Confirm something at ``url`` is really Ollama. Returns its version."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(f"{url}/api/version")
            if response.status_code == 200:
                return response.json().get("version", "unknown")
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    return None


async def scan_loopback() -> list[Found]:
    """Check this machine first -- by far the most common answer."""
    out = []
    for host in ("127.0.0.1", "[::1]"):
        url = f"http://{host}:{OLLAMA_PORT}"
        if version := await verify(url, timeout=1.5):
            out.append(Found(url=url, version=version, local=True))
            break   # the same server on both stacks is one server
    return out


async def scan_subnets(
    networks: list[ipaddress.IPv4Network] | None = None,
    port: int = OLLAMA_PORT,
    on_progress=None,
) -> list[Found]:
    """Sweep the local /24s for an open Ollama port."""
    networks = networks if networks is not None else private_subnets()
    if not networks:
        return []

    mine = set(local_ipv4_addresses())
    hosts: list[str] = []
    for network in networks[:2]:      # two interfaces is already unusual
        hosts.extend(str(h) for h in network.hosts() if str(h) not in mine)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    done = 0
    total = len(hosts)

    async def check(host: str) -> str | None:
        nonlocal done
        async with semaphore:
            open_ = await _port_open(host, port, CONNECT_TIMEOUT)
        done += 1
        if on_progress and done % 16 == 0:
            on_progress(done, total)
        return host if open_ else None

    results = await asyncio.gather(*(check(h) for h in hosts))
    candidates = [h for h in results if h]

    # An open port is not proof; ask each one whether it is Ollama.
    found: list[Found] = []
    verified = await asyncio.gather(
        *(verify(f"http://{h}:{port}") for h in candidates))
    for host, version in zip(candidates, verified):
        if version:
            found.append(Found(url=f"http://{host}:{port}", version=version, local=False))
    return found


async def discover(on_progress=None, scan_network: bool = True) -> list[Found]:
    """Loopback first, then the LAN. Returns everything that answered."""
    found = await scan_loopback()
    if scan_network:
        found.extend(await scan_subnets(on_progress=on_progress))
    return found
