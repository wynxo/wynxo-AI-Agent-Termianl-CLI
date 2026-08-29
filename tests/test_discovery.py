"""Discovery: loopback, the machine's own LAN address, and the network.

The point under test is that Ollama answering on *this* machine's LAN IP
(a server started with OLLAMA_HOST=0.0.0.0) is suggested, not skipped --
scan_subnets deliberately excludes the scanner's own address, so scan_device
exists to probe it separately."""

from __future__ import annotations

import asyncio

from wynxo import discovery as disc


class _FakeFound:
    def __init__(self, url, version, local):
        self.url, self.version, self.local = url, version, local


class TestScanDevice:
    def test_finds_ollama_on_own_lan_addresses(self, monkeypatch):
        monkeypatch.setattr(disc, "local_ipv4_addresses",
                            lambda: ["192.168.1.50", "10.0.0.7"])

        async def fake_verify(url, timeout=None):
            return "0.5.1" if url == "http://192.168.1.50:11434" else None

        monkeypatch.setattr(disc, "verify", fake_verify)
        found = asyncio.run(disc.scan_device())
        assert len(found) == 1
        assert found[0].url == "http://192.168.1.50:11434"
        assert found[0].local and found[0].where == "this machine"
        assert found[0].version == "0.5.1"

    def test_nothing_when_verify_fails_everywhere(self, monkeypatch):
        monkeypatch.setattr(disc, "local_ipv4_addresses",
                            lambda: ["192.168.1.50"])

        async def fake_verify(url, timeout=None):
            return None

        monkeypatch.setattr(disc, "verify", fake_verify)
        assert asyncio.run(disc.scan_device()) == []

    def test_no_addresses_means_no_hits(self, monkeypatch):
        monkeypatch.setattr(disc, "local_ipv4_addresses", lambda: [])
        monkeypatch.setattr(disc, "verify",
                            lambda url, timeout=None: "0.5.1")
        assert asyncio.run(disc.scan_device()) == []


class TestDiscover:
    def test_combines_loopback_device_and_network(self, monkeypatch):
        monkeypatch.setattr(disc, "scan_loopback", _scan(["127.0.0.1"]))
        monkeypatch.setattr(disc, "scan_device", _scan(["192.168.1.50"]))
        monkeypatch.setattr(disc, "scan_subnets", _scan(["10.0.0.9"]))
        found = asyncio.run(disc.discover())
        urls = [hit.url for hit in found]
        assert urls == ["http://127.0.0.1:11434",
                        "http://192.168.1.50:11434",
                        "http://10.0.0.9:11434"]

    def test_network_scan_can_be_skipped(self, monkeypatch):
        called = []

        async def scan_subnets(*a, **k):
            called.append(True)
            return [_FakeFound("http://10.0.0.9:11434", "0.5.1", False)]

        monkeypatch.setattr(disc, "scan_loopback", _scan([]))
        monkeypatch.setattr(disc, "scan_device", _scan([]))
        monkeypatch.setattr(disc, "scan_subnets", scan_subnets)
        assert asyncio.run(disc.discover(scan_network=False)) == []
        assert called == []


def _scan(urls):
    async def scan(*a, **k):
        return [_FakeFound(f"http://{u}:11434", "0.5.1", u.startswith("127.")
                           or u.startswith("192.168.1.50"))
                for u in urls]
    return scan