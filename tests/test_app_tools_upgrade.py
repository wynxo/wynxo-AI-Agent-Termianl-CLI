"""Regression tests for application discovery and generic terminal requests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.tools import apps as apps_module
from wynxo.tools import build_registry
from wynxo.tools.appcatalog import AppEntry, ApplicationCatalog, Sources
from wynxo.tools.apps import LaunchApplication, ListApplications


def catalog_with(*entries: AppEntry) -> ApplicationCatalog:
    catalog = ApplicationCatalog(sources=Sources())
    # The catalog scanner has its own platform tests. These tests care about
    # how already-discovered entries are searched and launched, so inject the
    # exact OS result and stay independent of the runner's desktop.
    catalog._entries = tuple(entries)
    return catalog


def test_list_applications_uses_the_same_fuzzy_matching_as_launch(tmp_path):
    catalog = catalog_with(
        AppEntry("Visual Studio Code", Path("/apps/code"), "path"),
        AppEntry("Steam", Path("/apps/steam"), "path"),
    )
    tool = ListApplications(tmp_path, catalog=catalog)

    result = asyncio.run(tool.invoke({"query": "vscode"}))

    assert result.ok
    assert result.metadata["applications"] == ["Visual Studio Code"]
    assert "Visual Studio Code" in result.output


def test_list_applications_terminal_category_returns_only_supported_terminals(tmp_path):
    catalog = catalog_with(
        AppEntry("Konsole", Path("/apps/konsole"), "path"),
        AppEntry("Kitty", Path("/apps/kitty"), "path"),
        AppEntry("Calculator", Path("/apps/kcalc"), "path"),
    )
    tool = ListApplications(tmp_path, catalog=catalog)

    result = asyncio.run(tool.invoke({"query": "terminal"}))

    assert result.ok
    assert set(result.metadata["applications"]) == {"Konsole", "Kitty"}
    assert "Calculator" not in result.output


def test_generic_terminal_request_prefers_the_users_terminal(monkeypatch, tmp_path):
    catalog = catalog_with(
        AppEntry("Konsole", Path("/apps/konsole"), "path"),
        AppEntry("Kitty", Path("/apps/kitty"), "path"),
    )
    launched = []

    async def fake_launch(argv):
        launched.append(argv)

    monkeypatch.setenv("TERMINAL", "kitty")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(apps_module, "_shell_launch", fake_launch)
    tool = LaunchApplication(tmp_path, catalog=catalog)

    result = asyncio.run(tool.invoke({"query": "terminal any"}))

    assert result.ok
    assert result.metadata["application"] == "Kitty"
    assert len(launched) == 1
    assert Path(launched[0][0]).name.lower() == "kitty"


def test_generic_terminal_can_run_a_command_without_model_guessing(monkeypatch, tmp_path):
    catalog = catalog_with(
        AppEntry("Konsole", Path("/apps/konsole"), "path"),
    )
    launched = []

    async def fake_launch(argv):
        launched.append(argv)

    monkeypatch.delenv("TERMINAL", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(apps_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(apps_module, "_shell_launch", fake_launch)
    tool = LaunchApplication(tmp_path, catalog=catalog)

    result = asyncio.run(tool.invoke({
        "query": "any terminal",
        "command": "printf hello",
    }))

    assert result.ok
    assert result.metadata["application"] == "Konsole"
    assert result.metadata["command"] == "printf hello"
    assert launched
    argv = launched[0]
    assert Path(argv[0]).name.lower() == "konsole"
    assert argv[-3:-1] == ["bash", "-lc"]
    assert "printf hello" in argv[-1]


def test_generic_terminal_request_fails_honestly_when_none_is_installed(tmp_path):
    catalog = catalog_with(
        AppEntry("Calculator", Path("/apps/kcalc"), "path"),
    )
    tool = LaunchApplication(tmp_path, catalog=catalog)

    result = asyncio.run(tool.invoke({"query": "terminal"}))

    assert not result.ok
    assert result.metadata["status"] == "not_found"
    assert "terminal" in result.error.lower()


def test_registry_exposes_read_only_app_discovery_and_shares_catalog(tmp_path):
    catalog = catalog_with(
        AppEntry("Konsole", Path("/apps/konsole"), "path"),
    )
    registry = build_registry(tmp_path, allow_shell=False, app_catalog=catalog)

    discovery = registry.get("list_applications")
    launcher = registry.get("launch_application")

    assert discovery is not None
    assert launcher is not None
    assert discovery.mutating is False
    assert discovery.catalog is launcher.catalog is catalog
