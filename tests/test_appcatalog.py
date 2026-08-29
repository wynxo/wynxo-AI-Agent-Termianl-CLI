"""The application catalog resolves against what the OS actually reports.

Every test here builds a fake catalog directory -- fake .lnk shortcuts, fake
.desktop entries, fake app bundles -- so matching, ambiguity, absence and
refresh are all tested without touching the real machine's applications. No
test may rely on a hardcoded application name existing in the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wynxo.tools.appcatalog import (
    ApplicationCatalog,
    Sources,
    condense,
    normalize_name,
)


def fake_machine(tmp_path: Path, shortcuts: dict[str, str]) -> ApplicationCatalog:
    """A Start Menu built from {name: filename} pairs, and nothing else."""
    programs = tmp_path / "Start Menu" / "Programs"
    programs.mkdir(parents=True, exist_ok=True)
    for filename in shortcuts:
        (programs / filename).write_text("", encoding="utf-8")
    return ApplicationCatalog(sources=Sources(
        shortcut_dirs=(programs,),
        use_app_paths=False,
    ))


def name_of(catalog: ApplicationCatalog, query: str) -> str | None:
    resolution = catalog.resolve(query)
    return resolution.entry.name if resolution.entry else None


# -- discovery builds the catalog from real shortcut files -------------------


def test_shortcuts_become_entries_named_after_their_file(tmp_path):
    catalog = fake_machine(tmp_path, {
        "Visual Studio Code.lnk": "", "LibreWolf.lnk": "", "Steam.lnk": "",
    })
    names = [e.name for e in catalog.entries()]
    assert names == ["LibreWolf", "Steam", "Visual Studio Code"]
    assert all(e.source == "start_menu" for e in catalog.entries())


def test_uninstall_shortcuts_are_not_applications(tmp_path):
    catalog = fake_machine(tmp_path, {
        "LibreWolf.lnk": "", "Uninstall LibreWolf.lnk": "",
    })
    assert [e.name for e in catalog.entries()] == ["LibreWolf"]


def test_a_newly_installed_application_is_found_after_a_refresh(tmp_path):
    """An application the developer never heard of, installed after the
    first scan, must appear with no source change."""
    catalog = fake_machine(tmp_path, {"Existing App.lnk": ""})
    assert catalog.resolve("Kiwano Sketchbook").status == "not_found"
    (next(iter(catalog.sources.shortcut_dirs)) / "Kiwano Sketchbook.lnk") \
        .write_text("", encoding="utf-8")
    catalog.refresh()
    assert catalog.resolve("Kiwano Sketchbook").matched is True


def test_the_catalog_is_scanned_once_and_then_cached(tmp_path, monkeypatch):
    catalog = fake_machine(tmp_path, {"App One.lnk": "", "App Two.lnk": ""})
    calls = []
    real_scan = ApplicationCatalog._scan_shortcuts

    def counting(self, offer):
        calls.append(1)
        return real_scan(self, offer)

    monkeypatch.setattr(ApplicationCatalog, "_scan_shortcuts", counting)
    catalog.entries()
    catalog.entries()
    catalog.resolve("app one")
    assert len(calls) == 1


# -- matching: the human phrases the spec asks for ---------------------------


@pytest.mark.parametrize("query", [
    "Visual Studio Code",
    "visual studio code",
    "VISUAL STUDIO CODE",
    "visual-studio-code",
    "Visual  Studio   Code",          # extra spacing
])
def test_exact_matches_ignoring_case_punctuation_and_spacing(tmp_path, query):
    catalog = fake_machine(tmp_path, {"Visual Studio Code.lnk": ""})
    assert name_of(catalog, query) == "Visual Studio Code"


def test_split_words_match_a_condensed_name(tmp_path):
    """'libre wolf' is how people say LibreWolf; the spaces fold away."""
    catalog = fake_machine(tmp_path, {"LibreWolf.lnk": ""})
    assert name_of(catalog, "libre wolf") == "LibreWolf"
    assert name_of(catalog, "Libre Wolf") == "LibreWolf"


def test_whole_word_subphrase_matches(tmp_path):
    catalog = fake_machine(tmp_path, {"Steam.lnk": "", "Visual Studio Code.lnk": ""})
    assert name_of(catalog, "studio code") == "Visual Studio Code"


def test_a_common_word_matching_several_apps_is_ambiguity_not_a_guess(tmp_path):
    catalog = fake_machine(tmp_path, {
        "Visual Studio Code.lnk": "", "Code Runner.lnk": "", "Steam.lnk": "",
    })
    resolution = catalog.resolve("code")
    assert resolution.status == "ambiguous"
    assert resolution.entry is None
    names = {c.name for c in resolution.candidates}
    assert names == {"Visual Studio Code", "Code Runner"}


def test_a_typo_still_finds_the_application(tmp_path):
    catalog = fake_machine(tmp_path, {"LibreWolf.lnk": ""})
    assert name_of(catalog, "librewlof") == "LibreWolf"


def test_an_abbreviation_finds_the_full_name(tmp_path):
    catalog = fake_machine(tmp_path, {"Visual Studio Code.lnk": ""})
    assert name_of(catalog, "vscode") == "Visual Studio Code"
    assert name_of(catalog, "vs code") == "Visual Studio Code"


@pytest.mark.parametrize("query", [
    "Zorg Editor 9000",
    "totally not installed",
    "qwertyuiop",
])
def test_no_match_is_a_clean_not_found(tmp_path, query):
    catalog = fake_machine(tmp_path, {
        "Visual Studio Code.lnk": "", "LibreWolf.lnk": "",
    })
    resolution = catalog.resolve(query)
    assert resolution.status == "not_found"
    assert resolution.entry is None
    assert resolution.candidates == ()


def test_an_ambiguous_result_never_launches_anything(tmp_path):
    catalog = fake_machine(tmp_path, {"Alpha Editor.lnk": "", "Beta Editor.lnk": ""})
    resolution = catalog.resolve("editor")
    assert resolution.status == "ambiguous"
    assert {c.name for c in resolution.candidates} == {"Alpha Editor", "Beta Editor"}


# -- what must never be matched ----------------------------------------------


@pytest.mark.parametrize("query", [
    r"C:\tools\evil.exe",
    "C:/random/path/app.lnk",
    "../somewhere/else",
    "setup.msi",
    "payload.bat",
])
def test_path_like_queries_are_refused_upfront(tmp_path, query):
    """The catalog is the only source of launch targets; a path from the
    model is arbitrary execution wearing an application's clothes."""
    catalog = fake_machine(tmp_path, {"Visual Studio Code.lnk": ""})
    assert catalog.resolve(query).status == "path_query"


def test_an_empty_query_finds_nothing(tmp_path):
    catalog = fake_machine(tmp_path, {"Steam.lnk": ""})
    assert catalog.resolve("").status == "not_found"
    assert catalog.resolve("   ").status == "not_found"


# -- other platforms' discovery, exercised with the same fakes ---------------


def test_macos_bundles_are_discovered_by_bundle_name(tmp_path):
    apps = tmp_path / "Applications"
    (apps / "Blender.app").mkdir(parents=True)
    catalog = ApplicationCatalog(sources=Sources(app_bundles=(apps,)))
    assert [e.name for e in catalog.entries()] == ["Blender"]
    assert catalog.entries()[0].source == "macos_app"
    assert name_of(catalog, "blender") == "Blender"


def test_linux_desktop_entries_use_their_display_name(tmp_path):
    apps = tmp_path / "applications"
    apps.mkdir(parents=True)
    (apps / "obsidian.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Obsidian\nExec=obsidian %U\n",
        encoding="utf-8")
    (apps / "hidden.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Secret Tool\nNoDisplay=true\n",
        encoding="utf-8")
    catalog = ApplicationCatalog(sources=Sources(desktop_dirs=(apps,)))
    names = [e.name for e in catalog.entries()]
    assert names == ["Obsidian"]
    assert name_of(catalog, "obsidian") == "Obsidian"


def test_path_executables_are_the_lowest_priority_source(tmp_path):
    """The same application reported by two sources appears once, and the
    entry a person would double-click wins."""
    programs = tmp_path / "Programs"
    programs.mkdir()
    (programs / "Steam.lnk").write_text("", encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "steam.exe").write_bytes(b"MZ")
    catalog = ApplicationCatalog(sources=Sources(
        shortcut_dirs=(programs,), path_dirs=(bindir,), use_app_paths=False))
    entries = catalog.entries()
    assert len(entries) == 1
    assert entries[0].source == "start_menu"


def test_the_windows_store_alias_directory_is_ignored(tmp_path):
    """Zero-byte execution aliases whose job is opening the Microsoft Store
    are not applications."""
    alias_dir = tmp_path / "Microsoft" / "WindowsApps"
    alias_dir.mkdir(parents=True)
    (alias_dir / "python.exe").write_bytes(b"")
    catalog = ApplicationCatalog(sources=Sources(
        path_dirs=_filtered(alias_dir), use_app_paths=False))
    assert catalog.entries() == ()


def _filtered(alias_dir: Path) -> tuple[Path, ...]:
    """Reproduce _windows_path_dirs' rule for a test-built PATH."""
    from wynxo.tools.appcatalog import _windows_path_dirs
    import os
    real_env = os.environ.get("PATH", "")
    os.environ["PATH"] = str(alias_dir)
    try:
        return _windows_path_dirs()
    finally:
        os.environ["PATH"] = real_env


# -- the normalizers, directly -----------------------------------------------


def test_normalization_folds_case_punctuation_and_accents():
    assert normalize_name("Visual Studio Code") == "visual studio code"
    assert normalize_name("Code::Blocks") == "code blocks"
    assert normalize_name("Café") == "cafe"
    assert condense("libre wolf") == "librewolf"


def test_resolution_reports_matched_for_a_single_exact_hit(tmp_path):
    catalog = fake_machine(tmp_path, {"Solo App.lnk": ""})
    resolution = catalog.resolve("solo app")
    assert resolution.status == "matched"
    assert resolution.entry is not None
    assert resolution.candidates == ()
