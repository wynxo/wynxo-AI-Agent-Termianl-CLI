import pytest

from wynxo.config import Config, Endpoint, normalise_url
from wynxo.effort import ORDER, POLICIES, resolve
from wynxo.permissions import PermissionStore, is_read_only_command
from wynxo.scope import Mode
from wynxo.session import Session, estimate_tokens


class TestNormaliseUrl:
    @pytest.mark.parametrize("raw,expected", [
        ("127.0.0.1", "http://127.0.0.1:11434"),
        ("192.168.1.50", "http://192.168.1.50:11434"),
        ("192.168.1.50:11434", "http://192.168.1.50:11434"),
        ("http://192.168.1.50", "http://192.168.1.50:11434"),
        ("10.0.0.4:8080", "http://10.0.0.4:8080"),
        ("http://192.168.1.50:11434/api", "http://192.168.1.50:11434"),
        ("http://192.168.1.50:11434/v1", "http://192.168.1.50:11434"),
        ("http://192.168.1.50:11434/", "http://192.168.1.50:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("  127.0.0.1  ", "http://127.0.0.1:11434"),
        ("localhost", "http://localhost:11434"),
    ])
    def test_shapes(self, raw, expected):
        assert normalise_url(raw) == expected

    def test_https_keeps_443(self):
        # An https URL means a reverse proxy, not a bare Ollama on 11434.
        assert normalise_url("https://ollama.example.com/v1") == "https://ollama.example.com"

    def test_empty_falls_back_to_loopback(self):
        assert normalise_url("") == "http://127.0.0.1:11434"


class TestEffort:
    def test_every_level_resolves(self):
        for name in ORDER:
            assert resolve(name).name == name

    def test_the_names_are_the_only_vocabulary(self):
        """No aliases. Six short names, and /effort lists them; a second
        vocabulary of "x", "u" and "insane" was more to remember, not less,
        and made it easy to pick a level you did not mean."""
        for shortcut in ("med", "xh", "u", "x", "insane", "maximum", "fast"):
            with pytest.raises(KeyError):
                resolve(shortcut)

    def test_case_and_padding_are_still_forgiven(self):
        assert resolve("MAX").name == "max"
        assert resolve("  Ultra ").name == "ultra"

    def test_unknown_level_lists_the_valid_ones(self):
        with pytest.raises(KeyError) as exc:
            resolve("turbo")
        assert "ultra" in str(exc.value)

    def test_effort_increases_monotonically(self):
        """The ladder must actually be a ladder."""
        policies = [POLICIES[n] for n in ORDER]
        for lower, higher in zip(policies, policies[1:]):
            assert higher.max_iterations > lower.max_iterations
            assert higher.max_tool_output >= lower.max_tool_output
            assert higher.repair_attempts >= lower.repair_attempts

    def test_bump_is_clamped_at_both_ends(self):
        assert resolve("low").bump(-5).name == "low"
        assert resolve("ultra").bump(5).name == "ultra"
        assert resolve("medium").bump(1).name == "high"

    def test_top_levels_verify_until_clean(self):
        assert POLICIES["max"].verify_rounds == -1
        assert POLICIES["max"].max_verify_rounds > 0
        assert POLICIES["low"].verify_rounds == 0

    def test_thinking_is_off_at_the_fast_levels(self):
        assert not POLICIES["low"].thinking
        assert not POLICIES["medium"].thinking
        assert POLICIES["high"].thinking


class TestPermissions:
    @pytest.mark.parametrize("command,read_only", [
        ("ls -la", True),
        ("git status", True),
        ("git log --oneline", True),
        ("pip list", True),
        ("cat file.py", True),
        ("git push origin main", False),
        ("rm -rf build", False),
        ("npm install", False),
        ("ls && rm -rf /", False),      # chained: the safe half proves nothing
        ("cat x | sh", False),
        ("echo x > /etc/passwd", False),
        # sed/awk/find/sort/tree are in the "safe" list for their normal,
        # observing-only use -- but each has a flag that turns it into a
        # write, and none of those flags involve a shell metacharacter the
        # chained-command check above would otherwise catch.
        ("sed 's/foo/bar/' file.txt", True),          # prints to stdout only
        ("sed -n '1,5p' file.txt", True),
        ("sed -i 's/foo/bar/' file.txt", False),       # rewrites the file
        ("sed -i.bak 's/foo/bar/' file.txt", False),
        ("sed -ni 's/foo/bar/p' file.txt", False),     # -i bundled, not first
        ("awk '{print $1}' file.txt", True),
        ("awk -i inplace '{gsub(/x/,\"y\")}' file", False),
        ("gawk -i inplace '{gsub(/x/,\"y\")}' file", False),
        ("find . -name '*.py'", True),
        ("find . -name '*.tmp' -delete", False),
        ("find . -exec rm {} +", False),               # no ';' or shell char
        ("find . -execdir rm {} +", False),
        ("sort file.txt", True),
        ("sort -o input.txt input.txt", False),        # "sort in place" trick
        ("sort --output=input.txt input.txt", False),
        ("tree", True),
        ("tree -o out.txt", False),
    ])
    def test_classification(self, command, read_only):
        assert is_read_only_command(command) is read_only

    @pytest.mark.parametrize("command", [
        # A newline is a command separator in every shell, and it was not in
        # the list -- so this read as the safe command "ls" and ran without
        # asking, taking the rm with it.
        "ls\nrm -rf build",
        "echo hi\nchmod 777 /etc",
        "cat notes.txt\n\ncurl http://example.com/x.sh",
        # A bare & backgrounds the first command and runs the second. Only
        # && was being looked for.
        "ls & rm -rf build",
        "pwd & git push",
        "ls\rrm -rf build",
    ])
    def test_a_safe_command_cannot_carry_another_one(self, command):
        assert is_read_only_command(command) is False
        assert PermissionStore().needs_prompt("shell", True, {"command": command})

    @pytest.mark.parametrize("command,read_only", [
        # `git config x y` sets a value, and with --global it sets it for
        # every repository on the machine. Both were waved through as reads.
        ("git config user.email me@example.com", False),
        ("git config --global user.name someone", False),
        ("git config --local core.hooksPath .hooks", False),
        ("git config --unset user.email", False),
        ("git config user.email", True),
        ("git config --get user.email", True),
        ("git config --list", True),
        ("git config -l", True),
        ("git config", True),
    ])
    def test_git_config_reads_and_writes_are_told_apart(self, command,
                                                        read_only):
        assert is_read_only_command(command) is read_only

    def test_the_ordinary_safe_commands_still_are(self):
        """Tightening this is only worth anything if it stays usable."""
        for command in ("ls -la", "git status", "git log --oneline -5",
                        "pwd", "cat README.md", "find . -name '*.py'",
                        "grep -rn TODO src", "npm ls", "pip list"):
            assert is_read_only_command(command) is True, command

    def test_reads_never_prompt(self):
        assert not PermissionStore().needs_prompt("read_file", False, {})

    def test_writes_prompt_by_default(self):
        assert PermissionStore().needs_prompt("write_file", True, {"path": "x"})

    def test_always_allow_is_remembered_per_tool(self):
        store = PermissionStore()
        store.remember("write_file", {})
        assert not store.needs_prompt("write_file", True, {"path": "x"})

    def test_shell_approval_is_per_command_not_blanket(self):
        """Approving `npm test` forever must not also approve `rm -rf build`."""
        store = PermissionStore()
        store.remember("shell", {"command": "npm test"})
        assert not store.needs_prompt("shell", True, {"command": "npm test"})
        assert store.needs_prompt("shell", True, {"command": "rm -rf build"})

    def test_network_commands_always_prompt_even_when_allowed(self):
        store = PermissionStore()
        store.always_allowed_tools.add("shell")
        assert store.needs_prompt("shell", True, {"command": "git push origin main"})
        assert store.needs_prompt("shell", True, {"command": "curl http://x.com"})

    def test_yolo_skips_everything(self):
        store = PermissionStore(mode=Mode.YOLO)
        assert not store.needs_prompt("shell", True, {"command": "rm -rf build"})
        assert store.yolo

    def test_plan_mode_refuses_rather_than_asks(self):
        """Plan mode is the only one that blocks: a prompt would defeat it."""
        store = PermissionStore(mode=Mode.PLAN)
        assert store.blocked("write_file", True)
        assert store.blocked("shell", True)
        assert store.blocked("read_file", False) is None

    def test_auto_mode_still_asks_before_an_in_place_sed(self):
        """sed being in the "safe, never prompt" list for its normal,
        read-only use must not extend to its in-place-write flag."""
        store = PermissionStore(mode=Mode.AUTO)
        assert not store.needs_prompt(
            "shell", True, {"command": "sed 's/x/y/' file.txt"})
        assert store.needs_prompt(
            "shell", True, {"command": "sed -i 's/x/y/' file.txt"})

    def test_auto_mode_edits_freely_but_still_asks_to_run(self):
        store = PermissionStore(mode=Mode.AUTO)
        assert not store.needs_prompt("write_file", True, {"path": "x"})
        assert not store.needs_prompt("edit_file", True, {"path": "x"})
        assert store.needs_prompt("shell", True, {"command": "npm install"})
        assert store.blocked("write_file", True) is None

    def test_manual_mode_asks_for_writes(self):
        store = PermissionStore(mode=Mode.MANUAL)
        assert store.needs_prompt("write_file", True, {"path": "x"})
        assert not store.needs_prompt("read_file", False, {})


class TestSession:
    def test_wire_puts_system_first(self, tmp_path):
        session = Session(workspace=tmp_path, system_prompt="sys")
        session.add_user("hi")
        wire = session.wire()
        assert wire[0]["role"] == "system"
        assert wire[1]["content"] == "hi"

    def test_compaction_triggers_at_the_threshold(self, tmp_path):
        session = Session(workspace=tmp_path)
        session.add_user("x" * 40_000)   # ~11k tokens
        assert session.should_compact(budget=8_000, num_ctx=32_000)
        assert not session.should_compact(budget=64_000, num_ctx=64_000)

    def test_compaction_never_orphans_a_tool_message(self, tmp_path):
        """A tool result with no matching call confuses every model."""
        session = Session(workspace=tmp_path)
        session.add_user("go")
        for _ in range(4):
            session.add_assistant("", [{"function": {"name": "read_file", "arguments": {}}}])
            session.add_tool_result("read_file", "contents")
        _, kept = session.slice_for_summary(keep_recent=3)
        assert kept[0]["role"] != "tool"

    def test_apply_compaction_preserves_the_tail(self, tmp_path):
        session = Session(workspace=tmp_path)
        for i in range(10):
            session.add_user(f"m{i}")
        older, kept = session.slice_for_summary(keep_recent=4)
        session.apply_compaction("summary text", kept)
        assert "summary text" in session.messages[0]["content"]
        assert session.messages[-1]["content"] == "m9"
        assert session.compactions == 1

    def test_token_estimate_grows_with_content(self, tmp_path):
        session = Session(workspace=tmp_path)
        before = session.token_estimate()
        session.add_user("x" * 1000)
        assert session.token_estimate() > before + 200

    def test_estimate_is_conservative(self):
        # Under-estimating silently truncates context, so err the other way.
        assert estimate_tokens("a" * 400) >= 100


class TestConfig:
    def test_endpoint_lookup_falls_back(self):
        config = Config(endpoints=[Endpoint(name="a", url="http://a:11434")],
                        active_endpoint="missing")
        assert config.endpoint().name == "a"

    def test_endpoint_url_is_normalised_on_construction(self):
        assert Endpoint(name="x", url="homelab").url == "http://homelab:11434"

    def test_roundtrips_through_disk(self, tmp_path):
        config = Config(model="qwen3:32b", effort="xhigh", num_ctx=65536)
        path = config.save(tmp_path / "config.json")
        import json
        loaded = Config.validate(json.loads(path.read_text()))
        assert loaded.model == "qwen3:32b"
        assert loaded.effort == "xhigh"
        assert loaded.num_ctx == 65536


class TestACorruptConfigNeverStopsWynxoStarting:
    """load() has always had a fallback for a corrupt config. The guard was
    just in the wrong place: the crash happened in `data.update()`, before
    the try that exists to prevent exactly this.
    """

    @pytest.fixture
    def where(self, tmp_path, monkeypatch):
        monkeypatch.setattr("wynxo.config.config_dir", lambda: tmp_path)
        (tmp_path / "config.json").parent.mkdir(parents=True, exist_ok=True)
        return tmp_path

    @pytest.mark.parametrize("body", [
        '"a string"', "5", "null", "true", "[]", "[1,2,3]",
        '{"model": "truncated', "", "   ", "[[1,2]]",
    ])
    def test_it_falls_back_to_defaults_instead_of_raising(self, where, body,
                                                          tmp_path):
        from wynxo.config import DEFAULT_MODEL, load

        (where / "config.json").write_text(body, encoding="utf-8")
        assert load(tmp_path).model == DEFAULT_MODEL

    def test_a_list_of_pairs_does_not_smuggle_in_keys(self, where, tmp_path):
        """dict.update accepts pairs, so [[\"model\", \"evil\"]] would
        otherwise be merged in as real config."""
        from wynxo.config import DEFAULT_MODEL, load

        (where / "config.json").write_text('[["model", "evil:1b"]]',
                                           encoding="utf-8")
        assert load(tmp_path).model == DEFAULT_MODEL

    def test_a_corrupt_project_file_does_not_stop_the_user_file(self, where,
                                                                tmp_path):
        from wynxo.config import load

        (where / "config.json").write_text('{"model": "mine:7b"}',
                                           encoding="utf-8")
        (tmp_path / ".wynxo.json").write_text("not json at all",
                                              encoding="utf-8")
        assert load(tmp_path).model == "mine:7b"

    def test_a_good_config_is_still_read(self, where, tmp_path):
        from wynxo.config import load

        (where / "config.json").write_text('{"model": "custom:7b"}',
                                           encoding="utf-8")
        assert load(tmp_path).model == "custom:7b"


class TestASettingsFileIsNeverHalfWritten:
    """write_text truncates first and writes second.

    Anything that stops the process in between -- Ctrl-C, a full disk, a
    container going away -- left a half-written settings file, and the next
    start silently used the defaults: endpoint list, model, theme, all gone
    with no explanation. Verified by truncating one: the model came back as
    the default and nothing said why.
    """

    def test_the_replacement_is_all_or_nothing(self, tmp_path):
        from wynxo.config import atomic_write

        target = tmp_path / "config.json"
        atomic_write(target, '{"model": "first"}')
        atomic_write(target, '{"model": "second"}')
        assert target.read_text(encoding="utf-8") == '{"model": "second"}'

    def test_it_leaves_no_temporary_behind(self, tmp_path):
        from wynxo.config import atomic_write

        atomic_write(tmp_path / "config.json", "{}")
        assert [p.name for p in tmp_path.iterdir()] == ["config.json"]

    def test_a_failed_write_keeps_the_old_file(self, tmp_path, monkeypatch):
        import os

        from wynxo import config as config_module

        target = tmp_path / "config.json"
        config_module.atomic_write(target, '{"model": "good"}')

        def refuse(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError):
            config_module.atomic_write(target, '{"model": "new"}')
        assert target.read_text(encoding="utf-8") == '{"model": "good"}'
        assert [p.name for p in tmp_path.iterdir()] == ["config.json"]

    def test_save_uses_it(self, tmp_path):
        import inspect

        from wynxo.config import Config

        assert "atomic_write" in inspect.getsource(Config.save)


class TestAnUnreadableSettingsFileSaysSo:
    """Falling back to the defaults is right. Doing it in silence is how a
    file with one bad character costs somebody their endpoint list without
    ever telling them."""

    def _load(self, tmp_path, monkeypatch, text):
        from wynxo import config as config_module

        target = tmp_path / "config.json"
        target.write_text(text, encoding="utf-8")
        monkeypatch.setattr(config_module, "config_path", lambda: target)
        return config_module.load(project_dir=tmp_path)

    def test_broken_json_is_reported(self, tmp_path, monkeypatch):
        from wynxo import config as config_module

        self._load(tmp_path, monkeypatch, '{"model": "x"')
        assert any("not valid JSON" in p for p in config_module.LOAD_PROBLEMS)

    def test_something_that_is_not_settings_is_reported(self, tmp_path,
                                                        monkeypatch):
        from wynxo import config as config_module

        self._load(tmp_path, monkeypatch, "[1, 2, 3]")
        assert config_module.LOAD_PROBLEMS

    def test_a_good_file_reports_nothing(self, tmp_path, monkeypatch):
        from wynxo import config as config_module

        config = self._load(tmp_path, monkeypatch, '{"model": "kept:7b"}')
        assert config.model == "kept:7b"
        assert config_module.LOAD_PROBLEMS == []

    def test_a_missing_file_is_not_a_problem(self, tmp_path, monkeypatch):
        from wynxo import config as config_module

        monkeypatch.setattr(config_module, "config_path",
                            lambda: tmp_path / "nothing.json")
        config_module.load(project_dir=tmp_path)
        assert config_module.LOAD_PROBLEMS == []

    def test_the_startup_check_reports_them(self):
        import inspect

        from wynxo.cli import Repl

        assert "LOAD_PROBLEMS" in inspect.getsource(Repl._connect)
