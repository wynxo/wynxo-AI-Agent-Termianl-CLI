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

    def test_aliases(self):
        assert resolve("med").name == "medium"
        assert resolve("MAX").name == "max"
        assert resolve("xh").name == "xhigh"

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
