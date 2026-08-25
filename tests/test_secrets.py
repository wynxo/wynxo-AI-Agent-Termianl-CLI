"""Keeping credentials out of the model's context and out of the logs.

The risk is concrete rather than theoretical: wynxo reads real files and
sends them to a model that is often on another machine (`--endpoint
192.168.1.50:11434` is the whole point of the endpoint list), and it writes
every tool result to a transcript kept for twenty sessions.

Two halves matter equally here. Catching secrets is the obvious one. Not
catching things that merely look like secrets is the other, and it is the
one that decides whether the feature is usable: an agent that masks hashes,
lockfile checksums and function calls cannot read its own project.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wynxo.secrets import Ignore, Shield, is_secret_file, redact


class TestWhichFilesAreCredentials:
    @pytest.mark.parametrize("name", [
        ".env", ".env.local", ".env.production", ".envrc", ".netrc",
        "id_rsa", "id_ed25519", "server.pem", "private.key", "keystore.jks",
        "credentials.json", "secrets.yaml", ".npmrc", ".pypirc",
        ".ssh/config", ".aws/credentials", ".gnupg/secring.gpg",
    ])
    def test_these_are(self, name):
        assert is_secret_file(Path(name)) is True

    @pytest.mark.parametrize("name", [
        "app.py", "README.md", ".env.example", ".env.sample",
        ".env.template", "id_rsa.pub", "package.json", "keyboard.py",
        "tokenizer.py", "settings.py", "public.pem.example",
    ])
    def test_these_are_not(self, name):
        assert is_secret_file(Path(name)) is False, (
            f"{name} would be refused, and it is not a secret")


class TestRedactionCatchesRealCredentials:
    @pytest.mark.parametrize("line", [
        'AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY',
        'api_key = "sk-proj-abc123def456ghi789jkl"',
        'STRIPE_KEY=sk_live_51HxxxxxxxxxxxxxxYY',
        'token: ghp_16C7e42F292c6912E7710c838347Ae178B4a',
        'password: "hunter2andmore"',
        'DATABASE_PASSWORD=s3cr3tp4ssw0rd',
        'apiKey: "abcdef1234567890abcdef"',
        'client_secret: GOCSPXabcdefghijklmnop',
        'AKIAIOSFODNN7EXAMPLE',
        'xoxb-123456789012-abcdefghijkl',
        'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.sig',
        'DATABASE_URL=postgres://user:realpassword@host:5432/db',
        'redis://default:AbCdEf123456@redis-host:6379',
    ])
    def test_it_is_masked(self, line):
        cleaned, count = redact(line)
        assert count > 0, f"missed: {line}"
        assert "[redacted by wynxo]" in cleaned

    def test_a_private_key_block_is_replaced_whole(self):
        body = ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
                "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU\n"
                "-----END OPENSSH PRIVATE KEY-----")
        cleaned, count = redact(body)
        assert count == 1
        assert "b3BlbnNzaC1rZXk" not in cleaned

    def test_the_name_survives_so_the_code_still_reads(self):
        """Masking the whole line would hide that the setting exists."""
        cleaned, _ = redact('API_KEY = "sk-proj-abc123def456ghi789"')
        assert "API_KEY" in cleaned


class TestRedactionLeavesOrdinaryCodeAlone:
    """The half that decides whether this is usable."""

    @pytest.mark.parametrize("line", [
        'def get_api_key():',
        'API_KEY = os.environ["API_KEY"]',
        'token = process.env.TOKEN',
        'const token = useToken();',
        'api_key = None',
        'password = ""',
        'API_KEY=your_api_key_here',
        'SECRET_KEY = "changeme"',
        'password: <your-password>',
        'API_KEY={{ vault_key }}',
        'DB_PASSWORD=${DB_PASSWORD}',
        'sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b"',
        'tokenizer = AutoTokenizer.from_pretrained("bert-base")',
        'monkey_patch = True',
        'keyboard_layout = "qwerty"',
        'MAX_TOKENS = 4096',
        'self.tokens = []',
        'token_name = "session"',
        'key_path = "/etc/ssl/private"',
        'PUBLIC_KEY=ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAB',
        'https://github.com/wynxo/repo.git',
        'mongodb://localhost:27017/mydb',
        'import hashlib',
        'key = row["key"]',
    ])
    def test_it_is_untouched(self, line):
        cleaned, count = redact(line)
        assert count == 0, f"false positive on: {line} -> {cleaned}"
        assert cleaned == line

    @pytest.mark.parametrize("line", [
        # Unquoted values that are names being referred to, not credentials.
        # Masking these rewrites working code into something that cannot run.
        'tokens=self.tokens',
        'key_bindings=bindings',
        'protect_secrets = enabled',
        'api_key = default_key',
        'secret: SETTINGS',
        'password=hashed',
        # Prose about code. A doc comment saying `token=self.token` is not a
        # leak, and neither is an escape sequence inside a test fixture.
        '# the setting `token=self.token` is read here',
        '"export TOKEN=keepme\\n"',
    ])
    def test_a_name_is_not_a_secret(self, line):
        cleaned, count = redact(line)
        assert count == 0, f"false positive on: {line} -> {cleaned}"

    def test_a_long_run_of_letters_is_still_a_secret(self):
        """The exemption above is shaped, not blanket: a bare unquoted value
        can still be a key, and `GOCSPX...` is exactly that."""
        _, count = redact("client_secret: GOCSPXabcdefghijklmnop")
        assert count == 1

    def test_the_whole_package_survives_unchanged(self):
        """One module proves little -- the redactor is only usable if it can
        read every file wynxo might be asked to open, including its own."""
        for source_file in sorted(Path("wynxo").rglob("*.py")):
            source = source_file.read_text(encoding="utf-8")
            cleaned, count = redact(source)
            assert count == 0, f"wynxo cannot read {source_file}"
            assert cleaned == source


class TestTheShieldInTheTools:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / ".env").write_text("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI\n")
        (tmp_path / "id_rsa").write_text(
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blb\n"
            "-----END OPENSSH PRIVATE KEY-----\n")
        (tmp_path / ".env.example").write_text("AWS_SECRET_ACCESS_KEY=xxx\n")
        (tmp_path / "settings.py").write_text(
            'DEBUG = True\nAPI_KEY = "sk-proj-abcdefghij1234567890"\n')
        return tmp_path

    def read(self, workspace, name, shield=None):
        from wynxo.tools.files import ReadFile, ReadInput

        tool = ReadFile(workspace=workspace, shield=shield)
        return asyncio.run(tool.run(ReadInput(path=name)))

    def test_a_credentials_file_is_refused(self, project):
        result = self.read(project, ".env")
        assert result.ok is False
        assert "wJalrXUtnFEMI" not in result.output

    def test_a_private_key_file_is_refused(self, project):
        assert self.read(project, "id_rsa").ok is False

    def test_a_sample_env_is_still_readable(self, project):
        """It is how a model learns which variables a project expects."""
        assert self.read(project, ".env.example").ok is True

    def test_an_ordinary_file_is_read_with_the_secret_masked(self, project):
        result = self.read(project, "settings.py")
        assert result.ok is True
        assert "DEBUG = True" in result.output
        assert "sk-proj-abcdefghij1234567890" not in result.output

    def test_the_model_is_told_something_was_masked(self, project):
        """Silently altering what it reads would make it reason about a file
        that does not exist."""
        assert "masked" in self.read(project, "settings.py").output

    def test_refusal_does_not_confirm_the_file_exists(self, project):
        """Which secrets a project has is itself worth withholding."""
        missing = self.read(project, ".env.production")
        assert missing.ok is False
        assert "does not exist" not in missing.output

    def test_grep_cannot_read_it_a_line_at_a_time(self, project):
        """A grep is a read with extra steps: matching inside a credentials
        file would hand the secret over a line at a time."""
        from wynxo.tools.search import Grep, GrepInput

        tool = Grep(workspace=project)
        # Searched for by its *name*, so a hit would print the value beside
        # it. The pattern itself is echoed back either way, and that is not
        # a leak -- the model is the one who wrote it.
        result = asyncio.run(tool.run(GrepInput(pattern="AWS_SECRET")))
        assert "wJalrXUtnFEMI" not in (result.output or "")

    def test_grep_masks_secrets_in_ordinary_files(self, project):
        from wynxo.tools.search import Grep, GrepInput

        tool = Grep(workspace=project)
        result = asyncio.run(tool.run(GrepInput(pattern="API_KEY")))
        assert "sk-proj-abcdefghij1234567890" not in (result.output or "")

    def test_a_tool_built_without_a_shield_still_protects(self, project):
        """No shield given must mean the protective one, not none -- a tool
        built by future code should not leak by default."""
        assert self.read(project, ".env").ok is False

    def test_turning_it_off_really_turns_it_off(self, project):
        result = self.read(project, ".env", shield=Shield.off(project))
        assert result.ok is True and "wJalrXUtnFEMI" in result.output

    def test_an_allowed_path_is_readable_again(self, project):
        shield = Shield(project)
        shield.allow(".env")
        assert self.read(project, ".env", shield=shield).ok is True


class TestTheIgnoreFile:
    def test_a_listed_glob_is_refused(self, tmp_path):
        (tmp_path / ".wynxoignore").write_text("*.log\nprivate/\n")
        (tmp_path / "debug.log").write_text("secret trace")
        shield = Shield(tmp_path)
        assert shield.blocks(tmp_path / "debug.log")

    def test_a_negation_puts_something_back(self, tmp_path):
        (tmp_path / ".wynxoignore").write_text("*.log\n!keep.log\n")
        shield = Shield(tmp_path)
        assert shield.blocks(tmp_path / "other.log")
        assert not shield.blocks(tmp_path / "keep.log")

    def test_comments_and_blanks_are_skipped(self, tmp_path):
        (tmp_path / ".wynxoignore").write_text("# a comment\n\n*.log\n")
        assert not Shield(tmp_path).blocks(tmp_path / "app.py")

    def test_a_missing_ignore_file_is_fine(self, tmp_path):
        assert Ignore.load(tmp_path).patterns == []

    def test_a_directory_name_matches_anywhere(self, tmp_path):
        (tmp_path / ".wynxoignore").write_text("vault\n")
        shield = Shield(tmp_path)
        assert shield.blocks(tmp_path / "deep" / "vault" / "key.txt")


class TestTheJournalIsScrubbedToo:
    """The quiet half of the same leak: the log keeps tool results for
    twenty sessions, and outlives any promise of a clean uninstall."""

    def test_a_secret_in_a_tool_result_is_not_written(self, tmp_path,
                                                      monkeypatch):
        from wynxo import journal as journal_module

        monkeypatch.setattr(journal_module, "data_dir", lambda: tmp_path)
        log = journal_module.Journal.open("test")
        log.tool_result("read_file", True,
                        'API_KEY = "sk_live_51Hxxxxxxxxxxxxxx"')
        assert "sk_live_51Hxxxxxxxxxxxxxx" not in log.path.read_text()

    def test_a_secret_the_user_typed_is_not_written(self, tmp_path,
                                                    monkeypatch):
        from wynxo import journal as journal_module

        monkeypatch.setattr(journal_module, "data_dir", lambda: tmp_path)
        log = journal_module.Journal.open("test")
        log.user("my token is ghp_16C7e42F292c6912E7710c838347Ae178B4a")
        assert "ghp_16C7e42F292c6912E7710c838347Ae178B4a" not in \
            log.path.read_text()

    def test_nested_tool_arguments_are_scrubbed(self, tmp_path, monkeypatch):
        from wynxo import journal as journal_module

        monkeypatch.setattr(journal_module, "data_dir", lambda: tmp_path)
        log = journal_module.Journal.open("test")
        log.tool("shell", {"command": "export T=ghp_16C7e42F292c6912E7710c83"})
        assert "ghp_16C7e42F292c6912E7710c83" not in log.path.read_text()


class TestTheSetting:
    def test_it_is_on_by_default(self):
        from wynxo.config import Config

        assert Config().protect_secrets is True

    def test_it_is_listed_in_help(self):
        from wynxo.cli import COMMANDS

        assert "/secrets" in COMMANDS
