from wynxo.permissions import PermissionStore, is_read_only_command
from wynxo.scope import Mode


def test_auto_always_prompts_for_github_writes():
    store = PermissionStore(mode=Mode.AUTO)
    assert store.needs_prompt("github_write", True, {"operation": "write"})


def test_remote_writes_cannot_be_remembered_as_always_allowed():
    store = PermissionStore(mode=Mode.AUTO)
    store.remember("github_write", {"operation": "write"})
    assert store.needs_prompt("github_write", True, {"operation": "write"})


def test_safe_pipeline_does_not_need_prompt():
    assert is_read_only_command("ls | head -10")
    assert is_read_only_command("git status | head -20")


def test_pipeline_with_mutation_is_not_read_only():
    assert not is_read_only_command("ls | rm -f build.txt")
