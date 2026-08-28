from __future__ import annotations


def install() -> None:
    from . import testing

    original = testing._node_agent
    if getattr(original, "_wynxo_bun_lock", False):
        return

    def node_agent(root):
        if (root / "bun.lock").is_file():
            return "bun"
        return original(root)

    node_agent._wynxo_bun_lock = True
    testing._node_agent = node_agent


install()
