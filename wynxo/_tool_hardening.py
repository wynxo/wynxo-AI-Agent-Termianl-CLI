from __future__ import annotations


def install() -> None:
    from .tools.base import Tool

    original = Tool.validate
    if getattr(original, "_wynxo_strict_args", False):
        return

    def validate(self, raw: dict):
        # Tool arguments are protocol data from a model. Silently dropping an
        # unknown key can turn a malformed request into a valid request with a
        # defaulted argument, which is particularly dangerous for mutating
        # tools. Config files keep the permissive Schema.validate() behavior.
        return self.Input.validate_strict(raw)

    validate._wynxo_strict_args = True
    Tool.validate = validate


install()
