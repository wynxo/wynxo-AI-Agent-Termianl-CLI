"""A small declarative schema layer: typed fields, JSON Schema, validation.

This exists so wynxo has no compiled dependencies. Pydantic would be the
natural choice, but ``pydantic-core`` is Rust, and PyPI's aarch64 wheels are
built against glibc while Android is Bionic -- so on Termux pip falls back to
a source build that needs a Rust toolchain and routinely runs a phone out of
memory. Every other dependency here is pure Python, so this one module is
what makes ``pip install wynxo`` work on a phone the same as on a laptop.

It is deliberately small. It covers exactly what tool inputs and the config
file need: scalars, lists, nested schemas, enums, bounds, and coercion.

Coercion matters more than it looks. A local model will hand you ``"5"`` for
an integer or ``"true"`` for a boolean often enough that rejecting those
outright would fail real tool calls for no good reason.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

MISSING = object()

_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ValidationError(ValueError):
    """One or more fields did not validate."""

    def __init__(self, errors: list[tuple[str, str]]):
        self.error_list = errors
        super().__init__("; ".join(f"{loc}: {msg}" for loc, msg in errors))

    def errors(self) -> list[dict]:
        return [{"loc": (loc,), "msg": msg} for loc, msg in self.error_list]


class Field:
    """One declared field.

    ``type`` is a Python type (``str``, ``int``, ``float``, ``bool``,
    ``list``) or a ``Schema`` subclass for a nested object.
    """

    __slots__ = ("type", "description", "default", "default_factory", "ge", "le",
                 "gt", "lt", "choices", "item_type", "transform", "name")

    def __init__(
        self,
        type: Any = str,
        description: str = "",
        *,
        default: Any = MISSING,
        default_factory: Callable[[], Any] | None = None,
        ge: float | None = None,
        le: float | None = None,
        gt: float | None = None,
        lt: float | None = None,
        choices: Iterable[str] | None = None,
        item_type: Any = None,
        transform: Callable[[Any], Any] | None = None,
    ):
        self.type = type
        self.description = description
        self.default = default
        self.default_factory = default_factory
        self.ge, self.le, self.gt, self.lt = ge, le, gt, lt
        self.choices = list(choices) if choices else None
        self.item_type = item_type
        self.transform = transform
        self.name = ""

    @property
    def required(self) -> bool:
        return self.default is MISSING and self.default_factory is None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return None if self.default is MISSING else self.default

    # -- schema ------------------------------------------------------------

    def json_schema(self) -> dict:
        if isinstance(self.type, type) and issubclass(self.type, Schema):
            schema: dict[str, Any] = self.type.json_schema()
        elif self.choices:
            schema = {"type": "string", "enum": list(self.choices)}
        else:
            schema = {"type": _JSON_TYPES.get(self.type, "string")}

        if self.type is list:
            if isinstance(self.item_type, type) and issubclass(self.item_type, Schema):
                schema["items"] = self.item_type.json_schema()
            else:
                schema["items"] = {"type": _JSON_TYPES.get(self.item_type or str, "string")}
        if self.description:
            schema["description"] = self.description
        for key, value in (("minimum", self.ge), ("maximum", self.le),
                           ("exclusiveMinimum", self.gt), ("exclusiveMaximum", self.lt)):
            if value is not None:
                schema[key] = value
        return schema

    # -- validation --------------------------------------------------------

    def validate(self, value: Any, loc: str) -> Any:
        errors: list[tuple[str, str]] = []
        result = self._coerce(value, loc, errors)
        if errors:
            raise ValidationError(errors)
        if self.transform is not None:
            result = self.transform(result)
        return result

    @property
    def nullable(self) -> bool:
        """A field whose default is None accepts None.

        This is what makes a saved config load again: an unset optional is
        written as JSON null, and reading it back must not be an error.
        """
        return self.default is None

    def _coerce(self, value: Any, loc: str, errors: list) -> Any:
        if value is None:
            if self.nullable or not self.required:
                return self.get_default() if not self.nullable else None
            errors.append((loc, "must not be null"))
            return None

        target = self.type

        if isinstance(target, type) and issubclass(target, Schema):
            if isinstance(value, target):
                return value
            if not isinstance(value, dict):
                errors.append((loc, f"expected an object, got {_name(value)}"))
                return None
            try:
                return target.validate(value)
            except ValidationError as exc:
                errors.extend((f"{loc}.{l}", m) for l, m in exc.error_list)
                return None

        if target is list:
            if not isinstance(value, list):
                errors.append((loc, f"expected a list, got {_name(value)}"))
                return None
            out = []
            item_field = Field(self.item_type or str) if self.item_type else None
            for i, item in enumerate(value):
                if item_field is None:
                    out.append(item)
                    continue
                out.append(item_field._coerce(item, f"{loc}[{i}]", errors))
            return out

        if self.choices is not None:
            text = str(value)
            if text not in self.choices:
                errors.append((loc, f"must be one of {', '.join(self.choices)}; got {text!r}"))
                return None
            return text

        if target is bool:
            if isinstance(value, bool):
                return value
            # A model that writes "true" means true. Rejecting that helps nobody.
            if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
                return value.strip().lower() in ("true", "yes", "1")
            if isinstance(value, int):
                return bool(value)
            errors.append((loc, f"expected a boolean, got {_name(value)}"))
            return None

        if target is int:
            if isinstance(value, bool):
                errors.append((loc, "expected an integer, got a boolean"))
                return None
            if isinstance(value, int):
                return self._bounded(value, loc, errors)
            if isinstance(value, float) and value.is_integer():
                return self._bounded(int(value), loc, errors)
            if isinstance(value, str):
                try:
                    return self._bounded(int(value.strip()), loc, errors)
                except ValueError:
                    pass
            errors.append((loc, f"expected an integer, got {_name(value)}"))
            return None

        if target is float:
            if isinstance(value, bool):
                errors.append((loc, "expected a number, got a boolean"))
                return None
            if isinstance(value, (int, float)):
                return self._bounded(float(value), loc, errors)
            if isinstance(value, str):
                try:
                    return self._bounded(float(value.strip()), loc, errors)
                except ValueError:
                    pass
            errors.append((loc, f"expected a number, got {_name(value)}"))
            return None

        if target is str:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if isinstance(value, (dict, list)):
                # A model sometimes sends structured content where text is
                # wanted. Serialising beats failing the whole call.
                return json.dumps(value)
            errors.append((loc, f"expected a string, got {_name(value)}"))
            return None

        return value

    def _bounded(self, value, loc: str, errors: list):
        if self.ge is self.le is self.gt is self.lt is None:
            return value
        # NaN and the infinities, before the comparisons that cannot catch
        # them: every comparison against NaN is false, so `value < self.ge`
        # waved it straight through every bounded field. Python's json
        # decoder accepts the literals NaN, Infinity and -Infinity as
        # extensions, and a NaN reaching asyncio.wait_for is not a short
        # timeout but no timeout at all -- the wait never expires. An int
        # field is saved by its own coercion (int(nan) raises); a float
        # field had nothing in the way.
        #
        # Declaring a range is what asks for this check: a field with no
        # bounds is left exactly as forgiving as it was.
        if isinstance(value, float) and (value != value or value in (
                float("inf"), float("-inf"))):
            errors.append((loc, "must be a finite number"))
            return value
        if self.ge is not None and value < self.ge:
            errors.append((loc, f"must be >= {self.ge}"))
        if self.le is not None and value > self.le:
            errors.append((loc, f"must be <= {self.le}"))
        if self.gt is not None and value <= self.gt:
            errors.append((loc, f"must be > {self.gt}"))
        if self.lt is not None and value >= self.lt:
            errors.append((loc, f"must be < {self.lt}"))
        return value


def _name(value: Any) -> str:
    return {str: "a string", int: "an integer", float: "a number", bool: "a boolean",
            list: "a list", dict: "an object", type(None): "null"}.get(
        type(value), type(value).__name__)


class Schema:
    """Base for declarative models. Subclasses declare ``Field`` attributes."""

    _fields: dict[str, Field] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field] = {}
        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, "_fields", {}) or {})
        for name, value in list(vars(cls).items()):
            if isinstance(value, Field):
                value.name = name
                fields[name] = value
                delattr(cls, name)
        cls._fields = fields

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self._fields)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected field(s): {', '.join(sorted(unknown))}")
        errors: list[tuple[str, str]] = []
        for name, field in self._fields.items():
            if name in kwargs:
                value = field._coerce(kwargs[name], name, errors)
                if field.transform is not None and not errors:
                    value = field.transform(value)
            elif field.required:
                errors.append((name, "field required"))
                continue
            else:
                value = field.get_default()
            object.__setattr__(self, name, value)
        if errors:
            raise ValidationError(errors)

    # -- construction ------------------------------------------------------

    @classmethod
    def validate(cls, data: Any) -> "Schema":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValidationError([("(root)", f"expected an object, got {_name(data)}")])
        return cls(**{k: v for k, v in data.items() if k in cls._fields})

    @classmethod
    def validate_strict(cls, data: Any) -> "Schema":
        """Like ``validate`` but rejects unknown keys, for tool arguments where
        a stray key usually means the model misread the schema."""
        if not isinstance(data, dict):
            raise ValidationError([("(root)", f"expected an object, got {_name(data)}")])
        unknown = sorted(set(data) - set(cls._fields))
        if unknown:
            known = ", ".join(cls._fields)
            raise ValidationError(
                [("(root)", f"unexpected field(s) {', '.join(unknown)}; expected: {known}")])
        return cls(**data)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for name in self._fields:
            value = getattr(self, name)
            if isinstance(value, Schema):
                out[name] = value.to_dict()
            elif isinstance(value, list):
                out[name] = [v.to_dict() if isinstance(v, Schema) else v for v in value]
            else:
                out[name] = value
        return out

    @classmethod
    def json_schema(cls) -> dict:
        properties = {name: field.json_schema() for name, field in cls._fields.items()}
        required = [name for name, field in cls._fields.items() if field.required]
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields)
        return f"{type(self).__name__}({inner})"

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and self.to_dict() == other.to_dict()
