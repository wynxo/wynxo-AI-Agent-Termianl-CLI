"""The boundary coercion helpers and the declarative Schema model they
back — everything the tools trust, so its behaviour is locked in here."""

from __future__ import annotations

import pytest

from wynxo.coerce import as_float, as_int, as_list, as_text
from wynxo.schema import Field, Schema, ValidationError


class TestAsText:
    def test_none_and_bool_are_absent(self):
        assert as_text(None) == ""
        assert as_text(True) == ""

    def test_string_passes_through(self):
        assert as_text("hi") == "hi"

    def test_number_and_object_are_stringified(self):
        assert as_text(42) == "42"

    def test_wrapped_in_a_dict_is_unwrapped(self):
        assert as_text({"content": "think"}) == "think"
        assert as_text({"value": 7}) == "7"

    def test_list_is_flattened(self):
        assert as_text(["a", "b"]) == "ab"

    def test_unknown_dict_payload_is_absent(self):
        assert as_text({"other": "x"}) == ""


class TestAsList:
    def test_list_and_tuple_pass(self):
        assert as_list([1, 2]) == [1, 2]
        assert as_list((1, 2)) == [1, 2]

    def test_single_object_is_wrapped(self):
        # Some shims send one tool call unwrapped.
        assert as_list({"name": "x"}) == [{"name": "x"}]

    def test_other_things_are_an_empty_list(self):
        assert as_list("nope") == []
        assert as_list(None) == []


class TestAsInt:
    def test_bool_is_not_a_count(self):
        assert as_int(True) == 0
        assert as_int(False) == 0

    def test_int_and_float(self):
        assert as_int(5) == 5
        assert as_int(5.9) == 5

    def test_numeric_string(self):
        assert as_int("128") == 128
        assert as_int("12.5") == 12

    def test_junk_is_zero(self):
        assert as_int("abc") == 0
        assert as_int(None) == 0
        assert as_int([]) == 0

    def test_nan_and_infinity_are_zero(self):
        assert as_int(float("nan")) == 0
        assert as_int(float("inf")) == 0


class TestAsFloat:
    def test_none_uses_the_given_default(self):
        assert as_float(None, default=5.0) == 5.0

    def test_number_and_string(self):
        assert as_float(3) == 3.0
        assert as_float("2.5") == 2.5

    def test_junk_uses_the_default(self):
        assert as_float("boom", default=-1.0) == -1.0
        assert as_float(True, default=1.0) == 1.0

    def test_nan_and_infinity_use_the_default(self):
        assert as_float(float("inf"), default=0.5) == 0.5
        assert as_float(float("nan"), default=0.5) == 0.5


class _User(Schema):
    name = Field(str, "their name")
    age = Field(int, default=0)


class _Repo(Schema):
    url = Field(str, "clone address")
    mirrors = Field(list, item_type=str, default_factory=list)


class TestSchema:
    def test_required_field_must_be_present(self):
        with pytest.raises(ValidationError):
            _User(name="a", age="not-an-int")

    def test_default_fills_an_absent_optional(self):
        user = _User(name="wyn")
        assert user.name == "wyn"
        assert user.age == 0
        assert _User(name="wyn").to_dict() == {"name": "wyn", "age": 0}

    def test_unknown_field_is_rejected_at_construction(self):
        with pytest.raises(TypeError):
            _User(name="a", bogus=1)

    def test_validate_strict_rejects_unknown_keys(self):
        with pytest.raises(ValidationError) as exc:
            _User.validate_strict({"name": "a", "surprise": True})
        message = "\n".join(m for _, m in exc.value.error_list)
        assert "surprise" in message

    def test_json_schema_lists_required(self):
        schema = _User.json_schema()
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"name"}
        assert schema["properties"]["age"]["type"] == "integer"

    def test_list_item_type_is_rendered(self):
        schema = _Repo.json_schema()
        assert schema["properties"]["mirrors"]["items"]["type"] == "string"

    def test_nested_schema_validates(self):
        class Wrap(Schema):
            user = Field(_User)

        wrap = Wrap.validate({"user": {"name": "x"}})
        assert wrap.user.name == "x"
        with pytest.raises(ValidationError):
            Wrap.validate({"user": {"name": "x", "age": "oops"}})

    def test_transform_and_choices(self):
        class Mode(Schema):
            level = Field(str, choices=("low", "high"), default="low")
            tidy = Field(str, transform=str.strip, default="")

        assert Mode(level="high").level == "high"
        assert Mode(tidy="  y ").tidy == "y"
        # choices are enforced: a value outside the set is rejected.
        with pytest.raises(ValidationError) as exc:
            Mode(level="nope", tidy="x")
        message = "\n".join(m for _, m in exc.value.error_list)
        assert "must be one of low, high" in message