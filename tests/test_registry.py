from __future__ import annotations

import pytest

from roco.tools import ToolContext, build_registry
from roco.tools.registry import ToolRegistry

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.tool(description="Add two numbers.", parameters=ADD_SCHEMA)
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool("explode")
    def _explode() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    return reg


def test_schema_shape(registry):
    schemas = registry.schemas()
    assert [s["function"]["name"] for s in schemas] == ["add", "explode"]
    assert schemas[0] == {
        "type": "function",
        "function": {"name": "add", "description": "Add two numbers.", "parameters": ADD_SCHEMA},
    }
    assert schemas[1]["function"]["description"] == "Always fails."


def test_call_with_dict_and_json_string(registry):
    assert registry.call("add", {"a": 2, "b": 3}) == "5"
    assert registry.call("add", '{"a": 1, "b": 1}') == "2"


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("nope", {}, "unknown tool"),
        ("add", "{not json", "invalid arguments"),
        ("add", "[1, 2]", "invalid arguments"),
        ("add", {"a": 1}, "missing required argument(s): b"),
        ("add", {"a": "x", "b": 1}, "failed"),  # TypeError inside -> still text
        ("explode", None, "explode failed: kaboom"),
    ],
)
def test_errors_come_back_as_text(registry, name, args, expected):
    result = registry.call(name, args)
    assert result.startswith("Error:")
    assert expected in result


def test_unknown_extra_arguments_are_dropped(registry):
    assert registry.call("add", {"a": 1, "b": 2, "c": 99}) == "3"


def test_result_conversion():
    reg = ToolRegistry()
    reg.register(lambda: None, name="nothing", description="x")
    reg.register(lambda: {"ok": True}, name="obj", description="x")
    assert reg.call("nothing") == "Done."
    assert reg.call("obj") == '{"ok": true}'


def test_registration_validation():
    reg = ToolRegistry()
    reg.register(lambda: 1, name="one", description="One.")
    with pytest.raises(ValueError, match="already registered"):
        reg.register(lambda: 1, name="one", description="One.")
    with pytest.raises(ValueError, match="description"):
        reg.register(lambda: 1, name="two")
    with pytest.raises(ValueError, match="JSON schema object"):
        reg.register(lambda: 1, name="three", description="x", parameters={"type": "array"})


def test_build_registry_loads_core_and_skips_broken_modules(config, bus):
    reg = build_registry(
        ToolContext(config=config, bus=bus),
        modules=("roco.tools.clock", "roco.tools.does_not_exist"),
    )
    assert "get_current_time" in reg
    assert reg.call("get_current_time")  # non-empty human-readable string
