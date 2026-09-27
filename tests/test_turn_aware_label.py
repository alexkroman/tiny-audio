"""Tests for scripts.turn_aware.label: request shape and response parsing (no API calls)."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.turn_aware.label import LABELS, MODEL, parse_label, request_params, user_message


def _msg(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_request_shares_one_cacheable_system_block_and_constrains_output():
    a = request_params("What's your account number?", "Sure, it's")
    b = request_params("", "Can you check")
    assert a["model"] == MODEL
    assert a["system"] == b["system"]  # identical prefix -> cacheable across the batch
    assert a["system"][-1]["cache_control"] == {"type": "ephemeral"}
    schema = a["output_config"]["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == list(LABELS)
    assert a["output_config"]["effort"] == "low"


def test_user_message_marks_missing_agent_question():
    assert (
        user_message("", "Hey there,") == "Agent's last question: (none)\nCaller so far: Hey there,"
    )


def test_parse_label():
    assert parse_label(_msg('{"label": "incomplete"}')) == "incomplete"
    assert parse_label(_msg('{"label": "maybe"}')) is None
    assert parse_label(_msg("not json")) is None
    assert parse_label(_msg('{"label": "complete"}', stop="refusal")) is None
    assert parse_label(_msg('{"label": "complete"}', stop="max_tokens")) is None
