import uuid

from app.services.chat_history import append_message, get_history


def test_history_round_trip():
    session_id = str(uuid.uuid4())
    append_message(session_id, "user", "hello")
    append_message(session_id, "assistant", "hi there")

    history = get_history(session_id, max_turns=5)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_is_isolated_per_session():
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    append_message(session_a, "user", "session a message")
    append_message(session_b, "user", "session b message")

    assert get_history(session_a, max_turns=5) == [
        {"role": "user", "content": "session a message"}
    ]
    assert get_history(session_b, max_turns=5) == [
        {"role": "user", "content": "session b message"}
    ]


def test_history_respects_max_turns():
    session_id = str(uuid.uuid4())
    for i in range(5):
        append_message(session_id, "user", f"q{i}")
        append_message(session_id, "assistant", f"a{i}")

    history = get_history(session_id, max_turns=2)

    assert len(history) == 4
    assert history[0]["content"] == "q3"
    assert history[-1]["content"] == "a4"


def test_unknown_session_returns_empty_history():
    assert get_history(str(uuid.uuid4()), max_turns=5) == []
