from app.services.llm_service import condense_question


def test_condense_question_returns_original_when_no_history():
    assert condense_question("what about it?", []) == "what about it?"


def test_condense_question_falls_back_on_error(mocker):
    mocker.patch(
        "app.services.llm_service.client.chat.completions.create",
        side_effect=RuntimeError("boom"),
    )
    history = [{"role": "user", "content": "Tell me about Paris"}]

    result = condense_question("what about it?", history)

    assert result == "what about it?"
