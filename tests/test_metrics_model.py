import pytest
from pydantic import ValidationError

from app.models.metrics import UnansweredQuery


def test_unanswered_query_accepts_no_answer_reason() -> None:
    record = UnansweredQuery(question="Вопрос", reason="no_answer")

    assert record.reason == "no_answer"


def test_unanswered_query_rejects_unknown_reason() -> None:
    with pytest.raises(ValidationError):
        UnansweredQuery(question="Вопрос", reason="unexpected")
