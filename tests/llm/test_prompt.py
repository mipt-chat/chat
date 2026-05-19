"""
Тесты для app/llm/prompt.py.
Проверяют фильтрацию чанков по порогу и сборку системного промпта.
"""


from app.llm.prompt import SCORE_THRESHOLD, build_prompt
from app.models.knowledge import RetrievedChunk


def _make_chunk(text: str, score: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_path": "instructions/test.txt"},
        score=score,
    )


class TestBuildPromptFiltering:
    def test_all_chunks_below_threshold_returns_false(self):
        chunks = [
            _make_chunk("текст 1", score=0.1, chunk_id="c1"),
            _make_chunk("текст 2", score=0.39, chunk_id="c2"),
        ]
        _, answered = build_prompt("вопрос", chunks)
        assert answered is False

    def test_all_chunks_below_threshold_returns_empty_prompt(self):
        chunks = [_make_chunk("текст", score=0.0, chunk_id="c1")]
        prompt, _ = build_prompt("вопрос", chunks)
        assert prompt == ""

    def test_empty_chunks_returns_false(self):
        _, answered = build_prompt("вопрос", [])
        assert answered is False

    def test_chunk_exactly_at_threshold_is_included(self):
        chunks = [_make_chunk("граничный чанк", score=SCORE_THRESHOLD, chunk_id="c1")]
        prompt, answered = build_prompt("вопрос", chunks)
        assert answered is True
        assert "граничный чанк" in prompt

    def test_chunk_just_below_threshold_is_excluded(self):
        chunks = [_make_chunk("исключённый", score=SCORE_THRESHOLD - 0.001, chunk_id="c1")]
        _, answered = build_prompt("вопрос", chunks)
        assert answered is False

    def test_mixed_chunks_only_relevant_included(self):
        chunks = [
            _make_chunk("релевантный", score=0.8, chunk_id="c1"),
            _make_chunk("нерелевантный", score=0.1, chunk_id="c2"),
        ]
        prompt, answered = build_prompt("вопрос", chunks)
        assert answered is True
        assert "релевантный" in prompt
        assert "нерелевантный" not in prompt

    def test_all_chunks_above_threshold_all_included(self):
        chunks = [
            _make_chunk("чанк А", score=0.5, chunk_id="c1"),
            _make_chunk("чанк Б", score=0.9, chunk_id="c2"),
        ]
        prompt, answered = build_prompt("вопрос", chunks)
        assert answered is True
        assert "чанк А" in prompt
        assert "чанк Б" in prompt


class TestBuildPromptStructure:
    def test_prompt_contains_context_label(self):
        chunks = [_make_chunk("содержимое", score=0.7, chunk_id="c1")]
        prompt, _ = build_prompt("вопрос", chunks)
        assert "КОНТЕКСТ" in prompt

    def test_chunks_are_numbered(self):
        chunks = [
            _make_chunk("первый", score=0.6, chunk_id="c1"),
            _make_chunk("второй", score=0.7, chunk_id="c2"),
        ]
        prompt, _ = build_prompt("вопрос", chunks)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_prompt_is_in_russian(self):
        chunks = [_make_chunk("данные", score=0.6, chunk_id="c1")]
        prompt, _ = build_prompt("вопрос", chunks)
        # Системный промпт должен содержать русскоязычные инструкции
        assert any(word in prompt for word in ["Отвечай", "базы знаний", "контекст"])

    def test_question_does_not_affect_filtering(self):
        """Вопрос не влияет на фильтрацию — только score чанков."""
        chunks = [_make_chunk("факт", score=0.6, chunk_id="c1")]
        _, answered1 = build_prompt("вопрос А", chunks)
        _, answered2 = build_prompt("вопрос Б", chunks)
        assert answered1 == answered2
