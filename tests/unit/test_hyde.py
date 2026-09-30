"""
Unit tests for HyDE (Hypothetical Document Embeddings) Query Expansion.

Verifies:
  1. Fallback to raw query when LLM API key is absent.
  2. Successful generation of hypothetical passage when LLM responds.
  3. Safe fallback to original query if LLM raises an error.
"""

from unittest.mock import MagicMock, patch

from rag.retrieval.hyde import generate_hypothetical_document


def test_hyde_fallback_when_no_api_key() -> None:
    """HyDE returns raw question if no API key is set."""
    with patch("rag.retrieval.hyde.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.llm.api_key = ""
        mock_settings.return_value = settings_mock

        res = generate_hypothetical_document("How does Python GC handle cyclic references?")
        assert res == "How does Python GC handle cyclic references?"


def test_hyde_successful_generation() -> None:
    """HyDE generates and returns hypothetical answer passage."""
    hypothetical_passage = (
        "Python's garbage collector resolves cyclic references through a generational "
        "cycle detection algorithm that tracks container objects."
    )

    with (
        patch("rag.retrieval.hyde.get_settings") as mock_settings,
        patch("rag.retrieval.hyde.generate") as mock_generate,
    ):
        settings_mock = MagicMock()
        settings_mock.llm.api_key = "test-key"
        mock_settings.return_value = settings_mock

        mock_generate.return_value = {
            "answer": hypothetical_passage,
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
        }

        result = generate_hypothetical_document("How does Python GC handle cyclic references?")
        assert result == hypothetical_passage
        mock_generate.assert_called_once()
        # Verify prompt format passed to generate
        messages = mock_generate.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert "How does Python GC handle cyclic references?" in messages[1]["content"]


def test_hyde_fallback_on_generation_failure() -> None:
    """If LLM call fails, HyDE logs warning and falls back to original question."""
    with (
        patch("rag.retrieval.hyde.get_settings") as mock_settings,
        patch("rag.retrieval.hyde.generate") as mock_generate,
    ):
        settings_mock = MagicMock()
        settings_mock.llm.api_key = "test-key"
        mock_settings.return_value = settings_mock

        mock_generate.side_effect = RuntimeError("Provider 503 Overloaded")

        result = generate_hypothetical_document("What is GIL?")
        assert result == "What is GIL?"
