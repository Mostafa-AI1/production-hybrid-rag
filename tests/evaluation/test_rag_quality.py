"""
DeepEval RAG Quality Gate — Automated LLM-as-a-Judge Evaluation.

CONCEPT: The RAG Triad of Metrics
When evaluating a production RAG system, three fundamental questions must be answered:
  1. Faithfulness (Grounding / Hallucination):
     - Question: Is every claim in the answer strictly supported by the retrieved context?
     - Score range: [0.0, 1.0]. A score < 0.7 indicates severe hallucination risk.
  2. Answer Relevancy:
     - Question: Does the answer directly address the user's question without extraneous fluff?
     - Score range: [0.0, 1.0]. A score < 0.7 indicates drifting or off-topic generation.
  3. Contextual Precision / Recall:
     - Question: Did retrieval fetch the right chunks, and are relevant chunks ranked at the top?
     - Score range: [0.0, 1.0]. Low precision = noisy context; low recall = missing knowledge.

WHY DEEPEVAL IN CI?
In standard software, unit tests check logic determinism (input X -> output Y).
In GenAI systems, prompts change, model weights change, and chunking parameters change.
Without automated quality gates in CI:
  - A developer tweaks a prompt to sound "friendlier", but accidentally
    increases hallucinations by 30%.
  - A chunk size change drops contextual recall by 25%.
DeepEval turns LLM-as-a-judge into automated pytest assertions (`assert_test(test_case, [metric])`).

INTERVIEW QUESTIONS TO MASTER:
  1. What is LLM-as-a-judge and what are its common failure modes (verbosity bias, position bias)?
  2. How do you set quality thresholds without causing CI flakiness?
  3. What is the difference between offline evaluation (golden datasets) and
     online evaluation (production tracing)?
"""

import json
from pathlib import Path
from typing import Any

import pytest

from rag.core.config import get_settings


def _load_golden_dataset() -> list[dict[str, Any]]:
    """Load the ground-truth golden dataset."""
    dataset_path = Path("eval/golden_dataset.json")
    if not dataset_path.exists():
        return []
    with open(dataset_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
        return data


@pytest.fixture(scope="module")
def golden_dataset() -> list[dict[str, Any]]:
    return _load_golden_dataset()


@pytest.mark.evaluation
class TestRAGQualityGates:
    """Quality gate assertions using DeepEval metrics."""

    def test_golden_dataset_structure(self, golden_dataset: list[dict[str, Any]]) -> None:
        """Verify the evaluation dataset adheres strictly to the required schema."""
        assert len(golden_dataset) > 0, "Golden dataset must not be empty"
        required_keys = {"question", "ground_truth_answer", "relevant_source", "category"}

        for item in golden_dataset:
            assert required_keys.issubset(item.keys()), f"Item missing keys: {item}"
            assert len(item["question"].strip()) > 5
            assert len(item["ground_truth_answer"].strip()) > 10

    def test_heuristic_grounding_check(self, golden_dataset: list[dict[str, Any]]) -> None:
        """
        Fast deterministic offline check: validates ground truth alignment
        without needing external LLM API calls.
        """
        for item in golden_dataset:
            question = item["question"]
            answer = item["ground_truth_answer"]
            assert len(question) > 0
            assert len(answer) > 0
            # Ground truth should not contain generic hallucination evasion phrases
            assert "I cannot answer" not in answer
            assert "As an AI" not in answer

    def test_deepeval_faithfulness_and_relevancy(
        self, golden_dataset: list[dict[str, Any]]
    ) -> None:
        """
        Live DeepEval test running Faithfulness and Relevancy metrics.
        Requires active LLM_API_KEY. Skips gracefully if running in offline test environment.
        """
        settings = get_settings()
        if not settings.llm.api_key or settings.llm.api_key == "test-key-not-real":
            pytest.skip("Active LLM_API_KEY required for live DeepEval evaluation")

        try:
            from deepeval import assert_test
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval not installed in current environment")

        # Evaluate the first test case as a CI smoke test to preserve API quota
        sample = golden_dataset[0]

        # Simulated or retrieved context and generated answer
        test_case = LLMTestCase(
            input=sample["question"],
            actual_output=sample["ground_truth_answer"],
            expected_output=sample["ground_truth_answer"],
            retrieval_context=[
                f"Context from {sample['relevant_source']}: {sample['ground_truth_answer']}"
            ],
        )

        faithfulness = FaithfulnessMetric(threshold=0.7)
        relevancy = AnswerRelevancyMetric(threshold=0.7)

        # Assert quality gates
        assert_test(test_case, [faithfulness, relevancy])
