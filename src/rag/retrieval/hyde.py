"""
HyDE: Hypothetical Document Embeddings for Query Expansion.

CONCEPT: The Vocabulary Mismatch & Query-Document Asymmetry in RAG
In dense retrieval, queries and documents are inherently asymmetrical:
  - Queries: Short, interrogative, incomplete ("python memory leak circular ref?")
  - Documents: Long, declarative, detailed ("Python's garbage collector uses reference counting...")

Because bi-encoder models are trained on cosine similarity between questions and passages,
vocabulary gaps or conceptual abstraction can cause standard dense retrieval to miss relevant
chunks.

HOW HyDE (Hypothetical Document Embeddings) WORKS:
  1. Take the user query.
  2. Ask an LLM: "Write a short, hypothetical passage answering this question."
  3. The LLM generates a plausible declarative answer (even if it contains minor hallucinations!).
  4. Embed this *hypothetical passage* instead of the raw question.
  5. Search the vector database using the hypothetical embedding.

WHY THIS WORKS:
The hypothetical document resides in the *same semantic and syntactic space* as the real document
chunks (declarative sentences, technical keywords, explanations). The dense search now performs
document-to-document similarity instead of query-to-document similarity.

INTERVIEW QUESTIONS TO MASTER:
  1. What is the Query-Document Asymmetry problem in bi-encoder retrieval?
  2. In what situations does HyDE hurt performance? (Answer: When the LLM hallucination is
     wildly off-topic, steering retrieval towards irrelevant embedding clusters).
  3. How does HyDE compare with Query Rewriting / Expansion (e.g. generating keyword synonyms)?
"""

from typing import Any

from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.generation.llm import generate

log = get_logger(__name__)

HYDE_PROMPT = (
    "You are a technical document writer. Write a short, clear, declarative technical paragraph "
    "(3-4 sentences) that directly answers the question below. Do not include introductory "
    "remarks or pleasantries.\n\nQUESTION:\n{question}\n\nPASSAGE:"
)


def generate_hypothetical_document(
    question: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Generate a hypothetical answer document for HyDE retrieval.

    Args:
        question: User query string.
        metadata: Optional tracing metadata.

    Returns:
        Hypothetical answer paragraph. If generation fails, falls back to the original question.
    """
    settings = get_settings()

    # If LLM API key is not configured (e.g. offline testing), fall back gracefully
    if not settings.llm.api_key:
        log.debug("hyde_skipped_no_api_key")
        return question

    messages = [
        {
            "role": "system",
            "content": "You write concise, authoritative technical documentation passages.",
        },
        {"role": "user", "content": HYDE_PROMPT.format(question=question)},
    ]

    try:
        result = generate(messages=messages, metadata=metadata)
        passage: str = str(result["answer"]).strip()
        log.info("hyde_passage_generated", passage_preview=passage[:80])
        return passage
    except Exception as e:
        log.warning("hyde_generation_failed_using_original_query", error=str(e))
        return question
