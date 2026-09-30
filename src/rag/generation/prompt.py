"""
Prompt templates for the RAG generation step.

CONCEPT: Prompt engineering for RAG has a few non-obvious requirements:

1. ANTI-HALLUCINATION INSTRUCTION: Without explicit instruction, LLMs tend to
   blend their parametric knowledge (what they learned during training) with the
   retrieved context. This produces answers that sound confident but are not
   grounded in the provided documents.

2. EXPLICIT FALLBACK: If the context doesn't contain the answer, the LLM
   should say so clearly rather than making something up. This is the most
   common failure mode in naive RAG systems.

3. CONTEXT FORMATTING: Structuring each chunk with a [Source] header makes
   it easier for the LLM to attribute information and easier for you to
   debug which chunk contributed to the answer.

4. TEMPERATURE MATTERS: We use temperature=0.1 (near-deterministic) for RAG.
   Higher temperatures increase creativity but also hallucination.
   RAG answers should be accurate, not creative.

INTERVIEW QUESTION: "How do you reduce hallucination in a RAG system?"
  - Retrieve more candidates, rerank precisely → better context quality
  - Use explicit "only answer from context" system prompt
  - Use low temperature for generation
  - Evaluate faithfulness metrics to catch regressions
  - Consider adding source citations so users can verify
"""

from typing import Any

RAG_SYSTEM_PROMPT = """\
You are a precise and helpful assistant. Your task is to answer the user's question \
using ONLY the information provided in the CONTEXT section below.

Rules you must follow without exception:
1. Base your answer exclusively on the provided context. Do not use your general \
knowledge or training data.
2. If the context does not contain sufficient information to answer the question, \
respond with: "I don't have enough information in the provided context to answer \
this question."
3. Be concise and direct. Do not pad your answer with unnecessary qualifications.
4. When the context contains conflicting information, note the conflict explicitly.
5. Do not invent source names, URLs, or specific facts not present in the context.\
"""

RAG_USER_TEMPLATE = """\
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:\
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a structured context string.

    Each chunk is labelled with its source file and index for traceability.
    This formatting helps the LLM correctly attribute information and makes
    debugging easier — you can see exactly which chunks were used.

    Args:
        chunks: List of reranked chunk dicts with "text" and "source_file" fields.

    Returns:
        Formatted context string to insert into the RAG_USER_TEMPLATE.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_file", "unknown")
        text = chunk.get("text", "").strip()
        parts.append(f"[Source {i}: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def build_rag_messages(
    question: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build the full OpenAI-format messages list for a RAG query.

    OpenAI-compatible APIs (Groq, Gemini, Ollama) use the messages format:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Args:
        question: The user's query.
        chunks: Reranked context chunks.

    Returns:
        Messages list ready to pass to the LLM client.
    """
    context = format_context(chunks)
    user_content = RAG_USER_TEMPLATE.format(context=context, question=question)
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
