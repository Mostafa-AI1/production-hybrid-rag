"""
Synthetic evaluation dataset generation script.

CONCEPT: Synthetic Data Generation for RAG Evaluation
In a production system, you need a ground-truth dataset of:
  (query, ground_truth_answer, relevant_context_chunk)
to measure whether your RAG pipeline is accurate (Faithfulness, Relevancy, Recall).

Problem: Hand-crafting 200 high-quality question-answer pairs takes days of manual effort.
Solution: Use an LLM to generate realistic questions and ground-truth answers directly from
ingested text chunks.

Methodology (inspired by Evol-Instruct & RAGAS synthetic generation):
  1. Parse documents into clean semantic chunks.
  2. For each chunk, prompt the LLM to generate diverse question types:
     - Factual: Direct lookup ("What is X?")
     - Analytical / Multi-hop: Understanding relationships ("Why does X require Y?")
     - Practical / Code: Practical usage ("How do I implement X using Y?")
  3. Extract both question and reference ground_truth_answer.
  4. Save structured outputs into eval/golden_dataset.json.

INTERVIEW QUESTIONS TO MASTER:
  1. What is the Evol-Instruct methodology, and why is question complexity evolution needed?
  2. What is data contamination / data leakage between evaluation sets and RAG training?
  3. Why should the evaluation dataset generation prompt explicitly instruct the LLM
     NOT to copy-paste verbatim from the context?
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag.core.logging import configure_logging, get_logger
from rag.generation.llm import generate
from rag.ingestion.chunker import chunk_document
from rag.ingestion.parser import parse_file

log = get_logger(__name__)

PROMPT_TEMPLATE = """You are an expert evaluator creating test datasets for a RAG system.
Given the technical documentation context, generate {num_questions} QA pair(s).

CONTEXT:
---
{chunk_text}
---

SOURCE FILE: {source_file}

REQUIREMENTS:
1. Each question must be something a real software engineer or developer would ask.
2. The answer must be completely faithful to the provided context (no external assumptions).
3. Choose category from: ["concepts", "best_practices", "implementation", "troubleshooting"].
4. Output MUST be valid JSON array of objects with the exact schema:
[
  {{
    "question": "string",
    "ground_truth_answer": "string",
    "relevant_source": "{source_file}",
    "category": "concepts"
  }}
]

Return ONLY the raw JSON array. No markdown code blocks, no preamble.
"""


def generate_qa_pairs_for_chunk(
    chunk_text: str,
    source_file: str,
    num_questions: int = 1,
) -> list[dict[str, Any]]:
    """
    Prompt LLM to produce question-answer pairs for a specific chunk.
    """
    prompt = PROMPT_TEMPLATE.format(
        chunk_text=chunk_text,
        source_file=source_file,
        num_questions=num_questions,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a senior technical benchmark creator. Return ONLY valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        result = generate(messages=messages)
        content = result["answer"].strip()

        # Strip optional markdown code fence if LLM returned it
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        pairs: list[dict[str, Any]] = json.loads(content)
        return pairs
    except Exception as e:
        log.warning("chunk_qa_generation_failed", error=str(e), source=source_file)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation dataset for RAG pipeline."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing source docs (.md, .txt, .pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Output path for golden dataset JSON",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=1,
        help="Number of questions to generate per chunk",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Max number of chunks to process (to avoid high LLM token costs during demos)",
    )
    args = parser.parse_args()

    configure_logging()
    log.info(
        "starting_eval_generation",
        docs_dir=str(args.docs_dir),
        output=str(args.output),
    )

    if not args.docs_dir.exists():
        log.error("docs_dir_not_found", docs_dir=str(args.docs_dir))
        print(f"Error: Directory '{args.docs_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    all_pairs: list[dict[str, Any]] = []

    # If output file already exists, load existing pairs to avoid overwriting unless intended
    if args.output.exists():
        try:
            with open(args.output, encoding="utf-8") as f:
                all_pairs = json.load(f)
            log.info("loaded_existing_pairs", count=len(all_pairs))
        except Exception:
            all_pairs = []

    processed_chunks = 0
    supported_extensions = {".md", ".txt", ".pdf"}

    for file_path in args.docs_dir.rglob("*"):
        if file_path.suffix.lower() not in supported_extensions:
            continue

        log.info("processing_document", path=str(file_path))
        try:
            doc = parse_file(file_path)
            chunks = chunk_document(doc)

            for chunk in chunks:
                if processed_chunks >= args.max_chunks:
                    break

                pairs = generate_qa_pairs_for_chunk(
                    chunk_text=chunk.text,
                    source_file=file_path.name,
                    num_questions=args.questions_per_chunk,
                )
                all_pairs.extend(pairs)
                processed_chunks += 1
                log.info("chunk_processed", chunk_index=chunk.chunk_index, generated=len(pairs))

        except Exception as e:
            log.error("file_processing_error", path=str(file_path), error=str(e))

        if processed_chunks >= args.max_chunks:
            log.info("reached_max_chunks_limit", limit=args.max_chunks)
            break

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    log.info("generation_complete", total_dataset_size=len(all_pairs), saved_to=str(args.output))
    print(f"Saved {len(all_pairs)} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
