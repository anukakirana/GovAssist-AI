"""RAG pipeline utilities for grounded question answering."""

from __future__ import annotations

from typing import Any, List, Optional


def build_prompt(question: str, context: str) -> str:
    """Create a prompt that tells the model to answer using the given context."""
    context_text = "\n\n".join(context) if isinstance(context, list) else context

    return (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}"
    )


def generate_grounded_answer(
    question: str,
    context: str | List[str],
    llm_client: Optional[Any] = None,
    model_name: str = "llama3",
) -> str:
    """Generate an answer grounded in the retrieved context."""
    if llm_client is None:
        raise ValueError("An LLM client is required.")

    prompt = build_prompt(question, context)
    response = llm_client.generate(model=model_name, prompt=prompt)
    return response.get("response", "")
