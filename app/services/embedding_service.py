"""Generate vector embeddings for text chunks and prepare them for Qdrant."""

from __future__ import annotations

from typing import Any, List, Optional


class EmbeddingService:
    """Wrap the embedding model used for document vectors."""

    def __init__(self, model_name: str = "nomic-embed-text", client: Optional[Any] = None):
        self.model_name = model_name
        self.client = client

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        if not text:
            return []

        if self.client is None:
            raise ValueError("An embedding client is required.")

        result = self.client.embeddings(model=self.model_name, prompt=text)
        return result.get("embedding", [])
