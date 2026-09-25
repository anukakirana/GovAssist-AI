"""Minimal Qdrant client wrapper for RAG indexing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class QdrantService:
    """Basic wrapper that stores document vectors in a Qdrant collection."""

    def __init__(self, client: Optional[Any] = None, collection_name: str = "govassist_documents"):
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int, distance: str = "Cosine") -> bool:
        """Ensure the collection exists, creating it if needed."""
        if self.client is None:
            raise ValueError("A Qdrant client is required.")

        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            self.client.create_collection(
                self.collection_name,
                vectors_config={"size": vector_size, "distance": distance},
            )
            return True

    def add_document(self, document_id: str, text: str, embedding: List[float]) -> Dict[str, Any]:
        """Store one document chunk with its embedding in Qdrant."""
        if self.client is None:
            raise ValueError("A Qdrant client is required.")

        point = {
            "id": document_id,
            "vector": embedding,
            "payload": {"text": text},
        }

        self.client.upsert(self.collection_name, [point])
        return {"id": document_id, "status": "stored"}

    def search_similar_documents(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar stored document chunks.

        This is the retrieval step used by the RAG workflow.
        """
        if self.client is None:
            raise ValueError("A Qdrant client is required.")

        if not query_embedding:
            return []

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
        )

        normalized_results = []
        for item in results:
            if isinstance(item, dict):
                score = item.get("score", 0.0)
                normalized_item = item
            else:
                score = item.score
                normalized_item = {
                    "id": str(item.id),
                    "score": score,
                    "payload": item.payload or {},
                }

            if score >= score_threshold:
                normalized_results.append(normalized_item)

        return normalized_results
