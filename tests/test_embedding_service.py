import unittest

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_client import QdrantService


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def embeddings(self, model, prompt):
        self.calls.append((model, prompt))
        return {"embedding": [0.1, 0.2, 0.3]}


class FakeQdrantClient:
    def __init__(self):
        self.collection_name = None
        self.points = []

    def get_collection(self, collection_name):
        if collection_name != "govassist_documents":
            raise ValueError("collection not found")
        return {"status": "exists"}

    def create_collection(self, collection_name, vectors_config):
        self.collection_name = collection_name
        self.vectors_config = vectors_config
        return True

    def upsert(self, collection_name, points):
        self.collection_name = collection_name
        self.points.extend(points)
        return {"status": "ok"}


class EmbeddingServiceTests(unittest.TestCase):
    def test_generate_embedding_uses_ollama_client(self):
        fake_client = FakeOllamaClient()
        service = EmbeddingService(model_name="nomic-embed-text", client=fake_client)

        embedding = service.generate_embedding("sample text")

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.assertEqual(fake_client.calls[0][0], "nomic-embed-text")

    def test_qdrant_service_stores_documents(self):
        fake_client = FakeQdrantClient()
        service = QdrantService(client=fake_client, collection_name="govassist_documents")

        service.ensure_collection(vector_size=3)
        service.add_document(document_id="doc-1", text="hello world", embedding=[0.1, 0.2, 0.3])

        self.assertEqual(service.collection_name, "govassist_documents")
        self.assertEqual(len(fake_client.points), 1)
        self.assertEqual(fake_client.points[0]["payload"]["text"], "hello world")


if __name__ == "__main__":
    unittest.main()
