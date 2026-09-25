import unittest

from app.services.qdrant_client import QdrantService


class FakeQdrantSearchClient:
    def __init__(self):
        self.points = []

    def get_collection(self, collection_name):
        return {"status": "exists"}

    def create_collection(self, collection_name, vectors_config):
        return True

    def upsert(self, collection_name, points):
        self.points.extend(points)
        return {"status": "ok"}

    def search(self, collection_name, query_vector, limit, score_threshold):
        return [
            {"id": "chunk-1", "score": 0.94, "payload": {"text": "This is relevant content."}},
            {"id": "chunk-2", "score": 0.80, "payload": {"text": "This is background info."}},
        ]


class SimilaritySearchTests(unittest.TestCase):
    def test_search_similar_documents(self):
        fake_client = FakeQdrantSearchClient()
        service = QdrantService(client=fake_client, collection_name="govassist_documents")

        result = service.search_similar_documents(query_embedding=[0.1, 0.2, 0.3], limit=2, score_threshold=0.5)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "chunk-1")
        self.assertIn("relevant content", result[0]["payload"]["text"].lower())


if __name__ == "__main__":
    unittest.main()
