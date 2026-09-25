from unittest.mock import MagicMock, patch

from app.models.chat import ChatRequest
from app.routes.chat import chat


def test_chat_uses_retrieved_context_when_no_context_given():
    fake_qdrant_service = MagicMock()
    fake_qdrant_service.search_similar_documents.return_value = [
        {"payload": {"text": "Eligibility rules say applicants must provide proof of income."}}
    ]

    fake_embedding_service = MagicMock()
    fake_embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]

    with patch("app.routes.chat.get_qdrant_service", return_value=fake_qdrant_service), patch(
        "app.routes.chat.EmbeddingService", return_value=fake_embedding_service
    ), patch(
        "app.routes.chat.generate_grounded_answer",
        return_value="Applicants must provide proof of income.",
    ):
        response = chat(ChatRequest(question="What proof do applicants need?"))

    assert response["answer"] == "Applicants must provide proof of income."
