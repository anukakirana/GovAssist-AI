from fastapi import APIRouter
from ollama import Client
from qdrant_client import QdrantClient

from app.config import OLLAMA_MODEL, OLLAMA_URL, QDRANT_COLLECTION, QDRANT_URL
from app.core.rag_pipeline import generate_grounded_answer
from app.models.chat import ChatRequest, ChatResponse
from app.services.embedding_service import EmbeddingService
from app.services.ollama_client import OllamaService
from app.services.qdrant_client import QdrantService

router = APIRouter()


def get_qdrant_service() -> QdrantService:
    """Create a Qdrant service backed by a real local client."""
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return QdrantService(client=qdrant_client, collection_name=QDRANT_COLLECTION)


def get_ollama_service() -> OllamaService:
    """Create an Ollama service backed by a real local client."""
    ollama_client = Client(host=OLLAMA_URL)
    return OllamaService(model_name=OLLAMA_MODEL, client=ollama_client)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a user question using retrieved document context."""
    question = request.question.strip()
    context = (request.context or "").strip()

    if not question:
        return {"answer": "Please provide a valid question."}

    ollama_service = get_ollama_service()

    if context:
        return {
            "answer": generate_grounded_answer(
                question=question,
                context=context,
                llm_client=ollama_service,
                model_name=OLLAMA_MODEL,
            )
        }

    qdrant_service = get_qdrant_service()
    embedding_service = EmbeddingService(model_name="nomic-embed-text", client=ollama_service)
    query_embedding = embedding_service.generate_embedding(question)

    if not query_embedding:
        return {"answer": "I could not generate an embedding for your question."}

    matches = qdrant_service.search_similar_documents(
        query_embedding=query_embedding,
        limit=3,
        score_threshold=0.0,
    )
    retrieved_context = [item.get("payload", {}).get("text", "") for item in matches]

    if not retrieved_context:
        return {
            "answer": "I could not find relevant document context. Please upload a PDF and index it before asking a question."
        }

    answer = generate_grounded_answer(
        question=question,
        context=retrieved_context,
        llm_client=ollama_service,
        model_name=OLLAMA_MODEL,
    )

    return {"answer": answer}
