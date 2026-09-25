from fastapi import APIRouter, File, UploadFile
from ollama import Client
from qdrant_client import QdrantClient
from uuid import uuid5, NAMESPACE_URL

from app.config import OLLAMA_URL, QDRANT_COLLECTION, QDRANT_URL
from app.core.pdf_processor import chunk_text, extract_text_from_pdf
from app.services.embedding_service import EmbeddingService
from app.services.ollama_client import OllamaService
from app.services.qdrant_client import QdrantService

router = APIRouter()


def index_pdf_document(file_path: str, filename: str, qdrant_service: QdrantService, embedding_service: EmbeddingService):
    """Extract, chunk, embed, and store PDF content in Qdrant."""
    try:
        text = extract_text_from_pdf(file_path)
    except Exception:
        return {"filename": filename, "chunks": 0, "status": "failed", "error": "Could not read PDF content"}

    if not text:
        return {"filename": filename, "chunks": 0, "status": "failed", "error": "No text found in PDF"}

    chunks = chunk_text(text, chunk_size=500, overlap=100)
    embedded_chunks = [
        (chunk, embedding_service.generate_embedding(chunk)) for chunk in chunks
    ]
    first_embedding = next(
        (embedding for _, embedding in embedded_chunks if embedding), None
    )
    if not first_embedding:
        return {
            "filename": filename,
            "chunks": 0,
            "status": "failed",
            "error": "Could not generate document embeddings",
        }

    qdrant_service.ensure_collection(vector_size=len(first_embedding))

    for index, (chunk, embedding) in enumerate(embedded_chunks):
        if not embedding:
            continue
        qdrant_service.add_document(
            document_id=str(uuid5(NAMESPACE_URL, f"{filename}-{index}")),
            text=chunk,
            embedding=embedding,
        )

    return {"filename": filename, "chunks": len(chunks), "status": "indexed"}


@router.get("/documents")
def list_documents():
    return {"documents": []}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and store the extracted chunks in Qdrant."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    qdrant_client = QdrantClient(url=QDRANT_URL)
    qdrant_service = QdrantService(
        client=qdrant_client, collection_name=QDRANT_COLLECTION
    )
    ollama_service = OllamaService(
        model_name="nomic-embed-text",
        client=Client(host=OLLAMA_URL),
    )
    embedding_service = EmbeddingService(
        model_name="nomic-embed-text",
        client=ollama_service,
    )

    file_contents = await file.read()
    temp_path = f"./tmp_{file.filename}"

    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_contents)

    result = index_pdf_document(
        file_path=temp_path,
        filename=file.filename,
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
    )

    return result
