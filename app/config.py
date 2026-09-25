"""Application configuration settings."""

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_NAME = "GovAssist-AI"
APP_VERSION = "0.1.0"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "govassist_documents")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
