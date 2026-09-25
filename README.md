# GovAssist-AI

This project is a beginner-friendly FastAPI application for a Retrieval-Augmented Generation (RAG) workflow using Qdrant and Ollama.

## Project structure

- `app/` - Main application package
  - `__init__.py` - Marks the directory as a Python package
  - `main.py` - Entry point for the FastAPI app
  - `config.py` - Central place for environment variables and app settings
  - `routes/` - HTTP endpoints grouped by feature
    - `__init__.py` - Exports router modules
    - `health.py` - Simple health check endpoint
    - `documents.py` - Endpoints for adding or listing documents
    - `chat.py` - Chat endpoint for asking questions
  - `services/` - External service logic
    - `__init__.py` - Service package marker
    - `qdrant_client.py` - Qdrant database connection logic
    - `ollama_client.py` - Ollama model interaction logic
  - `models/` - Pydantic request/response schemas
    - `__init__.py` - Models package marker
    - `chat.py` - Input/output models for chat requests
  - `core/` - Core application logic
    - `__init__.py` - Core package marker
    - `rag_pipeline.py` - Where the retrieval and prompt-building workflow will live

- `requirements.txt` - Python dependencies for the project
- `.gitignore` - Files and folders to ignore in Git

## How this fits the RAG idea

- Qdrant stores vector embeddings of documents
- Ollama hosts the LLM used to answer questions
- FastAPI exposes the API endpoints for the app
- The app will eventually:
  1. receive a user question,
  2. search relevant documents in Qdrant,
  3. pass context to Ollama,
  4. return the final answer

