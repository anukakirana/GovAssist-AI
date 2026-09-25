from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat_router, documents_router, health_router

app = FastAPI(title="GovAssist-AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/")
def read_root():
    return {"message": "GovAssist-AI is running"}
