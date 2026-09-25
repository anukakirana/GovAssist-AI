from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_endpoint_with_context_returns_answer():
    response = client.post(
        "/chat",
        json={
            "question": "What is the document about?",
            "context": "This document explains eligibility rules for government assistance.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_chat_endpoint_without_context_returns_guidance():
    response = client.post(
        "/chat",
        json={"question": "What is the document about?"},
    )

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert "upload" in answer or "context" in answer
