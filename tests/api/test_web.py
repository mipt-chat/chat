from fastapi.testclient import TestClient

from app.main import create_app


def test_web_root_serves_chat_client() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "CustomerSupportBot" in response.text
    assert "/chat/stream" in response.text
