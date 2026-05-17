import pytest
import json
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from tool_healing.api import build_app
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai import APIError, APIConnectionError, APITimeoutError

@pytest.fixture
def client(mock_openai, mock_httpx_client):
    app = build_app("http://upstream")
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_openai():
    with patch("tool_healing.api.AsyncOpenAI") as mock:
        instance = mock.return_value
        instance.post = AsyncMock()
        yield instance

@pytest.fixture
def mock_httpx_client():
    with patch("tool_healing.api.httpx.AsyncClient") as mock:
        mock_instance = mock.return_value.__aenter__.return_value
        mock_instance.request = AsyncMock()
        yield mock_instance

@pytest.fixture
def mock_tool_healing():
    with patch("tool_healing.api.strip_tool_call_markup") as mock_strip, \
         patch("tool_healing.api.parse_tool_calls_from_text") as mock_parse:
        yield mock_strip, mock_parse

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_completions_history_stripping(client, mock_openai, mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: f"stripped_{x}"
    
    mock_openai.post.return_value = ChatCompletion(
        id="1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(content="Hello", role="assistant"))],
        created=0,
        model="m",
        object="chat.completion"
    )
    
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<tool_call>old</tool_call>"}
    ]
    client.post("/v1/chat/completions", json={"messages": messages})
    
    # Check that history was stripped before calling upstream
    # Use index 1 for kwargs in call_args
    called_body = mock_openai.post.call_args[1]["body"]
    assert called_body["messages"][1]["content"] == "stripped_<tool_call>old</tool_call>"

def test_chat_completions_non_streaming(client, mock_openai, mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    
    mock_openai.post.return_value = ChatCompletion(
        id="1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(content="Hello", role="assistant"))],
        created=0,
        model="m",
        object="chat.completion"
    )
    
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello"

def test_chat_completions_with_tool_xml(client, mock_openai, mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    mock_strip.return_value = "Stripped content"
    mock_parse.return_value = [{"id": "tc1", "function": {"name": "f", "arguments": "{}"}}]
    
    mock_openai.post.return_value = ChatCompletion(
        id="1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(content="<tool_call>...</tool_call>", role="assistant"))],
        created=0,
        model="m",
        object="chat.completion"
    )
    
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Stripped content"
    assert data["choices"][0]["message"]["tool_calls"][0]["id"] == "tc1"

@patch("tool_healing.api.heal_chat_completion_stream")
def test_chat_completions_streaming(mock_heal, client, mock_openai):
    mock_openai.post.return_value = AsyncMock() # Mock stream
    
    async def mock_generator(stream, signals=None):
        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=ChoiceDelta(content="Hello"))
        ])
    
    mock_heal.side_effect = mock_generator
    
    response = client.post("/v1/chat/completions", json={"stream": True, "messages": []})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Check streaming output
    lines = [line for line in response.iter_lines() if line]
    assert lines[0].startswith("data: ")
    assert json.loads(lines[0][6:])["choices"][0]["delta"]["content"] == "Hello"
    assert lines[1] == "data: [DONE]"

@patch("tool_healing.api.heal_chat_completion_stream")
def test_chat_completions_streaming_error(mock_heal, client, mock_openai):
    mock_openai.post.return_value = AsyncMock()
    
    async def mock_generator(stream, signals=None):
        yield "not a chunk" # Will cause error in model_dump_json
        raise Exception("stream fail")
    
    mock_heal.side_effect = mock_generator
    
    response = client.post("/v1/chat/completions", json={"stream": True, "messages": []})
    assert response.status_code == 200
    lines = [line for line in response.iter_lines() if line]
    assert "Proxy error" in lines[0]

def test_chat_completions_connection_error(client, mock_openai):
    mock_openai.post.side_effect = APIConnectionError(message="fail", request=MagicMock())
    
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 502
    assert "Upstream connection error" in response.json()["error"]["message"]

def test_chat_completions_api_error(client, mock_openai):
    err = APIError("api error", MagicMock(), body={"error": "details"})
    err.status_code = 400
    mock_openai.post.side_effect = err
    
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json() == {"error": "details"}

def test_chat_completions_api_error_no_body(client, mock_openai):
    err = APIError("api error", MagicMock(), body=None)
    err.status_code = 500
    mock_openai.post.side_effect = err
    
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "api error"

def test_chat_completions_generic_exception(client, mock_openai):
    mock_openai.post.side_effect = Exception("boom")
    
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 502
    assert "boom" in response.json()["error"]["message"]

def test_passthrough(mock_httpx_client, client):
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.content = b"created"
    mock_resp.headers = {"Content-Type": "text/plain", "X-Custom": "val", "Content-Length": "7"}
    
    mock_httpx_client.request = AsyncMock(return_value=mock_resp)
    
    response = client.post("/other/path", content="body")
    assert response.status_code == 201
    assert response.text == "created"
    assert response.headers["x-custom"] == "val"
    # Note: Content-Length might be re-added by FastAPI/Starlette Response

def test_passthrough_error(mock_httpx_client, client):
    mock_httpx_client.request.side_effect = Exception("conn fail")
    
    response = client.get("/other/path")
    assert response.status_code == 502
    assert "Passthrough request failed" in response.json()["error"]["message"]

def test_passthrough_timeout(mock_httpx_client, client):
    mock_httpx_client.request.side_effect = httpx.TimeoutException("timeout")
    
    response = client.get("/other/path")
    assert response.status_code == 504
    assert "Upstream timeout" in response.json()["error"]["message"]

def test_passthrough_connect_error(mock_httpx_client, client):
    mock_httpx_client.request.side_effect = httpx.ConnectError("connect fail")
    
    response = client.get("/other/path")
    assert response.status_code == 502
    assert "Upstream connection error" in response.json()["error"]["message"]

def test_passthrough_logs_upstream_error(mock_httpx_client, client, caplog):
    import logging
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.content = b"Internal Server Error"
    mock_resp.headers = {"Content-Type": "text/plain"}
    mock_httpx_client.request.return_value = mock_resp
    
    with caplog.at_level(logging.ERROR):
        client.get("/other/path")
    
    assert "Upstream returned error 500 for /other/path" in caplog.text
