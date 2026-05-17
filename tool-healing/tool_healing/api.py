from __future__ import annotations

"""
API implementation for the Tool Healing Proxy.

This module provides a FastAPI application that acts as a proxy between
an OpenAI-compatible client and an upstream LLM server. It implements
Unsloth-style tool healing by detecting and converting tool call XML
in the assistant's content into structured tool calls.
"""

import json
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, AsyncStream
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletion, ChatCompletionChunk
from studio.backend.core.tool_healing  import strip_tool_call_markup, parse_tool_calls_from_text
from .streaming import heal_chat_completion_stream, DEFAULT_TOOL_SIGNALS
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize persistent client for passthrough
    async with httpx.AsyncClient(timeout=app.state.timeout) as client:
        app.state.hclient = client
        yield

def build_app(upstream_base_url: str, default_timeout: float = 600.0, signals: tuple[str, ...] = DEFAULT_TOOL_SIGNALS) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Args:
        upstream_base_url: The base URL of the upstream OpenAI-compatible API.
        default_timeout: The default timeout in seconds for upstream requests.
        signals: Tuple of strings that signal the start of a tool call.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(title="Tool Healing Proxy", lifespan=lifespan)
    app.state.timeout = default_timeout
    app.state.signals = signals
    client = AsyncOpenAI(base_url=f"{upstream_base_url}/v1", api_key="ignored", timeout=default_timeout)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """
        Handle OpenAI-compatible chat completion requests with tool healing.

        This endpoint:
        1. Strips stale tool-call XML from the incoming conversation history.
        2. Forwards the request to the upstream server.
        3. For non-streaming requests: parses any tool-call XML in the response
           and converts it to structured tool calls.
        4. For streaming requests: uses a state machine to strip tool XML and
           yield structured tool call chunks.
        """
        body = await request.json()
        
        # 1. Strip stale tool-call XML from conversation history
        if "messages" in body:
            for msg in body["messages"]:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg["content"] = strip_tool_call_markup(msg["content"], final=True)

        is_stream = body.get("stream", False)
        logger.info(f"Chat completion request: model={body.get('model')}, stream={is_stream}")

        try:
            if not is_stream:
                # Use generic post to ensure all fields are passed through
                resp = await client.post(
                    "/chat/completions",
                    body=body,
                    cast_to=ChatCompletion,
                )
                for choice in resp.choices:
                    content = choice.message.content or ""
                    if ("<tool_call>" in content or "<function=" in content) and not choice.message.tool_calls:
                        tool_calls = parse_tool_calls_from_text(content)
                        if tool_calls:
                            logger.info(f"Detected {len(tool_calls)} tool calls in non-streaming response")
                            choice.message.tool_calls = [
                                ChatCompletionMessageToolCall(
                                    id=tc["id"],
                                    type="function",
                                    function=ToolCallFunction(
                                        name=tc["function"]["name"],
                                        arguments=tc["function"]["arguments"]
                                    )
                                ) for tc in tool_calls
                            ]
                            choice.message.content = strip_tool_call_markup(content, final=True) or None
                return resp.model_dump(exclude_none=True)

            # Streaming: apply tool healing state machine
            stream = await client.post(
                "/chat/completions",
                body=body,
                cast_to=AsyncStream[ChatCompletionChunk],
                stream=True,
                stream_cls=AsyncStream[ChatCompletionChunk],
            )
            async def generate():
                try:
                    async for chunk in heal_chat_completion_stream(stream, signals=app.state.signals):
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'error': {'message': f'Proxy error: {str(e)}', 'type': 'proxy_error'}})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Upstream connection error: {str(e)}")
            return JSONResponse(status_code=502, content={"error": {"message": f"Upstream connection error: {str(e)}", "type": "proxy_error"}})
        except APIError as e:
            logger.error(f"Upstream API error: {str(e)}")
            status_code = getattr(e, "status_code", 500)
            body = getattr(e, "body", None)
            if body is None:
                body = {"error": {"message": str(e), "type": "upstream_error"}}
            return JSONResponse(status_code=status_code, content=body)
        except Exception as e:
            logger.error(f"Internal proxy error: {str(e)}")
            return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "proxy_error"}})

    @app.get("/health")
    async def health():
        """
        Simple health check endpoint.
        """
        return {"status": "ok"}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def passthrough(path: str, request: Request):
        """Pass all other endpoints through to upstream unchanged."""
        logger.info(f"Passthrough request: {request.method} /{path}")
        body = await request.body()
        
        # Filter hop-by-hop headers for the outgoing request
        exclude_headers = {
            "host", "connection", "keep-alive", "proxy-authenticate", 
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
        }
        headers = {
            k: v for k, v in request.headers.items() 
            if k.lower() not in exclude_headers
        }

        try:
            hclient: httpx.AsyncClient = request.app.state.hclient
            resp = await hclient.request(
                method=request.method,
                url=f"{upstream_base_url}/{path}",
                content=body,
                headers=headers,
            )
        except httpx.TimeoutException as e:
            logger.error(f"Upstream timeout during passthrough to /{path}: {str(e)}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": f"Upstream timeout: {str(e)}",
                        "type": "proxy_error",
                        "code": 504,
                    }
                }
            )
        except httpx.ConnectError as e:
            logger.error(f"Upstream connection error during passthrough to /{path}: {str(e)}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Upstream connection error: {str(e)}",
                        "type": "proxy_error",
                        "code": 502,
                    }
                }
            )
        except Exception as e:
            logger.error(f"Passthrough failed to /{path}: {str(e)}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Passthrough request failed: {str(e)}",
                        "type": "proxy_error",
                        "param": None,
                        "code": 502,
                    }
                }
            )
        
        # Log upstream errors (non-2xx)
        if resp.status_code >= 400:
            logger.error(f"Upstream returned error {resp.status_code} for /{path}: {resp.text[:200]}")

        # Build headers excluding hop-by-hop ones for the response
        exclude_resp_headers = {
            "content-length", "transfer-encoding", "content-encoding",
            "connection", "keep-alive", "proxy-authenticate", 
            "proxy-authorization", "te", "trailers", "upgrade"
        }
        headers = {
            k: v for k, v in resp.headers.items() 
            if k.lower() not in exclude_resp_headers
        }
        
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
            headers=headers,
        )

    return app
