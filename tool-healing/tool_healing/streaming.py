from __future__ import annotations

"""
Streaming implementation for Tool Healing.

This module provides a state machine and an async generator to handle
streaming chat completions. It detects tool call XML markers in the stream,
buffers or strips them, and converts them into structured tool call chunks
when the stream finishes.
"""

from typing import AsyncGenerator, Optional, AsyncIterator
import logging
from studio.backend.core.tool_healing import strip_tool_call_markup, parse_tool_calls_from_text
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta as Delta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction
)

DEFAULT_TOOL_SIGNALS = ("<tool_call>", "<function=")
logger = logging.getLogger(__name__)

class StreamHealer:
    """
    Encapsulates the state machine for detecting and stripping tool XML from a stream.

    The healer tracks the incoming tokens to decide whether the assistant is
    starting a tool call (XML format) or just providing regular text content.
    If a tool call is detected, it "drains" the content (stops emitting text)
    until the stream ends, at which point it parses the full XML.
    """

    def __init__(self, signals: tuple[str, ...] = DEFAULT_TOOL_SIGNALS, max_buffer: int = 32):
        """
        Initialize the StreamHealer.

        Args:
            signals: Tuple of strings that signal the start of a tool call.
            max_buffer: Maximum number of characters to buffer while checking
                if the start of the stream matches a tool call signal.
        """
        self.accum = ""
        self.emitted_len = 0
        self.is_draining = False
        self.is_streaming = False
        self.buffer = ""
        self.signals = signals
        self.max_buffer = max_buffer

    def process(self, token: str) -> str | None:
        """
        Process a new token from the stream.

        Args:
            token: The next text chunk from the assistant's content.

        Returns:
            The text diff that should be emitted to the client, or None if
            the token is being buffered or drained (part of a tool call).
        """
        self.accum += token
        if self.is_draining: return None
        if self.is_streaming: return self._emit_diff()
        
        self.buffer += token
        stripped = self.buffer.lstrip()
        if not stripped:
            self.is_streaming = True
            return self._emit_diff()
        
        if any(stripped.startswith(sig) for sig in self.signals):
            self.is_draining = True
            return None
        if any(sig.startswith(stripped) for sig in self.signals) and len(stripped) < self.max_buffer:
            return None
            
        self.is_streaming = True
        return self._emit_diff()

    def _emit_diff(self) -> str | None:
        """
        Calculate and return the clean text diff since the last emission.

        Uses `strip_tool_call_markup` to ensure any partial XML tags are not
        emitted prematurely.
        """
        cleaned = strip_tool_call_markup(self.accum)
        if len(cleaned) > self.emitted_len:
            diff = cleaned[self.emitted_len:]
            self.emitted_len = len(cleaned)
            return diff
        return None

    def finalize(self) -> tuple[list[dict], str | None]:
        """
        Finalize the stream and extract any tool calls or remaining text.

        Returns:
            A tuple of (parsed_tool_calls, remaining_content).
            `parsed_tool_calls` is a list of tool call dictionaries.
            `remaining_content` is any text that was buffered but not emitted.
        """
        if self.is_draining and any(sig in self.accum for sig in self.signals):
            tcs = parse_tool_calls_from_text(self.accum)
            if tcs: return tcs, None
        
        # At end of stream, use final=True to clean up any trailing incomplete tags
        # but the test test_streaming_draining_no_tool_calls_fallback expects 
        # that we don't lose content. Unsloth's strip_tool_call_markup with final=True
        # is very aggressive.
        cleaned = strip_tool_call_markup(self.accum, final=False)
        return [], cleaned[self.emitted_len:] if len(cleaned) > self.emitted_len else None

async def heal_chat_completion_stream(
    stream: AsyncIterator[ChatCompletionChunk],
    signals: tuple[str, ...] = DEFAULT_TOOL_SIGNALS
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """
    Applies tool healing to a stream of ChatCompletionChunk objects.

    This async generator wraps an upstream OpenAI-compatible stream. It uses
    `StreamHealer` to filter out tool XML markup from the `content` deltas
    and instead yields new chunks with structured `tool_calls` when a tool
    call is detected at the end of the stream.

    Args:
        stream: The upstream async iterator of chunks.
        signals: Tuple of strings that signal the start of a tool call.

    Yields:
        ChatCompletionChunk objects, possibly modified to strip XML or
        include structured tool calls.
    """
    healer = StreamHealer(signals=signals)
    last_chunk: Optional[ChatCompletionChunk] = None
    last_chunk_yielded = False

    async for chunk in stream:
        if not chunk.choices:
            yield chunk
            continue
            
        last_chunk = chunk
        last_chunk_yielded = False
        choice = chunk.choices[0]
        delta = choice.delta
        
        if delta.tool_calls:
            healer.is_streaming = True
            yield chunk
            last_chunk_yielded = True
            continue

        token = delta.content
        if not token:
            if not choice.finish_reason:
                yield chunk
                last_chunk_yielded = True
            continue

        diff = healer.process(token)
        if diff:
            delta.content = diff
            yield chunk
            last_chunk_yielded = True

    # Finalize
    tcs, final_text = healer.finalize()
    if tcs:
        logger.info(f"Detected {len(tcs)} tool calls in stream")
    if last_chunk and last_chunk.choices:
        base_id = last_chunk.id
        base_model = last_chunk.model
        base_created = last_chunk.created
        
        new_chunk_yielded = False
        if tcs:
            for i, tc in enumerate(tcs):
                yield ChatCompletionChunk(
                    id=base_id,
                    model=base_model,
                    created=base_created,
                    object="chat.completion.chunk",
                    choices=[Choice(index=0, delta=Delta(tool_calls=[
                        ChoiceDeltaToolCall(
                            index=i,
                            id=tc["id"],
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"]
                            )
                        )
                    ]))]
                )
            new_chunk_yielded = True
        elif final_text:
            yield ChatCompletionChunk(
                id=base_id,
                model=base_model,
                created=base_created,
                object="chat.completion.chunk",
                choices=[Choice(index=0, delta=Delta(content=final_text))]
            )
            new_chunk_yielded = True
            
        # Avoid redundant stop signal if last chunk already had one and no new content was added
        upstream_finish_reason = last_chunk.choices[0].finish_reason
        if new_chunk_yielded or not (last_chunk_yielded and upstream_finish_reason):
            yield ChatCompletionChunk(
                id=base_id,
                model=base_model,
                created=base_created,
                object="chat.completion.chunk",
                choices=[Choice(index=0, delta=Delta(), finish_reason=upstream_finish_reason or "stop")]
            )
