import pytest
import logging
from unittest.mock import MagicMock, patch
from tool_healing.streaming import StreamHealer, heal_chat_completion_stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta as Delta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

@pytest.fixture
def mock_tool_healing():
    with patch("tool_healing.streaming.strip_tool_call_markup") as mock_strip, \
         patch("tool_healing.streaming.parse_tool_calls_from_text") as mock_parse:
        yield mock_strip, mock_parse

def test_stream_healer_init():
    healer = StreamHealer()
    assert healer.accum == ""
    assert healer.emitted_len == 0
    assert not healer.is_draining
    assert not healer.is_streaming
    assert healer.buffer == ""

def test_stream_healer_process_plain_text(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    
    healer = StreamHealer()
    assert healer.process("Hello") == "Hello"
    assert healer.is_streaming
    assert healer.accum == "Hello"
    assert healer.emitted_len == 5

def test_stream_healer_process_tool_call_start(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    healer = StreamHealer()
    
    # "<tool" is a prefix of "<tool_call>"
    assert healer.process("<tool") is None
    assert not healer.is_streaming
    assert not healer.is_draining
    
    # Completing the tag
    assert healer.process("_call>") is None
    assert healer.is_draining
    assert not healer.is_streaming
    
    # Further tokens when draining
    assert healer.process("something") is None

def test_stream_healer_process_whitespace_then_text(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    healer = StreamHealer()
    
    # leading whitespace
    assert healer.process("  ") == "  "
    assert healer.is_streaming

def test_stream_healer_process_max_buffer(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    healer = StreamHealer(max_buffer=5)
    
    # "Long string" doesn't start with any signal and exceeds max_buffer prefix check
    assert healer.process("Long ") == "Long "
    assert healer.is_streaming

def test_stream_healer_finalize_no_tool(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    healer = StreamHealer()
    healer.process("Hello")
    
    tcs, final_text = healer.finalize()
    assert tcs == []
    assert final_text is None # already emitted

def test_stream_healer_finalize_with_remaining(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    healer = StreamHealer()
    healer.accum = "Hello world"
    healer.emitted_len = 5
    
    tcs, final_text = healer.finalize()
    assert tcs == []
    assert final_text == " world"

def test_stream_healer_finalize_draining_with_tools(mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    healer = StreamHealer()
    healer.process("<tool_call>")
    assert healer.is_draining
    
    mock_parse.return_value = [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]
    
    tcs, final_text = healer.finalize()
    assert tcs == [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]
    assert final_text is None

def test_stream_healer_finalize_draining_no_tools_fallback(mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    healer = StreamHealer()
    healer.process("<tool_call>")
    mock_parse.return_value = []
    mock_strip.side_effect = lambda x, final=False: x
    
    tcs, final_text = healer.finalize()
    assert tcs == []
    assert final_text == "<tool_call>"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_plain_text(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: x
    
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="Hello"))
        ])
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=" world"))
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    # 2 original chunks + 1 final chunk with finish_reason
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " world"
    assert chunks[2].choices[0].finish_reason == "stop"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_tool_xml(mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: ""
    mock_parse.return_value = [{"id": "tc1", "function": {"name": "func", "arguments": "{}"}}]
    
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="<tool_call>"))
        ])
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content='{"name": "func"}'))
        ])
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="</tool_call>"))
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    # No content chunks should have been yielded because they were swallowed by healer
    # 1 tool call chunk + 1 final chunk
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.tool_calls[0].id == "tc1"
    assert chunks[1].choices[0].finish_reason == "stop"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_existing_tool_calls(mock_tool_healing):
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(tool_calls=[ChoiceDeltaToolCall(index=0, id="tc1", type="function")]))
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    assert len(chunks) == 2 # original + final stop
    assert chunks[0].choices[0].delta.tool_calls[0].id == "tc1"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_no_choices(mock_tool_healing):
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert chunks[0].choices == []

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_empty_delta(mock_tool_healing):
    async def source():
        # Empty delta but not finished
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=None))
        ])
        # Delta with finish reason
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=None), finish_reason="stop")
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    # Should get 2 chunks (one from finalize with finish_reason, one stop reason)
    # Actually, the code says:
    # if not token:
    #     if not choice.finish_reason:
    #         yield chunk
    #     continue
    # 1. First chunk has None token and no finish_reason -> yielded.
    # 2. Second chunk has None token but has finish_reason -> not yielded.
    # 3. Finalize yields a chunk from final_text (which is None here) or just continues.
    # 4. Finalize yields a stop chunk.
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content is None
    assert chunks[1].choices[0].finish_reason == "stop"

def test_stream_healer_emit_diff_none(mock_tool_healing):
    mock_strip, _ = mock_tool_healing
    healer = StreamHealer()
    healer.emitted_len = 10
    mock_strip.return_value = "short" # len < emitted_len
    assert healer._emit_diff() is None

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_final_text(mock_tool_healing):
    mock_strip, mock_parse = mock_tool_healing
    # First call (in process) returns "", second call (in finalize) returns "recovered"
    mock_strip.side_effect = ["", "recovered"]
    mock_parse.return_value = []
    
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="swallowed"))
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    # 1. process("swallowed") calls mock_strip -> returns "" -> diff is "", not yielded
    # 2. finalize() calls mock_strip -> returns "recovered" -> hits line 124
    # 3. yielded chunk with "recovered"
    # 4. yielded chunk with finish_reason "stop"
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "recovered"
    assert chunks[1].choices[0].finish_reason == "stop"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_no_redundant_stop():
    # Scenario: Upstream already provides a stop signal, and we don't add any new content.
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="Hello"), finish_reason=None)
        ])
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=None), finish_reason="stop")
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].finish_reason == "stop"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_stop_reason_preserved():
    # Scenario: Upstream provides a different stop reason (e.g. length)
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="Hello"), finish_reason=None)
        ])
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=None), finish_reason="length")
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[1].choices[0].finish_reason == "length"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_force_stop_on_new_content():
    # Scenario: Healer buffers content that looks like a tool call, but it's not.
    # It recovers this content in finalize().
    # It must yield the recovered content AND then a stop chunk.
    
    async def source():
        # "<tool" will be buffered by StreamHealer
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="<tool"), finish_reason=None)
        ])
        # Upstream finishes
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content=None), finish_reason="stop")
        ])

    chunks = []
    async for chunk in heal_chat_completion_stream(source()):
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "<tool"
    assert chunks[1].choices[0].finish_reason == "stop"

@pytest.mark.asyncio
async def test_heal_chat_completion_stream_logs_tool_call(mock_tool_healing, caplog):
    mock_strip, mock_parse = mock_tool_healing
    mock_strip.side_effect = lambda x, final=False: ""
    mock_parse.return_value = [{"id": "tc1", "function": {"name": "func", "arguments": "{}"}}]
    
    async def source():
        yield ChatCompletionChunk(id="1", model="m", created=0, object="chat.completion.chunk", choices=[
            Choice(index=0, delta=Delta(content="<tool_call>"))
        ])

    with caplog.at_level(logging.INFO):
        async for _ in heal_chat_completion_stream(source()):
            pass
    
    assert "Detected 1 tool calls in stream" in caplog.text
