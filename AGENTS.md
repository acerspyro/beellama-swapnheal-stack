# Agent Architecture: Multi-Model "Swap" Strategy

This project leverages a dual-model setup managed by `llama-swap`. This allows for high-performance reasoning and efficient tool execution within limited VRAM by swapping models on demand.

## 1. Model Roles & Specializations

As configured in `llama-swap/config.yaml`, the system is divided into two distinct logical layers:

| Model Role | Model | Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **Logic/Brain** | `qwen-logic` | `reasoning: on` | Planning, architectural decisions, complex logic, and code review. |
| **Tools/Hands** | `gemma-tools` | `reasoning: off` | MCP tool execution, web search, file operations, and fast iterative tasks. |

## 2. MCP Integration Strategy

### Recommended: MCP Proxy (Gateway)
For this project, we recommend running an **MCP Proxy** (e.g., [mcp-gateway](https://github.com/lastmile-ai/mcp-gateway)) as a separate service in `docker-compose.yml`.

**Why a Proxy?**
*   **Decoupling**: Your agent (OpenCode) doesn't need to manage MCP process lifecycles.
*   **Shared Tools**: Both Open WebUI and OpenCode can access the same set of tools (search, git, filesystem).
*   **Protocol Translation**: The proxy converts MCP tools into standard OpenAI Function Calling, which `gemma-tools` is optimized to handle.

### Configuration Example
Add this to your `docker-compose.yml` to enable a centralized MCP gateway:

```yaml
  mcp-gateway:
    image: ghcr.io/lastmile-ai/mcp-gateway:latest
    environment:
      - LLM_PROVIDER=openai
      - OPENAI_API_BASE_URL=http://llama-swap:8000/v1
      - OPENAI_API_KEY=llama-swap
    volumes:
      - ./mcp-config.json:/app/config.json
    ports:
      - "3000:3000"
```

## 3. Web & Docs Search

To ensure your agent actually performs searches, you must provide it with the right tools and the right "nudge."

### Required MCP Servers
1.  **Web Search**: Use the `brave-search` or `google-search` MCP server.
2.  **Web Fetch**: Use the `fetch` MCP server to read documentation pages once URLs are found.
3.  **Local Docs**: Use the `filesystem` MCP server to index and search local documentation in this repo.

### System Prompting (The "Gunpoint" Fix)
Modern models (especially Qwen/Gemma) often try to answer from memory to be "helpful." You must explicitly instruct them to use tools. Update your OpenCode system prompt:

```markdown
# Operational Mandate
- NEVER guess about technical details. If you aren't 100% sure, use the `google_search` or `brave_search` tool.
- When researching a new library or API, always `fetch` the official documentation URL.
- Use `qwen-logic` for the initial plan, then switch to `gemma-tools` for execution.
```

## 4. Python Environment

To ensure stability and avoid conflicts with the host system, this project uses a dedicated virtual environment.

- **Virtual Environment**: Always use the `.venv` located in the project root.
- **Strict Requirement**: NEVER use `--break-system-packages` with `pip`. If a package is missing, install it within the `.venv`.
- **Activation**: Ensure your execution environment points to `.venv/bin/python` or `.venv/bin/pip`.

## 5. Maintenance: Tool Healing Feature Parity

To guarantee feature parity with the official Unsloth implementation, this project depends on the full `unsloth` package.

- **Dependency**: The `unsloth` package is included in `tool-healing/requirements.txt`.
- **Logic**: Tool healing logic is imported from `studio.backend.core.tool_healing`.
