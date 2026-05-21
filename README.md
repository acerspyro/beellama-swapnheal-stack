# Beellama + llama-swap + Unsloth Tool Healing Stack

This stack is a deployment framework designed to run multiple Large Language Models (LLMs) on a single GPU by swapping them in and out of VRAM. This allows you to use a large, capable model for reasoning and a smaller, faster model for tool use without needing enough memory to hold both simultaneously.

You can use whatever GPU you want, but the base config was written with ROCm in mind. The stack is designed to be flexible and can be adapted to different hardware configurations.

Make sure to build beellama with the correct flags for your system.

## Features

- Model Swapping: Uses llama-swap to manage VRAM by loading the appropriate model for the current task.
- Tool Healing Proxy: Automatically corrects LLM output formatting to ensure tool calls work reliably.
- Unified Tool Access: Connects to various external tools and data sources via the Model Context Protocol (MCP).
- AMD GPU Support: Optimized for ROCm and Docker.
- Web Interface: Includes Open WebUI for a familiar chat experience.

## System Architecture

The project consists of several services managed by Docker Compose:

1. llama-swap: The core service that manages model lifecycles.
   - Logic Model (qwen-logic): A Qwen-35B model used for complex reasoning and planning.
   - Tools Model (gemma-tools): A Gemma-26B model used for fast tool execution.
2. tool-healing: A FastAPI proxy that intercepts model output, fixes tool call formatting, and ensures compatibility with standard APIs.
3. mcp-proxy: A gateway that combines multiple MCP servers into a single interface.
4. open-webui: The user interface for interacting with the models.
5. nginx: Handles network routing and provides external access.

## Setup

### Prerequisites

- Docker and Docker Compose.
- Python 3.11 or newer (required for local development only).

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Add your API keys to the .env file (for example, TAVILY_API_KEY for search tools).

### Running the System

Start the services using Docker Compose:

```bash
docker-compose up --build -d
```

The services will be available at:
- Open WebUI: http://localhost (via NGINX) or http://localhost:8080 (direct).
- MCP Proxy: http://localhost:3000.
- Tool-Healing Proxy: http://localhost:8081.

## Project Structure

- tool-healing/: Source code and tests for the tool-healing proxy.
- llama-swap/: Configuration and Dockerfile for the model-swapping engine.
- mcp-proxy/: Configuration for MCP servers.
- nginx/: Reverse proxy configuration.

## License

This project is licensed under the AGPL-3.0 License. See the LICENSE.md file for details.
