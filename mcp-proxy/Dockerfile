FROM astral/uv:0.11-debian

# Install NodeJS 24
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash -
RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENTRYPOINT ["uvx", "mcp-proxy", \
  "--host", "0.0.0.0", \
  "--port", "3000", \
  "--pass-environment", \
  "--named-server-config", "/app/servers.json" \
]
