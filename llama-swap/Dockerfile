# ==========================================
# Stage 1: Build beellama & llama-swap
# ==========================================
FROM rocm/dev-ubuntu-24.04:7.2.3-complete AS builder

# Set non-interactive apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies including Go for llama-swap
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    golang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. Clone and build BeeLlama with ROCm
RUN git clone https://github.com/Anbeeld/beellama.cpp.git && \
    cd beellama.cpp && \
    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=${GPU_ARCH} -DGGML_NATIVE=ON && \
    cmake --build build -j$(nproc)

# 2. Clone and build llama-swap (mostlygeek/llama-swap)
RUN git clone https://github.com/mostlygeek/llama-swap.git && \
    cd llama-swap && \
    go build -o /build/llama-swap-bin main.go

# ==========================================
# Stage 2: Final Runtime Image
# ==========================================
FROM rocm/dev-ubuntu-24.04:7.2.3-complete

WORKDIR /app

# Copy compiled binaries from the builder stage
# (Assuming beellama-server is generated in build/bin or build directory)
COPY --from=builder /build/beellama.cpp/build/bin/beellama-server /usr/local/bin/beellama-server
COPY --from=builder /build/llama-swap-bin /usr/local/bin/llama-swap

RUN mkdir -p /root/.cache/huggingface

EXPOSE 8080

# Start llama-swap pointing to the baked-in config
ENTRYPOINT ["llama-swap", "-config", "/app/llama-swap.yaml", "-port", "8080"]