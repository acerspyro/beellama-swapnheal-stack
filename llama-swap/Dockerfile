# ==========================================
# Stage 1: Build beellama & llama-swap
# ==========================================
FROM rocm/dev-ubuntu-24.04:7.2.3-complete AS builder

ARG TARGETOS
ARG TARGETARCH

# Set non-interactive apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install NodeJS 24
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash -
RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*


# Install build dependencies including Go for llama-swap
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    golang \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. Clone BeeLlama and llama-swap
RUN git clone --depth 1 --branch v0.1.2 https://github.com/Anbeeld/beellama.cpp.git
RUN git clone --depth 1 --branch v216 https://github.com/mostlygeek/llama-swap.git


# 2. Build BeeLlama with ROCm
RUN cmake -S beellama.cpp -B beellama.cpp/build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=${GPU_ARCH} \
    -DLLAMA_OPENSSL=ON \
    -DGGML_NATIVE=ON
RUN cmake --build beellama.cpp/build -j$(nproc)

# 3. Build llama-swap (mostlygeek/llama-swap)
RUN cd llama-swap && make all
RUN cp "/build/llama-swap/build/llama-swap-${TARGETOS}-${TARGETARCH}" /build/llama-swap-bin

# ==========================================
# Stage 2: Final Runtime Image
# ==========================================
FROM rocm/dev-ubuntu-24.04:7.2.3-complete

WORKDIR /app

# Copy compiled binaries from the builder stage
# (Assuming beellama-server is generated in build/bin or build directory)
COPY --from=builder /build/beellama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/beellama.cpp/build/bin/lib*.so* /usr/local/lib/
COPY --from=builder /build/llama-swap-bin /usr/local/bin/llama-swap
RUN ldconfig

RUN mkdir -p /root/.cache/huggingface

EXPOSE 8080

# Start llama-swap pointing to the baked-in config
ENTRYPOINT ["llama-swap", "-config", "/app/llama-swap.yaml", "-listen", "0.0.0.0:8080"]
