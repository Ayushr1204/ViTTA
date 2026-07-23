# ============================================================================
# ViTTA — GPU-enabled Docker Image
# Base: NVIDIA CUDA 12.4 runtime (matches PyTorch cu124 wheels)
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python packages into a venv
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System deps needed to build Python packages (e.g. OpenCV, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create a virtual-env so we can cleanly copy it to the runtime stage
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch + CUDA wheels first (large layer, cached separately)
RUN pip install --no-cache-dir \
        torch==2.6.0+cu124 \
        torchvision==0.21.0+cu124 \
        torchaudio==2.6.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# Copy only dependency manifests first for better layer caching
COPY pyproject.toml README.md ./
COPY vitta/ vitta/

# Install the project (pulls remaining deps: fastapi, ultralytics, etc.)
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to run the app
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system deps for OpenCV and general operation
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Make sure Python finds the right interpreter
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# ── Application code ──
COPY web_app.py .
COPY main.py .
COPY vitta/ vitta/
COPY static/ static/
COPY templates/ templates/

# ── Model weights (baked in for zero-config deployment) ──
COPY IDD_YOLO/best.pt IDD_YOLO/best.pt
COPY yolo26n.pt yolo26n.pt

# ── Output directory ──
RUN mkdir -p /app/output

# ── Expose the web app port ──
EXPOSE 8000

# ── Health check (optional — helps Docker Desktop show container status) ──
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Default command ──
CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
