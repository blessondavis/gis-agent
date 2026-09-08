# Headless QGIS + GDAL come from the official QGIS image, which ships the
# qgis_process CLI this project drives. That is the supported headless entry
# point -- the popular qgis_mcp servers instead need a QGIS Desktop GUI with a
# human starting a socket plugin, which cannot run in a container.
FROM qgis/qgis:ltr

# cpu  -> small image, SAM 3 runs on CPU (slow but works anywhere)
# cu128 -> CUDA 12.8 wheels, required for RTX 50-series (Blackwell, sm_120)
ARG TORCH_VARIANT=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    GISAGENT_DATA_DIR=/data \
    HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv resolves and installs far faster than pip, which matters for torch
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install torch from the chosen index FIRST so the later resolve treats it as
# already satisfied and does not pull a different build.
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# Dependency layer: cached unless the manifest changes.
COPY pyproject.toml README.md ./
RUN mkdir -p src/gisagent && touch src/gisagent/__init__.py \
    && uv pip install --no-cache .

COPY src ./src
COPY web ./web
RUN uv pip install --no-cache --no-deps .

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "gisagent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
