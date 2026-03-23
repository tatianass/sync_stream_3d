# Use the same base you provided
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS uv_base

# Install only essential system libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# 1. LAYER DEPENDENCIES: Copy ONLY files needed for install
COPY pyproject.toml uv.lock ./

# 2. CACHE LIBS: Install dependencies without the project code
# This layer is now cached even if you change your Python scripts!
RUN uv sync --frozen --no-install-project --no-dev

# 3. COPY SOURCE: Only copy the code needed for Task 2 & 3
COPY syncstream/ ./syncstream/
COPY main.py ./

# 4. FINAL SYNC: Install the project itself
RUN uv sync --frozen --no-dev

# Ensure output directory exists as per Task 2 requirements [cite: 15]
RUN mkdir -p output

CMD ["uv", "run", "python", "main.py"]
